use serde_json::Value;
use std::format;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::client::{LLMClient, LLMConfig};
use crate::error::{AgentError, Result};
use crate::memory::{AgentMemory, Message, Role};
use crate::pool::ToolPool;

pub struct Agent {
    pub name: String,
    pub system_prompt: String,
    client: LLMClient,
    tools: Option<Arc<RwLock<ToolPool>>>,
    pub memory: Arc<dyn AgentMemory>, 
}

impl Agent {
    pub fn new(
        name: &str, 
        system_prompt: &str, 
        config: LLMConfig, 
        memory: Arc<dyn AgentMemory>
    ) -> Self {
        Self {
            name: name.to_string(),
            system_prompt: system_prompt.to_string(),
            client: LLMClient::new(config),
            tools: None,
            memory,
        }
    }

    pub fn set_tools(&mut self, pool: Arc<RwLock<ToolPool>>) {
        self.tools = Some(pool);
    }

    pub fn register_tool(&mut self, tool: Box<dyn crate::tools::Tool>) {
        if self.tools.is_none() {
            self.tools = Some(Arc::new(RwLock::new(crate::pool::ToolPool::new())));
        }
        
        if let Some(pool_arc) = &self.tools {
            if let Ok(mut pool) = pool_arc.try_write() {
                pool.register_tool(tool);
            } else {
                eprintln!(
                    "[SYSTEM WARNING] Agent tools are locked (a background task might be running). Cannot add new tool: {}", 
                    tool.name()
                );
            }
        }
    }

    pub async fn run_with_stream(
        &self,
        session_id: &str, 
        user_input: &str,
        callback: Option<Box<dyn Fn(String) + Send + Sync>>,
        max_steps: usize
    ) -> Result<String> {
        
        let current_history = self.memory.get_history(session_id).await?;
        
        let has_system_prompt = current_history.iter().any(|m| m.role == Role::System);
        if !has_system_prompt && !self.system_prompt.is_empty() {
            self.memory.add_message(session_id, Message::system(&self.system_prompt)).await?;
        }

        self.memory.add_message(session_id, Message::user(user_input)).await?;

        let schemas = if let Some(pool) = &self.tools {
            Some(pool.read().await.get_tool_schemas())
        } else {
            None
        };
        let mut consecutive_errors = 0; 

        for _step in 0..max_steps {
            let current_messages = self.memory.get_history(session_id).await?;

            let response_msg = match self
                .client
                .send_stream_request(current_messages, schemas.clone(), &callback)
                .await 
            {
                Ok(msg) => msg,
                Err(e) => {
                    let err_str = e.to_string().to_lowercase();
                    
                    let is_tool_format_err = err_str.contains("tool_use_failed") 
                        || (err_str.contains("invalid") && err_str.contains("argument"))
                        || err_str.contains("validation failed")
                        || err_str.contains("invalid_request_error");
                        
                    let is_critical_api_err = err_str.contains("api_key") 
                        || err_str.contains("unauthorized")
                        || err_str.contains("insufficient_quota")
                        || err_str.contains("rate_limit");

                    if is_tool_format_err && !is_critical_api_err {
                        
                        println!("[API GUARDRAIL] LLM Provider rejected the malformed tool call! Self-Healing activated.");
                        
                        self.memory.add_message(session_id, Message::user(
                            &format!("SYSTEM WARNING: Your last tool call was rejected by the API server! Details: {}. Please fix the parameter types (e.g., use integer instead of string) and try again.", err_str)
                        )).await?;
                        
                        consecutive_errors += 1;
                        if consecutive_errors >= 3 {
                            return Err(AgentError::InternalError(
                                "Critical Error: LLM hit API validation errors 3 times in a row. Death loop prevented.".to_string()
                            ));
                        }
                        
                        continue; 
                    } else {
                        return Err(e);
                    }
                }
            };

            self.memory.add_message(session_id, response_msg.clone()).await?;

            if let Some(tool_calls) = response_msg.tool_calls {
                let tools_arc = match &self.tools {
                    Some(t) => t.clone(),
                    None => {
                        return Err(AgentError::InternalError(
                            "Agent called a tool but the tool pool is not defined.".to_string(),
                        ));
                    }
                };

                let calls_array = tool_calls.as_array().cloned().unwrap_or_default();
                let mut tasks = Vec::new();

                for tc in calls_array {
                    let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                    let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}").to_string();
                    let call_id = tc["id"].as_str().unwrap_or_default().to_string();
                    
                    let pool_clone = tools_arc.clone();

                    let task = tokio::spawn(async move {
                        let args_val: Value = match serde_json::from_str(&args_str) {
                            Ok(val) => val,
                            Err(e) => {
                                return format!("ERROR: The JSON arguments you sent could not be parsed. Syntax error: {}. Please fix the JSON format and try again.", e);
                            }
                        };
                        
                        if name.is_empty() {
                            "ERROR: LLM made a call with an empty/invalid tool name.".to_string()
                        } else {
                            let pool = pool_clone.read().await;
                            match pool.execute_tool(&name, args_val).await {
                                Ok(output) => output,
                                Err(e) => format!("ERROR: Tool execution failed. Details: {}", e),
                            }
                        }
                    });
                    
                    tasks.push((call_id, task));
                }

                let mut has_error_in_this_step = false;

                for (call_id, handle) in tasks {
                    let tool_result = match handle.await {
                        Ok(res) => res,
                        Err(join_err) => {
                            if join_err.is_panic() {
                                "ERROR: System panicked during tool execution (Thread Panic).".to_string()
                            } else {
                                "ERROR: Thread was cancelled during tool execution (Task Cancelled).".to_string()
                            }
                        }
                    };

                    if tool_result.starts_with("ERROR:") {
                        has_error_in_this_step = true;
                    }

                    self.memory.add_message(session_id, Message {
                        role: Role::Tool,
                        content: Some(tool_result),
                        tool_calls: None,
                        tool_call_id: Some(call_id),
                    }).await?;
                }

                if has_error_in_this_step {
                    consecutive_errors += 1;
                    if consecutive_errors >= 3 {
                        return Err(AgentError::InternalError(
                            "Critical Error: LLM called an invalid tool 3 times in a row. Death loop prevented.".to_string()
                        ));
                    }
                } else {
                    consecutive_errors = 0; 
                }
                
                continue;
            } else {
                return Ok(response_msg.content.unwrap_or_default());
            }
        }

        Err(AgentError::InternalError(
            format!("Agent reached the step limit (max_steps={}).", max_steps)
        ))
    }
}