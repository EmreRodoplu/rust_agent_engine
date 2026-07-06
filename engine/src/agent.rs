use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::client::{LLMClient, LLMConfig};
use crate::error::{AgentError, Result};
use crate::memory::{ConversationHistory, Message, Role};
use crate::pool::ToolPool;

pub struct Agent {
    pub name: String,
    pub system_prompt: String,
    client: LLMClient,
    tools: Option<Arc<ToolPool>>,
    pub history: Mutex<ConversationHistory>, 
}

impl Agent {
    pub fn new(name: &str, system_prompt: &str, config: LLMConfig) -> Self {
        let mut history = ConversationHistory::new();
        history.add_system_prompt(system_prompt);

        Self {
            name: name.to_string(),
            system_prompt: system_prompt.to_string(),
            client: LLMClient::new(config),
            tools: None,
            history: Mutex::new(history), 
        }
    }

    pub fn set_tools(&mut self, pool: Arc<ToolPool>) {
        self.tools = Some(pool);
    }

    pub fn register_tool(&mut self, tool: Box<dyn crate::tools::Tool>) {
        if self.tools.is_none() {
            self.tools = Some(Arc::new(crate::pool::ToolPool::new()));
        }
        if let Some(pool_arc) = &mut self.tools {
            // Arc içindeki veriyi değiştirmek için güvenli erişim
            if let Some(pool) = Arc::get_mut(pool_arc) {
                pool.register_tool(tool);
            }
        }
    }

    pub async fn run_with_stream(
        &self,
        user_input: &str,
        callback: Option<Box<dyn Fn(String) + Send + Sync>>,
    ) -> Result<String> {
        {
            
            let mut history = self.history.lock().await; 
            history.add_user_message(user_input);
        }

        let schemas = self.tools.as_ref().map(|p| p.get_tool_schemas());
        let max_steps = 15; 
        let mut consecutive_errors = 0; 

        for _step in 0..max_steps {
            let current_messages = {
                let h = self.history.lock().await;
                h.messages.clone()
            };

            let response_msg = self
                .client
                .send_stream_request(current_messages, schemas.clone(), &callback)
                .await?;

            {
                let mut history = self.history.lock().await;
                history.add_message(response_msg.clone());
            }

            if let Some(tool_calls) = response_msg.tool_calls {
                let tools_arc = match &self.tools {
                    Some(t) => t.clone(),
                    None => {
                        return Err(AgentError::InternalError(
                            "Ajan tool çağırdı ama havuz tanımlı değil.".to_string(),
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
                        let args_val: Value = serde_json::from_str(&args_str).unwrap_or(serde_json::json!({}));
                        
                        let tool_result = if name.is_empty() {
                            "HATA: LLM boş/geçersiz bir tool adıyla çağrı yaptı.".to_string()
                        } else {
                            match pool_clone.execute_tool(&name, args_val).await {
                                Ok(output) => output,
                                Err(e) => format!("HATA: Araç çalıştırılamadı. Detay: {}", e),
                            }
                        };
                        
                        (call_id, tool_result)
                    });
                    
                    tasks.push(task);
                }

                let results = futures::future::join_all(tasks).await;
                let mut has_error_in_this_step = false;

                let mut history = self.history.lock().await;
                for res in results {
                    if let Ok((call_id, tool_result)) = res {
                        if tool_result.starts_with("HATA:") {
                            has_error_in_this_step = true;
                        }

                        history.add_message(Message {
                            role: Role::Tool,
                            content: Some(tool_result),
                            tool_calls: None,
                            tool_call_id: Some(call_id),
                        });
                    }
                }

                if has_error_in_this_step {
                    consecutive_errors += 1;
                    if consecutive_errors >= 3 {
                        return Err(AgentError::InternalError(
                            "Kritik Hata: LLM üst üste 3 kez geçersiz araç çağırdı. Ölümcül döngü (Death Loop) engellendi.".to_string()
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
            format!("Ajan adım sınırına ulaştı (max_steps={}).", max_steps)
        ))
    }
}