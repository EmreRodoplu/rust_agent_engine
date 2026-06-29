use serde_json::Value;
use std::sync::Mutex;

use crate::client::{LLMClient, LLMConfig};
use crate::error::{AgentError, Result};
use crate::memory::{ConversationHistory, Message, Role};
use crate::pool::ToolPool;

pub struct Agent {
    pub name: String,
    pub system_prompt: String,
    client: LLMClient,
    tools: Option<ToolPool>,
    pub history: Mutex<ConversationHistory>
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
            history: Mutex::new(history)
        }
    }

    pub fn set_tools(&mut self, pool: ToolPool) {
        self.tools = Some(pool);
    }
    
    pub fn register_tool(&mut self, tool: Box<dyn crate::tools::Tool>) {
        if self.tools.is_none() {
            self.tools = Some(crate::pool::ToolPool::new());
        }
        if let Some(pool) = &mut self.tools {
            pool.register_tool(tool);
        }
    }

    pub async fn run_with_stream(
        &self, 
        user_input: &str, 
        callback: Option<Box<dyn Fn(String) + Send + Sync>>
    ) -> Result<String> {
        {
            let mut history = self.history.lock().unwrap();
            history.add_user_message(user_input);
        }

        let schemas = self.tools.as_ref().map(|p| p.get_tool_schemas());
        loop {
            let current_messages = {
                let h = self.history.lock().unwrap();
                h.messages.clone()
            };
            let response_msg = self.client.send_stream_request(current_messages, schemas.clone(), &callback).await?;
            {
                let mut history = self.history.lock().unwrap();
                history.add_message(response_msg.clone());
            }

            if let Some(tool_calls) = response_msg.tool_calls {
                if let Some(tools) = &self.tools {
                    for tc in tool_calls.as_array().unwrap() {
                        let f = &tc["function"];
                        let name = f["name"].as_str().unwrap();
                        let args_str = f["arguments"].as_str().unwrap_or("{}");

                        let args_val: serde_json::Value = serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
                        let result = tools.execute_tool(name, args_val).await?;
                        {
                            let mut history = self.history.lock().unwrap();
                            history.add_message(crate::memory::Message {
                                role: crate::memory::Role::User,
                                content: Some(result),
                                tool_calls: None,
                                tool_call_id: Some(tc["id"].as_str().unwrap().to_string()),
                            });
                        }
                    }
                }
                continue; 
            } else {
                return Ok(response_msg.content.unwrap_or_default());
            }
        }
    }

    pub async fn run(&self, user_input: &str, prune: Option<usize>) -> Result<String> {
        let mut history = self.history.lock().unwrap();
        history.prune(match prune {
            Some(n) => n,
            None => 20,
        });
        history.add_user_message(user_input);

        let max_steps = 10;

        for _step in 0..max_steps {
            let schemas = self.tools.as_ref().map(|p| p.get_tool_schemas());
            let response_json = self.client.send_request(&history, schemas).await?;
            let choice = response_json["choices"][0]["message"]
                .as_object()
                .ok_or_else(|| {
                    AgentError::InternalError("API geçersiz choice yapısı döndürdü".to_string())
                })?;

            let assistant_msg: Message = serde_json::from_value(Value::Object(choice.clone()))?;
            history.add_message(assistant_msg.clone());

            if let Some(tool_calls) = assistant_msg.tool_calls {
                if let Some(calls_array) = tool_calls.as_array() {
                    let pool = self.tools.as_ref().ok_or_else(|| {
                        AgentError::InternalError(
                            "Ajan araç çağırdı ama havuz tanımlı değil".to_string(),
                        )
                    })?;

                    for call in calls_array {
                        let call_id = call["id"].as_str().unwrap_or_default().to_string();
                        let func_name = call["function"]["name"].as_str().unwrap_or_default();
                        let func_args_str =
                            call["function"]["arguments"].as_str().unwrap_or_default();
                        let func_args: Value =
                            serde_json::from_str(func_args_str).unwrap_or(Value::Null);

                        let tool_result = match pool.execute_tool(func_name, func_args).await {
                            Ok(output) => output,
                            Err(e) => format!("HATA: Araç çalıştırılamadı. Detay: {}", e),
                        };

                        history.add_message(Message {
                            role: Role::Tool,
                            content: Some(tool_result),
                            tool_calls: None,
                            tool_call_id: Some(call_id),
                        });
                    }
                    continue; 
                }
            }

            if let Some(final_text) = assistant_msg.content {
                return Ok(final_text);
            }
        }

        Err(AgentError::InternalError(
            "Ajan adım sınırına ulaştı.".to_string(),
        ))
    }
}
