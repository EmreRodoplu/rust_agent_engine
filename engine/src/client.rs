use reqwest::Client;
use serde_json::{json, Value};
use futures::StreamExt;

use crate::error::{AgentError, Result};
use crate::memory::ConversationHistory;

#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

impl LLMConfig {
    pub fn new(model: &str, api_key: &str, provider: Option<&str>, base_url: Option<&str>) -> Self {
        let resolved_url = match base_url {
            Some(url) => url.to_string(),
            None => match provider {
                Some("gemini") => "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions".to_string(),
                Some("openai") => "https://api.openai.com/v1/chat/completions".to_string(),
                Some("ollama") => "http://localhost:11434/v1/chat/completions".to_string(),
                Some("anthropic") => "https://api.anthropic.com/v1/complete".to_string(),
                _ => "Gemini/OpenAI/Anthropic/Ollama uyumlu varsayılan URL giriniz".to_string(),
            },
        };

        Self {
            base_url: resolved_url,
            model: model.to_string(),
            api_key: api_key.to_string(),
        }
    }
}

pub struct LLMClient {
    http_client: Client,
    config: LLMConfig,
}

impl LLMClient {
    pub fn new(config: LLMConfig) -> Self {
        Self {
            http_client: Client::new(),
            config,
        }
    }

    pub async fn send_request(
        &self,
        history: &ConversationHistory,
        tools_schema: Option<Vec<Value>>,
    ) -> Result<Value> {
        let mut payload = json!({
            "model": self.config.model,
            "messages": history.messages,
        });

        if let Some(schemas) = tools_schema {
            if !schemas.is_empty() {
                payload["tools"] = json!(schemas);
                payload["tool_choice"] = json!("auto");
            }
        }

        let response = self
            .http_client
            .post(&self.config.base_url)
            .bearer_auth(&self.config.api_key)
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status_code = response.status();
            let err_msg = response.text().await.unwrap_or_default();
            return Err(AgentError::InternalError(format!(
                "API Hatası [{}]: {}",
                status_code, err_msg
            )));
        }

        let response_json: Value = response.json().await?;

        Ok(response_json)
    }
    
    pub async fn send_stream_request(
        &self,
        messages: Vec<crate::memory::Message>, 
        schemas: Option<Vec<serde_json::Value>>,
        callback: &Option<Box<dyn Fn(String) + Send + Sync>>,
    ) -> crate::error::Result<crate::memory::Message> {        
        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": true 
        });

        if let Some(s) = schemas {
            body.as_object_mut().unwrap().insert("tools".to_string(), serde_json::json!(s));
        }

        let response = self.http_client
            .post(&self.config.base_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::AgentError::InternalError(format!("Ağ hatası: {}", e)))?;

        
        let mut stream = response.bytes_stream();
        
        let mut full_text = String::new();
        let mut tool_id = String::new();
        let mut tool_name = String::new();
        let mut tool_args = String::new();
        let mut line_buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| crate::error::AgentError::InternalError(format!("Stream buffer error: {}", e)))?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            
            line_buffer.push_str(&chunk_str);

            while let Some(newline_pos) = line_buffer.find('\n') {
                let line = line_buffer[..newline_pos].to_string();
                line_buffer.drain(..=newline_pos);

                let trimmed = line.trim();
                if trimmed.starts_with("data: ") {
                    let data = &trimmed[6..];
                    if data == "[DONE]" { continue; } 
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        let delta = &json["choices"][0]["delta"];
                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                            full_text.push_str(content);
                            if let Some(cb) = callback {
                                cb(content.to_string());
                            }
                        }

                        if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                            if let Some(tc) = tool_calls.first() {
                                if let Some(id) = tc.get("id").and_then(|i| i.as_str()) { tool_id = id.to_string(); }
                                if let Some(func) = tc.get("function") {
                                    if let Some(name) = func.get("name").and_then(|n| n.as_str()) { tool_name.push_str(name); }
                                    if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) { tool_args.push_str(args); }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !tool_name.is_empty() {
            let tc_id = if tool_id.is_empty() { "call_local".to_string() } else { tool_id };
            let tc = serde_json::json!([{
                "id": tc_id,
                "type": "function",
                "function": { "name": tool_name, "arguments": tool_args }
            }]);

            Ok(crate::memory::Message {
                role: crate::memory::Role::Assistant,
                content: None,
                tool_calls: Some(tc),
                tool_call_id: None,
            })
        } else {
            Ok(crate::memory::Message {
                role: crate::memory::Role::Assistant,
                content: Some(full_text),
                tool_calls: None,
                tool_call_id: None,
            })
        }
    }
}