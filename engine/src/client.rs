use reqwest::Client;
use serde_json::{json, Value};
use futures::StreamExt;
use std::collections::BTreeMap;
use std::time::Duration;

use crate::error::{AgentError, Result};
use crate::memory::{ConversationHistory, Role};

#[derive(Clone)]
pub struct LLMConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>, 
    pub provider: String, 
}

impl std::fmt::Debug for LLMConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLMConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("provider", &self.provider)
            .field("api_key", &"***") 
            .finish()
    }
}

impl LLMConfig {
    pub fn new(
        model: &str,
        api_key: Option<&str>,
        provider: Option<&str>,
        base_url: Option<&str>,
    ) -> Result<Self> {
        let provider_str = provider.unwrap_or("openai").to_string();

        let resolved_url = match base_url {
            Some(url) => url.to_string(),
            None => match provider_str.as_str() {
                "gemini" => "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions".to_string(),
                "openai" => "https://api.openai.com/v1/chat/completions".to_string(),
                "ollama" => "http://localhost:11434/v1/chat/completions".to_string(),
                "anthropic" => "https://api.anthropic.com/v1/messages".to_string(),
                other => {
                    return Err(AgentError::InternalError(format!(
                        "Bilinmeyen provider: '{}'. Bilinen değerler: gemini, openai, ollama, anthropic. \
                         Farklı bir sağlayıcı kullanacaksan base_url'i elle belirt.",
                        other
                    )));
                }
            },
        };

        Ok(Self {
            base_url: resolved_url,
            model: model.to_string(),
            api_key: api_key.map(|k| k.to_string()),
            provider: provider_str,
        })
    }
}

pub struct LLMClient {
    http_client: Client,
    config: LLMConfig,
}

impl LLMClient {
    pub fn new(config: LLMConfig) -> Self {
        Self {
            http_client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .unwrap_or_else(|_| Client::new()),
            config,
        }
    }

    pub async fn send_request(
        &self,
        history: &ConversationHistory,
        tools_schema: Option<Vec<Value>>,
    ) -> Result<Value> {
        if self.config.provider == "anthropic" {
            return self.send_anthropic_request(history, tools_schema).await;
        }

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

        let mut request_builder = self.http_client.post(&self.config.base_url);

        if let Some(key) = &self.config.api_key {
            request_builder = request_builder.bearer_auth(key);
        }

        let response = request_builder
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

    async fn send_anthropic_request(
        &self,
        history: &ConversationHistory,
        tools_schema: Option<Vec<Value>>,
    ) -> Result<Value> {
        let system_prompt = history
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.content.clone())
            .unwrap_or_default();

        let filtered_messages: Vec<Value> = history
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| {
                json!({
                    "role": if m.role == Role::Assistant { "assistant" } else { "user" },
                    "content": m.content.clone().unwrap_or_default()
                })
            })
            .collect();

        let mut payload = json!({
            "model": self.config.model,
            "system": system_prompt,
            "messages": filtered_messages,
            "max_tokens": 4096
        });

        if let Some(schemas) = tools_schema {
            if !schemas.is_empty() {
                payload["tools"] = json!(schemas);
            }
        }

        let mut request_builder = self.http_client.post(&self.config.base_url)
            .header("anthropic-version", "2023-06-01");

        if let Some(key) = &self.config.api_key {
            request_builder = request_builder.header("x-api-key", key);
        }

        let response = request_builder
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status_code = response.status();
            let err_msg = response.text().await.unwrap_or_default();
            return Err(AgentError::InternalError(format!(
                "Anthropic API Hatası [{}]: {}",
                status_code, err_msg
            )));
        }

        Ok(response.json().await?)
    }

    pub async fn send_stream_request(
        &self,
        messages: Vec<crate::memory::Message>,
        schemas: Option<Vec<serde_json::Value>>,
        callback: &Option<Box<dyn Fn(String) + Send + Sync>>,
    ) -> crate::error::Result<crate::memory::Message> {
        
        if self.config.provider == "anthropic" {
            return Err(AgentError::InternalError(
                "Anthropic için stream (SSE) desteği henüz eklenmedi. Lütfen 'run' metodunu kullanın.".to_string()
            ));
        }

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": true
        });

        if let Some(s) = schemas {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), serde_json::json!(s));
        }

        let mut request_builder = self.http_client.post(&self.config.base_url);

        if let Some(key) = &self.config.api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = request_builder
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::InternalError(format!("Ağ hatası: {}", e)))?;

        if !response.status().is_success() {
            let status_code = response.status();
            let err_msg = response.text().await.unwrap_or_default();
            return Err(AgentError::InternalError(format!(
                "API Hatası [{}]: {}",
                status_code, err_msg
            )));
        }

        let mut stream = response.bytes_stream();
        let mut full_text = String::new();
        let mut tool_calls: BTreeMap<i64, (String, String, String)> = BTreeMap::new();
        
        let mut byte_buffer: Vec<u8> = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| AgentError::InternalError(format!("Stream buffer error: {}", e)))?;
            
            byte_buffer.extend_from_slice(&chunk);

            while let Some(newline_pos) = byte_buffer.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = byte_buffer.drain(..=newline_pos).collect();
                let line_str = String::from_utf8_lossy(&line_bytes);
                let trimmed = line_str.trim();

                if trimmed.starts_with("data: ") {
                    let data = &trimmed[6..];
                    if data == "[DONE]" {
                        continue;
                    }
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        let delta = &json["choices"][0]["delta"];
                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                            full_text.push_str(content);
                            if let Some(cb) = callback {
                                cb(content.to_string());
                            }
                        }

                        if let Some(tc_array) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                            for tc in tc_array {
                                let idx = tc.get("index").and_then(|i| i.as_i64()).unwrap_or(0);
                                let entry = tool_calls.entry(idx).or_insert_with(|| {
                                    (String::new(), String::new(), String::new())
                                });
                                if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                                    entry.0.push_str(id);
                                }
                                if let Some(func) = tc.get("function") {
                                    if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                                        entry.1.push_str(name);
                                    }
                                    if let Some(args) =
                                        func.get("arguments").and_then(|a| a.as_str())
                                    {
                                        entry.2.push_str(args);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !tool_calls.is_empty() {
            let tc_json: Vec<serde_json::Value> = tool_calls
                .into_iter()
                .map(|(idx, (id, name, args))| {
                    let tc_id = if id.is_empty() {
                        format!("call_local_{}", idx)
                    } else {
                        id
                    };
                    serde_json::json!({
                        "id": tc_id,
                        "type": "function",
                        "function": { "name": name, "arguments": args }
                    })
                })
                .collect();

            Ok(crate::memory::Message {
                role: crate::memory::Role::Assistant,
                content: None,
                tool_calls: Some(serde_json::json!(tc_json)),
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