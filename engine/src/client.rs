use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::format;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration as TokioDuration};
use std::sync::Arc;

use crate::error::{AgentError, Result};
use crate::memory::Role;

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
                        "Unknown provider: '{}'. Known values: gemini, openai, ollama, anthropic. If you want to use a different provider, specify the base_url manually.",
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

#[derive(Clone)]
pub struct LLMClient {
    http_client: Client,
    config: LLMConfig,
    semaphore: Arc<Semaphore>, 
}

impl LLMClient {
    pub fn new(config: LLMConfig) -> Self {
        Self {
            http_client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .unwrap_or_else(|_| Client::new()),
            config,
            semaphore: Arc::new(Semaphore::new(50)), 
        }
    }

    pub async fn send_request(
        &self,
        messages: Vec<crate::memory::Message>,
        tools_schema: Option<Vec<Value>>,
    ) -> Result<Value> {
        if self.config.provider == "anthropic" {
            return self.send_anthropic_request(messages, tools_schema).await;
        }

        let mut payload = json!({
            "model": self.config.model,
            "messages": messages,
        });

        if let Some(schemas) = tools_schema {
            if !schemas.is_empty() {
                payload["tools"] = json!(schemas);
                payload["tool_choice"] = json!("auto");
            }
        }

        let _permit = self.semaphore.acquire().await
            .map_err(|_| AgentError::InternalError("Failed to acquire semaphore".to_string()))?;
        let max_attempts = 3;
        let mut backoff_ms = 1500;
        let mut attempt = 0;

        loop {
            attempt += 1;
            
            let mut request_builder = self.http_client.post(&self.config.base_url);
            if let Some(key) = &self.config.api_key {
                request_builder = request_builder.bearer_auth(key);
            }

            match request_builder.json(&payload).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        let response_json: Value = response.json().await
                            .map_err(|e| AgentError::InternalError(e.to_string()))?;
                        return Ok(response_json);
                    }
                    
                    let status = response.status();
                    if status.as_u16() == 429 || status.is_server_error() {
                        if attempt >= max_attempts {
                            let err_msg = response.text().await.unwrap_or_default();
                            return Err(AgentError::InternalError(format!("API Error [{}]: {}", status, err_msg)));
                        }
                        println!("[Warning] API Rate Limit hit ({}). Retrying in {} ms...", status, backoff_ms);
                        sleep(TokioDuration::from_millis(backoff_ms)).await;
                        backoff_ms *= 2;
                        continue;
                    } 
                    let err_msg = response.text().await.unwrap_or_default();
                    return Err(AgentError::InternalError(format!("API Error [{}]: {}", status, err_msg)));
                }
                Err(e) => {
                    if attempt >= max_attempts {
                        return Err(AgentError::InternalError(format!("Network Error: {}", e)));
                    }
                    println!("[Warning] Network error: {}. Retrying in {} ms...", e, backoff_ms);
                    sleep(TokioDuration::from_millis(backoff_ms)).await;
                    backoff_ms *= 2;
                    continue;
                }
            }
        }
    }

    async fn send_anthropic_request(
        &self,
        messages: Vec<crate::memory::Message>,
        tools_schema: Option<Vec<Value>>,
    ) -> Result<Value> {
        
        let system_prompt = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .filter_map(|m| m.content.clone())
            .collect::<Vec<String>>()
            .join("\n\n");

        let filtered_messages: Vec<Value> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| {
                if m.role == Role::Tool {
                    json!({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                                "content": m.content.clone().unwrap_or_default()
                            }
                        ]
                    })
                } else {
                    json!({
                        "role": if m.role == Role::Assistant { "assistant" } else { "user" },
                        "content": m.content.clone().unwrap_or_default()
                    })
                }
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

        let _permit = self.semaphore.acquire().await
            .map_err(|_| AgentError::InternalError("Failed to acquire semaphore".to_string()))?;

        let max_attempts = 3;
        let mut backoff_ms = 1500;
        let mut attempt = 0;

        loop {
            attempt += 1;
            
            let mut request_builder = self
                .http_client
                .post(&self.config.base_url)
                .header("anthropic-version", "2023-06-01");

            if let Some(key) = &self.config.api_key {
                request_builder = request_builder.header("x-api-key", key);
            }

            match request_builder.json(&payload).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        let response_json: Value = response.json().await
                            .map_err(|e| AgentError::InternalError(e.to_string()))?;
                        return Ok(response_json);
                    }
                    
                    let status = response.status();
                    if status.as_u16() == 429 || status.is_server_error() {
                        if attempt >= max_attempts {
                            let err_msg = response.text().await.unwrap_or_default();
                            return Err(AgentError::InternalError(format!("Anthropic API error [{}]: {}", status, err_msg)));
                        }
                        println!("[Warning] Anthropic Rate Limit hit ({}). Retrying in {} ms...", status, backoff_ms);
                        sleep(TokioDuration::from_millis(backoff_ms)).await;
                        backoff_ms *= 2;
                        continue;
                    } 
                    
                    let err_msg = response.text().await.unwrap_or_default();
                    return Err(AgentError::InternalError(format!("Anthropic API error [{}]: {}", status, err_msg)));
                }
                Err(e) => {
                    if attempt >= max_attempts {
                        return Err(AgentError::InternalError(format!("Network Error: {}", e)));
                    }
                    println!("[Warning] Network error: {}. Retrying in {} ms...", e, backoff_ms);
                    sleep(TokioDuration::from_millis(backoff_ms)).await;
                    backoff_ms *= 2;
                    continue;
                }
            }
        }
    }

    pub async fn send_stream_request(
        &self,
        messages: Vec<crate::memory::Message>,
        schemas: Option<Vec<serde_json::Value>>,
        callback: &Option<Box<dyn Fn(String) + Send + Sync>>,
    ) -> crate::error::Result<crate::memory::Message> {
        if self.config.provider == "anthropic" {
            return Err(AgentError::InternalError(
                "Stream (SSE) support for Anthropic has not been added yet. Please use the 'run' method.".to_string()
            ));
        }

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": true
        });

        if let Some(s) = schemas {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("tools".to_string(), serde_json::json!(s));
            }
        }

        let _permit = self.semaphore.acquire().await
            .map_err(|_| AgentError::InternalError("Failed to acquire semaphore".to_string()))?;

        let max_attempts = 3;
        let mut backoff_ms = 1500;
        let mut attempt = 0;
        
        let response = loop {
            attempt += 1;
            
            let mut request_builder = self.http_client.post(&self.config.base_url);
            if let Some(key) = &self.config.api_key {
                request_builder = request_builder.header("Authorization", format!("Bearer {}", key));
            }

            match request_builder.json(&body).send().await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        break resp;
                    }
                    
                    let status = resp.status();
                    if status.as_u16() == 429 || status.is_server_error() {
                        if attempt >= max_attempts {
                            let err_msg = resp.text().await.unwrap_or_default();
                            return Err(AgentError::InternalError(format!("API Error [{}]: {}", status, err_msg)));
                        }
                        println!("[Warning] Stream API Rate Limit hit ({}). Retrying in {} ms...", status, backoff_ms);
                        sleep(TokioDuration::from_millis(backoff_ms)).await;
                        backoff_ms *= 2;
                        continue;
                    } 
                    
                    let err_msg = resp.text().await.unwrap_or_default();
                    return Err(AgentError::InternalError(format!("API Error [{}]: {}", status, err_msg)));
                }
                Err(e) => {
                    if attempt >= max_attempts {
                        return Err(AgentError::InternalError(format!("Network Error: {}", e)));
                    }
                    println!("[Warning] Stream Network error: {}. Retrying in {} ms...", e, backoff_ms);
                    sleep(TokioDuration::from_millis(backoff_ms)).await;
                    backoff_ms *= 2;
                    continue;
                }
            }
        };

        let mut stream = response.bytes_stream();
        let mut full_text = String::new();
        let mut tool_calls: BTreeMap<usize, (String, String, String)> = BTreeMap::new();
        let mut byte_buffer: Vec<u8> = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| AgentError::InternalError(format!("Stream buffer error: {}", e)))?;

            byte_buffer.extend_from_slice(&chunk);

            while let Some(newline_pos) = byte_buffer.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = byte_buffer.drain(..=newline_pos).collect();
                let line_str = String::from_utf8_lossy(&line_bytes);
                let trimmed = line_str.trim();

                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.starts_with(':') {
                    continue;
                }

                if let Some(event_str) = trimmed.strip_prefix("event:") {
                    let event_type = event_str.trim();
                    if event_type == "error" {
                        eprintln!("\n[API NOTICE] The server sent an error event! Reading error details...");
                    }
                    continue;
                }

                if let Some(data_str) = trimmed.strip_prefix("data:") {
                    let data = data_str.trim();
                    if data == "[DONE]" {
                        continue;
                    }

                    match serde_json::from_str::<serde_json::Value>(data) {
                        Ok(json) => {
                            if let Some(err) = json.get("error") {
                                return Err(AgentError::InternalError(format!(
                                    "API stream error: {}",
                                    err
                                )));
                            }

                            if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                                if let Some(choice) = choices.get(0) {
                                    if let Some(delta) = choice.get("delta") {
                                        if let Some(content) =
                                            delta.get("content").and_then(|c| c.as_str())
                                        {
                                            full_text.push_str(content);
                                            if let Some(cb) = callback {
                                                cb(content.to_string());
                                            }
                                        }

                                        if let Some(tc_array) =
                                            delta.get("tool_calls").and_then(|tc| tc.as_array())
                                        {
                                            for tc in tc_array {
                                                let idx = tc
                                                    .get("index")
                                                    .and_then(|i| i.as_u64())
                                                    .unwrap_or(0)
                                                    as usize;
                                                let entry =
                                                    tool_calls.entry(idx).or_insert_with(|| {
                                                        (
                                                            String::new(),
                                                            String::new(),
                                                            String::new(),
                                                        )
                                                    });
                                                if let Some(id) =
                                                    tc.get("id").and_then(|i| i.as_str())
                                                {
                                                    entry.0.push_str(id);
                                                }
                                                if let Some(func) = tc.get("function") {
                                                    if let Some(name) =
                                                        func.get("name").and_then(|n| n.as_str())
                                                    {
                                                        entry.1.push_str(name);
                                                    }
                                                    if let Some(args) = func
                                                        .get("arguments")
                                                        .and_then(|a| a.as_str())
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
                        Err(e) => {
                            eprintln!("\n[SYSTEM WARNING] Could not parse the incoming stream chunk: {}. Data: {}", e, data);
                        }
                    }
                } else if trimmed.starts_with("{") {
                    if let Ok(err_json) = serde_json::from_str::<serde_json::Value>(trimmed) {
                        if err_json.get("error").is_some() {
                            return Err(AgentError::InternalError(format!(
                                "Hidden stream error detected: {}",
                                err_json
                            )));
                        }
                    }
                    eprintln!(
                        "\n[SYSTEM WARNING] Unexpected data format received: {}",
                        trimmed
                    );
                } else {
                    eprintln!(
                        "\n[SYSTEM WARNING] Not JSON and not in data format. Skipping: {}",
                        trimmed
                    );
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
                content: if full_text.trim().is_empty() {
                    None
                } else {
                    Some(full_text)
                },
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