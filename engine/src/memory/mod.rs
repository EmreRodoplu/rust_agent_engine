use serde::{Deserialize, Serialize};
use serde_json::Value;
use async_trait::async_trait;
use tiktoken_rs::cl100k_base; 

pub mod in_memory;
pub mod redis_memory;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}


#[async_trait]
pub trait AgentMemory: Send + Sync {
    async fn add_message(&self, session_id: &str, msg: Message) -> anyhow::Result<()>;
    
    async fn get_history(&self, session_id: &str) -> anyhow::Result<Vec<Message>>;
    
    async fn set_history(&self, session_id: &str, messages: Vec<Message>) -> anyhow::Result<()>;
    
    async fn clear(&self, session_id: &str) -> anyhow::Result<()>;

    async fn get_active_sessions(&self) -> anyhow::Result<Vec<String>>;

    async fn prune_by_tokens(&self, session_id: &str, max_tokens: usize) -> anyhow::Result<()> {
        let mut messages = self.get_history(session_id).await?;
        
        if messages.is_empty() {
            return Ok(());
        }

        let bpe = cl100k_base().map_err(|e| anyhow::anyhow!("Could not initialize the tokenizer: {}", e))?;

        loop {
            let mut current_tokens = 0;
            for msg in &messages {
                if let Some(content) = &msg.content {
                    current_tokens += bpe.encode_with_special_tokens(content).len();
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    let tool_str = serde_json::to_string(tool_calls).unwrap_or_default();
                    current_tokens += bpe.encode_with_special_tokens(&tool_str).len();
                }
            }

            if current_tokens <= max_tokens {
                break;
            }

            let remove_index = if let Some(first_msg) = messages.get(0) {
                if first_msg.role == Role::System {
                    if messages.len() > 1 { 1 } else { break; }
                } else { 0 }
            } else { break; };

            messages.remove(remove_index);
        }

        self.set_history(session_id, messages).await?;
        Ok(())
    }
}