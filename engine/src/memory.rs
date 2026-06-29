use serde::{Deserialize, Serialize};
use serde_json::Value;

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

#[derive(Debug, Clone)]
pub struct ConversationHistory {
    pub messages: Vec<Message>,
}

impl ConversationHistory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn add_system_prompt(&mut self, prompt: &str) {
        self.messages.push(Message {
            role: Role::System,
            content: Some(prompt.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(Message {
            role: Role::User,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn prune(&mut self, max_messages: usize) {
        if self.messages.len() > max_messages {
            let system_prompt = self.messages.remove(0);
            let drain_count = self.messages.len() - (max_messages - 1);
            self.messages.drain(0..drain_count);
            self.messages.insert(0, system_prompt);
        }
    }
}
