use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use super::{AgentMemory, Message};

#[derive(Clone)]
pub struct InMemoryHistory {
    pub store: Arc<RwLock<HashMap<String, Vec<Message>>>>,
}

impl InMemoryHistory {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl AgentMemory for InMemoryHistory {
    async fn add_message(&self, session_id: &str, msg: Message) -> anyhow::Result<()> {
        let mut store = self.store.write().await;
        store.entry(session_id.to_string()).or_default().push(msg);
        Ok(())
    }

    async fn get_history(&self, session_id: &str) -> anyhow::Result<Vec<Message>> {
        let store = self.store.read().await;
        Ok(store.get(session_id).cloned().unwrap_or_default())
    }

    async fn set_history(&self, session_id: &str, messages: Vec<Message>) -> anyhow::Result<()> {
        let mut store = self.store.write().await;
        store.insert(session_id.to_string(), messages);
        Ok(())
    }

    async fn clear(&self, session_id: &str) -> anyhow::Result<()> {
        let mut store = self.store.write().await;
        store.remove(session_id);
        Ok(())
    }

    async fn get_active_sessions(&self) -> anyhow::Result<Vec<String>> {
        let store = self.store.read().await;
        Ok(store.keys().cloned().collect())
    }
}