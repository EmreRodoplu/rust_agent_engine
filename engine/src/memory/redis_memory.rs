use std::format;

use async_trait::async_trait;
use redis::AsyncCommands;

use super::{AgentMemory, Message};

pub struct RedisHistory {
    client: redis::Client,
    prefix: String,
    ttl_seconds: Option<usize>, 
}

impl RedisHistory {
    pub fn new(redis_url: &str, ttl_seconds: Option<usize>) -> anyhow::Result<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| anyhow::anyhow!("Redis connection failed: {}", e))?;

        Ok(Self {
            client,
            prefix: "agent_session:".to_string(),
            ttl_seconds,
        })
    }

    fn get_key(&self, session_id: &str) -> String {
        format!("{}{}", self.prefix, session_id)
    }
}

#[async_trait]
impl AgentMemory for RedisHistory {
    async fn add_message(&self, session_id: &str, msg: Message) -> anyhow::Result<()> {
        let key = self.get_key(session_id);
        let json_msg = serde_json::to_string(&msg)?;

        let mut con = self.client.get_multiplexed_async_connection().await
            .map_err(|e| anyhow::anyhow!("Failed to get Redis connection: {}", e))?;

        let _: () = redis::cmd("RPUSH").arg(&key).arg(json_msg).query_async(&mut con).await
            .map_err(|e| anyhow::anyhow!("Failed to write to Redis: {}", e))?;

        if let Some(ttl) = self.ttl_seconds {
            let _: () = redis::cmd("EXPIRE").arg(&key).arg(ttl as i64).query_async(&mut con).await
                .map_err(|e| anyhow::anyhow!("Failed to set TTL on Redis key: {}", e))?;
        }

        Ok(())
    }

    async fn get_history(&self, session_id: &str) -> anyhow::Result<Vec<Message>> {
        let key = self.get_key(session_id);
        
        let mut con = self.client.get_multiplexed_async_connection().await
            .map_err(|e| anyhow::anyhow!("Failed to get Redis connection: {}", e))?;

        let items: Vec<String> = con.lrange(&key, 0, -1).await
            .map_err(|e| anyhow::anyhow!("Failed to read from Redis: {}", e))?;

        let mut messages = Vec::new();
        for item in items {
            let msg: Message = serde_json::from_str(&item)?;
            messages.push(msg);
        }

        Ok(messages)
    }

    async fn set_history(&self, session_id: &str, messages: Vec<Message>) -> anyhow::Result<()> {
        let key = self.get_key(session_id);
        let mut con = self.client.get_multiplexed_async_connection().await
            .map_err(|e| anyhow::anyhow!("Failed to get Redis connection: {}", e))?;

        
        let _: () = redis::cmd("DEL").arg(&key).query_async(&mut con).await
            .map_err(|e| anyhow::anyhow!("Failed to clear old Redis key: {}", e))?;

        for msg in messages {
            let json_msg = serde_json::to_string(&msg)?;
            let _: () = redis::cmd("RPUSH").arg(&key).arg(json_msg).query_async(&mut con).await
                .map_err(|e| anyhow::anyhow!("Failed to set new history in Redis: {}", e))?;
        }

        if let Some(ttl) = self.ttl_seconds {
            let _: () = redis::cmd("EXPIRE").arg(&key).arg(ttl as i64).query_async(&mut con).await
                .map_err(|e| anyhow::anyhow!("Failed to set TTL: {}", e))?;
        }

        Ok(())
    }

    async fn clear(&self, session_id: &str) -> anyhow::Result<()> {
        let key = self.get_key(session_id);
        let mut con = self.client.get_multiplexed_async_connection().await
            .map_err(|e| anyhow::anyhow!("Failed to get Redis connection: {}", e))?;

        let _: () = redis::cmd("DEL").arg(&key).query_async(&mut con).await
            .map_err(|e| anyhow::anyhow!("Failed to delete Redis key: {}", e))?;
            
        Ok(())
    }

    async fn get_active_sessions(&self) -> anyhow::Result<Vec<String>> {
        let mut con = self.client.get_multiplexed_async_connection().await
            .map_err(|e| anyhow::anyhow!("Failed to get Redis connection: {}", e))?;

        let keys: Vec<String> = redis::cmd("KEYS").arg(format!("{}*", self.prefix)).query_async(&mut con).await
            .map_err(|e| anyhow::anyhow!("Failed to retrieve active sessions from Redis: {}", e))?;

        let session_ids: Vec<String> = keys.into_iter()
            .map(|key| key.trim_start_matches(&self.prefix).to_string())
            .collect();

        Ok(session_ids)
    }
}