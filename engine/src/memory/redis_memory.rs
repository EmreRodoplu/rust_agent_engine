use async_trait::async_trait;
use redis::AsyncCommands;
use redis::aio::MultiplexedConnection;
use tokio::sync::OnceCell;

use super::{AgentMemory, Message};

pub struct RedisHistory {
    client: redis::Client,
    conn: OnceCell<MultiplexedConnection>, 
    prefix: String,
    ttl_seconds: Option<usize>, 
}

impl RedisHistory {
    pub fn new(redis_url: &str, ttl_seconds: Option<usize>) -> anyhow::Result<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| anyhow::anyhow!("Redis connection failed: {}", e))?;

        Ok(Self {
            client,
            conn: OnceCell::new(),
            prefix: "agent_session:".to_string(),
            ttl_seconds,
        })
    }

    async fn get_conn(&self) -> anyhow::Result<MultiplexedConnection> {
        let conn = self.conn.get_or_try_init(|| async {
            self.client.get_multiplexed_async_connection().await
        }).await.map_err(|e| anyhow::anyhow!("Failed to establish multiplexed connection: {}", e))?;
        
        Ok(conn.clone())
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

        let mut con = self.get_conn().await?;

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
        let mut con = self.get_conn().await?;

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
        let mut con = self.get_conn().await?;
        let mut pipe = redis::pipe();
        pipe.atomic().cmd("DEL").arg(&key);

        if !messages.is_empty() {
            let json_msgs: Result<Vec<String>, _> = messages.iter().map(|m| serde_json::to_string(m)).collect();
            pipe.cmd("RPUSH").arg(&key).arg(json_msgs?);
        }

        if let Some(ttl) = self.ttl_seconds {
            pipe.cmd("EXPIRE").arg(&key).arg(ttl as i64);
        }

        let _: () = pipe.query_async(&mut con).await
            .map_err(|e| anyhow::anyhow!("Failed to set new history in Redis via Pipeline: {}", e))?;

        Ok(())
    }

    async fn clear(&self, session_id: &str) -> anyhow::Result<()> {
        let key = self.get_key(session_id);
        let mut con = self.get_conn().await?;

        let _: () = redis::cmd("DEL").arg(&key).query_async(&mut con).await
            .map_err(|e| anyhow::anyhow!("Failed to delete Redis key: {}", e))?;
            
        Ok(())
    }

    async fn get_active_sessions(&self) -> anyhow::Result<Vec<String>> {
        let mut con = self.get_conn().await?;

        let mut iter: redis::AsyncIter<String> = redis::cmd("SCAN")
            .arg("MATCH")
            .arg(format!("{}*", self.prefix))
            .arg("COUNT")
            .arg(1000) 
            .clone()
            .iter_async(&mut con)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to retrieve active sessions from Redis using SCAN: {}", e))?;

        let mut keys = Vec::new();
        while let Some(key) = iter.next_item().await {
            keys.push(key);
        }

        let session_ids: Vec<String> = keys.into_iter()
            .map(|key| key.trim_start_matches(&self.prefix).to_string())
            .collect();

        Ok(session_ids)
    }
}