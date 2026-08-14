use async_trait::async_trait;
use redis::AsyncCommands;
use std::collections::HashMap;

use crate::rag::{Chunk, VectorStore, SearchResult}; 

pub struct RedisVectorStore {
    client: redis::Client,
}

impl RedisVectorStore {
    pub fn new(redis_url: &str) -> anyhow::Result<Self> {
        let client = redis::Client::open(redis_url)?;
        Ok(Self { client })
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[async_trait]
impl VectorStore for RedisVectorStore {
    async fn add_chunks(&self, collection: &str, chunks: Vec<Chunk>) -> anyhow::Result<()> {
        let mut conn = self.client.get_multiplexed_async_connection().await?;
        let redis_key = format!("rag_collection:{}", collection);

        for chunk in chunks {
            let chunk_json = serde_json::to_string(&chunk)?;
            let chunk_id = if chunk.id.is_empty() {
                uuid::Uuid::new_v4().to_string()
            } else {
                chunk.id.clone()
            };
            
            let _: () = conn.hset(&redis_key, chunk_id, chunk_json).await?;
        }
        Ok(())
    }

    async fn search(&self, collection: &str, query_vector: Vec<f32>, limit: usize) -> anyhow::Result<Vec<SearchResult>> {
        let mut conn = self.client.get_multiplexed_async_connection().await?;
        let redis_key = format!("rag_collection:{}", collection);

        let chunk_map: HashMap<String, String> = conn.hgetall(&redis_key).await?;
        
        let mut scored_chunks: Vec<(f32, Chunk)> = Vec::new();

        for (_, chunk_json) in chunk_map {
            if let Ok(chunk) = serde_json::from_str::<Chunk>(&chunk_json) {
                if let Some(ref emb) = chunk.embedding {
                    let score = cosine_similarity(&query_vector, emb);
                    scored_chunks.push((score, chunk));
                }
            }
        }

        scored_chunks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let results = scored_chunks
            .into_iter()
            .take(limit)
            .map(|(score, chunk)| SearchResult { chunk, score })
            .collect();
            
        Ok(results)
    }

    async fn clear_collection(&self, collection: &str) -> anyhow::Result<()> {
        let mut conn = self.client.get_multiplexed_async_connection().await?;
        let redis_key = format!("rag_collection:{}", collection);
        
        if conn.exists(&redis_key).await? {
            let _: () = conn.del(&redis_key).await?;
        } else {
            eprintln!("Warning: collection '{}' does not exist; delete operation skipped.", collection);
        }
        Ok(())
    }

    async fn get_chunk_count(&self, collection: &str) -> anyhow::Result<usize> {
        let mut conn = self.client.get_multiplexed_tokio_connection().await?;
        let redis_key = format!("rag_collection:{}", collection);
        let count: usize = conn.hlen(&redis_key).await?;
        
        Ok(count)
    }
}