use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::rag::{Chunk, SearchResult, VectorStore};

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for (a, b) in v1.iter().zip(v2.iter()) {
        dot_product += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

pub struct InMemoryVectorStore {
    collections: Arc<RwLock<HashMap<String, Vec<Chunk>>>>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add_chunks(&self, collection: &str, chunks: Vec<Chunk>) -> anyhow::Result<()> {
        let mut map = self.collections.write().await;
        let coll = map.entry(collection.to_string()).or_insert_with(Vec::new);
        coll.extend(chunks);
        
        Ok(())
    }

    async fn search(
        &self, 
        collection: &str, 
        query_embedding: Vec<f32>, 
        limit: usize
    ) -> anyhow::Result<Vec<SearchResult>> {
        let map = self.collections.read().await;
        
        let Some(chunks) = map.get(collection) else {
            return Ok(Vec::new()); 
        };

        let mut results = Vec::new();

        for chunk in chunks {
            if let Some(ref embedding) = chunk.embedding {
                let score = cosine_similarity(&query_embedding, embedding);
                results.push(SearchResult {
                    chunk: chunk.clone(),
                    score,
                });
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    async fn clear_collection(&self, collection: &str) -> anyhow::Result<()> {
        let mut map = self.collections.write().await;
        if map.contains_key(collection) {
            map.remove(collection);
        }
        else {
            eprintln!("Warning: collection '{}' does not exist; delete operation skipped.", collection);
        }

        Ok(())
            
        
    }

    async fn get_chunk_count(&self, collection: &str) -> anyhow::Result<usize> {
        let store = self.collections.read().await;
        if let Some(chunks) = store.get(collection) {
            Ok(chunks.len())
        } else {
            Ok(0)
        }
        
    }
}