use async_trait::async_trait;
use crate::rag::{Chunk, VectorStore, SearchResult};

pub struct RedisVectorStore {
    client: redis::Client,
}

impl RedisVectorStore {
    pub fn new(redis_url: &str) -> anyhow::Result<Self> {
        let client = redis::Client::open(redis_url)?;
        Ok(Self { client })
    }

    async fn ensure_index(&self, collection: &str, dim: usize) -> anyhow::Result<()> {
        let mut conn = self.client.get_multiplexed_async_connection().await?;
        let index_name = format!("idx:{}", collection);
        let prefix = format!("rag:{}:", collection); 

        let info_check: redis::RedisResult<redis::Value> = redis::cmd("FT.INFO")
            .arg(&index_name)
            .query_async(&mut conn).await;

        if info_check.is_err() {
            let _: () = redis::cmd("FT.CREATE")
                .arg(&index_name)
                .arg("ON").arg("HASH")
                .arg("PREFIX").arg("1").arg(&prefix)
                .arg("SCHEMA")
                .arg("content").arg("TEXT")
                .arg("vector").arg("VECTOR").arg("HNSW") 
                .arg("6")
                .arg("TYPE").arg("FLOAT32")
                .arg("DIM").arg(dim)
                .arg("DISTANCE_METRIC").arg("COSINE")
                .query_async(&mut conn).await?;
            println!("[Vector Store] Created new HNSW index: {}", index_name);
        }
        Ok(())
    }
}

fn f32_vec_to_bytes(vec: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vec.len() * 4);
    for &v in vec {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    bytes
}

#[async_trait]
impl VectorStore for RedisVectorStore {
    async fn add_chunks(&self, collection: &str, chunks: Vec<Chunk>) -> anyhow::Result<()> {
        if chunks.is_empty() { return Ok(()); }
        
        let dim = chunks[0].embedding.as_ref().map(|v| v.len()).unwrap_or(1536);
        self.ensure_index(collection, dim).await?;

        let mut conn = self.client.get_multiplexed_async_connection().await?;

        for chunk in chunks {
            let chunk_json = serde_json::to_string(&chunk)?;
            let chunk_id = if chunk.id.is_empty() {
                uuid::Uuid::new_v4().to_string()
            } else {
                chunk.id.clone()
            };
            
            let key = format!("rag:{}:{}", collection, chunk_id);
            let vector_bytes = if let Some(ref emb) = chunk.embedding {
                f32_vec_to_bytes(emb)
            } else {
                continue; 
            };

            let _: () = redis::cmd("HSET")
                .arg(&key)
                .arg("content").arg(&chunk.content) 
                .arg("vector").arg(vector_bytes)    
                .arg("data").arg(chunk_json)        
                .query_async(&mut conn).await?;
        }
        Ok(())
    }

    async fn search(&self, collection: &str, query_vector: Vec<f32>, limit: usize) -> anyhow::Result<Vec<SearchResult>> {
        let mut conn = self.client.get_multiplexed_async_connection().await?;
        let index_name = format!("idx:{}", collection);
        let query_bytes = f32_vec_to_bytes(&query_vector);

        let result: redis::RedisResult<redis::Value> = redis::cmd("FT.SEARCH")
            .arg(&index_name)
            .arg(format!("*=>[KNN {} @vector $query_vec AS score]", limit))
            .arg("PARAMS").arg("2")
            .arg("query_vec").arg(query_bytes)
            .arg("DIALECT").arg("2")
            .arg("RETURN").arg("2").arg("data").arg("score")
            .query_async(&mut conn).await;

        let result = match result {
            Ok(val) => val,
            Err(e) => {
                if e.to_string().contains("Unknown Index name") {
                    return Ok(Vec::new());
                }
                return Err(e.into());
            }
        };

        let mut matches = Vec::new();

        if let redis::Value::Bulk(arr) = result {
            if arr.is_empty() { return Ok(matches); }
            
            let mut i = 1; 
            while i < arr.len() {
                if i + 1 < arr.len() {
                    if let redis::Value::Bulk(ref fields) = arr[i + 1] {
                        let mut chunk_opt: Option<Chunk> = None;
                        let mut distance: f32 = 0.0;

                        let mut j = 0;
                        while j < fields.len() {
                            if let redis::Value::Data(ref field_name_bytes) = fields[j] {
                                let name = String::from_utf8_lossy(field_name_bytes);
                                if j + 1 < fields.len() {
                                    if let redis::Value::Data(ref field_val_bytes) = fields[j+1] {
                                        if name == "data" {
                                            let json_str = String::from_utf8_lossy(field_val_bytes);
                                            chunk_opt = serde_json::from_str(&json_str).ok();
                                        } else if name == "score" {
                                            let score_str = String::from_utf8_lossy(field_val_bytes);
                                            distance = score_str.parse().unwrap_or(0.0);
                                        }
                                    }
                                }
                            }
                            j += 2;
                        }
                        if let Some(chunk) = chunk_opt {
                            matches.push(SearchResult { chunk, score: 1.0 - distance });
                        }
                    }
                }
                i += 2;
            }
        }

        matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(matches)
    }

    async fn clear_collection(&self, collection: &str) -> anyhow::Result<()> {
        let mut conn = self.client.get_multiplexed_async_connection().await?;
        let index_name = format!("idx:{}", collection);
        
        let _: redis::RedisResult<()> = redis::cmd("FT.DROPINDEX")
            .arg(&index_name)
            .arg("DD")
            .query_async(&mut conn).await;
            
        Ok(())
    }

    async fn get_chunk_count(&self, collection: &str) -> anyhow::Result<usize> {
        let mut conn = self.client.get_multiplexed_async_connection().await?;
        let index_name = format!("idx:{}", collection);

        let info: redis::RedisResult<redis::Value> = redis::cmd("FT.INFO")
            .arg(&index_name)
            .query_async(&mut conn).await;

        if let Ok(redis::Value::Bulk(arr)) = info {
            let mut i = 0;
            while i < arr.len() {
                if let redis::Value::Data(ref key_bytes) = arr[i] {
                    let key = String::from_utf8_lossy(key_bytes);
                    if key == "num_docs" && i + 1 < arr.len() {
                        if let redis::Value::Data(ref val_bytes) = arr[i+1] {
                            let count_str = String::from_utf8_lossy(val_bytes);
                            return Ok(count_str.parse().unwrap_or(0));
                        } else if let redis::Value::Int(count) = arr[i+1] {
                            return Ok(count as usize);
                        }
                    }
                }
                i += 1;
            }
        }
        Ok(0)
    }
}