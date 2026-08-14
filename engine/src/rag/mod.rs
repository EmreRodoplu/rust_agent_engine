use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;


// pub mod parser;
pub mod chunker;
pub mod embedder;
pub mod vector_store;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub document_id: String,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32, 
}

#[async_trait]
pub trait Parser: Send + Sync {
    async fn parse(&self, file_path: &str, metadata: Value) -> anyhow::Result<Document>;
}

pub trait Chunker: Send + Sync {
    fn chunk(&self, document: &Document) -> anyhow::Result<Vec<Chunk>>;
}

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed_chunks(&self, chunks: &mut [Chunk]) -> anyhow::Result<()>;
    
    async fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>>;
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    
    async fn add_chunks(&self, collection: &str, chunks: Vec<Chunk>) -> anyhow::Result<()>;
    
    async fn search(
        &self, 
        collection: &str, 
        query_embedding: Vec<f32>, 
        limit: usize
    ) -> anyhow::Result<Vec<SearchResult>>;
    
    async fn clear_collection(&self, collection: &str) -> anyhow::Result<()>;

    async fn get_chunk_count(&self, collection: &str) -> anyhow::Result<usize>;
}