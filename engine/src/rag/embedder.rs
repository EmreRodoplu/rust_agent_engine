use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use super::{Chunk, Embedder};

pub struct UniversalEmbedder {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
}

impl UniversalEmbedder {
    pub fn new(
        api_key: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            client: Client::new(),
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1/embeddings".to_string()),
            api_key,
            model: model.unwrap_or_else(|| "text-embedding-3-small".to_string()),
        })
    }
}  

#[derive(Deserialize)]
struct StandardEmbeddingResponse {
    data: Vec<StandardEmbeddingData>,
}

#[derive(Deserialize)]
struct StandardEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[async_trait]
impl Embedder for UniversalEmbedder {
    async fn embed_chunks(&self, chunks: &mut [Chunk]) -> anyhow::Result<()> {
        if chunks.is_empty() { return Ok(()); }
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

        let mut req = self.client.post(&self.base_url);
        
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let res = req
            .json(&json!({ "model": self.model, "input": texts }))
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status();
            let err_text = res.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Embedding API error [{}]: {}", status, err_text));
        }

        let mut parsed: StandardEmbeddingResponse = res.json().await?;
        parsed.data.sort_by_key(|d| d.index);

        for (chunk, data) in chunks.iter_mut().zip(parsed.data) {
            chunk.embedding = Some(data.embedding);
        }
        Ok(())
    }

    async fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let mut req = self.client.post(&self.base_url);
        
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let res = req
            .json(&json!({ "model": self.model, "input": query }))
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status();
            let err_text = res.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Embedding API error [{}]: {}", status, err_text));
        }

        let parsed: StandardEmbeddingResponse = res.json().await?;
        parsed.data.into_iter().next().map(|d| d.embedding)
            .ok_or_else(|| anyhow::anyhow!("Could not generate a vector; the API returned an empty response."))
    }
}