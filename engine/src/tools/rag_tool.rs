use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::tools::Tool;
use crate::error::Result; 
use crate::rag::{Embedder, VectorStore};

pub struct RagSearchTool {
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    collection: String, 
    limit: usize,       
}

impl RagSearchTool {
    pub fn new(
        embedder: Arc<dyn Embedder>,
        vector_store: Arc<dyn VectorStore>,
        collection: String,
        limit: usize,
    ) -> Self {
        Self {
            embedder,
            vector_store,
            collection,
            limit,
        }
    }
}

#[async_trait]
impl Tool for RagSearchTool {
    fn name(&self) -> String {
        format!("search_{}", self.collection.replace("-", "_"))
    }

    fn description(&self) -> String {
        format!(
            "Searches the documents and knowledge base in the '{}' collection using semantic search. Use this tool only to find information related to {}.", 
            self.collection, self.collection
        )
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "search_text": {
                    "type": "string",
                    "description": format!("A sentence or question to search for in the '{}' documents.", self.collection)
                }
            },
            "required": ["search_text"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let query = args["search_text"].as_str().unwrap_or("");
        println!("\n[RUST RAG TOOL TRIGGERED] Target: '{}' | Query: {}", self.collection, query);
        
        if query.trim().is_empty() {
            return Ok("Please provide a valid search text.".to_string());
        }

        let query_embedding = match self.embedder.embed_query(query).await {
            Ok(emb) => emb,
            Err(e) => return Ok(format!("An error occurred while processing the query (embedding): {}", e)),
        };

        let results = match self.vector_store.search(&self.collection, query_embedding, self.limit).await {
            Ok(res) => res,
            Err(e) => return Ok(format!("An error occurred while searching the database: {}", e)),
        };

        if results.is_empty() {
            return Ok("No relevant documents were found in the search results.".to_string());
        }

        let mut formatted_response = String::from("Here are the most relevant text chunks found in the knowledge base:\n\n");
        
        for (i, res) in results.iter().enumerate() {
            formatted_response.push_str(&format!(
                "--- SECTION {} (Similarity Score: {:.2}) ---\n{}\n\n",
                i + 1,
                res.score,
                res.chunk.content
            ));
        }

        Ok(formatted_response)
    }
}