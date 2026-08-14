use pyo3::prelude::*;
use std::sync::{Arc, OnceLock};
use tokio::runtime::Runtime;
use uuid;
use rust_agent_engine_core::rag::{Document, Chunker, Embedder, VectorStore};
use rust_agent_engine_core::rag::chunker::RecursiveChunker;
use rust_agent_engine_core::rag::embedder::UniversalEmbedder;
use rust_agent_engine_core::rag::vector_store::in_memory::InMemoryVectorStore;
use rust_agent_engine_core::rag::vector_store::redis_store::RedisVectorStore;

static SHARED_RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn get_runtime() -> &'static Runtime {
    SHARED_RUNTIME.get_or_init(|| {
        Runtime::new().expect("[Critical Error] Failed to create Tokio runtime! Engine cannot start.")
    })
}

#[pyclass(name = "RagEngine")]
pub struct PyRagEngine {
    pub embedder: Arc<dyn Embedder>,
    pub vector_store: Arc<dyn VectorStore>,
}

#[pymethods]
impl PyRagEngine {
    #[new]
    #[pyo3(signature = (model=None, base_url=None, api_key=None, redis_vectorstore_url=None))]
    fn new(
        model: Option<String>,
        base_url: Option<String>,
        api_key: Option<String>,
        redis_vectorstore_url: Option<String>,
    ) -> PyResult<Self> {
        
        let embedder = UniversalEmbedder::new(api_key, model, base_url)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let vector_store: Arc<dyn VectorStore> = match redis_vectorstore_url {
            Some(url) => {
                let store = RedisVectorStore::new(&url)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Redis connection error: {}", e)))?;
                Arc::new(store)
            }
            None => Arc::new(InMemoryVectorStore::new()),
        };

        Ok(Self { 
            embedder: Arc::new(embedder) as Arc<dyn Embedder>, 
            vector_store 
        })
    }

    #[pyo3(signature = (collection, text, source_name))]
    fn load_document(&self, py: Python, collection: String, text: String, source_name: String) -> PyResult<()> {
        let embedder = self.embedder.clone();
        let vector_store = self.vector_store.clone();
        
        let result = py.allow_threads(|| {
            get_runtime().block_on(async {
                let unique_id = uuid::Uuid::new_v4().to_string();
                
                let doc = Document { 
                    id: unique_id, 
                    content: text, 
                    metadata: serde_json::json!({"source": source_name}) 
                };
                
                let chunker = RecursiveChunker::new(500, 50);
                let mut chunks = chunker.chunk(&doc)
                    .map_err(|e| anyhow::anyhow!("Chunking failed: {}", e))?;
                
                embedder.embed_chunks(&mut chunks).await?;
                vector_store.add_chunks(&collection, chunks).await?;
                
                Ok::<(), anyhow::Error>(())
            })
        });

        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("RAG loading error: {}", e))),
        }
    }

    #[pyo3(signature = (collection, query, limit=3))]
    fn search(&self, py: Python, collection: String, query: String, limit: usize) -> PyResult<Vec<String>> {
        let embedder = self.embedder.clone();
        let vector_store = self.vector_store.clone();

        let result = py.allow_threads(|| {
            get_runtime().block_on(async {
                let query_vector = embedder.embed_query(&query).await?;
                let search_results = vector_store.search(&collection, query_vector, limit).await?;
                
                let texts: Vec<String> = search_results.into_iter().map(|res| res.chunk.content).collect();
                Ok::<Vec<String>, anyhow::Error>(texts)
            })
        });

        match result {
            Ok(texts) => Ok(texts),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("RAG search error: {}", e))),
        }
    }

    #[pyo3(signature = (collection))]
    fn get_collection_status(&self, py: Python, collection: String) -> PyResult<String> {
        let vector_store = self.vector_store.clone();
        
        let result = py.allow_threads(|| {
            get_runtime().block_on(async {
                vector_store.get_chunk_count(&collection).await
            })
        });

        match result {
            Ok(count) => {
                if count == 0 {
                    Ok(format!("Collection '{}' is empty (0 chunks). There is currently no data in the database.", collection))
                } else {
                    Ok(format!("Collection '{}' is active. It contains a total of {} chunks.", collection, count))
                }
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Status query error: {}", e))),
        }
    }

    #[pyo3(signature = (collection))]
    fn drop_collection(&self, py: Python, collection: String) -> PyResult<()> {
        let vector_store = self.vector_store.clone();
        
        let result = py.allow_threads(|| {
            get_runtime().block_on(async {
                vector_store.clear_collection(&collection).await
            })
        });

        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Collection deletion error: {}", e))),
        }
    }
}