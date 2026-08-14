use pyo3::prelude::*;
use std::sync::Arc;
use rust_agent_engine_core::memory::AgentMemory as RustAgentMemory;
use rust_agent_engine_core::memory::in_memory::InMemoryHistory;
use rust_agent_engine_core::memory::redis_memory::RedisHistory;

#[pyclass(name = "AgentMemory")]
#[derive(Clone)]
pub struct PyAgentMemory {
    pub inner: Arc<dyn RustAgentMemory>,
}

#[pymethods]
impl PyAgentMemory {
    #[staticmethod]
    pub fn in_memory() -> Self {
        let mem = InMemoryHistory::new();
        Self {
            inner: Arc::new(mem),
        }
    }
    
    #[staticmethod]
    #[pyo3(signature = (redis_url, ttl_seconds=None))]
    pub fn redis(redis_url: &str, ttl_seconds: Option<usize>) -> PyResult<Self> {
        let mem = RedisHistory::new(redis_url, ttl_seconds)
            .map_err(|e| pyo3::exceptions::PyConnectionError::new_err(e.to_string()))?;
        
        Ok(Self {
            inner: Arc::new(mem),
        })
    }
}