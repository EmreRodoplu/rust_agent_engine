use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyAny, PyDict};
use std::sync::{Arc, OnceLock};
use tokio::runtime::Runtime;
use rust_agent_engine_core::memory::AgentMemory;
use rust_agent_engine_core::memory::in_memory::InMemoryHistory; 
use rust_agent_engine_core::client::LLMConfig;
use rust_agent_engine_core::agent::Agent;
use rust_agent_engine_core::error::AgentError;
use rust_agent_engine_core::tools::rag_tool::RagSearchTool;
use serde_json::json;
use crate::utils::infer_json_schema_type;

use crate::tools::PythonTool;
use crate::memory::PyAgentMemory; 

use crate::rag::PyRagEngine; 

static SHARED_RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn get_runtime() -> &'static Runtime {
    SHARED_RUNTIME.get_or_init(|| {
        Runtime::new().expect("[Critical Error] Failed to create Tokio runtime! Engine cannot start.")
    })
}

#[pyclass(name = "LLMConfig")]
#[derive(Clone)]
pub struct PyLLMConfig {
    pub inner: LLMConfig,
}

#[pymethods]
impl PyLLMConfig {
    #[new]
    #[pyo3(signature = (model, api_key=None, provider=None, base_url=None))]
    pub fn new(
        model: &str,
        api_key: Option<&str>,
        provider: Option<&str>,
        base_url: Option<&str>,
    ) -> PyResult<Self> {
        let inner = LLMConfig::new(model, api_key, provider, base_url)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "Agent")]
pub struct PyAgent {
    inner: Agent, 
}

#[pymethods]
impl PyAgent {
    #[new]
    #[pyo3(signature = (name, system_prompt, config, memory=None))] 
    pub fn new(
        name: &str, 
        system_prompt: &str, 
        config: &PyLLMConfig,
        memory: Option<PyAgentMemory> 
    ) -> PyResult<Self> {
        let mem_arc: Arc<dyn AgentMemory> = match memory {
            Some(m) => m.inner,
            None => Arc::new(InMemoryHistory::new()),
        };
        
        Ok(Self {
            inner: Agent::new(name, system_prompt, config.inner.clone(), mem_arc),
        })
    }

    #[pyo3(signature = (user_input, session_id=None, stream_callback=None, max_tokens=None, max_steps=None))]
    pub fn run(
        &self,
        py: Python,
        user_input: &str,
        session_id: Option<&str>, 
        stream_callback: Option<PyObject>,
        max_tokens: Option<usize>,
        max_steps: Option<usize>
    ) -> PyResult<String> {
        
        let actual_session_id = session_id.unwrap_or("default_session");
        
        let cb = stream_callback.map(|py_cb: Py<PyAny>| {
            let py_cb = std::sync::Arc::new(py_cb);

            Box::new(move |token: String| {
                let cb_clone = py_cb.clone();
                Python::with_gil(|py| {
                    if let Err(e) = cb_clone.bind(py).call1((token,)) {
                        e.print(py);
                    }
                });
            }) as Box<dyn Fn(String) + Send + Sync>
        });
        
        let session_id_str = actual_session_id.to_string();
        let user_input_str = user_input.to_string();
        let result: Result<String, AgentError> = py.allow_threads(|| {
            get_runtime().block_on(async { 
                self.inner.run_with_stream(&session_id_str, &user_input_str, cb, max_steps.unwrap_or(15)).await
            })
        });
        let prune_session_str = actual_session_id.to_string();
        py.allow_threads(|| {
            get_runtime().block_on(async {
                let limit = max_tokens.unwrap_or(4096);
                if let Err(e) = self.inner.memory.prune_by_tokens(&prune_session_str, limit).await {
                    eprintln!("[Memory Warning] Failed to prune session '{}': {}", prune_session_str, e);
                }
            });
        });

        result.map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (func))]
    pub fn register_tool(&mut self, py: Python, func: PyObject) -> PyResult<PyObject> {
        let f = func.bind(py);
        let name: String = f.getattr("__name__")?.extract()?;
        let description: String = f
            .getattr("__doc__")
            .and_then(|d| d.extract())
            .unwrap_or_else(|_| "No description provided.".to_string());

        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        let inspect = py.import_bound("inspect")?;
        let sig = inspect.getattr("signature")?.call1((&func,))?; 
        let builtins = py.import_bound("builtins")?;
        let params_mapping = sig.getattr("parameters")?;
        let params_obj = builtins.getattr("dict")?.call1((params_mapping,))?;
        let params = params_obj.downcast_into::<PyDict>()?;

        let empty = inspect.getattr("_empty")?;

        for (key, val) in params {
            let param_name: String = key.extract()?;
            if param_name == "self" || param_name == "return" { continue; }

            let annotation = val.getattr("annotation")?;
            
            let json_type = if annotation.is(&empty) {
                json!({ "type": "string" })
            } else {
                infer_json_schema_type(&annotation)
            };

            properties.insert(param_name.clone(), json_type);

            let default_val = val.getattr("default")?;
            if default_val.is(&empty) {
                required.push(serde_json::Value::String(param_name));
            }
        }

        let schema_params = json!({
            "type": "object",
            "properties": properties,
            "required": required
        });

        let python_tool = PythonTool {
            name,
            description,
            schema_params,
            func: func.clone(),
        };

        self.inner.register_tool(Box::new(python_tool));
        Ok(func)
    }
    
    #[pyo3(signature = (rag_engine, collection, limit=3))]
    pub fn add_rag_tool(&mut self, rag_engine: &PyRagEngine, collection: String, limit: usize) {
        let tool = RagSearchTool::new(
            rag_engine.embedder.clone(), 
            rag_engine.vector_store.clone(), 
            collection, 
            limit
        );
        self.inner.register_tool(Box::new(tool));
    }

    #[pyo3(signature = ())]
    pub fn get_active_sessions(&self, py: Python) -> PyResult<Vec<String>> {
        let result = py.allow_threads(|| {
            get_runtime().block_on(async {
                self.inner.memory.get_active_sessions().await
            })
        });
        
        match result {
            Ok(sessions) => Ok(sessions),
            Err(e) => Err(PyRuntimeError::new_err(format!("Could not read memory sessions: {}", e))),
        }
    }
}