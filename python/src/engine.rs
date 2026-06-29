use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use tokio::runtime::Runtime;
use rust_agent_engine_core::client::LLMConfig;
use rust_agent_engine_core::agent::Agent;
use pyo3::types::PyDict;
use rust_agent_engine_core::error::AgentError;
use serde_json::json;

use crate::tools::PythonTool;

#[pyclass(name = "LLMConfig")]
#[derive(Clone)]
pub struct PyLLMConfig { pub inner: LLMConfig }

#[pymethods]
impl PyLLMConfig {
    #[new]
    #[pyo3(signature = (model, api_key, provider=None, base_url=None))]
    pub fn new(model: &str, api_key: &str, provider: Option<&str>, base_url: Option<&str>) -> Self {
        Self { inner: LLMConfig::new(model, api_key, provider, base_url) }
    }
}

#[pyclass(name = "Agent")]
pub struct PyAgent {
    inner: Agent,
    rt: Runtime, 
}

#[pymethods]
impl PyAgent {
    #[new]
    pub fn new(name: &str, system_prompt: &str, config: &PyLLMConfig) -> PyResult<Self> {
        let rt = Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("Tokio Runtime başlatılamadı: {}", e))
        })?;

        Ok(Self {
            inner: Agent::new(name, system_prompt, config.inner.clone()),
            rt,
        })
    }
    #[pyo3(signature = (user_input, stream_callback=None, prune=None))]
    pub fn run(&self, py: Python, user_input: &str, stream_callback: Option<PyObject>, prune: Option<usize>) -> PyResult<String> {
        let cb = stream_callback.map(|py_cb| {
            let py_cb = std::sync::Arc::new(py_cb);
            
            Box::new(move |token: String| {
                let cb_clone = py_cb.clone();
                Python::with_gil(|py| {
                    let _ = cb_clone.bind(py).call1((token,));
                });
            }) as Box<dyn Fn(String) + Send + Sync>
        });
        
        let result: Result<String, AgentError> = py.allow_threads(|| {
            self.rt.block_on(async {
                self.inner.run_with_stream(user_input, cb).await
            })
        });
        
        let mut history = self.inner.history.lock().unwrap();
        history.prune(match prune {
            Some(n) => n,
            None => 20,
        });
        match result {
            Ok(output) => Ok(output),
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    #[pyo3(signature = (func))]
    pub fn register_tool(&mut self, py: Python, func: PyObject) -> PyResult<()> {
        let f = func.bind(py);
        let raw_name: String = f.getattr("__name__")?.extract()?;
        let raw_doc: String = f.getattr("__doc__")
            .and_then(|d| d.extract())
            .unwrap_or_else(|_| "Açıklama bulunamadı.".to_string());
        let name: &'static str = Box::leak(raw_name.into_boxed_str());
        let description: &'static str = Box::leak(raw_doc.into_boxed_str());
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        
        if let Ok(annotations) = f.getattr("__annotations__") {
            if let Ok(dict) = annotations.downcast::<PyDict>() {
                for (key, val) in dict {
                    let k: String = key.extract()?;
                    if k == "return" { continue; } 
                    
                    let type_name: String = val.getattr("__name__")
                        .and_then(|n| n.extract())
                        .unwrap_or_else(|_| "string".to_string());
                
                    let json_type = match type_name.as_str() {
                        "int" => "integer",
                        "float" => "number",
                        "bool" => "boolean",
                        _ => "string",
                    };
                    
                    properties.insert(k.clone(), json!({ "type": json_type }));
                    required.push(serde_json::Value::String(k));
                }
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
        Ok(())
    }
}
