use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyAny, PyDict, PyTuple};
use std::sync::OnceLock;
use tokio::runtime::Runtime;
use rust_agent_engine_core::client::LLMConfig;
use rust_agent_engine_core::agent::Agent;
use rust_agent_engine_core::error::AgentError;
use serde_json::json;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::tools::PythonTool;

static SHARED_RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn get_runtime() -> &'static Runtime {
    SHARED_RUNTIME.get_or_init(|| Runtime::new().expect("Tokio Runtime başlatılamadı!"))
}

#[gen_stub_pyclass]
#[pyclass(name = "LLMConfig")]
#[derive(Clone)]
pub struct PyLLMConfig {
    pub inner: LLMConfig,
}

#[gen_stub_pymethods]
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

#[gen_stub_pyclass]
#[pyclass(name = "Agent")]
pub struct PyAgent {
    inner: Agent, 
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgent {
    #[new]
    pub fn new(name: &str, system_prompt: &str, config: &PyLLMConfig) -> PyResult<Self> {
        Ok(Self {
            inner: Agent::new(name, system_prompt, config.inner.clone()),
        })
    }

    #[pyo3(signature = (user_input, stream_callback=None, prune=None))]
    pub fn run(
        &self,
        py: Python,
        user_input: &str,
        stream_callback: Option<PyObject>,
        prune: Option<usize>,
    ) -> PyResult<String> {
        let cb = stream_callback.map(|py_cb| {
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

        let result: Result<String, AgentError> = py.allow_threads(|| {
            get_runtime().block_on(async { 
                self.inner.run_with_stream(user_input, cb).await 
            })
        });

        py.allow_threads(|| {
            get_runtime().block_on(async {
                let mut history = self.inner.history.lock().await;
                history.prune(prune.unwrap_or(20));
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
            .unwrap_or_else(|_| "Açıklama bulunamadı.".to_string());

        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        let inspect = py.import_bound("inspect")?;
        let sig = inspect.getattr("signature")?.call1((f,))?;
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
}

fn infer_json_schema_type(annotation: &Bound<'_, PyAny>) -> serde_json::Value {
    if let (Ok(origin), Ok(args)) = (
        annotation.getattr("__origin__"),
        annotation.getattr("__args__"),
    ) {
        let origin_name: String = origin
            .getattr("__name__")
            .and_then(|n| n.extract())
            .unwrap_or_default();

        let args_vec: Vec<Bound<'_, PyAny>> = args
            .downcast_into::<PyTuple>()
            .map(|t| t.iter().collect())
            .unwrap_or_default();

        let is_none_type = |a: &Bound<'_, PyAny>| -> bool {
            a.getattr("__name__")
                .and_then(|n| n.extract::<String>())
                .unwrap_or_default()
                == "NoneType"
        };
        
        let mut has_none = false;
        let mut inner_type: Option<&Bound<'_, PyAny>> = None;
        for a in &args_vec {
            if is_none_type(a) {
                has_none = true;
            } else if inner_type.is_none() {
                inner_type = Some(a);
            }
        }
        if has_none {
            if let Some(inner) = inner_type {
                return infer_json_schema_type(inner);
            }
        }

        if origin_name == "list" {
            return json!({ "type": "array", "items": { "type": "string" } });
        }
        if origin_name == "dict" {
            return json!({ "type": "object" });
        }
    }

    let type_name: String = annotation
        .getattr("__name__")
        .and_then(|n| n.extract())
        .unwrap_or_else(|_| "string".to_string());

    let json_type = match type_name.as_str() {
        "int" => "integer",
        "float" => "number",
        "bool" => "boolean",
        "list" => "array",
        "dict" => "object",
        _ => "string",
    };

    json!({ "type": json_type })
}