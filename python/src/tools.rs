use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use async_trait::async_trait;
use rust_agent_engine_core::tools::Tool;
use rust_agent_engine_core::error::AgentError;

pub struct PythonTool {
    pub name: &'static str,
    pub description: &'static str,
    pub schema_params: Value,
    pub func: PyObject, 
}

#[async_trait]
impl Tool for PythonTool {
    fn name(&self) -> &'static str { self.name }
    fn description(&self) -> &'static str { self.description }
    fn schema(&self) -> Value { self.schema_params.clone() }
    
    async fn execute(&self, args: Value) -> Result<String, AgentError> {
        let func = self.func.clone();
        let args_str = args.to_string(); 
        
        let result = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let json_module = py.import_bound("json")?;
                let kwargs = json_module.getattr("loads")?.call1((args_str,))?.downcast_into::<PyDict>()?;
                let res = func.bind(py).call((), Some(&kwargs))?;
                res.str()?.extract::<String>()
            })
        })
        .await
        .map_err(|e| AgentError::InternalError(format!("Thread Hatası: {}", e)))?
        .map_err(|e| AgentError::InternalError(format!("Python Fonksiyon Hatası: {}", e)))?;
        Ok(result)
    }
}