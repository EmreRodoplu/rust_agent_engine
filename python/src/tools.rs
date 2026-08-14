use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use async_trait::async_trait;
use rust_agent_engine_core::tools::Tool;
use rust_agent_engine_core::error::AgentError;

pub struct PythonTool {
    pub name: String,        
    pub description: String, 
    pub schema_params: Value,
    pub func: PyObject,
}

#[async_trait]
impl Tool for PythonTool {
    fn name(&self) -> String { self.name.clone() }
    fn description(&self) -> String { self.description.clone() }
    fn schema(&self) -> Value { self.schema_params.clone() }

    async fn execute(&self, args: Value) -> Result<String, AgentError> {
        let func = self.func.clone();
        let args_str = args.to_string();

        
        let result = tokio::task::spawn_blocking(move || -> Result<String, AgentError> {
            Python::with_gil(|py| {
               
                let json_module = py.import_bound("json")
                    .map_err(|e| AgentError::ToolExecutionError(format!("Could not load the JSON module: {}", e)))?;
                
                let kwargs = json_module
                    .getattr("loads")
                    .map_err(|e| AgentError::ToolExecutionError(format!("The 'loads' method was not found: {}", e)))?
                    .call1((&args_str,))
                    .map_err(|e| AgentError::ToolExecutionError(format!("JSON parse error: {}", e)))?
                    .downcast_into::<PyDict>()
                    .map_err(|e| AgentError::ToolExecutionError(format!("Dictionary conversion error: {}", e)))?;

                
                match func.bind(py).call((), Some(&kwargs)) {
                    Ok(res) => {
                        let out_str: String = res.str()
                            .map_err(|e| AgentError::ToolExecutionError(format!("Could not convert output to string: {}", e)))?
                            .extract()
                            .map_err(|e| AgentError::ToolExecutionError(format!("String extraction error: {}", e)))?;
                        Ok(out_str)
                    },
                    Err(py_err) => {
                        let err_msg = py_err.value_bound(py).to_string();
                        
                        println!("[SECURITY WALL] The agent sent invalid parameters. Self-healing triggered!");
                        Err(AgentError::ToolExecutionError(
                            format!("(PYDANTIC SECURITY WALL) Function parameters are invalid: {}. Please correct the JSON data types to match the schema and try again.", err_msg)
                        ))
                    }
                }
            })
        })
        .await
        .map_err(|e| AgentError::ToolExecutionError(format!("Thread panic: {}", e)))??; 
        
        Ok(result)
    }
}