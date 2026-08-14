use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rust_agent_engine_core::tools::schedule::{ScheduledTask, TaskManager, TaskAction};
use std::sync::OnceLock;
use tokio::runtime::Runtime;

static SHARED_RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn get_runtime() -> &'static Runtime {
    SHARED_RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create Tokio runtime!"))
}

#[pyclass(name = "TaskManager")]
pub struct PyTaskManager {
    inner: TaskManager,
}

#[pymethods]
impl PyTaskManager {
    #[new]
    #[pyo3(signature = (redis_url=None))] 
    pub fn new(redis_url: Option<&str>) -> Self {
        let inner = TaskManager::new(redis_url);
        
        Self {
            inner
        }
    }

    #[pyo3(signature = (callback))]
    pub fn start_daemon(&self, callback: PyObject) {
        let manager = self.inner.clone();
        
        let executor = move |task: ScheduledTask| -> Result<(), String> {
            Python::with_gil(|py| {
                let (action_type, payload, args_str) = match task.action {
                    TaskAction::AutonomousGoal { prompt } => ("autonomous_goal", prompt, "".to_string()),
                    TaskAction::ExecuteTool { tool_name, args } => ("execute_tool", tool_name, args.to_string()),
                };

                let call_result = callback.call1(py, (task.id.clone(), action_type, payload, args_str));

                match call_result {
                    Ok(_) => Ok(()),
                    Err(e) => Err(format!("Python Callback crashed during execution: {:?}", e)),
                }
            })
        };

        let _guard = get_runtime().enter(); 
        manager.start_daemon(executor);
    }

    #[pyo3(signature = (prompt, execute_at_iso=None, delay_in_seconds=None))]
    pub fn add_autonomous_task(&self, prompt: &str, execute_at_iso: Option<String>, delay_in_seconds: Option<i64>) -> PyResult<()> {
        let task = ScheduledTask::new_autonomous(prompt.to_string(), execute_at_iso, delay_in_seconds)
            .map_err(|e| PyValueError::new_err(e))?;
        
        let manager = self.inner.clone();
        
        get_runtime().spawn(async move {
            if let Err(e) = manager.add_task(task).await {
                eprintln!("[TaskManager Binding] Failed to add task to backend: {}", e);
            }
        });

        Ok(())
    }

    #[pyo3(signature = (tool_name, args_json, execute_at_iso=None, delay_in_seconds=None))]
    pub fn add_tool_task(&self, tool_name: &str, args_json: &str, execute_at_iso: Option<String>, delay_in_seconds: Option<i64>) -> PyResult<()> {
        let args: serde_json::Value = serde_json::from_str(args_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON argument: {}", e)))?;

        let task = ScheduledTask::new_tool_execution(tool_name.to_string(), args, execute_at_iso, delay_in_seconds)
            .map_err(|e| PyValueError::new_err(e))?;
            
        let manager = self.inner.clone();
        
        get_runtime().spawn(async move {
            if let Err(e) = manager.add_task(task).await {
                eprintln!("[TaskManager Binding] Failed to add task to backend: {}", e);
            }
        });

        Ok(())
    }
}