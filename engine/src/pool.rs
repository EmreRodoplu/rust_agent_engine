use std::collections::HashMap;
use serde_json::{json, Value};

use crate::tools::Tool; 
use crate::error::{AgentError, Result};
pub struct ToolPool {
    tools: HashMap<String, Box<dyn Tool>>,
    cached_schemas: Vec<Value>,
}

impl ToolPool {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            cached_schemas: Vec::new(),
        }
    }
    pub fn register_tool(&mut self, tool: Box<dyn Tool>) {
        let full_schema = json!({
            "type": "function",
            "function": {
                "name": tool.name(),
                "description": tool.description(),
                "parameters": tool.schema() 
            }
        });
        
        self.cached_schemas.push(full_schema);
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get_tool_schemas(&self) -> Vec<Value> {
        self.cached_schemas.clone()
    }

    pub async fn execute_tool(&self, name: &str, args: Value) -> Result<String> {
        
        let tool = self.tools.get(name).ok_or_else(|| {
            AgentError::InternalError(format!("Kayıtlı olmayan bir araç çağrıldı: {}", name))
        })?;
        
        tool.execute(args).await
    }
}