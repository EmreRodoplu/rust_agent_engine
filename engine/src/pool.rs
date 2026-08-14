use serde_json::{json, Value};
use std::collections::HashMap;
use crate::error::{AgentError, Result};
use crate::tools::Tool;

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
        let tool_name = tool.name();
        let full_schema = json!({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool.description(),
                "parameters": tool.schema()
            }
        });

        self.cached_schemas
            .retain(|s| s["function"]["name"].as_str() != Some(tool_name.as_str()));
        self.cached_schemas.push(full_schema);
        self.tools.insert(tool_name, tool);
    }

    pub fn get_tool_schemas(&self) -> Vec<Value> {
        self.cached_schemas.clone()
    }

    pub async fn execute_tool(&self, name: &str, args: Value) -> Result<String> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| AgentError::ToolNotFound(name.to_string()))?;

        tool.execute(args).await
    }
}
