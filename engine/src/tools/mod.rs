use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

pub mod rag_tool;
pub mod schedule;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> String;

    fn description(&self) -> String;

    fn schema(&self) -> Value;

    async fn execute(&self, args: Value) -> Result<String>;
}