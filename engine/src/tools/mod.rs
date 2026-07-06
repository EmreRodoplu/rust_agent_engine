use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> String;

    fn description(&self) -> String;

    fn schema(&self) -> Value;

    async fn execute(&self, args: Value) -> Result<String>;
}