pub mod math;
use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result; 

#[async_trait]
pub trait Tool: Send + Sync {
    
    fn name(&self) -> &'static str;

    fn description(&self) -> &'static str;

    fn schema(&self) -> Value;

    async fn execute(&self, args: Value) -> Result<String>;
}