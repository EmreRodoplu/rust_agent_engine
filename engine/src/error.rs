use thiserror::Error;
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Could not contact the LLM API or the connection was lost: {0}")]
    ApiError(#[from] reqwest::Error),

    #[error("The LLM returned invalid JSON or the data schema did not match: {0}")]
    ParseError(#[from] serde_json::Error),
    
    #[error("System or memory layer error: {0}")]
    SystemError(#[from] anyhow::Error),

    #[error("An error occurred while executing the tool: {0}")]
    ToolExecutionError(String),

    #[error("Critical: the LLM attempted to call a non-existent tool named '{0}'!")]
    ToolNotFound(String),

    #[error("Communication error with the MCP server: {0}")]
    McpError(String),

    #[error("Unexpected internal system error: {0}")]
    InternalError(String),
}
pub type Result<T> = std::result::Result<T, AgentError>;