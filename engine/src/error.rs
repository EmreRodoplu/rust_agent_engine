use thiserror::Error;
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("LLM API ile iletişim kurulamadı veya bağlantı koptu: {0}")]
    ApiError(#[from] reqwest::Error),

    #[error("LLM geçersiz bir JSON döndürdü veya veri şeması uyuşmazlığı: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Araç (Tool) çalıştırılırken bir hata oluştu: {0}")]
    ToolExecutionError(String),

    #[error("Kritik: LLM hafızada olmayan '{0}' adında uydurma bir araç çağırmaya çalıştı!")]
    ToolNotFound(String),

    #[error("MCP Sunucusu ile iletişim hatası: {0}")]
    McpError(String),

    #[error("Sistem içi beklenmeyen hata: {0}")]
    InternalError(String),
}
pub type Result<T> = std::result::Result<T, AgentError>;