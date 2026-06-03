use std::time::Duration;

/// Unified error type for all LLM provider operations.
///
/// This enum normalizes errors from different providers (OpenAI, Anthropic, etc.)
/// into a canonical set of error types. Provider-specific errors are converted
/// to these types when crossing the provider boundary.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Request was malformed or contained invalid parameters.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// API key missing or invalid.
    #[error("unauthorized")]
    Unauthorized,

    /// Key does not have permission for this operation.
    #[error("forbidden")]
    Forbidden,

    /// Rate limit hit; caller should back off.
    #[error("rate limited (retry after {retry_after:?})")]
    RateLimit { retry_after: Option<Duration> },

    /// Provider is overloaded; retry after delay.
    #[error("server overloaded (retry after {retry_after:?})")]
    Overloaded { retry_after: Option<Duration> },

    /// Unrecoverable server-side error.
    #[error("server error: {0}")]
    ServerError(String),

    /// Request or response timed out.
    #[error("request timed out")]
    Timeout,

    /// Error within an active SSE stream.
    #[error("stream error: {0}")]
    Stream(String),

    /// HTTP transport failure.
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    /// JSON serialisation or deserialisation failed.
    #[error("decode error: {0}")]
    Decode(#[from] serde_json::Error),

    /// Provider returned a structured error object.
    #[error("provider error {code}: {message}")]
    Provider { code: String, message: String },
}
