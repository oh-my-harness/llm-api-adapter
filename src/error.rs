use std::time::Duration;

/// Normalised error type for all LLM provider operations.
///
/// Maps provider-specific errors (OpenAI, Anthropic, DeepSeek, etc.) into
/// canonical variants. Callers can match on these variants to implement
/// retry logic, error reporting, or graceful degradation without coupling
/// to any specific provider's wire format.
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

/// Normalised error type for search provider operations.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
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

    /// Unrecoverable server-side error.
    #[error("server error: {0}")]
    ServerError(String),

    /// Request or response timed out.
    #[error("request timed out")]
    Timeout,

    /// HTTP transport failure.
    #[error("network error: {0}")]
    Network(reqwest::Error),

    /// JSON serialisation or deserialisation failed.
    #[error("decode error: {0}")]
    Decode(#[from] serde_json::Error),

    /// Provider response was syntactically valid but missing required data.
    #[error("invalid provider response: {0}")]
    InvalidResponse(String),

    /// Requested provider or feature is unsupported.
    #[error("unsupported provider: {0}")]
    UnsupportedProvider(String),

    /// Provider returned a structured error object.
    #[error("provider error {code}: {message}")]
    Provider { code: String, message: String },
}

impl From<reqwest::Error> for SearchError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            Self::Timeout
        } else {
            Self::Network(err)
        }
    }
}
