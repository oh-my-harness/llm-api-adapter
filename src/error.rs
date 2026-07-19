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

    /// The SSE stream ended abnormally before a `MessageStop` event was
    /// emitted (e.g. connection dropped mid-stream, or `[DONE]` arrived
    /// without `finish_reason` + `usage`).
    ///
    /// - `received_chunks`: number of stream events received before the
    ///   abnormal termination. The unit depends on which layer produced the
    ///   error — see below.
    ///   - **Layer 1** (provider `parse_stream` tail / SSE-`Err` arm): counts
    ///     SSE `data` events off the wire, including the `[DONE]` marker and
    ///     keepalive lines under tolerant parsing — not only successfully
    ///     deserialised chunks.
    ///   - **Layer 2** (`StreamHandle::collect` backstop, reached only by
    ///     direct `from_raw_stream` callers or a future provider lacking
    ///     Layer 1): counts `StreamEvent`s consumed after `flat_map` fan-out,
    ///     which may differ from the SSE frame count when one frame yields
    ///
    ///   Callers should treat the value as a rough diagnostic, not a precise
    ///   wire-level counter, unless they know which layer they received it
    ///   from. 0 = immediate drop / unknown.
    /// - `finish_reason`: the provider's `finish_reason`/`stop_reason` if one
    ///   was observed in any chunk before the drop. `None` means not even a
    ///   finish reason arrived — the most severe case (model never signalled
    ///   completion). `Some` means the model did finish but usage/stream
    ///   termination was lost.
    #[error(
        "stream incomplete: {received_chunks} chunks received, finish_reason={finish_reason:?}"
    )]
    StreamIncomplete {
        received_chunks: u32,
        finish_reason: Option<String>,
    },

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
