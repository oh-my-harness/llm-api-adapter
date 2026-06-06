use serde::{Deserialize, Serialize};

// ── Request ──────────────────────────────────────────────────────────────────

/// OpenAI Chat Completions API request.
#[derive(Debug, Serialize)]
pub(crate) struct OaiRequest {
    /// Model identifier (e.g., "gpt-4o").
    pub model: String,
    /// List of messages in the conversation.
    pub messages: Vec<OaiMessage>,
    /// List of tools available to the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OaiTool>,
    /// Tool choice mode: "auto", "required", specific tool, or null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Sampling temperature (0.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Nucleus sampling parameter (0.0 to 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Enable streaming responses.
    pub stream: bool,
    /// Stream options (e.g., include_usage).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<OaiStreamOptions>,
    /// Seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Frequency penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Unique identifier for tracking requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Reasoning effort for o-series / GPT-5 models ("low", "medium", "high", etc.).
    /// Has no effect on models that do not support reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Response format for structured output.
    /// `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {...}}`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
}

/// Options for streaming responses.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OaiStreamOptions {
    /// Include usage information in the final chunk.
    pub include_usage: bool,
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OaiMessage {
    /// Role: "system", "user", "assistant", or "tool".
    pub role: String,
    /// Message content (text or multipart).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OaiContent>,
    /// Tool calls made by the assistant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OaiToolCall>,
    /// Tool call ID (for tool role responses).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// DeepSeek reasoning content (present when parse_reasoning_content is enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Message content: either a single string or array of content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum OaiContent {
    /// Simple text string.
    Text(String),
    /// Array of mixed content (text, images, etc.).
    Parts(Vec<OaiContentPart>),
}

/// A single content part in a multipart message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OaiContentPart {
    /// Text content.
    Text { text: String },
    /// Image URL reference.
    ImageUrl { image_url: OaiImageUrl },
}

/// Image URL metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OaiImageUrl {
    /// The image URL.
    pub url: String,
}

/// A tool available to the model.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OaiTool {
    /// Tool type (always "function").
    pub r#type: String,
    /// Function definition.
    pub function: OaiFunction,
}

/// Function definition.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OaiFunction {
    /// Function name.
    pub name: String,
    /// Function description.
    pub description: String,
    /// JSON Schema describing parameters.
    pub parameters: serde_json::Value,
}

/// A tool call made by the assistant.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct OaiToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// Call type (always "function").
    pub r#type: String,
    /// Function call details.
    pub function: OaiToolCallFunction,
}

/// Function call details.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct OaiToolCallFunction {
    /// Function name.
    pub name: String,
    /// JSON-encoded string of arguments (not a parsed object).
    pub arguments: String,
}

// ── Response ─────────────────────────────────────────────────────────────────

/// OpenAI Chat Completions API response (non-streaming).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OaiResponse {
    /// Response ID.
    pub id: String,
    /// Model used.
    pub model: String,
    /// Completion choices.
    pub choices: Vec<OaiChoice>,
    /// Token usage information.
    #[serde(default)]
    pub usage: Option<OaiUsage>,
}

/// A completion choice.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OaiChoice {
    /// The generated message.
    pub message: OaiMessage,
    /// Reason for stopping (e.g., "stop", "tool_calls", "length").
    pub finish_reason: Option<String>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct OaiUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
    /// Detailed prompt token breakdown (OpenAI).
    #[serde(default)]
    pub prompt_tokens_details: Option<OaiPromptTokensDetails>,
    /// Detailed completion token breakdown (OpenAI: includes reasoning_tokens).
    #[serde(default)]
    pub completion_tokens_details: Option<OaiCompletionTokensDetails>,
    /// DeepSeek cache hit tokens.
    #[serde(default)]
    pub prompt_cache_hit_tokens: u32,
}

/// Detailed breakdown of prompt tokens.
#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct OaiPromptTokensDetails {
    /// Number of cached prompt tokens (OpenAI prompt caching).
    #[serde(default)]
    pub cached_tokens: u32,
}

/// Detailed breakdown of completion tokens.
#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct OaiCompletionTokensDetails {
    /// Tokens consumed by internal reasoning (o-series / GPT-5).
    #[serde(default)]
    pub reasoning_tokens: u32,
}

// ── Streaming ─────────────────────────────────────────────────────────────────

/// A chunk in a streaming response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // id/model are part of the wire schema; we read them via Value when needed.
pub(crate) struct OaiChunk {
    /// Response ID.
    #[serde(default)]
    pub id: String,
    /// Model used.
    #[serde(default)]
    pub model: String,
    /// Stream choices (delta updates).
    #[serde(default)]
    pub choices: Vec<OaiStreamChoice>,
    /// Token usage (only in final chunk if include_usage is true).
    #[serde(default)]
    pub usage: Option<OaiUsage>,
}

/// A choice in a streaming response.
#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamChoice {
    /// Delta content update.
    pub delta: OaiDelta,
    /// Finish reason (only in final chunk).
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming update.
#[derive(Debug, Deserialize, Default)]
pub(crate) struct OaiDelta {
    /// Text content delta.
    #[serde(default)]
    pub content: Option<String>,
    /// Tool call deltas.
    #[serde(default)]
    pub tool_calls: Vec<OaiStreamToolCall>,
    /// DeepSeek reasoning content delta.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

/// A tool call delta in a streaming response.
#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamToolCall {
    /// Index in the tool_calls array.
    pub index: usize,
    /// Tool call ID (only in first delta).
    #[serde(default)]
    pub id: Option<String>,
    /// Function call details (only if present).
    #[serde(default)]
    pub function: Option<OaiStreamFunction>,
}

/// Function call details in a streaming response.
#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamFunction {
    /// Function name (only in first delta).
    #[serde(default)]
    pub name: Option<String>,
    /// JSON-encoded string of arguments (streamed incrementally).
    #[serde(default)]
    pub arguments: Option<String>,
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// OpenAI error response.
#[derive(Debug, Deserialize)]
pub(crate) struct OaiErrorResponse {
    /// Error details.
    pub error: OaiErrorBody,
}

/// Error body.
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // message/type kept for future error-detail surfacing; only `code` is read today.
pub(crate) struct OaiErrorBody {
    /// Error message.
    pub message: String,
    /// Error type (e.g., "invalid_request_error").
    pub r#type: String,
    /// Error code (if available).
    #[serde(default)]
    pub code: Option<String>,
}
