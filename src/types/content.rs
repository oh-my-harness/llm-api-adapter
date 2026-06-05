/// Source of image data for vision-capable requests.
///
/// OpenAI, Anthropic, and DeepSeek all support image input. Use [`Url`](ImageSource::Url)
/// for publicly accessible images, or [`Base64`](ImageSource::Base64) for inline data.
///
/// # Provider notes
///
/// | Provider   | Url | Base64 |
/// |------------|-----|--------|
/// | OpenAI     | Yes | Yes    |
/// | Anthropic  | Yes | Yes    |
/// | DeepSeek   | Yes | Yes    |
#[derive(Debug, Clone, PartialEq)]
pub enum ImageSource {
    /// Publicly accessible URL the provider fetches.
    Url(String),
    /// Inline base64-encoded image. `media_type` must be a valid MIME type
    /// (e.g. `"image/jpeg"`, `"image/png"`, `"image/webp"`).
    Base64 { media_type: String, data: String },
}

/// Content that a user or tool result produces as input to a request.
///
/// Construct a `Vec<RequestContent>` and pass it to `Message::User` or
/// `Message::Tool`. Each variant enforces the correct content type at compile time.
#[derive(Debug, Clone, PartialEq)]
pub enum RequestContent {
    /// Plain text.
    Text(String),
    /// Image (URL or base64). Not all providers or models support images;
    /// check the target model's capabilities.
    Image(ImageSource),
}

/// A tool call emitted by the model.
///
/// The `id` is assigned by the provider and **must** be echoed back verbatim
/// in the corresponding `Message::Tool` so the model can correlate the result
/// with this invocation.
///
/// Callers should validate `input` (a JSON object of named arguments) against
/// the tool's `input_schema` before executing the tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInvocation {
    /// Provider-assigned call ID. Required for tool result routing — must match
    /// `Message::Tool.invocation_id` in the follow-up turn.
    pub id: String,
    /// Name of the tool being called, matching a `ToolDef.name`.
    pub name: String,
    /// Arguments the model produced. Callers should validate this JSON against
    /// the corresponding `ToolDef.input_schema` before execution.
    pub input: serde_json::Value,
}

/// Content that the model produces in a response.
///
/// A `ChatResponse` holds a `Vec<ResponseContent>` representing the model's
/// output as ordered content blocks. Convenience methods on `ChatResponse`
/// — [`text()`](crate::types::ChatResponse::text),
/// [`tool_invocations()`](crate::types::ChatResponse::tool_invocations),
/// [`reasoning()`](crate::types::ChatResponse::reasoning) — iterate or
/// concatenate specific variants.
///
/// # Variant semantics
///
/// | Variant | When it appears |
/// |---------|----------------|
/// | [`Text`](ResponseContent::Text) | The model's visible output; present in every response. |
/// | [`ToolInvocation`](ResponseContent::ToolInvocation) | The model wants to call a tool (`stop_reason == ToolUse`). |
/// | [`Reasoning`](ResponseContent::Reasoning) | Chain-of-thought (OpenAI `reasoning_content`, DeepSeek `reasoning_content`, Anthropic `thinking`). |
///
/// # `signature` caveat
///
/// `Reasoning.signature` is populated **only** by Anthropic's non-streaming
/// endpoint. It is required for multi-turn extended thinking.  Use
/// `AnthropicNativeResponse::into_message()` to preserve it across turns.
/// All other providers and streaming responses set `signature: None`.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseContent {
    /// Plain text output from the model.
    Text(String),
    /// Tool the model wants to invoke. Corresponds to `StopReason::ToolUse`.
    ToolInvocation(ToolInvocation),
    /// Chain-of-thought reasoning text.
    ///
    /// `signature` is populated only by **Anthropic non-streaming** responses
    /// and must be preserved for multi-turn extended thinking.  For all other
    /// providers (OpenAI, DeepSeek) and streaming responses it is `None`.
    Reasoning {
        text: String,
        /// Anthropic-only thinking signature. `None` for all other providers
        /// and streaming responses. Pass through
        /// `AnthropicNativeResponse::into_message()` to preserve it.
        signature: Option<String>,
    },
}
