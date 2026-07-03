use super::content::{ResponseContent, ToolInvocation};

/// Why the model stopped generating.
///
/// Each variant corresponds to a termination condition that the model or
/// provider signals at the end of a response.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// The model finished its response naturally (no truncation).
    EndTurn,
    /// The response was truncated because it reached `max_tokens`.
    MaxTokens,
    /// The model encountered a user-specified stop sequence.
    StopSequence,
    /// The model wants to call a tool. The response will contain one or more
    /// [`ResponseContent::ToolInvocation`] blocks.
    ToolUse,
    /// The provider's content moderation filter triggered.
    ContentFilter,
    /// Any other provider-specific stop reason not covered by the variants above.
    Other(String),
}

/// Token usage for a single request/response pair.
///
/// Provider-specific fields are populated according to the API response:
///
/// | Field | OpenAI | Anthropic | DeepSeek |
/// |-------|--------|-----------|----------|
/// | `input_tokens` | Yes | Yes | Yes |
/// | `output_tokens` | Yes | Yes | Yes |
/// | `cached_input_tokens` | Yes | Yes | Yes |
/// | `cache_creation_input_tokens` | No | Yes | No |
/// | `reasoning_tokens` | Yes (o-series) | Yes (thinking) | Yes (thinking) |
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Usage {
    /// Tokens in the prompt (input / request).
    pub input_tokens: u32,
    /// Tokens generated (output / completion).
    pub output_tokens: u32,
    /// Prompt tokens served from the provider's cache. Supported by OpenAI
    /// (prompt caching), Anthropic (prompt caching), and DeepSeek (context caching).
    pub cached_input_tokens: u32,
    /// Tokens written to the Anthropic prompt cache. Always 0 for OpenAI and DeepSeek.
    pub cache_creation_input_tokens: u32,
    /// Reasoning/thinking tokens consumed. Populated by OpenAI o-series
    /// (`completion_tokens_details.reasoning_tokens`), Anthropic extended
    /// thinking (`output_tokens_details.thinking_tokens`), and DeepSeek
    /// thinking mode. Always 0 for non-reasoning models.
    pub reasoning_tokens: u32,
}

/// A complete (non-streaming) response from the model.
///
/// Provides convenience accessors for common tasks:
///
/// - [`text()`](ChatResponse::text) — concatenate all `Text` blocks
/// - [`tool_invocations()`](ChatResponse::tool_invocations) — iterate tool calls
/// - [`reasoning()`](ChatResponse::reasoning) — extract the first reasoning block
/// - [`into_message()`](ChatResponse::into_message) — create an `Assistant` message for multi-turn
///
/// Can also be obtained from a streaming response via
/// [`StreamHandle::collect()`](crate::stream_handle::StreamHandle::collect).
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Provider-assigned message ID. May be empty if collected from streaming.
    pub id: String,
    /// Model that generated this response.
    pub model: String,
    /// Ordered content blocks (text, tool invocations, reasoning).
    pub content: Vec<ResponseContent>,
    /// Reason the model stopped generating.
    pub stop_reason: StopReason,
    /// Token usage for this exchange.
    pub usage: Usage,
}

impl ChatResponse {
    /// Convert into a `Message::Assistant` for multi-turn continuations.
    ///
    /// This preserves all content blocks (including `Reasoning` with signatures)
    /// so the next request can continue the conversation seamlessly.
    pub fn into_message(self) -> super::message::Message {
        super::message::Message::Assistant(self.content)
    }

    /// Concatenate all [`Text`](ResponseContent::Text) blocks in order.
    ///
    /// Non-text blocks (tool invocations, reasoning) are silently skipped.
    /// This is the simplest way to get the model's visible text output.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| {
                if let ResponseContent::Text(t) = c {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Iterate over all [`ToolInvocation`](ToolInvocation) blocks.
    ///
    /// Returns an empty iterator if the model did not call any tools
    /// (i.e. `stop_reason != ToolUse`).
    pub fn tool_invocations(&self) -> impl Iterator<Item = &ToolInvocation> {
        self.content.iter().filter_map(|c| {
            if let ResponseContent::ToolInvocation(ti) = c {
                Some(ti)
            } else {
                None
            }
        })
    }

    /// Return the first reasoning block text, if any.
    ///
    /// Only present when the model produces chain-of-thought output (OpenAI
    /// `reasoning_content`, DeepSeek `reasoning_content`, or Anthropic
    /// `thinking`).
    pub fn reasoning(&self) -> Option<&str> {
        self.content.iter().find_map(|c| {
            if let ResponseContent::Reasoning { text, .. } = c {
                Some(text.as_str())
            } else {
                None
            }
        })
    }
}
