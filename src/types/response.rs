use super::content::{ResponseContent, ToolInvocation};

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// Natural end of the model's response.
    EndTurn,
    /// Hit the `max_tokens` limit.
    MaxTokens,
    /// A stop sequence was matched.
    StopSequence,
    /// Model wants to call a tool.
    ToolUse,
    /// Content policy triggered.
    ContentFilter,
    /// Any other provider-specific stop reason.
    Other(String),
}

/// Token usage for a single request/response pair.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Usage {
    /// Tokens in the prompt (input).
    pub input_tokens: u32,
    /// Tokens generated (output).
    pub output_tokens: u32,
    /// Prompt tokens served from provider cache (OpenAI/DeepSeek).
    pub cached_input_tokens: u32,
    /// Tokens written to Anthropic prompt cache; always 0 for other providers.
    pub cache_creation_input_tokens: u32,
    /// Reasoning tokens consumed (DeepSeek thinking mode); 0 for other providers.
    pub reasoning_tokens: u32,
}

/// A complete (non-streaming) response from the model.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Provider-assigned message ID.
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
    pub fn into_message(self) -> super::message::Message {
        super::message::Message::Assistant(self.content)
    }

    /// Concatenate all `Text` blocks in order.
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

    /// Iterate over all `ToolInvocation` blocks.
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
