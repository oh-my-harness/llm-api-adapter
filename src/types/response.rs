use super::content::{ResponseContent, ToolInvocation};

pub const BILLING_USAGE_PROFILE_ID: &str = "llm-harness-billing-usage";
pub const BILLING_USAGE_PROFILE_VERSION: &str = "1";
pub const BILLING_USAGE_PROFILE_CONTRACT: &str = "input_tokens=gross_input;cache_read_tokens=input_subset;cache_creation_input_tokens=input_subset;cache_read_and_creation=disjoint;output_tokens=gross_output;reasoning_tokens=output_subset;uncached_input=input_tokens-cache_read_tokens-cache_creation_input_tokens";
pub const BILLING_USAGE_PROFILE_DIGEST: &str =
    "sha256:6afb4df926d32f3917ab28fb7ea30af87b20d7b81391d800c076f637c7eb73fb";

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
    /// Gross prompt tokens, including cache reads and cache writes.
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

impl Usage {
    /// Returns prompt tokens billed at the normal, non-cache input rate.
    pub fn billable_uncached_input_tokens(&self) -> Result<u32, BillingUsageProfileError> {
        let cached = self
            .cached_input_tokens
            .checked_add(self.cache_creation_input_tokens)
            .ok_or_else(|| {
                BillingUsageProfileError::new("cache input token categories overflow")
            })?;
        self.input_tokens.checked_sub(cached).ok_or_else(|| {
            BillingUsageProfileError::new(format!(
                "gross input tokens ({}) are smaller than cache read plus cache creation tokens ({cached})",
                self.input_tokens
            ))
        })
    }

    /// Validates the canonical gross-plus-exclusive-cache billing profile.
    pub fn validate_billing_profile(&self) -> Result<(), BillingUsageProfileError> {
        self.billable_uncached_input_tokens()?;
        if self.reasoning_tokens > self.output_tokens {
            return Err(BillingUsageProfileError::new(format!(
                "reasoning tokens ({}) exceed gross output tokens ({})",
                self.reasoning_tokens, self.output_tokens
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct BillingUsageProfileError {
    message: String,
}

impl BillingUsageProfileError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod billing_profile_tests {
    use super::*;

    #[test]
    fn rejects_cache_categories_larger_than_gross_input() {
        let usage = Usage {
            input_tokens: 100,
            cached_input_tokens: 80,
            cache_creation_input_tokens: 30,
            ..Default::default()
        };

        let error = usage.validate_billing_profile().unwrap_err();
        assert!(error.to_string().contains("smaller than cache read"));
    }

    #[test]
    fn rejects_reasoning_larger_than_gross_output() {
        let usage = Usage {
            output_tokens: 20,
            reasoning_tokens: 21,
            ..Default::default()
        };

        let error = usage.validate_billing_profile().unwrap_err();
        assert!(error.to_string().contains("exceed gross output"));
    }
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
