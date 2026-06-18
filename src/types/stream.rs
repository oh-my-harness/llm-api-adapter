use super::response::{StopReason, Usage};

/// Identifies what type of content a stream block carries.
///
/// Emitted in [`StreamEvent::ContentStart`] to tell the consumer what kind of
/// content deltas to expect for the given block index.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentKind {
    /// Plain text output. Followed by [`StreamEvent::TextDelta`] events.
    Text,
    /// Chain-of-thought reasoning. Followed by [`StreamEvent::ReasoningDelta`] events.
    Reasoning,
    /// A tool call — id and name are known at block start, arguments arrive
    /// as [`StreamEvent::ToolDelta`] events.
    ToolInvocation {
        /// Provider-assigned call ID, matching `ToolInvocation.id`.
        id: String,
        /// Name of the tool being called, matching `ToolDef.name`.
        name: String,
    },
}

/// A single event in a streaming response.
///
/// The stream follows a strict lifecycle per content block:
///
/// 1. [`ContentStart`](StreamEvent::ContentStart) — announces the start of a new block
/// 2. N × delta events (`TextDelta` | `ReasoningDelta` | `ToolDelta`) — incremental content
/// 3. [`ContentStop`](StreamEvent::ContentStop) — the block is complete
/// 4. [`MessageStop`](StreamEvent::MessageStop) — all blocks are done, carries final metadata
///
/// Multiple blocks can be interleaved (each has a unique `index`).
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// A new content block has started. The `kind` determines which delta
    /// events to expect.
    ContentStart { index: usize, kind: ContentKind },
    /// Incremental text for a text block (index matches a prior `ContentStart::Text`).
    TextDelta { index: usize, text: String },
    /// Incremental reasoning text for a reasoning block.
    ReasoningDelta { index: usize, text: String },
    /// Incremental JSON fragment for a tool-call arguments block.
    /// Fragments should be concatenated to build the full arguments JSON.
    ToolDelta { index: usize, arguments: String },
    /// A content block has finished.
    ContentStop {
        index: usize,
        /// Anthropic-specific reasoning block signature (if provided by the provider).
        /// `None` for text/tool blocks or non-Anthropic providers.
        signature: Option<String>,
    },
    /// The full message has finished; carries final stop reason and token usage.
    /// No further events follow this one.
    MessageStop {
        stop_reason: StopReason,
        usage: Usage,
    },
}
