use super::response::{StopReason, Usage};

/// Identifies what type of content a stream block carries.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentKind {
    /// Plain text output.
    Text,
    /// Chain-of-thought reasoning.
    Reasoning,
    /// A tool call — id and name are known at block start, arguments arrive as deltas.
    ToolInvocation {
        /// Provider-assigned call ID.
        id: String,
        /// Name of the tool being called.
        name: String,
    },
}

/// A single event in a streaming response.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// A new content block has started.
    ContentStart { index: usize, kind: ContentKind },
    /// Incremental text for a text block.
    TextDelta { index: usize, text: String },
    /// Incremental reasoning text for a reasoning block.
    ReasoningDelta { index: usize, text: String },
    /// Incremental JSON fragment for a tool-call arguments block.
    ToolDelta { index: usize, arguments: String },
    /// A content block has finished.
    ContentStop { index: usize },
    /// The full message has finished; carries final usage.
    MessageStop { stop_reason: StopReason, usage: Usage },
}
