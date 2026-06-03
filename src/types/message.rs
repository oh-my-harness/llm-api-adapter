use super::content::{RequestContent, ResponseContent};

/// A single message in a conversation, with role encoded as an enum variant.
///
/// Reasoning content can only appear inside `Assistant` — any other placement
/// is a compile-time error.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// System-level instruction (not all providers support this role).
    System(String),
    /// Human turn; may contain text and images.
    User(Vec<RequestContent>),
    /// Model turn; may contain text, tool invocations, and reasoning.
    Assistant(Vec<ResponseContent>),
    /// Tool result returned to the model after it issued a `ToolInvocation`.
    Tool {
        /// Matches `ToolInvocation.id` from the preceding `Assistant` message.
        invocation_id: String,
        /// Result content (text or images).
        content: Vec<RequestContent>,
        /// Set to `true` when the tool execution failed.
        is_error: bool,
    },
}
