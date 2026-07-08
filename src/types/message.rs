use super::content::{RequestContent, ResponseContent};

/// A single message in a conversation, with role encoded as an enum variant.
///
/// The role is enforced at compile time — you cannot put [`ResponseContent`] in
/// a [`Message::User`], nor [`RequestContent`] in a [`Message::Assistant`].
///
/// # Variant semantics
///
/// | Variant | Contents | Notes |
/// |---------|----------|-------|
/// | [`System`](Message::System) | One string | Provider-dependent; not all providers support a system role (Anthropic does, OpenAI supports it via a special message type). |
/// | [`User`](Message::User) | `Vec<RequestContent>` (text + images) | Standard human turn. |
/// | [`Assistant`](Message::Assistant) | `Vec<ResponseContent>` (text + tool calls + reasoning) | Model turn. Pass a prior response via `ChatResponse::into_message()` to continue a conversation. |
/// | [`Tool`](Message::Tool) | invocation_id + result content | Tool result turn. Every `Tool` message **must** be preceded by an `Assistant` message containing a matching `ToolInvocation`. |
///
/// # Tool call protocol
///
/// 1. The model replies with an `Assistant` message containing one or more
///    `ResponseContent::ToolInvocation` blocks.
/// 2. You execute each tool, collect results into `Tool` messages.
/// 3. Append the `Assistant` + all `Tool` messages to the conversation and
///    call the provider again. The model sees the tool results and continues.
///
/// Provider adapters own provider-specific wire normalization. For example,
/// OpenAI-compatible adapters encode [`Message::Tool`] as `role: "tool"`,
/// while Anthropic encodes tool results as `tool_result` blocks inside `user`
/// messages and normalizes adjacent user/tool-result turns to satisfy its
/// alternation rules. Callers should keep using this canonical protocol instead
/// of applying provider-specific message workarounds before invoking adapters.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// System-level instruction.
    ///
    /// Not all providers support a system message. When they do, behaviour
    /// varies: Anthropic treats it as a separate `system` parameter, OpenAI
    /// maps it to a `system`-role message, DeepSeek mirrors OpenAI.
    System(String),
    /// Human turn; may contain text and images.
    User(Vec<RequestContent>),
    /// Model turn; may contain text, tool invocations, and reasoning content.
    Assistant(Vec<ResponseContent>),
    /// Tool result returned to the model after it issued a `ToolInvocation`.
    ///
    /// The `invocation_id` **must** match the `id` from the corresponding
    /// `ToolInvocation` in the preceding `Assistant` message.
    Tool {
        /// Matches `ToolInvocation.id` from the preceding `Assistant` message.
        /// The provider uses this to correlate the result with the original call.
        invocation_id: String,
        /// Result content (text or images).
        content: Vec<RequestContent>,
        /// Set to `true` to indicate the tool execution failed.
        /// The model may then decide to retry or apologise.
        is_error: bool,
    },
}
