use serde::{Deserialize, Serialize};

/// Provider-neutral specification of model thinking/reasoning intensity.
///
/// Set via [`crate::types::ChatRequest::thinking`] on the universal path.
/// Each provider's converter translates this into its own wire format:
/// - Anthropic: `{type:"enabled", budget_tokens}` (4.6) or
///   `{type:"adaptive", effort}` (4.7+)
/// - OpenAI: top-level `reasoning_effort` string ("minimal".."xhigh")
/// - DeepSeek: `extra_body.thinking` object with `type` + `reasoning_effort`
///
/// **Cross-provider semantics of `Budget(n)`:** The exact token value is only
/// preserved on Anthropic 4.6 (`budget_tokens: n`). On OpenAI it is
/// interval-mapped to one of the 5 discrete effort levels; on DeepSeek it is
/// binary-mapped to `"high"` or `"max"`; on Anthropic 4.7+ it degrades to
/// `adaptive + aggressive` (the value is ignored).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingLevel {
    /// Disable thinking entirely.
    ///
    /// On the universal path, `Some(Off)` explicitly disables thinking
    /// (e.g. DeepSeek injects `{type:"disabled"}`). `None` (the default)
    /// injects no thinking field at all — DeepSeek's default thinking stays on.
    Off,
    /// Minimal reasoning effort.
    ///
    /// Maps to OpenAI `"minimal"`, Anthropic adaptive `"standard"`,
    /// DeepSeek `"high"` (lowest supported).
    Minimal,
    /// Low reasoning effort.
    ///
    /// Maps to OpenAI `"low"`, Anthropic adaptive `"standard"`,
    /// DeepSeek `"high"` (lowest supported).
    Low,
    /// Medium reasoning effort (balanced).
    ///
    /// Maps to OpenAI `"medium"`, Anthropic adaptive `"standard"`,
    /// DeepSeek `"high"` (lowest supported).
    Medium,
    /// High reasoning effort.
    ///
    /// Maps to OpenAI `"high"`, Anthropic adaptive `"aggressive"`,
    /// DeepSeek `"high"`.
    High,
    /// Extra-high reasoning effort (maximum on OpenAI).
    ///
    /// Maps to OpenAI `"xhigh"`, Anthropic adaptive `"aggressive"`,
    /// DeepSeek `"max"`.
    XHigh,
    /// Explicit token budget (Anthropic 4.6 semantics).
    ///
    /// On adaptive-only models (4.7+), degrades to adaptive+aggressive.
    /// On OpenAI, interval-mapped to one of the 5 discrete levels.
    /// On DeepSeek, binary-mapped to `"high"` (n ≤ 8192) or `"max"` (n > 8192).
    Budget(u32),
}
