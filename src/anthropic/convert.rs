use crate::LlmError;
use crate::types::{
    ChatRequest, ChatResponse, Message, RequestContent, ResponseContent, StopReason, ToolChoice,
    ToolInvocation, Usage,
};

use super::schema::{
    AnthropicContentBlock, AnthropicContentBlocks, AnthropicImageSource, AnthropicResponse,
    AnthropicResponseBlock, AnthropicUsage,
};

// ── AnthropicExt ──────────────────────────────────────────────────────────────

/// Provider-specific extensions for Anthropic requests.
///
/// These fields allow fine-tuning of Anthropic-specific behavior beyond the
/// canonical ChatRequest.
#[derive(Debug, Clone, Default)]
pub struct AnthropicExt {
    /// Only sample from the top K tokens; not all Anthropic models support this.
    pub top_k: Option<u32>,
    /// Disallow parallel tool use (default: false = parallel allowed).
    pub disable_parallel_tool_use: bool,
    /// Extended thinking configuration.
    pub thinking: Option<ThinkingConfig>,
    /// Metadata about the request.
    pub metadata: Option<AnthropicMetadata>,
    /// Hints for prompt caching applied to specific content blocks.
    pub cache_hints: Vec<CacheHint>,
    /// JSON schema for structured output via Strict Tool Use.
    ///
    /// When set, `to_anthropic()` injects a synthetic `"structured_result"` tool
    /// carrying this schema and forces `tool_choice` to that tool.  The caller
    /// extracts the result from `resp.tool_invocations().next()`.
    pub structured_output: Option<serde_json::Value>,
}

/// Extended thinking configuration for the native Anthropic path.
///
/// Choose the variant that matches the target model:
/// - 4.6 models (Sonnet/Opus): [`ThinkingConfig::Budget`]
/// - 4.7+ models (Opus 4.7 / Mythos): [`ThinkingConfig::Adaptive`]
///
/// If you are unsure which variant to use, prefer the canonical path
/// ([`crate::types::ChatRequest::thinking`] with [`crate::types::ThinkingLevel`]),
/// which auto-detects the model version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingConfig {
    /// 4.6 fixed budget: `{type:"enabled", budget_tokens}`.
    Budget {
        /// Maximum tokens to spend on thinking (must be ≥ 1024 and ≤ `max_tokens`).
        budget_tokens: u32,
    },
    /// 4.7+ adaptive: `thinking: {type:"adaptive"}` plus a top-level
    /// `output_config: {effort}` (effort is never nested inside `thinking`
    /// on the wire — the API rejects that shape).
    Adaptive {
        /// Effort level: one of `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`.
        effort: String,
    },
}

/// Metadata about the request.
#[derive(Debug, Clone)]
pub struct AnthropicMetadata {
    /// An external identifier for the user.
    pub user_id: Option<String>,
}

/// A cache control hint for a content block.
#[derive(Debug, Clone)]
pub struct CacheHint {
    /// Index of the content block to apply caching to.
    pub block_index: usize,
    /// Whether to mark this block as ephemeral cache.
    pub ephemeral: bool,
}

impl AnthropicExt {
    /// Create a new builder for AnthropicExt.
    pub fn builder() -> AnthropicExtBuilder {
        AnthropicExtBuilder::default()
    }
}

/// Builder for AnthropicExt with fluent API.
#[derive(Debug, Default)]
pub struct AnthropicExtBuilder {
    top_k: Option<u32>,
    disable_parallel_tool_use: bool,
    thinking: Option<ThinkingConfig>,
    metadata: Option<AnthropicMetadata>,
    cache_hints: Vec<CacheHint>,
    structured_output: Option<serde_json::Value>,
}

impl AnthropicExtBuilder {
    /// Set top_k sampling.
    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Disable parallel tool use.
    pub fn disable_parallel_tool_use(mut self) -> Self {
        self.disable_parallel_tool_use = true;
        self
    }

    /// Set extended thinking configuration.
    pub fn thinking(mut self, config: ThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }

    /// Set request metadata.
    pub fn metadata(mut self, meta: AnthropicMetadata) -> Self {
        self.metadata = Some(meta);
        self
    }

    /// Add a cache hint for a specific content block index.
    pub fn cache_hint(mut self, hint: CacheHint) -> Self {
        self.cache_hints.push(hint);
        self
    }

    /// Force the model to return a structured JSON result conforming to `schema`.
    ///
    /// Implemented via Strict Tool Use: a synthetic `"structured_result"` tool is
    /// injected into the request and `tool_choice` is forced to that tool.
    pub fn structured_output(mut self, schema: serde_json::Value) -> Self {
        self.structured_output = Some(schema);
        self
    }

    /// Build the AnthropicExt instance.
    pub fn build(self) -> AnthropicExt {
        AnthropicExt {
            top_k: self.top_k,
            disable_parallel_tool_use: self.disable_parallel_tool_use,
            thinking: self.thinking,
            metadata: self.metadata,
            cache_hints: self.cache_hints,
            structured_output: self.structured_output,
        }
    }
}

// ── AnthropicRequest ──────────────────────────────────────────────────────────

/// A ChatRequest paired with Anthropic provider-specific extensions.
pub struct AnthropicRequest {
    /// The canonical chat request.
    pub base: ChatRequest,
    /// Anthropic-specific extensions.
    pub ext: AnthropicExt,
}

// ── Conversion: Canonical → Anthropic ─────────────────────────────────────────

/// Convert a canonical ChatRequest to the Anthropic wire format.
///
/// # Arguments
/// * `req` - The canonical chat request
/// * `ext` - Optional Anthropic-specific extensions
/// * `stream` - Whether to enable streaming
///
/// # Returns
/// An `AnthropicRequest` ready to send to the Anthropic API, or an `LlmError`
/// if conversion fails (e.g., message alternation violation).
pub(crate) fn to_anthropic(
    req: &ChatRequest,
    ext: Option<&AnthropicExt>,
    stream: bool,
) -> Result<super::schema::AnthropicRequest, LlmError> {
    let default_ext = AnthropicExt::default();
    let ext = ext.unwrap_or(&default_ext);

    // Split system messages from the rest.
    let (system_blocks, conversation_messages): (Vec<&Message>, Vec<&Message>) = req
        .messages()
        .iter()
        .partition(|m| matches!(m, Message::System(_)));

    // Convert system messages to AnthropicContentBlocks (joined with '\n').
    let system = if system_blocks.is_empty() {
        None
    } else {
        let texts: Vec<&str> = system_blocks
            .iter()
            .filter_map(|m| {
                if let Message::System(t) = m {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        Some(AnthropicContentBlocks::Text(texts.join("\n")))
    };

    // Convert conversation messages (non-system) to wire format.
    let mut wire_messages = Vec::new();
    for msg in &conversation_messages {
        match msg {
            Message::User(content) => {
                wire_messages.push(user_content_to_wire(content));
            }
            Message::Assistant(content) => {
                wire_messages.push(assistant_content_to_wire(content));
            }
            Message::Tool {
                invocation_id,
                content,
                is_error,
            } => {
                // Tool results become a user message with a tool_result block.
                let tool_result_block = AnthropicContentBlock::ToolResult {
                    tool_use_id: invocation_id.clone(),
                    content: request_content_to_blocks(content),
                    is_error: if *is_error { Some(true) } else { None },
                };
                wire_messages.push(AnthropicWireMessageCompat {
                    role: "user".into(),
                    content: AnthropicContentBlocks::Blocks(vec![tool_result_block]),
                });
            }
            Message::System(_) => {
                // Already handled above — skip.
            }
        }
    }

    let wire_messages = normalize_anthropic_wire_messages(wire_messages);

    reject_leading_tool_result(&wire_messages)?;

    // Enforce alternation: user/assistant must alternate and first must be user.
    check_alternation(&wire_messages)?;

    // Convert tools.
    let mut tools: Vec<super::schema::AnthropicTool> = req
        .tools()
        .iter()
        .map(|t| super::schema::AnthropicTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect();

    // Inject synthetic structured-output tool if requested.
    if let Some(schema) = &ext.structured_output {
        tools.push(super::schema::AnthropicTool {
            name: "structured_result".into(),
            description: "Return the structured result.".into(),
            input_schema: schema.clone(),
        });
    }

    // Convert tool choice.
    let tool_choice = if ext.structured_output.is_some() {
        // Force the synthetic tool regardless of any canonical tool_choice.
        Some(serde_json::json!({"type": "tool", "name": "structured_result"}))
    } else if req.tools().is_empty() {
        None
    } else {
        Some(tool_choice_to_anthropic(req.tool_choice()))
    };

    let params = req.params();

    // Convert thinking + effort config: ext.thinking (native) overrides req.thinking
    // (canonical). The native ThinkingConfig enum lets callers explicitly choose
    // Budget (4.6) or Adaptive (4.7+). The canonical ThinkingLevel path auto-detects
    // via is_adaptive_model(model). ThinkingLevel::Off means "omit thinking field
    // and output_config.effort entirely".
    //
    // `effort` is returned separately from the thinking config because the API
    // expects it as a top-level `output_config.effort` sibling, not nested inside
    // `thinking` (nesting it there is rejected: "thinking.adaptive.effort: Extra
    // inputs are not permitted").
    let (thinking, effort) = if let Some(native) = ext.thinking.as_ref() {
        match native {
            ThinkingConfig::Budget { budget_tokens } => (
                Some(super::schema::AnthropicThinkingConfig {
                    r#type: "enabled".into(),
                    budget_tokens: Some(clamp_budget(*budget_tokens, params.max_tokens())),
                }),
                None,
            ),
            ThinkingConfig::Adaptive { effort } => (
                Some(super::schema::AnthropicThinkingConfig {
                    r#type: "adaptive".into(),
                    budget_tokens: None,
                }),
                Some(effort.clone()),
            ),
        }
    } else if let Some(spec) = req.thinking() {
        thinking_to_anthropic(spec, req.model(), params.max_tokens())
    } else {
        (None, None)
    };

    let output_config =
        effort.map(|effort| super::schema::AnthropicOutputConfig {
            effort: Some(effort),
        });

    // Convert metadata.
    let metadata = ext
        .metadata
        .as_ref()
        .map(|m| super::schema::AnthropicMetadata {
            user_id: m.user_id.clone(),
        });

    // Convert messages from our helper struct back to schema type.
    let messages: Vec<super::schema::AnthropicWireMessage> = wire_messages
        .into_iter()
        .map(|wm| super::schema::AnthropicWireMessage {
            role: wm.role,
            content: wm.content,
        })
        .collect();

    // Build the cache hints map for content block cache_control.
    let messages_with_cache = apply_cache_hints(messages, &ext.cache_hints);

    Ok(super::schema::AnthropicRequest {
        model: req.model().into(),
        max_tokens: params.max_tokens(),
        messages: messages_with_cache,
        system,
        tools,
        tool_choice,
        temperature: params.temperature(),
        top_p: params.top_p(),
        top_k: ext.top_k,
        stop_sequences: params.stop_sequences().to_vec(),
        stream,
        thinking,
        output_config,
        metadata,
        disable_parallel_tool_use: if ext.disable_parallel_tool_use {
            Some(true)
        } else {
            None
        },
    })
}

/// A local helper to hold wire messages during construction (avoids borrowing issues
/// when appending tool results).
struct AnthropicWireMessageCompat {
    role: String,
    content: AnthropicContentBlocks,
}

/// Merge adjacent wire messages with the same role.
///
/// Anthropic expresses tool results as user content blocks, so canonical
/// `Tool + Tool` and `User + Tool` sequences naturally become adjacent user
/// messages. Normalizing them here keeps the runtime/provider boundary clean:
/// callers can pass canonical messages, and this adapter owns Anthropic's
/// alternation requirement.
fn normalize_anthropic_wire_messages(
    messages: Vec<AnthropicWireMessageCompat>,
) -> Vec<AnthropicWireMessageCompat> {
    let mut normalized: Vec<AnthropicWireMessageCompat> = Vec::new();

    for message in messages {
        if let Some(last) = normalized.last_mut()
            && last.role == message.role
        {
            append_anthropic_content(&mut last.content, message.content);
            continue;
        }

        normalized.push(message);
    }

    normalized
}

fn append_anthropic_content(target: &mut AnthropicContentBlocks, addition: AnthropicContentBlocks) {
    let mut blocks = anthropic_content_into_blocks(std::mem::replace(
        target,
        AnthropicContentBlocks::Blocks(Vec::new()),
    ));
    blocks.extend(anthropic_content_into_blocks(addition));
    *target = AnthropicContentBlocks::Blocks(blocks);
}

fn anthropic_content_into_blocks(content: AnthropicContentBlocks) -> Vec<AnthropicContentBlock> {
    match content {
        AnthropicContentBlocks::Text(text) => vec![AnthropicContentBlock::Text {
            text,
            cache_control: None,
        }],
        AnthropicContentBlocks::Blocks(blocks) => blocks,
    }
}

fn reject_leading_tool_result(messages: &[AnthropicWireMessageCompat]) -> Result<(), LlmError> {
    let Some(first) = messages.first() else {
        return Ok(());
    };

    if first.role == "user" && content_starts_with_tool_result(&first.content) {
        return Err(LlmError::InvalidRequest(
            "Anthropic tool results require a preceding assistant tool invocation".into(),
        ));
    }

    Ok(())
}

fn content_starts_with_tool_result(content: &AnthropicContentBlocks) -> bool {
    match content {
        AnthropicContentBlocks::Text(_) => false,
        AnthropicContentBlocks::Blocks(blocks) => {
            matches!(
                blocks.first(),
                Some(AnthropicContentBlock::ToolResult { .. })
            )
        }
    }
}

/// Convert user request content to an Anthropic wire message.
fn user_content_to_wire(content: &[RequestContent]) -> AnthropicWireMessageCompat {
    let blocks = request_content_to_blocks(content);
    AnthropicWireMessageCompat {
        role: "user".into(),
        content: blocks,
    }
}

/// Convert assistant response content to an Anthropic wire message.
fn assistant_content_to_wire(content: &[ResponseContent]) -> AnthropicWireMessageCompat {
    let mut blocks = Vec::new();
    for item in content {
        match item {
            ResponseContent::Text(t) => {
                blocks.push(AnthropicContentBlock::Text {
                    text: t.clone(),
                    cache_control: None,
                });
            }
            ResponseContent::ToolInvocation(ti) => {
                blocks.push(AnthropicContentBlock::ToolUse {
                    id: ti.id.clone(),
                    name: ti.name.clone(),
                    input: ti.input.clone(),
                });
            }
            ResponseContent::Reasoning {
                text,
                signature: Some(sig),
            } => {
                // Re-encode as Thinking block with its signature for multi-turn compatibility.
                blocks.push(AnthropicContentBlock::Thinking {
                    thinking: text.clone(),
                    signature: sig.clone(),
                });
            }
            ResponseContent::Reasoning {
                text,
                signature: None,
            } => {
                // Without a signature we cannot emit a Thinking block — the API would
                // reject it. This path is hit when (a) using the universal Provider::chat()
                // path, (b) collecting a streamed response (chat_stream_native), or
                // (c) mixing non-Anthropic reasoning into an Anthropic request.
                // Encode as text as a best-effort fallback.
                tracing::warn!(
                    "Reasoning block has no signature; encoding as text for Anthropic request"
                );
                blocks.push(AnthropicContentBlock::Text {
                    text: text.clone(),
                    cache_control: None,
                });
            }
        }
    }
    let content = if blocks.len() == 1 {
        match &blocks[0] {
            AnthropicContentBlock::Text { text, .. } => AnthropicContentBlocks::Text(text.clone()),
            _ => AnthropicContentBlocks::Blocks(blocks),
        }
    } else {
        AnthropicContentBlocks::Blocks(blocks)
    };
    AnthropicWireMessageCompat {
        role: "assistant".into(),
        content,
    }
}

/// Convert canonical `RequestContent` items into `AnthropicContentBlocks`.
fn request_content_to_blocks(content: &[RequestContent]) -> AnthropicContentBlocks {
    let mut blocks: Vec<AnthropicContentBlock> = Vec::new();
    for item in content {
        match item {
            RequestContent::Text(t) => {
                blocks.push(AnthropicContentBlock::Text {
                    text: t.clone(),
                    cache_control: None,
                });
            }
            RequestContent::Image(img) => {
                let (source_type, media_type, data, url) = match img {
                    crate::types::ImageSource::Url(u) => {
                        ("url".into(), None, None, Some(u.clone()))
                    }
                    crate::types::ImageSource::Base64 { media_type, data } => (
                        "base64".into(),
                        Some(media_type.clone()),
                        Some(data.clone()),
                        None,
                    ),
                };
                blocks.push(AnthropicContentBlock::Image {
                    source: AnthropicImageSource {
                        source_type,
                        media_type,
                        data,
                        url,
                    },
                });
            }
        }
    }

    if blocks.len() == 1
        && let AnthropicContentBlock::Text { text, .. } = &blocks[0]
    {
        return AnthropicContentBlocks::Text(text.clone());
    }
    AnthropicContentBlocks::Blocks(blocks)
}

/// Enforce that messages alternate user/assistant and start with user.
///
/// Anthropic requires messages to strictly alternate between `user` and `assistant`
/// roles, with the first message being from the user.
fn check_alternation(messages: &[AnthropicWireMessageCompat]) -> Result<(), LlmError> {
    if messages.is_empty() {
        return Err(LlmError::InvalidRequest(
            "Anthropic requires at least one user message".into(),
        ));
    }

    if messages[0].role != "user" {
        return Err(LlmError::InvalidRequest(
            "Anthropic requires the first message to have role 'user'".into(),
        ));
    }

    for i in 1..messages.len() {
        let prev = &messages[i - 1].role;
        let curr = &messages[i].role;
        if prev == curr {
            return Err(LlmError::InvalidRequest(format!(
                "Anthropic messages must alternate user/assistant; got two '{prev}' messages in a row"
            )));
        }
    }

    Ok(())
}

/// Apply cache control hints to content blocks in messages.
fn apply_cache_hints(
    mut messages: Vec<super::schema::AnthropicWireMessage>,
    hints: &[CacheHint],
) -> Vec<super::schema::AnthropicWireMessage> {
    let mut flat_index = 0usize;
    for msg in &mut messages {
        // Take the content to modify it.
        let new_content = match std::mem::replace(
            &mut msg.content,
            AnthropicContentBlocks::Text(String::new()),
        ) {
            AnthropicContentBlocks::Blocks(mut blocks) => {
                for block in &mut blocks {
                    for hint in hints {
                        if hint.block_index == flat_index && hint.ephemeral {
                            apply_cache_control_to_block(block);
                        }
                    }
                    flat_index += 1;
                }
                AnthropicContentBlocks::Blocks(blocks)
            }
            AnthropicContentBlocks::Text(text) => {
                let hint_matches = hints
                    .iter()
                    .any(|h| h.block_index == flat_index && h.ephemeral);
                flat_index += 1;
                if hint_matches {
                    let mut block = AnthropicContentBlock::Text {
                        text,
                        cache_control: None,
                    };
                    apply_cache_control_to_block(&mut block);
                    AnthropicContentBlocks::Blocks(vec![block])
                } else {
                    AnthropicContentBlocks::Text(text)
                }
            }
        };
        msg.content = new_content;
    }

    messages
}

/// Add ephemeral cache_control to a content block if it supports it.
fn apply_cache_control_to_block(block: &mut AnthropicContentBlock) {
    if let AnthropicContentBlock::Text { cache_control, .. } = block {
        *cache_control = Some(super::schema::AnthropicCacheControl {
            r#type: "ephemeral".into(),
        });
    }
}

/// Convert a canonical ToolChoice to an Anthropic tool_choice JSON value.
fn tool_choice_to_anthropic(tc: &ToolChoice) -> serde_json::Value {
    match tc {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::None => serde_json::json!({"type": "none"}),
        ToolChoice::Required => serde_json::json!({"type": "any"}),
        ToolChoice::Specific(name) => serde_json::json!({
            "type": "tool",
            "name": name
        }),
    }
}

// ── Thinking conversion ──────────────────────────────────────────────────────

/// Detect Anthropic 4.7+ / Mythos models that use adaptive thinking wire format.
///
/// This is a heuristic based on model name substrings. The canonical path
/// ([`thinking_to_anthropic`]) uses this to decide between `enabled` (4.6)
/// and `adaptive` (4.7+) wire formats. For the native path, prefer explicitly
/// choosing the [`ThinkingConfig`] variant that matches your model.
pub(crate) fn is_adaptive_model(model: &str) -> bool {
    let m = model.to_lowercase();
    // 4.7+ and later models use the adaptive wire format. Budget_tokens is
    // deprecated/rejected on these. We match by version substring to cover
    // current and foreseeable model names (opus-4-7, opus-4-8, sonnet-5,
    // fable-5, mythos, etc.). For unknown future models, prefer the native
    // ThinkingConfig::Adaptive path explicitly.
    m.contains("4-7")
        || m.contains("4-8")
        || m.contains("4-9")
        || m.contains("sonnet-5")
        || m.contains("fable")
        || m.contains("mythos")
}

/// Minimum budget_tokens Anthropic accepts (per official docs).
const MIN_BUDGET_TOKENS: u32 = 1024;

/// Clamp a thinking budget to the Anthropic constraints: ≥ 1024 and < max_tokens.
///
/// If `max_tokens` is too small to leave room for the minimum budget, the
/// budget is clamped to `max_tokens - 1` (which will still be rejected by the
/// API — the caller's `max_tokens` is simply too small for thinking).
fn clamp_budget(budget: u32, max_tokens: u32) -> u32 {
    let upper = max_tokens.saturating_sub(1);
    budget
        .max(MIN_BUDGET_TOKENS)
        .min(upper.max(MIN_BUDGET_TOKENS))
}

/// Map a canonical [`ThinkingLevel`] to an Anthropic thinking config and,
/// on adaptive models, an `output_config.effort` value.
///
/// Returns `(None, None)` for [`ThinkingLevel::Off`] — callers should omit
/// both the thinking field and output_config.effort entirely in that case.
///
/// The `model` string is used by [`is_adaptive_model`] to choose between
/// the 4.6 (`enabled` + `budget_tokens`) and 4.7+ (`adaptive` type, with a
/// separate top-level `effort`) wire formats. `max_tokens` is used to clamp
/// `budget_tokens` so it stays within the Anthropic constraint
/// (≥ 1024, < max_tokens).
///
/// `effort` values sent to the API must be one of the levels Anthropic
/// actually accepts: `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"` —
/// arbitrary strings like `"standard"`/`"aggressive"` are rejected.
pub(crate) fn thinking_to_anthropic(
    spec: crate::types::ThinkingLevel,
    model: &str,
    max_tokens: u32,
) -> (Option<super::schema::AnthropicThinkingConfig>, Option<String>) {
    use crate::types::ThinkingLevel;
    if spec == ThinkingLevel::Off {
        return (None, None);
    }
    if is_adaptive_model(model) {
        let effort = match spec {
            ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High => "high",
            ThinkingLevel::XHigh => "xhigh",
            ThinkingLevel::Budget(_) => "high",
            ThinkingLevel::Off => unreachable!("handled above"),
        };
        (
            Some(super::schema::AnthropicThinkingConfig {
                r#type: "adaptive".into(),
                budget_tokens: None,
            }),
            Some(effort.into()),
        )
    } else {
        let budget = match spec {
            ThinkingLevel::Minimal => 2048,
            ThinkingLevel::Low => 4096,
            ThinkingLevel::Medium => 8192,
            ThinkingLevel::High => 16384,
            ThinkingLevel::XHigh => 32768,
            ThinkingLevel::Budget(n) => n,
            ThinkingLevel::Off => unreachable!("handled above"),
        };
        (
            Some(super::schema::AnthropicThinkingConfig {
                r#type: "enabled".into(),
                budget_tokens: Some(clamp_budget(budget, max_tokens)),
            }),
            None,
        )
    }
}

// ── Conversion: Anthropic → Canonical ─────────────────────────────────────────

/// Convert an Anthropic response to a canonical ChatResponse.
///
/// # Arguments
/// * `resp` - The Anthropic response
///
/// # Returns
/// A ChatResponse in canonical form, or an LlmError if conversion fails.
pub(crate) fn from_anthropic(resp: AnthropicResponse) -> Result<ChatResponse, LlmError> {
    let content: Vec<ResponseContent> = resp
        .content
        .into_iter()
        .filter_map(anthropic_response_block_to_canonical)
        .collect();

    let stop_reason = parse_anthropic_stop_reason(resp.stop_reason.as_deref());
    let usage = anthropic_usage_to_canonical(resp.usage);

    Ok(ChatResponse {
        id: resp.id,
        model: resp.model,
        content,
        stop_reason,
        usage,
    })
}

/// Convert an Anthropic response block to a canonical ResponseContent.
fn anthropic_response_block_to_canonical(block: AnthropicResponseBlock) -> Option<ResponseContent> {
    match block {
        AnthropicResponseBlock::Text { text } if !text.is_empty() => {
            Some(ResponseContent::Text(text))
        }
        AnthropicResponseBlock::Text { .. } => None,
        AnthropicResponseBlock::ToolUse { id, name, input } => {
            Some(ResponseContent::ToolInvocation(ToolInvocation {
                id,
                name,
                input,
            }))
        }
        AnthropicResponseBlock::Thinking {
            thinking,
            signature,
        } if !thinking.is_empty() => {
            let sig = if signature.is_empty() {
                None
            } else {
                Some(signature)
            };
            Some(ResponseContent::Reasoning {
                text: thinking,
                signature: sig,
            })
        }
        AnthropicResponseBlock::Thinking { .. } => None,
        AnthropicResponseBlock::RedactedThinking { data } => Some(ResponseContent::Reasoning {
            text: format!("[redacted_thinking: {data}]"),
            signature: None,
        }),
    }
}

/// Parse an Anthropic stop_reason string to a canonical StopReason.
fn parse_anthropic_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("end_turn") => StopReason::EndTurn,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some("tool_use") => StopReason::ToolUse,
        Some(other) => match other {
            "pause_turn" | "refusal" => StopReason::ContentFilter,
            _ => StopReason::Other(other.into()),
        },
        None => StopReason::Other("unknown".into()),
    }
}

/// Convert AnthropicUsage to canonical Usage.
pub(crate) fn anthropic_usage_to_canonical(u: AnthropicUsage) -> Usage {
    Usage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        cached_input_tokens: u.cache_read_input_tokens,
        cache_creation_input_tokens: u.cache_creation_input_tokens,
        reasoning_tokens: u.output_tokens_details.thinking_tokens,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ImageSource, RequestContent as RC, ResponseContent as RespC, ToolDef, ToolInvocation as TI,
    };

    #[test]
    fn ext_builder_defaults() {
        let ext = AnthropicExt::builder().build();
        assert!(ext.top_k.is_none());
        assert!(!ext.disable_parallel_tool_use);
        assert!(ext.thinking.is_none());
        assert!(ext.metadata.is_none());
        assert!(ext.cache_hints.is_empty());
    }

    #[test]
    fn ext_builder_with_fields() {
        let ext = AnthropicExt::builder()
            .top_k(50)
            .disable_parallel_tool_use()
            .thinking(ThinkingConfig::Budget {
                budget_tokens: 2000,
            })
            .metadata(AnthropicMetadata {
                user_id: Some("user_123".into()),
            })
            .cache_hint(CacheHint {
                block_index: 0,
                ephemeral: true,
            })
            .build();

        assert_eq!(ext.top_k, Some(50));
        assert!(ext.disable_parallel_tool_use);
        assert_eq!(
            ext.thinking.as_ref(),
            Some(&ThinkingConfig::Budget {
                budget_tokens: 2000
            })
        );
        assert_eq!(
            ext.metadata.as_ref().and_then(|m| m.user_id.as_deref()),
            Some("user_123")
        );
        assert_eq!(ext.cache_hints.len(), 1);
    }

    #[test]
    fn system_message_extracted_to_top_level() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::System("You are helpful.".into()))
            .message(Message::User(vec![RC::Text("Hello".into())]))
            .build();
        let wire = to_anthropic(&req, None, false).unwrap();
        // System should be extracted to top-level
        assert!(wire.system.is_some());
        if let Some(AnthropicContentBlocks::Text(s)) = &wire.system {
            assert_eq!(s, "You are helpful.");
        } else {
            panic!("expected Text system");
        }
        // Only the user message remains in the messages array
        assert_eq!(wire.messages.len(), 1);
        assert_eq!(wire.messages[0].role, "user");
    }

    #[test]
    fn multiple_system_messages_joined() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::System("Be concise.".into()))
            .message(Message::System("Be helpful.".into()))
            .message(Message::User(vec![RC::Text("Hi".into())]))
            .build();
        let wire = to_anthropic(&req, None, false).unwrap();
        if let Some(AnthropicContentBlocks::Text(s)) = &wire.system {
            assert_eq!(s, "Be concise.\nBe helpful.");
        } else {
            panic!("expected Text system");
        }
        assert_eq!(wire.messages.len(), 1);
    }

    #[test]
    fn tool_result_becomes_user_message_with_tool_result_block() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("What's the weather?".into())]))
            .message(Message::Assistant(vec![RespC::ToolInvocation(TI {
                id: "toolu_01".into(),
                name: "get_weather".into(),
                input: serde_json::json!({"location": "SF"}),
            })]))
            .message(Message::Tool {
                invocation_id: "toolu_01".into(),
                content: vec![RC::Text("22°C".into())],
                is_error: false,
            })
            .build();
        let wire = to_anthropic(&req, None, false).unwrap();
        // Should be 3 messages: user, assistant, user
        assert_eq!(wire.messages.len(), 3);
        assert_eq!(wire.messages[0].role, "user");
        assert_eq!(wire.messages[1].role, "assistant");
        assert_eq!(wire.messages[2].role, "user");
        // The last user message should contain a tool_result block
        match &wire.messages[2].content {
            AnthropicContentBlocks::Blocks(blocks) => {
                assert!(matches!(
                    blocks[0],
                    AnthropicContentBlock::ToolResult { .. }
                ));
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    fn adjacent_user_messages_are_normalized() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hello".into())]))
            .message(Message::User(vec![RC::Text("World".into())]))
            .build();
        let wire = to_anthropic(&req, None, false).unwrap();

        assert_eq!(wire.messages.len(), 1);
        assert_eq!(wire.messages[0].role, "user");
        match &wire.messages[0].content {
            AnthropicContentBlocks::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(
                    matches!(&blocks[0], AnthropicContentBlock::Text { text, .. } if text == "Hello")
                );
                assert!(
                    matches!(&blocks[1], AnthropicContentBlock::Text { text, .. } if text == "World")
                );
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    fn cache_hint_on_single_text_message_is_applied() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hello, cache me.".into())]))
            .build();

        let ext = AnthropicExt::builder()
            .cache_hint(CacheHint {
                block_index: 0,
                ephemeral: true,
            })
            .build();

        let wire = to_anthropic(&req, Some(&ext), false).unwrap();

        // The first message's content should be Blocks form (array), not Text
        // form (plain string), because the cache hint forced an upgrade.
        match &wire.messages[0].content {
            AnthropicContentBlocks::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    AnthropicContentBlock::Text {
                        text,
                        cache_control,
                    } => {
                        assert_eq!(text, "Hello, cache me.");
                        assert!(cache_control.is_some(), "cache_control must be set");
                    }
                    other => panic!("expected Text block, got {other:?}"),
                }
            }
            other => panic!("expected Blocks form (cache hint should upgrade Text), got {other:?}"),
        }
    }

    #[test]
    fn adjacent_tool_results_are_normalized_into_one_user_message() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Use both tools.".into())]))
            .message(Message::Assistant(vec![
                RespC::ToolInvocation(TI {
                    id: "toolu_01".into(),
                    name: "first".into(),
                    input: serde_json::json!({}),
                }),
                RespC::ToolInvocation(TI {
                    id: "toolu_02".into(),
                    name: "second".into(),
                    input: serde_json::json!({}),
                }),
            ]))
            .message(Message::Tool {
                invocation_id: "toolu_01".into(),
                content: vec![RC::Text("first result".into())],
                is_error: false,
            })
            .message(Message::Tool {
                invocation_id: "toolu_02".into(),
                content: vec![RC::Text("second result".into())],
                is_error: true,
            })
            .build();

        let wire = to_anthropic(&req, None, false).unwrap();

        assert_eq!(wire.messages.len(), 3);
        assert_eq!(wire.messages[0].role, "user");
        assert_eq!(wire.messages[1].role, "assistant");
        assert_eq!(wire.messages[2].role, "user");
        match &wire.messages[2].content {
            AnthropicContentBlocks::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(
                    &blocks[0],
                    AnthropicContentBlock::ToolResult { tool_use_id, is_error, .. }
                        if tool_use_id == "toolu_01" && is_error.is_none()
                ));
                assert!(matches!(
                    &blocks[1],
                    AnthropicContentBlock::ToolResult { tool_use_id, is_error, .. }
                        if tool_use_id == "toolu_02" && *is_error == Some(true)
                ));
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    fn tool_result_after_user_is_appended_to_that_user_message() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Context for result.".into())]))
            .message(Message::Tool {
                invocation_id: "toolu_01".into(),
                content: vec![RC::Text("result".into())],
                is_error: false,
            })
            .build();

        let wire = to_anthropic(&req, None, false).unwrap();

        assert_eq!(wire.messages.len(), 1);
        assert_eq!(wire.messages[0].role, "user");
        match &wire.messages[0].content {
            AnthropicContentBlocks::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(
                    matches!(&blocks[0], AnthropicContentBlock::Text { text, .. } if text == "Context for result.")
                );
                assert!(matches!(
                    &blocks[1],
                    AnthropicContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "toolu_01"
                ));
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    fn adjacent_assistant_messages_are_normalized() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hi".into())]))
            .message(Message::Assistant(vec![RespC::Text("Thinking.".into())]))
            .message(Message::Assistant(vec![RespC::ToolInvocation(TI {
                id: "toolu_01".into(),
                name: "lookup".into(),
                input: serde_json::json!({"q": "x"}),
            })]))
            .build();

        let wire = to_anthropic(&req, None, false).unwrap();

        assert_eq!(wire.messages.len(), 2);
        assert_eq!(wire.messages[1].role, "assistant");
        match &wire.messages[1].content {
            AnthropicContentBlocks::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(
                    matches!(&blocks[0], AnthropicContentBlock::Text { text, .. } if text == "Thinking.")
                );
                assert!(matches!(
                    &blocks[1],
                    AnthropicContentBlock::ToolUse { id, name, .. }
                        if id == "toolu_01" && name == "lookup"
                ));
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    fn anthropic_still_rejects_conversation_starting_without_user() {
        let assistant_first = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::Assistant(vec![RespC::Text("Hello".into())]))
            .build();
        let err = to_anthropic(&assistant_first, None, false).unwrap_err();
        assert!(err.to_string().contains("first message"));

        let tool_first = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::Tool {
                invocation_id: "toolu_01".into(),
                content: vec![RC::Text("result".into())],
                is_error: false,
            })
            .build();
        let err = to_anthropic(&tool_first, None, false).unwrap_err();
        assert!(err.to_string().contains("preceding assistant"));
    }

    #[test]
    fn to_anthropic_basic_request() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hello".into())]))
            .message(Message::Assistant(vec![RespC::Text("Hi there!".into())]))
            .temperature(0.7)
            .top_p(0.9)
            .stop_sequence("\n\n")
            .build();
        let wire = to_anthropic(&req, None, false).unwrap();
        assert_eq!(wire.model, "claude-sonnet-4-20250514");
        assert_eq!(wire.max_tokens, 1024);
        assert_eq!(wire.temperature, Some(0.7));
        assert_eq!(wire.top_p, Some(0.9));
        assert_eq!(wire.stop_sequences, vec!["\n\n"]);
        assert!(!wire.stream);
        assert!(wire.thinking.is_none());
        assert!(wire.metadata.is_none());
        assert_eq!(wire.messages.len(), 2);
        assert_eq!(wire.messages[0].role, "user");
        assert_eq!(wire.messages[1].role, "assistant");
    }

    #[test]
    fn from_anthropic_converts_text_response() {
        let resp = AnthropicResponse {
            id: "msg_01abcd".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![AnthropicResponseBlock::Text {
                text: "Hello! How can I help?".into(),
            }],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                ..Default::default()
            },
        };
        let canon = from_anthropic(resp).unwrap();
        assert_eq!(canon.id, "msg_01abcd");
        assert_eq!(canon.model, "claude-sonnet-4-20250514");
        assert_eq!(canon.text(), "Hello! How can I help?");
        assert!(matches!(canon.stop_reason, StopReason::EndTurn));
        assert_eq!(canon.usage.input_tokens, 10);
        assert_eq!(canon.usage.output_tokens, 5);
        assert_eq!(canon.usage.cached_input_tokens, 0);
        assert_eq!(canon.usage.cache_creation_input_tokens, 0);
        assert_eq!(canon.usage.reasoning_tokens, 0);
    }

    #[test]
    fn from_anthropic_converts_thinking_to_reasoning() {
        let resp = AnthropicResponse {
            id: "msg_02xyz".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![
                AnthropicResponseBlock::Thinking {
                    thinking: "Let me reason here...".into(),
                    signature: "sig_abc123".into(),
                },
                AnthropicResponseBlock::Text {
                    text: "The answer is 42.".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 15,
                output_tokens: 100,
                cache_creation_input_tokens: 10,
                cache_read_input_tokens: 5,
                output_tokens_details: super::super::schema::AnthropicOutputTokensDetails {
                    thinking_tokens: 80,
                },
            },
        };
        let canon = from_anthropic(resp).unwrap();
        assert_eq!(canon.content.len(), 2);
        // First block should be Reasoning (from Thinking)
        match &canon.content[0] {
            ResponseContent::Reasoning { text, signature } => {
                assert_eq!(text, "Let me reason here...");
                assert_eq!(signature.as_deref(), Some("sig_abc123"));
            }
            other => panic!("expected Reasoning, got {other:?}"),
        }
        // Second block should be Text
        match &canon.content[1] {
            ResponseContent::Text(t) => assert_eq!(t, "The answer is 42."),
            other => panic!("expected Text, got {other:?}"),
        }
        assert_eq!(canon.usage.reasoning_tokens, 80);
        assert_eq!(canon.usage.cached_input_tokens, 5);
        assert_eq!(canon.usage.cache_creation_input_tokens, 10);
    }

    #[test]
    fn from_anthropic_converts_tool_use() {
        let resp = AnthropicResponse {
            id: "msg_03efg".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![
                AnthropicResponseBlock::Text {
                    text: "Let me check the weather.".into(),
                },
                AnthropicResponseBlock::ToolUse {
                    id: "toolu_01def".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"location": "SF"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 20,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                ..Default::default()
            },
        };
        let canon = from_anthropic(resp).unwrap();
        assert_eq!(canon.content.len(), 2);
        // Second block should be ToolInvocation
        match &canon.content[1] {
            ResponseContent::ToolInvocation(ti) => {
                assert_eq!(ti.id, "toolu_01def");
                assert_eq!(ti.name, "get_weather");
                assert_eq!(
                    ti.input.get("location").and_then(|v| v.as_str()),
                    Some("SF")
                );
            }
            other => panic!("expected ToolInvocation, got {other:?}"),
        }
        assert!(matches!(canon.stop_reason, StopReason::ToolUse));
    }

    #[test]
    fn tool_choice_mapping() {
        // Test all 4 mappings through to_anthropic with a tool defined
        let req_auto = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hi".into())]))
            .tool(ToolDef {
                name: "get_weather".into(),
                description: "Get weather".into(),
                input_schema: serde_json::json!({"type": "object"}),
            })
            .tool_choice(ToolChoice::Auto)
            .build();
        let wire = to_anthropic(&req_auto, None, false).unwrap();
        assert_eq!(wire.tool_choice, Some(serde_json::json!({"type": "auto"})));

        let req_none = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hi".into())]))
            .tool(ToolDef {
                name: "get_weather".into(),
                description: "Get weather".into(),
                input_schema: serde_json::json!({"type": "object"}),
            })
            .tool_choice(ToolChoice::None)
            .build();
        let wire = to_anthropic(&req_none, None, false).unwrap();
        assert_eq!(wire.tool_choice, Some(serde_json::json!({"type": "none"})));

        let req_required = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hi".into())]))
            .tool(ToolDef {
                name: "get_weather".into(),
                description: "Get weather".into(),
                input_schema: serde_json::json!({"type": "object"}),
            })
            .tool_choice(ToolChoice::Required)
            .build();
        let wire = to_anthropic(&req_required, None, false).unwrap();
        assert_eq!(wire.tool_choice, Some(serde_json::json!({"type": "any"})));

        let req_specific = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hi".into())]))
            .tool(ToolDef {
                name: "get_weather".into(),
                description: "Get weather".into(),
                input_schema: serde_json::json!({"type": "object"}),
            })
            .tool_choice(ToolChoice::Specific("get_weather".into()))
            .build();
        let wire = to_anthropic(&req_specific, None, false).unwrap();
        assert_eq!(
            wire.tool_choice,
            Some(serde_json::json!({"type": "tool", "name": "get_weather"}))
        );
    }

    #[test]
    fn from_anthropic_converts_redacted_thinking() {
        let resp = AnthropicResponse {
            id: "msg_04".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![AnthropicResponseBlock::RedactedThinking {
                data: "sensitive".into(),
            }],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                ..Default::default()
            },
        };
        let canon = from_anthropic(resp).unwrap();
        assert_eq!(canon.content.len(), 1);
        match &canon.content[0] {
            ResponseContent::Reasoning { text, signature } => {
                assert_eq!(text, "[redacted_thinking: sensitive]");
                assert!(signature.is_none());
            }
            other => panic!("expected Reasoning, got {other:?}"),
        }
    }

    #[test]
    fn stop_reason_mapping_end_turn() {
        let resp = AnthropicResponse {
            id: "msg_05".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        };
        let canon = from_anthropic(resp).unwrap();
        assert!(matches!(canon.stop_reason, StopReason::EndTurn));
    }

    #[test]
    fn stop_reason_mapping_stop_sequence() {
        let resp = AnthropicResponse {
            id: "msg_06".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![],
            stop_reason: Some("stop_sequence".into()),
            stop_sequence: Some("\n\n".into()),
            usage: AnthropicUsage::default(),
        };
        let canon = from_anthropic(resp).unwrap();
        assert!(matches!(canon.stop_reason, StopReason::StopSequence));
    }

    #[test]
    fn image_conversion_url() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![
                RC::Text("Describe this image:".into()),
                RC::Image(ImageSource::Url("https://example.com/img.png".into())),
            ]))
            .build();
        let wire = to_anthropic(&req, None, false).unwrap();
        assert_eq!(wire.messages.len(), 1);
        match &wire.messages[0].content {
            AnthropicContentBlocks::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[1] {
                    AnthropicContentBlock::Image { source } => {
                        assert_eq!(source.source_type, "url");
                        assert_eq!(source.url.as_deref(), Some("https://example.com/img.png"));
                    }
                    other => panic!("expected Image, got {other:?}"),
                }
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    fn image_conversion_base64() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Image(ImageSource::Base64 {
                media_type: "image/png".into(),
                data: "iVBORw0KGgo=".into(),
            })]))
            .build();
        let wire = to_anthropic(&req, None, false).unwrap();
        match &wire.messages[0].content {
            AnthropicContentBlocks::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    AnthropicContentBlock::Image { source } => {
                        assert_eq!(source.source_type, "base64");
                        assert_eq!(source.media_type.as_deref(), Some("image/png"));
                        assert_eq!(source.data.as_deref(), Some("iVBORw0KGgo="));
                        assert!(source.url.is_none());
                    }
                    other => panic!("expected Image, got {other:?}"),
                }
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    fn from_anthropic_filters_empty_text_blocks() {
        let resp = AnthropicResponse {
            id: "msg_07".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![
                AnthropicResponseBlock::Text { text: "".into() },
                AnthropicResponseBlock::Text {
                    text: "Hello".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        };
        let canon = from_anthropic(resp).unwrap();
        assert_eq!(canon.content.len(), 1);
        assert_eq!(canon.text(), "Hello");
    }

    #[test]
    fn ext_builder_with_structured_output() {
        let schema = serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}});
        let ext = AnthropicExt::builder()
            .structured_output(schema.clone())
            .build();
        assert_eq!(ext.structured_output, Some(schema));
    }

    #[test]
    fn structured_output_injects_tool_and_forces_choice() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        });
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Answer the question.".into())]))
            .build();
        let ext = AnthropicExt::builder()
            .structured_output(schema.clone())
            .build();

        let wire = to_anthropic(&req, Some(&ext), false).unwrap();

        assert_eq!(wire.tools.len(), 1, "expected exactly one injected tool");
        assert_eq!(wire.tools[0].name, "structured_result");
        assert_eq!(
            wire.tools[0].input_schema, schema,
            "injected schema must match the one provided"
        );
        assert_eq!(
            wire.tool_choice,
            Some(serde_json::json!({"type": "tool", "name": "structured_result"})),
            "tool_choice must be forced to structured_result"
        );
    }

    #[test]
    fn anthropic_request_struct() {
        let base = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hi".into())]))
            .build();
        let ext = AnthropicExt::builder().top_k(40).build();
        let req = AnthropicRequest { base, ext };
        assert_eq!(req.base.model(), "claude-sonnet-4-20250514");
        assert_eq!(req.ext.top_k, Some(40));
    }

    #[test]
    fn anthropic_native_response_into_message_preserves_signature() {
        use crate::anthropic::AnthropicNativeResponse;
        let resp = AnthropicResponse {
            id: "msg_sig".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![
                AnthropicResponseBlock::Thinking {
                    thinking: "Let me think...".into(),
                    signature: "sig_abc".into(),
                },
                AnthropicResponseBlock::Text {
                    text: "Answer.".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        };
        let native = AnthropicNativeResponse::from_wire(resp).unwrap();
        let msg = native.into_message();
        match msg {
            crate::types::Message::Assistant(content) => match &content[0] {
                ResponseContent::Reasoning { text, signature } => {
                    assert_eq!(text, "Let me think...");
                    assert_eq!(signature.as_deref(), Some("sig_abc"));
                }
                other => panic!("expected Reasoning, got {other:?}"),
            },
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn anthropic_native_response_into_canonical_strips_signature() {
        use crate::anthropic::AnthropicNativeResponse;
        let resp = AnthropicResponse {
            id: "msg_sig2".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![
                AnthropicResponseBlock::Thinking {
                    thinking: "Let me think...".into(),
                    signature: "sig_xyz".into(),
                },
                AnthropicResponseBlock::Text {
                    text: "Answer.".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        };
        let native = AnthropicNativeResponse::from_wire(resp).unwrap();
        let canon = native.into_canonical();
        match &canon.content[0] {
            ResponseContent::Reasoning { text, signature } => {
                assert_eq!(text, "Let me think...");
                assert!(signature.is_none(), "into_canonical must strip signature");
            }
            other => panic!("expected Reasoning, got {other:?}"),
        }
    }

    #[test]
    fn anthropic_46_budget_mapping() {
        use crate::types::ThinkingLevel;
        let (cfg, effort) =
            thinking_to_anthropic(ThinkingLevel::Medium, "claude-sonnet-4-6", 32768);
        let cfg = cfg.unwrap();
        assert_eq!(cfg.r#type, "enabled");
        assert_eq!(cfg.budget_tokens, Some(8192));
        assert!(effort.is_none());
    }

    #[test]
    fn anthropic_46_budget_explicit() {
        use crate::types::ThinkingLevel;
        let (cfg, _) =
            thinking_to_anthropic(ThinkingLevel::Budget(5000), "claude-sonnet-4-6", 32768);
        let cfg = cfg.unwrap();
        assert_eq!(cfg.r#type, "enabled");
        assert_eq!(cfg.budget_tokens, Some(5000));
    }

    #[test]
    fn anthropic_46_budget_clamped_to_max_tokens() {
        // Budget(8192) with max_tokens=2048 → clamped to 2047 (< max_tokens).
        use crate::types::ThinkingLevel;
        let (cfg, _) = thinking_to_anthropic(ThinkingLevel::Medium, "claude-sonnet-4-6", 2048);
        assert_eq!(cfg.unwrap().budget_tokens, Some(2047));
    }

    #[test]
    fn anthropic_46_budget_below_minimum_clamped_up() {
        // Budget(100) → clamped up to MIN_BUDGET_TOKENS (1024).
        use crate::types::ThinkingLevel;
        let (cfg, _) =
            thinking_to_anthropic(ThinkingLevel::Budget(100), "claude-sonnet-4-6", 32768);
        assert_eq!(cfg.unwrap().budget_tokens, Some(1024));
    }

    #[test]
    fn anthropic_47_adaptive_medium() {
        use crate::types::ThinkingLevel;
        let (cfg, effort) =
            thinking_to_anthropic(ThinkingLevel::Medium, "claude-opus-4-7", 32768);
        let cfg = cfg.unwrap();
        assert_eq!(cfg.r#type, "adaptive");
        assert_eq!(effort.as_deref(), Some("medium"));
        assert!(cfg.budget_tokens.is_none());
    }

    #[test]
    fn anthropic_47_adaptive_high() {
        use crate::types::ThinkingLevel;
        let (cfg, effort) = thinking_to_anthropic(ThinkingLevel::High, "claude-opus-4-7", 32768);
        assert_eq!(cfg.unwrap().r#type, "adaptive");
        assert_eq!(effort.as_deref(), Some("high"));
    }

    #[test]
    fn anthropic_47_budget_degrades_to_high_effort() {
        use crate::types::ThinkingLevel;
        let (cfg, effort) =
            thinking_to_anthropic(ThinkingLevel::Budget(5000), "claude-opus-4-7", 32768);
        assert_eq!(cfg.unwrap().r#type, "adaptive");
        assert_eq!(effort.as_deref(), Some("high"));
    }

    #[test]
    fn anthropic_off_returns_none() {
        use crate::types::ThinkingLevel;
        let (cfg, effort) = thinking_to_anthropic(ThinkingLevel::Off, "claude-sonnet-4-6", 32768);
        assert!(cfg.is_none());
        assert!(effort.is_none());
        let (cfg, effort) = thinking_to_anthropic(ThinkingLevel::Off, "claude-opus-4-7", 32768);
        assert!(cfg.is_none());
        assert!(effort.is_none());
    }

    #[test]
    fn anthropic_47_adaptive_effort_is_top_level_output_config_not_nested_in_thinking() {
        // Regression test: effort must never be serialized inside `thinking` —
        // Anthropic rejects `thinking.adaptive.effort` as an unexpected field.
        use crate::types::{ChatRequest, Message, RequestContent as RC, ThinkingLevel};

        let req = ChatRequest::builder("claude-opus-4-7", 1024)
            .message(Message::User(vec![RC::Text("hi".into())]))
            .thinking(ThinkingLevel::Medium)
            .build();
        let wire = to_anthropic(&req, None, false).unwrap();

        let thinking_json = serde_json::to_value(wire.thinking.as_ref().unwrap()).unwrap();
        assert!(
            thinking_json.get("effort").is_none(),
            "effort must not be nested inside thinking: {thinking_json:?}"
        );

        let output_config = wire.output_config.as_ref().unwrap();
        assert_eq!(output_config.effort.as_deref(), Some("medium"));
    }

    #[test]
    fn is_adaptive_model_detection() {
        assert!(is_adaptive_model("claude-opus-4-7"));
        assert!(is_adaptive_model("claude-opus-4-7-2026"));
        assert!(is_adaptive_model("mythos-preview"));
        assert!(is_adaptive_model("claude-opus-4-8"));
        assert!(is_adaptive_model("claude-sonnet-5"));
        assert!(is_adaptive_model("claude-fable-5"));
        assert!(!is_adaptive_model("claude-sonnet-4-6"));
        assert!(!is_adaptive_model("claude-3-5-sonnet"));
    }
}
