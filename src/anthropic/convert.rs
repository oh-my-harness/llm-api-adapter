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
}

/// Extended thinking configuration.
#[derive(Debug, Clone)]
pub struct ThinkingConfig {
    /// Maximum tokens to spend on extended thinking.
    pub budget_tokens: u32,
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

    /// Build the AnthropicExt instance.
    pub fn build(self) -> AnthropicExt {
        AnthropicExt {
            top_k: self.top_k,
            disable_parallel_tool_use: self.disable_parallel_tool_use,
            thinking: self.thinking,
            metadata: self.metadata,
            cache_hints: self.cache_hints,
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
                // If the last message was assistant, we need alternation;
                // but we just ensure it's user. Tool results become user messages
                // handled separately below.
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
                // If the last message in wire_messages is already a user, append to its blocks.
                let appended = if let Some(last) = wire_messages.last_mut() {
                    if last.role == "user" {
                        match &mut last.content {
                            AnthropicContentBlocks::Blocks(blocks) => {
                                blocks.push(tool_result_block.clone());
                                true
                            }
                            AnthropicContentBlocks::Text(t) => {
                                // Promote Text to Blocks and append.
                                let text_block = AnthropicContentBlock::Text {
                                    text: std::mem::take(t),
                                    cache_control: None,
                                };
                                let blocks = vec![text_block, tool_result_block.clone()];
                                last.content = AnthropicContentBlocks::Blocks(blocks);
                                true
                            }
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !appended {
                    // Need a new user message.
                    wire_messages.push(AnthropicWireMessageCompat {
                        role: "user".into(),
                        content: AnthropicContentBlocks::Blocks(vec![tool_result_block]),
                    });
                }
            }
            Message::System(_) => {
                // Already handled above — skip.
            }
        }
    }

    // Enforce alternation: user/assistant must alternate and first must be user.
    check_alternation(&wire_messages)?;

    // Convert tools.
    let tools: Vec<super::schema::AnthropicTool> = req
        .tools()
        .iter()
        .map(|t| super::schema::AnthropicTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect();

    // Convert tool choice.
    let tool_choice = if req.tools().is_empty() {
        None
    } else {
        Some(tool_choice_to_anthropic(req.tool_choice()))
    };

    let params = req.params();

    // Convert thinking config.
    let thinking = ext
        .thinking
        .as_ref()
        .map(|t| super::schema::AnthropicThinkingConfig {
            r#type: "enabled".into(),
            budget_tokens: Some(t.budget_tokens),
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
            ResponseContent::Reasoning(r) => {
                // Anthropic does not accept reasoning in requests; encode as text
                // for round-trip compatibility.
                blocks.push(AnthropicContentBlock::Text {
                    text: r.clone(),
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
    // Build a flat index across all content blocks in all messages.
    let mut flat_index = 0usize;
    for msg in &mut messages {
        match &msg.content {
            AnthropicContentBlocks::Blocks(blocks) => {
                for _block in blocks {
                    for hint in hints {
                        if hint.block_index == flat_index && hint.ephemeral {
                            // We need to mutate the block — we can't borrow mutably here
                            // since blocks is behind a shared ref. Collect the info and do it in a second pass.
                        }
                    }
                    flat_index += 1;
                }
            }
            AnthropicContentBlocks::Text(_) => {
                flat_index += 1;
            }
        }
    }

    // Second pass: actually apply hints.
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
            other @ AnthropicContentBlocks::Text(_) => {
                flat_index += 1;
                other
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
        AnthropicResponseBlock::Text { text } if !text.is_empty() => Some(ResponseContent::Text(text)),
        AnthropicResponseBlock::Text { .. } => None,
        AnthropicResponseBlock::ToolUse { id, name, input } => {
            Some(ResponseContent::ToolInvocation(ToolInvocation { id, name, input }))
        }
        AnthropicResponseBlock::Thinking { thinking, signature: _ } if !thinking.is_empty() => {
            Some(ResponseContent::Reasoning(thinking))
        }
        AnthropicResponseBlock::Thinking { .. } => None,
        AnthropicResponseBlock::RedactedThinking { data } => {
            Some(ResponseContent::Reasoning(format!("[redacted_thinking: {data}]")))
        }
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
        reasoning_tokens: 0,
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
            .thinking(ThinkingConfig {
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
        assert_eq!(ext.thinking.as_ref().map(|t| t.budget_tokens), Some(2000));
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
    fn alternation_enforced() {
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hello".into())]))
            .message(Message::User(vec![RC::Text("World".into())]))
            .build();
        let result = to_anthropic(&req, None, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("alternat"));
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
            },
        };
        let canon = from_anthropic(resp).unwrap();
        assert_eq!(canon.content.len(), 2);
        // First block should be Reasoning (from Thinking)
        match &canon.content[0] {
            ResponseContent::Reasoning(r) => assert_eq!(r, "Let me reason here..."),
            other => panic!("expected Reasoning, got {other:?}"),
        }
        // Second block should be Text
        match &canon.content[1] {
            ResponseContent::Text(t) => assert_eq!(t, "The answer is 42."),
            other => panic!("expected Text, got {other:?}"),
        }
        // Usage mapping
        assert_eq!(canon.usage.cached_input_tokens, 5);
        assert_eq!(canon.usage.cache_creation_input_tokens, 10);
        assert_eq!(canon.usage.reasoning_tokens, 0);
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
            },
        };
        let canon = from_anthropic(resp).unwrap();
        assert_eq!(canon.content.len(), 1);
        match &canon.content[0] {
            ResponseContent::Reasoning(r) => {
                assert_eq!(r, "[redacted_thinking: sensitive]");
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
                AnthropicResponseBlock::Text { text: "Hello".into() },
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
    fn anthropic_request_struct() {
        let base = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Hi".into())]))
            .build();
        let ext = AnthropicExt::builder().top_k(40).build();
        let req = AnthropicRequest { base, ext };
        assert_eq!(req.base.model(), "claude-sonnet-4-20250514");
        assert_eq!(req.ext.top_k, Some(40));
    }
}
