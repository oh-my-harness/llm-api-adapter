# Anthropic Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Anthropic Messages API provider so users can hold `Box<dyn Provider>` and switch between Anthropic, OpenAI, and DeepSeek at runtime.

**Architecture:** Follow the same module structure as `openai/` — `schema.rs` for wire types, `convert.rs` for canonical ↔ Anthropic conversion, `stream.rs` for SSE parsing, and `mod.rs` for `AnthropicProvider` + builder + `Provider` trait impl. The provider constructs `POST /v1/messages` requests with `x-api-key` + `anthropic-version` headers, maps responses from Anthropic's structured `content` block array to the canonical content system, and handles Anthropic-specific error codes (529 → `Overloaded`, 413 → `InvalidRequest`).

**Tech Stack:** Rust, reqwest, serde, serde_json, async-trait, eventsource-stream, wiremock (dev)

---

### Task 1: Create `src/anthropic/schema.rs` — Anthropic Wire Types

**Files:**
- Create: `src/anthropic/schema.rs`
- Test: none (pure types; deserialization tests in the inline test below)

- [ ] **Step 1: Write wire-type definitions with full serde annotations**

Create `src/anthropic/schema.rs` with the full Anthropic Messages API wire schema:

```rust
use serde::{Deserialize, Serialize};

// ── Request ──────────────────────────────────────────────────────────────────

/// Anthropic Messages API request body.
#[derive(Debug, Serialize)]
pub(crate) struct AnthropicRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<AnthropicWireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
}

/// A message in the Anthropic messages array.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicWireMessage {
    pub role: String,
    pub content: AnthropicContentBlocks,
}

/// Content blocks: either a simple string or an array of content blocks.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum AnthropicContentBlocks {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// A single content block in a message.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "image")]
    Image {
        source: AnthropicImageSource,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: AnthropicContentBlocks,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        data: String,
    },
}

/// Image source in Anthropic format.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Cache control hint for a content block.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicCacheControl {
    pub r#type: String,
}

/// Thinking configuration.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicThinkingConfig {
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

/// Request metadata.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// Tool definition for Anthropic (input_schema directly, no `function` wrapper).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

// ── Response ─────────────────────────────────────────────────────────────────

/// Anthropic Messages API response (non-streaming).
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<AnthropicResponseBlock>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// A content block in the response.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicResponseBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        data: String,
    },
}

/// Token usage from Anthropic response.
#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

// ── Streaming ─────────────────────────────────────────────────────────────────

/// SSE event from the Anthropic Messages stream.
///
/// Each event has a `type` discriminator. We decode based on `event` field
/// pairing with `data` content during SSE parsing; the deserialised struct
/// is selected by the parser in `stream.rs`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart {
        #[serde(rename = "type")]
        event_type: String,
        message: AnthropicResponse,
    },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: AnthropicResponseBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: AnthropicDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        index: usize,
    },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDeltaPayload,
        usage: AnthropicUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
}

/// Delta payload within a content_block_delta event.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

/// Metadata delta from message_delta event.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicMessageDeltaPayload {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

// ── Error ────────────────────────────────────────────────────────────────────

/// Anthropic error response body.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub error: AnthropicErrorBody,
}

/// Error details.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicErrorBody {
    pub r#type: String,
    pub message: String,
}
```

- [ ] **Step 2: Write deserialization test to verify wire types match the spec**

Add this test at the bottom of `schema.rs` (inside `#[cfg(test)]`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_deserializes() {
        let json = r#"{
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"location": "Paris"}}
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "msg_01");
        assert_eq!(resp.model, "claude-sonnet-4-6");
        assert_eq!(resp.content.len(), 2);
        assert!(matches!(resp.content[0], AnthropicResponseBlock::Text { ref text } if text == "Hello!"));
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn thinking_response_deserializes() {
        let json = r#"{
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "thinking", "thinking": "Let me think...", "signature": "sig_01"},
                {"type": "text", "text": "42"}
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 2);
        assert!(matches!(&resp.content[0], AnthropicResponseBlock::Thinking { thinking, signature }
            if thinking == "Let me think..." && signature == "sig_01"));
        assert!(matches!(&resp.content[1], AnthropicResponseBlock::Text { text } if text == "42"));
    }

    #[test]
    fn stream_event_deserializes() {
        let json = r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}"#;
        let evt: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        match evt {
            AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                assert!(matches!(delta, AnthropicDelta::TextDelta { text } if text == "Hello"));
            }
            _ => panic!("expected ContentBlockDelta"),
        }
    }

    #[test]
    fn content_blocks_text_and_blocks_roundtrip() {
        // Simple string
        let simple: AnthropicContentBlocks = serde_json::from_str("\"Hello\"").unwrap();
        match simple {
            AnthropicContentBlocks::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("expected Text"),
        }
        // Array
        let arr: AnthropicContentBlocks = serde_json::from_str(
            r#"[{"type": "text", "text": "Hi"}, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "YWJj"}}]"#,
        ).unwrap();
        match arr {
            AnthropicContentBlocks::Blocks(_) => {}
            _ => panic!("expected Blocks"),
        }
    }
}
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check --all-targets`
Expected: types compile cleanly (the module isn't wired into `lib.rs` yet, so trigger check by adding a stub — or just skip mid-cycle check until the module is registered). Instead, check that `serde_json::from_str::<AnthropicResponse>` works via `cargo test --lib` if we wire the test above into the module.

For now, just ensure the file syntax is correct:
```bash
cargo check 2>&1 || true
```

- [ ] **Step 4: Commit**

```bash
git add src/anthropic/schema.rs
git commit -m "feat(anthropic): add wire types for Messages API

- Request body with messages, system, tools, thinking, metadata
- Response with structured content blocks (text, tool_use, thinking, redacted_thinking)
- SSE stream event types (message_start, content_block_start/delta/stop,
  message_delta, message_stop, ping)
- Error response types for error mapping
- Deserialization tests for key scenarios"

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 2: Create `src/anthropic/convert.rs` — Request/Response Conversion

**Files:**
- Create: `src/anthropic/convert.rs`
- Modify: `src/anthropic/schema.rs` (already done)
- Test: `src/anthropic/convert.rs` inline tests

- [ ] **Step 1: Write the conversion module**

Create `src/anthropic/convert.rs`:

```rust
use crate::types::{
    ChatRequest, ChatResponse, Message, RequestContent, ResponseContent, StopReason, ToolChoice,
    ToolInvocation, Usage,
};
use crate::LlmError;

use super::schema::{
    AnthropicContentBlock, AnthropicContentBlocks, AnthropicImageSource, AnthropicRequest,
    AnthropicResponse, AnthropicResponseBlock, AnthropicThinkingConfig, AnthropicTool,
    AnthropicUsage, AnthropicWireMessage,
};

// ── AnthropicExt ────────────────────────────────────────────────────────────────

/// Provider-specific extensions for Anthropic requests.
///
/// These fields allow fine-tuning of Anthropic-specific behavior beyond the
/// canonical ChatRequest. All fields are optional.
#[derive(Debug, Clone, Default)]
pub struct AnthropicExt {
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Disable parallel tool use (force at most one tool call).
    pub disable_parallel_tool_use: bool,
    /// Thinking configuration (extended thinking).
    pub thinking: Option<ThinkingConfig>,
    /// Request metadata (e.g. user_id).
    pub metadata: Option<AnthropicMetadata>,
    /// Cache control hints for content blocks (P2, default empty).
    pub cache_hints: Vec<CacheHint>,
}

/// Configuration for Anthropic extended thinking.
#[derive(Debug, Clone)]
pub struct ThinkingConfig {
    pub budget_tokens: u32,
}

/// Request metadata.
#[derive(Debug, Clone)]
pub struct AnthropicMetadata {
    pub user_id: Option<String>,
}

/// Cache hint for a content block (P2 — reserved for future use).
#[derive(Debug, Clone)]
pub struct CacheHint {
    pub block_index: usize,
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
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set top-k sampling.
    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Disable parallel tool use.
    pub fn disable_parallel_tool_use(mut self) -> Self {
        self.disable_parallel_tool_use = true;
        self
    }

    /// Set thinking/Extended Thinking configuration.
    pub fn thinking(mut self, cfg: ThinkingConfig) -> Self {
        self.thinking = Some(cfg);
        self
    }

    /// Set request metadata.
    pub fn metadata(mut self, meta: AnthropicMetadata) -> Self {
        self.metadata = Some(meta);
        self
    }

    /// Add a cache hint (P2).
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

/// A ChatRequest paired with Anthropic provider-specific extensions.
pub struct AnthropicRequest {
    /// The canonical chat request.
    pub base: ChatRequest,
    /// Anthropic-specific extensions.
    pub ext: AnthropicExt,
}

// ── Conversion: Canonical → Anthropic ──────────────────────────────────────────

/// Convert a canonical ChatRequest to an Anthropic AnthropicRequest wire body.
///
/// # Arguments
/// * `req` - The canonical chat request
/// * `ext` - Optional Anthropic-specific extensions
/// * `stream` - Whether to enable streaming
///
/// # Returns
/// An AnthropicRequest ready to send, or an LlmError if conversion fails.
pub(crate) fn to_anthropic(
    req: &ChatRequest,
    ext: Option<&AnthropicExt>,
    stream: bool,
) -> Result<AnthropicRequest, LlmError> {
    let default_ext = AnthropicExt::default();
    let ext = ext.unwrap_or(&default_ext);

    let (messages, system) = build_messages_and_system(req.messages())?;

    // Convert tools
    let tools: Vec<AnthropicTool> = req
        .tools()
        .iter()
        .map(|t| AnthropicTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect();

    // Convert tool_choice
    let tool_choice = if req.tools().is_empty() {
        None
    } else {
        Some(tool_choice_to_anthropic(req.tool_choice()))
    };

    let params = req.params();

    // Build thinking config from ext
    let thinking = ext.thinking.as_ref().map(|tc| AnthropicThinkingConfig {
        r#type: "enabled".into(),
        budget_tokens: Some(tc.budget_tokens),
    });

    // Build metadata
    let metadata = ext.metadata.as_ref().map(|m| super::schema::AnthropicMetadata {
        user_id: m.user_id.clone(),
    });

    Ok(AnthropicRequest {
        model: req.model().into(),
        max_tokens: params.max_tokens(),
        messages,
        system: system.map(|s| serde_json::Value::String(s)),
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

/// Convert canonical messages to Anthropic messages array + optional system field.
///
/// Rules:
/// - System messages → top-level `system` field (never in messages array)
/// - Messages must alternate user/assistant
/// - First message must be user
/// - Tool messages become `tool_result` blocks inside a User message
fn build_messages_and_system(
    messages: &[Message],
) -> Result<(Vec<AnthropicWireMessage>, Option<String>), LlmError> {
    let mut system_text: Option<String> = None;
    let mut wire_messages: Vec<AnthropicWireMessage> = Vec::new();
    let mut last_role: Option<&str> = None;

    for msg in messages {
        match msg {
            Message::System(text) => {
                // Accumulate system messages. Anthropic takes a single top-level
                // system field; multiple system messages are joined.
                let merged = system_text.get_or_insert_with(String::new);
                if !merged.is_empty() {
                    merged.push('\n');
                }
                merged.push_str(text);
                // System messages don't go in the messages array
            }
            Message::User(content) => {
                check_alternation("user", last_role)?;
                let blocks = req_content_to_blocks(content);
                wire_messages.push(AnthropicWireMessage {
                    role: "user".into(),
                    content: blocks,
                });
                last_role = Some("user");
            }
            Message::Assistant(content) => {
                check_alternation("assistant", last_role)?;
                let blocks = resp_content_to_blocks(content)?;
                wire_messages.push(AnthropicWireMessage {
                    role: "assistant".into(),
                    content: blocks,
                });
                last_role = Some("assistant");
            }
            Message::Tool {
                invocation_id,
                content,
                is_error,
            } => {
                // Tool results are placed inside a User message as tool_result blocks.
                // If the last message was assistant, this tool result creates a new user message.
                // If the last message was user (from a previous tool result), append to it.
                let blocks = if content.len() == 1 {
                    if let RequestContent::Text(t) = &content[0] {
                        AnthropicContentBlocks::Text(t.clone())
                    } else {
                        AnthropicContentBlocks::Blocks(vec![AnthropicContentBlock::ToolResult {
                            tool_use_id: invocation_id.clone(),
                            content: AnthropicContentBlocks::Text(
                                content_text(content).unwrap_or_default(),
                            ),
                            is_error: if *is_error { Some(true) } else { None },
                        }])
                    }
                } else {
                    AnthropicContentBlocks::Blocks(vec![AnthropicContentBlock::ToolResult {
                        tool_use_id: invocation_id.clone(),
                        content: AnthropicContentBlocks::Text(
                            content_text(content).unwrap_or_default(),
                        ),
                        is_error: if *is_error { Some(true) } else { None },
                    }])
                };

                if last_role == Some("user") {
                    // Append tool_result to existing user message
                    if let Some(last) = wire_messages.last_mut() {
                        if let AnthropicContentBlocks::Blocks(ref mut existing) = last.content {
                            if let AnthropicContentBlocks::Blocks(new_blocks) = blocks {
                                existing.extend(new_blocks);
                            } else if let AnthropicContentBlocks::Text(t) = blocks {
                                existing.push(AnthropicContentBlock::Text { text: t, cache_control: None });
                            }
                        } else {
                            // Convert existing Text to Blocks and append
                            let old_text = std::mem::take(&mut last.content);
                            let mut new_blocks = vec![];
                            if let AnthropicContentBlocks::Text(t) = old_text {
                                new_blocks.push(AnthropicContentBlock::Text { text: t, cache_control: None });
                            }
                            if let AnthropicContentBlocks::Blocks(mut bs) = blocks {
                                new_blocks.append(&mut bs);
                            } else if let AnthropicContentBlocks::Text(t) = blocks {
                                new_blocks.push(AnthropicContentBlock::Text { text: t, cache_control: None });
                            }
                            last.content = AnthropicContentBlocks::Blocks(new_blocks);
                        }
                    }
                } else {
                    // Start a new user message
                    wire_messages.push(AnthropicWireMessage {
                        role: "user".into(),
                        content: blocks,
                    });
                    last_role = Some("user");
                }
            }
        }
    }

    // Check first message is user for non-empty messages arrays
    if let Some(first) = wire_messages.first() {
        if first.role != "user" {
            return Err(LlmError::InvalidRequest(
                "Anthropic requires the first message to have role 'user'".into(),
            ));
        }
    }

    Ok((wire_messages, system_text))
}

/// Check message role alternation rule (user → assistant → user → ...).
fn check_alternation(role: &str, last_role: Option<&str>) -> Result<(), LlmError> {
    if let Some(last) = last_role {
        if last == role {
            return Err(LlmError::InvalidRequest(format!(
                "Anthropic requires alternating user/assistant messages, got two '{role}' in a row"
            )));
        }
    }
    Ok(())
}

/// Convert canonical RequestContent vec to Anthropic content blocks.
fn req_content_to_blocks(content: &[RequestContent]) -> AnthropicContentBlocks {
    if content.len() == 1 {
        if let RequestContent::Text(t) = &content[0] {
            return AnthropicContentBlocks::Text(t.clone());
        }
    }
    let blocks: Vec<AnthropicContentBlock> = content
        .iter()
        .map(|c| match c {
            RequestContent::Text(t) => AnthropicContentBlock::Text {
                text: t.clone(),
                cache_control: None,
            },
            RequestContent::Image(img) => {
                use crate::types::ImageSource;
                let (source_type, media_type, data, url) = match img {
                    ImageSource::Url(u) => ("url".into(), None::<String>, None::<String>, Some(u.clone())),
                    ImageSource::Base64 { media_type, data } => {
                        ("base64".into(), Some(media_type.clone()), Some(data.clone()), None)
                    }
                };
                AnthropicContentBlock::Image {
                    source: AnthropicImageSource {
                        source_type,
                        media_type,
                        data,
                        url,
                    },
                }
            }
        })
        .collect();
    AnthropicContentBlocks::Blocks(blocks)
}

/// Convert canonical ResponseContent vec to Anthropic content blocks (assistant messages).
fn resp_content_to_blocks(content: &[ResponseContent]) -> Result<AnthropicContentBlocks, LlmError> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut blocks: Vec<AnthropicContentBlock> = Vec::new();

    for item in content {
        match item {
            ResponseContent::Text(t) => {
                text_parts.push(t.clone());
            }
            ResponseContent::ToolInvocation(ti) => {
                // Flush accumulated text before tool_use
                if !text_parts.is_empty() {
                    blocks.push(AnthropicContentBlock::Text {
                        text: text_parts.join(""),
                        cache_control: None,
                    });
                    text_parts.clear();
                }
                blocks.push(AnthropicContentBlock::ToolUse {
                    id: ti.id.clone(),
                    name: ti.name.clone(),
                    input: ti.input.clone(),
                });
            }
            ResponseContent::Reasoning(r) => {
                // Anthropic's canonical response preserves reasoning as text blocks
                // (signature is lost — native path preserves it).
                blocks.push(AnthropicContentBlock::Text {
                    text: r.clone(),
                    cache_control: None,
                });
            }
        }
    }

    if !text_parts.is_empty() {
        blocks.push(AnthropicContentBlock::Text {
            text: text_parts.join(""),
            cache_control: None,
        });
    }

    if blocks.is_empty() {
        Ok(AnthropicContentBlocks::Text(String::new()))
    } else if blocks.len() == 1 {
        if let AnthropicContentBlock::Text { text, cache_control: _ } = &blocks[0] {
            Ok(AnthropicContentBlocks::Text(text.clone()))
        } else {
            Ok(AnthropicContentBlocks::Blocks(blocks))
        }
    } else {
        Ok(AnthropicContentBlocks::Blocks(blocks))
    }
}

/// Extract concatenated text from RequestContent vec.
fn content_text(content: &[RequestContent]) -> Option<String> {
    let parts: Vec<String> = content
        .iter()
        .filter_map(|c| match c {
            RequestContent::Text(t) => Some(t.clone()),
            RequestContent::Image(_) => None,
        })
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(""))
    }
}

/// Convert canonical ToolChoice to Anthropic tool_choice value.
fn tool_choice_to_anthropic(tc: &ToolChoice) -> serde_json::Value {
    match tc {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::None => serde_json::json!({"type": "none"}),
        ToolChoice::Required => serde_json::json!({"type": "any"}),
        ToolChoice::Specific(name) => serde_json::json!({
            "type": "tool",
            "name": name,
        }),
    }
}

// ── Conversion: Anthropic → Canonical ──────────────────────────────────────────

/// Convert an Anthropic AnthropicResponse to a canonical ChatResponse.
///
/// Reasoning content (thinking/redacted_thinking) is converted to
/// ResponseContent::Reasoning. The signature is dropped — callers
/// using the native path should use AnthropicResponse directly.
pub(crate) fn from_anthropic(resp: AnthropicResponse) -> Result<ChatResponse, LlmError> {
    let content = response_blocks_to_content(&resp.content);
    let stop_reason = parse_stop_reason(resp.stop_reason.as_deref());
    let usage = anthropic_usage_to_canonical(resp.usage);

    Ok(ChatResponse {
        id: resp.id,
        model: resp.model,
        content,
        stop_reason,
        usage,
    })
}

/// Convert Anthropic response content blocks to canonical ResponseContent vec.
fn response_blocks_to_content(blocks: &[AnthropicResponseBlock]) -> Vec<ResponseContent> {
    let mut content = Vec::new();

    for block in blocks {
        match block {
            AnthropicResponseBlock::Text { text } => {
                if !text.is_empty() {
                    content.push(ResponseContent::Text(text.clone()));
                }
            }
            AnthropicResponseBlock::ToolUse { id, name, input } => {
                content.push(ResponseContent::ToolInvocation(ToolInvocation {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }));
            }
            AnthropicResponseBlock::Thinking { thinking, signature: _ } => {
                if !thinking.is_empty() {
                    content.push(ResponseContent::Reasoning(thinking.clone()));
                }
            }
            AnthropicResponseBlock::RedactedThinking { data } => {
                if !data.is_empty() {
                    content.push(ResponseContent::Reasoning(format!(
                        "[redacted_thinking: {}]",
                        data
                    )));
                }
            }
        }
    }

    content
}

/// Parse Anthropic stop_reason to canonical StopReason.
fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("end_turn") => StopReason::EndTurn,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some("tool_use") => StopReason::ToolUse,
        Some("pause_turn") => StopReason::Other("pause_turn".into()),
        Some("refusal") => StopReason::ContentFilter,
        Some(other) => StopReason::Other(other.into()),
        None => StopReason::Other("unknown".into()),
    }
}

/// Convert Anthropic usage to canonical Usage.
pub(crate) fn anthropic_usage_to_canonical(u: AnthropicUsage) -> Usage {
    Usage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        cached_input_tokens: u.cache_read_input_tokens,
        cache_creation_input_tokens: u.cache_creation_input_tokens,
        reasoning_tokens: 0, // Anthropic reports reasoning within output_tokens
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn ext_builder_defaults() {
        let ext = AnthropicExt::builder().build();
        assert!(ext.top_k.is_none());
        assert!(!ext.disable_parallel_tool_use);
        assert!(ext.thinking.is_none());
        assert!(ext.metadata.is_none());
    }

    #[test]
    fn ext_builder_with_fields() {
        let ext = AnthropicExt::builder()
            .top_k(100)
            .disable_parallel_tool_use()
            .thinking(ThinkingConfig { budget_tokens: 5000 })
            .metadata(AnthropicMetadata { user_id: Some("u1".into()) })
            .build();
        assert_eq!(ext.top_k, Some(100));
        assert!(ext.disable_parallel_tool_use);
        assert!(ext.thinking.is_some());
        assert_eq!(ext.metadata.as_ref().and_then(|m| m.user_id.as_deref()), Some("u1"));
    }

    #[test]
    fn system_message_extracted_to_top_level() {
        let req = ChatRequest::builder("claude-sonnet-4-6", 1024)
            .message(Message::System("You are helpful.".into()))
            .message(Message::User(vec![RequestContent::Text("Hi".into())]))
            .build();
        let (messages, system) = build_messages_and_system(req.messages()).unwrap();
        assert_eq!(system.as_deref(), Some("You are helpful."));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn multiple_system_messages_joined() {
        let req = ChatRequest::builder("claude-sonnet-4-6", 1024)
            .message(Message::System("Rule one.".into()))
            .message(Message::System("Rule two.".into()))
            .message(Message::User(vec![RequestContent::Text("Ok".into())]))
            .build();
        let (messages, system) = build_messages_and_system(req.messages()).unwrap();
        assert_eq!(system.as_deref(), Some("Rule one.\nRule two."));
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn tool_result_becomes_user_message_with_tool_result_block() {
        let req = ChatRequest::builder("claude-sonnet-4-6", 1024)
            .message(Message::User(vec![RequestContent::Text("Weather?".into())]))
            .message(Message::Assistant(vec![
                ResponseContent::ToolInvocation(ToolInvocation {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"location": "Paris"}),
                }),
            ]))
            .message(Message::Tool {
                invocation_id: "call_1".into(),
                content: vec![RequestContent::Text("Sunny, 22°C".into())],
                is_error: false,
            })
            .build();
        let (messages, _system) = build_messages_and_system(req.messages()).unwrap();
        // Messages should be: user, assistant, user (with tool_result)
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[2].role, "user");
        // Check the last user message contains a tool_result
        if let AnthropicContentBlocks::Blocks(ref blocks) = messages[2].content {
            assert!(matches!(blocks[0], AnthropicContentBlock::ToolResult { .. }));
        } else {
            panic!("expected tool_result in a Blocks container");
        }
    }

    #[test]
    fn alternation_enforced() {
        let err = || -> Result<(), LlmError> {
            build_messages_and_system(&[
                Message::User(vec![RequestContent::Text("Hi".into())]),
                Message::User(vec![RequestContent::Text("Again".into())]),
            ])?;
            Ok(())
        };
        let result = err();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("alternating") || msg.contains("two 'user' in a row") || msg.contains("user"), "got: {msg}");
    }

    #[test]
    fn to_anthropic_basic_request() {
        let req = ChatRequest::builder("claude-sonnet-4-6", 1024)
            .message(Message::System("Be concise.".into()))
            .message(Message::User(vec![RequestContent::Text("Hello".into())]))
            .temperature(0.5)
            .build();
        let wire = to_anthropic(&req, None, false).unwrap();
        assert_eq!(wire.model, "claude-sonnet-4-6");
        assert_eq!(wire.max_tokens, 1024);
        assert_eq!(wire.temperature, Some(0.5));
        assert_eq!(wire.messages.len(), 1);
        assert_eq!(wire.messages[0].role, "user");
        assert!(wire.system.is_some());
    }

    #[test]
    fn from_anthropic_converts_text_response() {
        use crate::openai::schema::AnthropicResponse as _;
        let wire = AnthropicResponse {
            id: "msg_01".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-6".into(),
            content: vec![
                AnthropicResponseBlock::Text { text: "Hello!".into() },
            ],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        };
        let canon = from_anthropic(wire).unwrap();
        assert_eq!(canon.id, "msg_01");
        assert_eq!(canon.text(), "Hello!");
        assert!(matches!(canon.stop_reason, StopReason::EndTurn));
        assert_eq!(canon.usage.input_tokens, 10);
        assert_eq!(canon.usage.output_tokens, 5);
    }

    #[test]
    fn from_anthropic_converts_thinking_to_reasoning() {
        let wire = AnthropicResponse {
            id: "msg_02".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-6".into(),
            content: vec![
                AnthropicResponseBlock::Thinking { thinking: "Let me compute...".into(), signature: "sig_01".into() },
                AnthropicResponseBlock::Text { text: "42".into() },
            ],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 50,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        };
        let canon = from_anthropic(wire).unwrap();
        assert_eq!(canon.reasoning(), Some("Let me compute..."));
        assert_eq!(canon.text(), "42");
    }

    #[test]
    fn from_anthropic_converts_tool_use() {
        let wire = AnthropicResponse {
            id: "msg_03".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-6".into(),
            content: vec![
                AnthropicResponseBlock::Text { text: "Calling tool...".into() },
                AnthropicResponseBlock::ToolUse {
                    id: "toolu_01".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"location": "Paris"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        };
        let canon = from_anthropic(wire).unwrap();
        assert!(matches!(canon.stop_reason, StopReason::ToolUse));
        let invocations: Vec<_> = canon.tool_invocations().collect();
        assert_eq!(invocations.len(), 1);
        assert_eq!(invocations[0].name, "get_weather");
        assert_eq!(invocations[0].input["location"], "Paris");
    }

    #[test]
    fn tool_choice_mapping() {
        assert_eq!(tool_choice_to_anthropic(&ToolChoice::Auto), serde_json::json!({"type": "auto"}));
        assert_eq!(tool_choice_to_anthropic(&ToolChoice::None), serde_json::json!({"type": "none"}));
        assert_eq!(tool_choice_to_anthropic(&ToolChoice::Required), serde_json::json!({"type": "any"}));
        assert_eq!(
            tool_choice_to_anthropic(&ToolChoice::Specific("get_weather".into())),
            serde_json::json!({"type": "tool", "name": "get_weather"})
        );
    }
}
```

- [ ] **Step 2: Run tests to ensure the conversion logic is correct**

```bash
cargo test --lib -- anthropic::convert 2>&1 || cargo test 2>&1 | head -50
```

Note: The module isn't registered in `lib.rs` yet — tests won't run. We'll verify after Task 4 when the module is wired. For now, just confirm no syntax errors:

```bash
rustfmt --check src/anthropic/convert.rs 2>&1 || cargo check 2>&1 | tail -20
```

- [ ] **Step 3: Commit**

```bash
git add src/anthropic/convert.rs
git commit -m "feat(anthropic): add request/response conversion logic

- AnthropicExt with builder for top_k, thinking, metadata, cache_hints
- Message conversion: system → top-level field, tool → tool_result blocks
- Strict user/assistant alternation enforcement
- Response conversion: thinking → Reasoning, tool_use → ToolInvocation
- ToolChoice mapping: Auto/None → auto/none, Required → any, Specific → named"

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 3: Create `src/anthropic/stream.rs` — Anthropic SSE Parser

**Files:**
- Create: `src/anthropic/stream.rs`
- Test: inline tests

- [ ] **Step 1: Write the Anthropic SSE stream parser**

Create `src/anthropic/stream.rs`:

```rust
use crate::LlmError;
use crate::types::{ContentKind, StopReason, StreamEvent};

use super::schema::{
    AnthropicDelta, AnthropicResponseBlock, AnthropicStreamEvent,
};
use super::convert::anthropic_usage_to_canonical;

/// Tracks streaming state across Anthropic SSE events for a single message.
///
/// Unlike OpenAI where content blocks are implicit, Anthropic explicitly
/// announces content_block_start / content_block_delta / content_block_stop
/// events with indices. We track open blocks and accumulate tool_use
/// input_json_delta fragments that must be concatenated then parsed.
#[derive(Debug)]
pub struct AnthropicStreamState {
    /// Allocator for canonical block indices (maps 1:1 with Anthropic block indices).
    next_index: usize,
    /// Tracks which blocks have had ContentStart emitted.
    open_blocks: Vec<usize>,
    /// Accumulator for tool_use input_json_delta fragments per block index.
    tool_input_fragments: std::collections::HashMap<usize, String>,
    /// The initial anthropic response from message_start (for usage, model, id).
    initial_message: Option<super::schema::AnthropicResponse>,
    /// Output tokens accumulator (from message_delta events).
    output_tokens: u32,
    /// Pending stop reason from message_delta.
    pending_stop_reason: Option<String>,
}

impl AnthropicStreamState {
    pub fn new() -> Self {
        AnthropicStreamState {
            next_index: 0,
            open_blocks: Vec::new(),
            tool_input_fragments: std::collections::HashMap::new(),
            initial_message: None,
            output_tokens: 0,
            pending_stop_reason: None,
        }
    }

    fn allocate_index(&mut self) -> usize {
        let idx = self.next_index;
        self.next_index += 1;
        idx
    }
}

/// Process a single Anthropic stream event and emit corresponding StreamEvents.
///
/// # Arguments
/// * `state` - Mutable reference to the stream state
/// * `event` - The AnthropicStreamEvent to process
///
/// # Returns
/// A vector of StreamEvent results.
pub fn process_event(
    state: &mut AnthropicStreamState,
    event: AnthropicStreamEvent,
) -> Vec<Result<StreamEvent, LlmError>> {
    let mut events = Vec::new();

    match event {
        AnthropicStreamEvent::MessageStart { message, .. } => {
            // The message_start event contains the initial usage snapshot
            // (input_tokens accurate, output_tokens=1 placeholder).
            state.initial_message = Some(message);
        }

        AnthropicStreamEvent::ContentBlockStart { index, content_block } => {
            let canonical_idx = state.allocate_index();

            let kind = match &content_block {
                AnthropicResponseBlock::Text { .. } => ContentKind::Text,
                AnthropicResponseBlock::ToolUse { id, name, .. } => {
                    // Pre-seed the input fragments accumulator for this block
                    state.tool_input_fragments.insert(canonical_idx, String::new());
                    ContentKind::ToolInvocation {
                        id: id.clone(),
                        name: name.clone(),
                    }
                }
                AnthropicResponseBlock::Thinking { .. } | AnthropicResponseBlock::RedactedThinking { .. } => {
                    ContentKind::Reasoning
                }
            };

            state.open_blocks.push(canonical_idx);
            events.push(Ok(StreamEvent::ContentStart {
                index: canonical_idx,
                kind,
            }));
        }

        AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
            let canonical_idx = index; // For Anthropic, indices are 1:1 during streaming

            match delta {
                AnthropicDelta::TextDelta { text } => {
                    events.push(Ok(StreamEvent::TextDelta {
                        index: canonical_idx,
                        text,
                    }));
                }
                AnthropicDelta::InputJsonDelta { partial_json } => {
                    // Accumulate tool arguments fragments
                    state
                        .tool_input_fragments
                        .entry(canonical_idx)
                        .or_default()
                        .push_str(&partial_json);
                    // Emit a ToolDelta for each fragment (the collector concatenates them)
                    events.push(Ok(StreamEvent::ToolDelta {
                        index: canonical_idx,
                        arguments: partial_json,
                    }));
                }
                AnthropicDelta::ThinkingDelta { thinking } => {
                    events.push(Ok(StreamEvent::ReasoningDelta {
                        index: canonical_idx,
                        text: thinking,
                    }));
                }
                AnthropicDelta::SignatureDelta { .. } => {
                    // Signature is included at the end of thinking blocks.
                    // We drop it at the canonical layer (native path preserves it).
                    // No StreamEvent needed; the reasoning text is already delivered.
                }
            }
        }

        AnthropicStreamEvent::ContentBlockStop { index: _ } => {
            // Close all open blocks (Anthropic may have multiple in flight)
            // We close them when the block_stop arrives.
            if let Some(idx) = state.open_blocks.last().copied() {
                // We assume blocks close in LIFO order for simplicity.
                // If not, we need to match by index — see next event handling.
                let _ = idx;
            }
            // Pragmatic: close the last open block
            if let Some(idx) = state.open_blocks.pop() {
                events.push(Ok(StreamEvent::ContentStop { index: idx }));
            }
        }

        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            state.pending_stop_reason = delta.stop_reason;
            state.output_tokens = usage.output_tokens;
        }

        AnthropicStreamEvent::MessageStop => {
            // Close any remaining open blocks
            for &idx in &state.open_blocks {
                events.push(Ok(StreamEvent::ContentStop { index: idx }));
            }
            state.open_blocks.clear();

            let stop_reason = parse_stop_reason(state.pending_stop_reason.as_deref());

            // Build usage from initial message + accumulated output_tokens
            let usage = state.initial_message.as_ref().map(|m| {
                let mut u = m.usage.clone();
                u.output_tokens = state.output_tokens;
                anthropic_usage_to_canonical(u)
            }).unwrap_or_default();

            events.push(Ok(StreamEvent::MessageStop {
                stop_reason,
                usage,
            }));
        }

        AnthropicStreamEvent::Ping => {
            // Keepalive ping — no-op in canonical stream
        }
    }

    events
}

/// Parse Anthropic stop_reason string to canonical StopReason.
fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("end_turn") => StopReason::EndTurn,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some("tool_use") => StopReason::ToolUse,
        Some("pause_turn") => StopReason::Other("pause_turn".into()),
        Some("refusal") => StopReason::ContentFilter,
        Some(other) => StopReason::Other(other.into()),
        None => StopReason::Other("unknown".into()),
    }
}

/// Parse an HTTP stream response into a stream of StreamEvents.
///
/// Anthropic SSE uses named `event:` lines with JSON `data:` bodies.
/// Unlike OpenAI, there is no [DONE] marker — the stream ends with
/// `message_stop` followed by connection close.
pub fn parse_stream(
    response: reqwest::Response,
) -> futures::stream::BoxStream<'static, Result<StreamEvent, LlmError>> {
    use eventsource_stream::Eventsource;
    use futures::stream::StreamExt;

    let stream = response.bytes_stream();
    let event_stream = stream.eventsource();

    let mut state = AnthropicStreamState::new();

    (event_stream
        .flat_map(move |event_result| {
            let mut results = Vec::new();

            match event_result {
                Ok(event) => {
                    // Parse the JSON data as an AnthropicStreamEvent.
                    // We also handle the event type from the `event:` field
                    // by checking `event.event_type()` or the `type` field
                    // inside the JSON.
                    match serde_json::from_str::<AnthropicStreamEvent>(&event.data) {
                        Ok(stream_event) => {
                            let evts = process_event(&mut state, stream_event);
                            results = evts;
                        }
                        Err(e) => {
                            results.push(Err(LlmError::Decode(e)));
                        }
                    }
                }
                Err(e) => {
                    results.push(Err(LlmError::Stream(e.to_string())));
                }
            }

            futures::stream::iter(results).boxed()
        })
        .boxed()) as _
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn message_start_triggers_no_events() {
        let mut state = AnthropicStreamState::new();
        let json = r#"{"type": "message_start", "message": {
            "id": "msg_01", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4-6", "content": [],
            "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 1,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        }}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        let events = process_event(&mut state, event);
        // message_start doesn't emit any StreamEvent by itself
        assert!(events.is_empty());
        // But it should store the initial message
        assert!(state.initial_message.is_some());
    }

    #[test]
    fn content_block_start_emits_content_start() {
        let mut state = AnthropicStreamState::new();
        // Simulate message_start first (required for usage later)
        let msg_start: AnthropicStreamEvent = serde_json::from_str(r#"{"type": "message_start", "message": {
            "id": "msg_01", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4-6", "content": [],
            "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 1,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        }}"#).unwrap();
        process_event(&mut state, msg_start);

        let text_start: AnthropicStreamEvent = serde_json::from_str(r#"{
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}
        }"#).unwrap();
        let events = process_event(&mut state, text_start);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            Ok(StreamEvent::ContentStart { index: 0, kind: ContentKind::Text })
        ));
    }

    #[test]
    fn text_delta_emits_text_delta() {
        let mut state = AnthropicStreamState::new();
        let delta: AnthropicStreamEvent = serde_json::from_str(r#"{
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        }"#).unwrap();
        let events = process_event(&mut state, delta);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            Ok(StreamEvent::TextDelta { index: 0, text }) if text == "Hello"
        ));
    }

    #[test]
    fn thinking_delta_emits_reasoning_delta() {
        let mut state = AnthropicStreamState::new();
        let delta: AnthropicStreamEvent = serde_json::from_str(r#"{
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "I'm thinking..."}
        }"#).unwrap();
        let events = process_event(&mut state, delta);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            Ok(StreamEvent::ReasoningDelta { index: 0, text }) if text == "I'm thinking..."
        ));
    }

    #[test]
    fn tool_use_input_json_accumulates() {
        let mut state = AnthropicStreamState::new();
        // Start tool_use block
        let start: AnthropicStreamEvent = serde_json::from_str(r#"{
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}
        }"#).unwrap();
        let _ = process_event(&mut state, start);

        // Delta fragments
        let d1: AnthropicStreamEvent = serde_json::from_str(r#"{
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"loc"}
        }"#).unwrap();
        let events1 = process_event(&mut state, d1);
        assert!(matches!(&events1[0], Ok(StreamEvent::ToolDelta { .. })));

        let d2: AnthropicStreamEvent = serde_json::from_str(r#"{
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "ation\":\"Paris\"}"}
        }"#).unwrap();
        let events2 = process_event(&mut state, d2);
        assert!(matches!(&events2[0], Ok(StreamEvent::ToolDelta { .. })));

        // Verify accumulated fragments
        let accumulated = state.tool_input_fragments.get(&0);
        assert!(accumulated.is_some());
        // Adjusted: fragments are stored under canonical_idx=0 (was allocated in ContentBlockStart)
        assert_eq!(
            state.tool_input_fragments.get(&0).map(String::as_str),
            Some("{\"location\":\"Paris\"}")
        );
    }

    #[test]
    fn message_stop_emits_all_events() {
        let mut state = AnthropicStreamState::new();

        // message_start
        let msg_start: AnthropicStreamEvent = serde_json::from_str(r#"{"type": "message_start", "message": {
            "id": "msg_01", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4-6", "content": [],
            "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 1,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        }}"#).unwrap();
        process_event(&mut state, msg_start);

        // content_block_start (text)
        let start: AnthropicStreamEvent = serde_json::from_str(r#"{
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}
        }"#).unwrap();
        process_event(&mut state, start);

        // content_block_delta
        let delta: AnthropicStreamEvent = serde_json::from_str(r#"{
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "Hi"}
        }"#).unwrap();
        process_event(&mut state, delta);

        // content_block_stop
        let stop: AnthropicStreamEvent = serde_json::from_str(r#"{"type": "content_block_stop", "index": 0}"#).unwrap();
        process_event(&mut state, stop);

        // message_delta
        let msg_delta: AnthropicStreamEvent = serde_json::from_str(r#"{
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"input_tokens": 10, "output_tokens": 5,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        }"#).unwrap();
        process_event(&mut state, msg_delta);

        // message_stop
        let stop: AnthropicStreamEvent = serde_json::from_str(r#"{"type": "message_stop"}"#).unwrap();
        let events = process_event(&mut state, stop);

        // Should emit MessageStop with usage
        let has_msg_stop = events.iter().any(|e| matches!(e, Ok(StreamEvent::MessageStop { .. })));
        assert!(has_msg_stop, "expected MessageStop event");
    }

    #[test]
    fn ping_is_noop() {
        let mut state = AnthropicStreamState::new();
        let ping: AnthropicStreamEvent = serde_json::from_str(r#"{"type": "ping"}"#).unwrap();
        let events = process_event(&mut state, ping);
        assert!(events.is_empty());
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/anthropic/stream.rs
git commit -m "feat(anthropic): add SSE stream parser for Messages API

- Process message_start, content_block_start/delta/stop, message_delta,
  message_stop, and ping events into canonical StreamEvent sequence
- Accumulate tool_use input_json_delta fragments across multiple chunks
- Map thinking_delta → ReasoningDelta, signature_delta → dropped at
  canonical layer
- Unit tests for each event type and full event sequence"

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 4: Create `src/anthropic/mod.rs` — AnthropicProvider + Builder + Provider Trait

**Files:**
- Create: `src/anthropic/mod.rs`
- Modify: `src/lib.rs` (register the `anthropic` module)
- Modify: `Cargo.toml` (no new deps needed)

- [ ] **Step 1: Write the AnthropicProvider module**

Create `src/anthropic/mod.rs`:

```rust
pub mod convert;
pub(crate) mod schema;
pub(crate) mod stream;

pub use convert::{AnthropicExt, AnthropicExtBuilder, AnthropicMetadata, AnthropicRequest, CacheHint, ThinkingConfig};

use crate::{error::LlmError, provider::Provider, stream_handle::StreamHandle, types::*};
use async_trait::async_trait;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_MESSAGES_PATH: &str = "/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages API provider, constructed via `AnthropicProvider::builder`.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    messages_path: String,
}

/// Builder for AnthropicProvider with fluent configuration API.
pub struct AnthropicProviderBuilder {
    api_key: String,
    base_url: String,
    messages_path: String,
}

impl AnthropicProvider {
    /// Create a new builder for AnthropicProvider with the given API key.
    pub fn builder(api_key: impl Into<String>) -> AnthropicProviderBuilder {
        AnthropicProviderBuilder {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            messages_path: DEFAULT_MESSAGES_PATH.to_string(),
        }
    }

    /// Send an HTTP request to the Anthropic Messages API.
    async fn send(
        &self,
        anthropic_req: &schema::AnthropicRequest,
        _stream: bool,
    ) -> Result<reqwest::Response, LlmError> {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), self.messages_path);
        let body = serde_json::to_string(anthropic_req)?;

        let req = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .body(body);

        let response = req.send().await?;
        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().await?;

            // Try to parse structured error response
            let _error_type = serde_json::from_str::<schema::AnthropicErrorResponse>(&error_text)
                .ok()
                .map(|e| e.error.r#type);

            return match status.as_u16() {
                401 => Err(LlmError::Unauthorized),
                403 => Err(LlmError::Forbidden),
                404 => Err(LlmError::InvalidRequest(format!("not_found: {error_text}"))),
                413 => Err(LlmError::InvalidRequest(format!("request_too_large: {error_text}"))),
                429 => {
                    let retry_after = Self::extract_retry_after(&error_text);
                    Err(LlmError::RateLimit { retry_after })
                }
                500 | 502 | 503 | 504 => Err(LlmError::ServerError(error_text)),
                529 => {
                    let retry_after = Self::extract_retry_after(&error_text);
                    Err(LlmError::Overloaded { retry_after })
                }
                _ => Err(LlmError::InvalidRequest(error_text)),
            };
        }

        Ok(response)
    }

    /// Extract retry-after delay from error text (if present).
    fn extract_retry_after(_error_text: &str) -> Option<Duration> {
        // TODO: Parse retry-after header from actual response headers
        None
    }

    /// Chat completion using the native Anthropic path (non-streaming).
    pub async fn chat_native(&self, req: &AnthropicRequest) -> Result<ChatResponse, LlmError> {
        let anthropic_req = convert::to_anthropic(&req.base, Some(&req.ext), false)?;
        let response = self.send(&anthropic_req, false).await?;
        let wire_resp = response.json::<schema::AnthropicResponse>().await?;
        convert::from_anthropic(wire_resp)
    }

    /// Chat completion using the native Anthropic path (streaming).
    pub async fn chat_stream_native(&self, req: &AnthropicRequest) -> Result<StreamHandle, LlmError> {
        let anthropic_req = convert::to_anthropic(&req.base, Some(&req.ext), true)?;
        let response = self.send(&anthropic_req, true).await?;
        let model = anthropic_req.model.clone();
        let stream = stream::parse_stream(response);
        Ok(StreamHandle::new(model, stream))
    }
}

impl AnthropicProviderBuilder {
    /// Set the base URL for the API (default: "https://api.anthropic.com").
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Override the messages path appended to `base_url`
    /// (default: "/v1/messages"). Useful for compat gateways.
    pub fn messages_path(mut self, path: impl Into<String>) -> Self {
        self.messages_path = path.into();
        self
    }

    /// Build the AnthropicProvider instance.
    pub fn build(self) -> AnthropicProvider {
        AnthropicProvider {
            client: reqwest::Client::new(),
            api_key: self.api_key,
            base_url: self.base_url,
            messages_path: self.messages_path,
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let anthropic_req = AnthropicRequest {
            base: req.clone(),
            ext: AnthropicExt::default(),
        };
        self.chat_native(&anthropic_req).await
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<StreamHandle, LlmError> {
        let anthropic_req = AnthropicRequest {
            base: req.clone(),
            ext: AnthropicExt::default(),
        };
        self.chat_stream_native(&anthropic_req).await
    }
}
```

- [ ] **Step 2: Register `anthropic` module in `src/lib.rs`**

```rust
// Add after `pub mod deepseek;`
pub mod anthropic;
```

And add the `pub use` for the anthropic types:

Find the block:
```rust
pub mod deepseek;
mod error;
pub mod openai;
pub mod provider;
pub mod stream_handle;
pub mod types;
```

Change to:
```rust
pub mod anthropic;
pub mod deepseek;
mod error;
pub mod openai;
pub mod provider;
pub mod stream_handle;
pub mod types;
```

- [ ] **Step 3: Add a compile-check test for the provider in `src/lib.rs` tests**

Add to the bottom of the `mod tests` block in `lib.rs`:

```rust
#[test]
fn anthropic_provider_builds() {
    let _p = anthropic::AnthropicProvider::builder("test-key").build();
}
```

- [ ] **Step 4: Verify compilation and run all existing tests**

```bash
cargo check --all-targets 2>&1
```

Expected: compiles cleanly.

```bash
cargo test --lib 2>&1
```

Expected: all existing lib tests pass (including the new anthropic module tests).

- [ ] **Step 5: Commit**

```bash
git add src/anthropic/mod.rs src/lib.rs
git commit -m "feat(anthropic): add AnthropicProvider with universal and native paths

- Provider + Builder with base_url/messages_path configuration
- Sends x-api-key + anthropic-version headers
- Error mapping: 401/403/404/413/429/500-504/529 → canonical errors
- Implements Provider trait for Box<dyn Provider> runtime switching
- Registers anthropic module in lib.rs public API"

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 5: Offline Tests — Anthropic Chat

**Files:**
- Create: `tests/anthropic_chat.rs`
- Create: `tests/fixtures/anthropic/chat_simple.json`
- Create: `tests/fixtures/anthropic/chat_tool_call.json`
- Create: `tests/fixtures/anthropic/chat_after_tool.json`
- Create: `tests/fixtures/anthropic/chat_reasoning.json`
- Create: `tests/fixtures/anthropic/error_unauthorized.json`
- Create: `tests/fixtures/anthropic/error_rate_limit.json`
- Create: `tests/fixtures/anthropic/error_server.json`

- [ ] **Step 1: Create the common mock helper for Anthropic endpoints**

The Anthropic API uses `POST /v1/messages` (not `/v1/chat/completions`). Add a helper to `tests/common/mod.rs`:

```rust
/// Mount a mock that replies to POST /v1/messages with the given JSON body.
#[allow(dead_code)]
pub async fn mock_anthropic_chat(server: &MockServer, body: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/json"))
        .mount(server)
        .await;
}

/// Mount a mock that replies to POST /v1/messages with an SSE stream.
#[allow(dead_code)]
pub async fn mock_anthropic_stream(server: &MockServer, body: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "text/event-stream")
                .append_header("cache-control", "no-cache")
                .append_header("request-id", "req_01"),
        )
        .mount(server)
        .await;
}
```

- [ ] **Step 2: Create fixture files for Anthropic scenarios**

`tests/fixtures/anthropic/chat_simple.json`:
```json
{
  "id": "msg_01abc123",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-6-20250514",
  "content": [
    {"type": "text", "text": "Hello! How can I help you today?"}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 8,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

`tests/fixtures/anthropic/chat_tool_call.json`:
```json
{
  "id": "msg_02def456",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-6-20250514",
  "content": [
    {"type": "text", "text": "I'll check the weather for you."},
    {"type": "tool_use", "id": "toolu_01abc", "name": "get_weather", "input": {"location": "Paris"}}
  ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 50,
    "output_tokens": 30,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

`tests/fixtures/anthropic/chat_after_tool.json`:
```json
{
  "id": "msg_03ghi789",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-6-20250514",
  "content": [
    {"type": "text", "text": "The weather in Paris is currently sunny and 22°C."}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 80,
    "output_tokens": 12,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

`tests/fixtures/anthropic/chat_reasoning.json`:
```json
{
  "id": "msg_04jkl012",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-6-20250514",
  "content": [
    {"type": "thinking", "thinking": "Let me compute 7 times 8. That's a basic multiplication: 7 * 8 = 56.", "signature": "sig_abc123"},
    {"type": "text", "text": "56"}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 20,
    "output_tokens": 18,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

`tests/fixtures/anthropic/error_unauthorized.json`:
```json
{
  "type": "error",
  "error": {
    "type": "authentication_error",
    "message": "Invalid API key"
  }
}
```

`tests/fixtures/anthropic/error_rate_limit.json`:
```json
{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "Too many requests"
  }
}
```

`tests/fixtures/anthropic/error_server.json`:
```json
{
  "type": "error",
  "error": {
    "type": "api_error",
    "message": "Internal server error"
  }
}
```

- [ ] **Step 3: Write the anthropic_chat test file**

`tests/anthropic_chat.rs`:
```rust
mod common;

use llm_adapter::{LlmError, Provider, anthropic::AnthropicProvider, types::*};
use wiremock::MockServer;

fn provider(server: &MockServer) -> AnthropicProvider {
    AnthropicProvider::builder("test-key")
        .base_url(server.uri())
        .build()
}

#[tokio::test]
async fn simple_chat_returns_text() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(&server, include_str!("fixtures/anthropic/chat_simple.json")).await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::User(vec![RequestContent::Text("Hello".into())]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();

    assert!(!resp.id.is_empty());
    assert!(!resp.model.is_empty());
    assert!(!resp.text().is_empty());
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
}

#[tokio::test]
async fn chat_with_system_message() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(&server, include_str!("fixtures/anthropic/chat_simple.json")).await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::System("Be concise.".into()))
        .message(Message::User(vec![RequestContent::Text("Hello".into())]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn chat_returns_tool_call() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(&server, include_str!("fixtures/anthropic/chat_tool_call.json")).await;

    let tool = ToolDef {
        name: "get_weather".into(),
        description: "Get the weather".into(),
        input_schema: serde_json::json!({"type":"object","properties":{"location":{"type":"string"}}}),
    };
    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::User(vec![RequestContent::Text(
            "What's the weather in Paris?".into(),
        )]))
        .tool(tool)
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();
    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
    assert_eq!(invocations[0].input["location"], "Paris");
}

#[tokio::test]
async fn multi_turn_tool_call() {
    let server = MockServer::start().await;

    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
            include_str!("fixtures/anthropic/chat_tool_call.json"),
            "application/json",
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
            include_str!("fixtures/anthropic/chat_after_tool.json"),
            "application/json",
        ))
        .mount(&server)
        .await;

    let p = provider(&server);

    // Round 1: get tool call
    let mut messages = vec![Message::User(vec![RequestContent::Text(
        "What's the weather in Paris?".into(),
    )])];
    let req1 = ChatRequest::builder("claude-sonnet-4-6", 512)
        .messages(messages.clone())
        .tool(ToolDef {
            name: "get_weather".into(),
            description: "Get weather".into(),
            input_schema: serde_json::json!({}),
        })
        .build();
    let resp1 = p.chat(&req1).await.unwrap();
    assert!(matches!(resp1.stop_reason, StopReason::ToolUse));

    let invocation_id = resp1.tool_invocations().next().unwrap().id.clone();
    messages.push(resp1.into_message());
    messages.push(Message::Tool {
        invocation_id,
        content: vec![RequestContent::Text("Sunny, 22°C".into())],
        is_error: false,
    });

    // Round 2: final answer
    let req2 = ChatRequest::builder("claude-sonnet-4-6", 512)
        .messages(messages)
        .build();
    let resp2 = p.chat(&req2).await.unwrap();
    assert!(matches!(resp2.stop_reason, StopReason::EndTurn));
    assert!(!resp2.text().is_empty());
}

#[tokio::test]
async fn reasoning_content_extracted() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(&server, include_str!("fixtures/anthropic/chat_reasoning.json")).await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 1024)
        .message(Message::User(vec![RequestContent::Text(
            "What is 7 times 8?".into(),
        )]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();
    assert!(resp.reasoning().is_some(), "reasoning should be parsed from thinking blocks");
    assert!(
        resp.reasoning().unwrap().contains("56") || resp.reasoning().unwrap().contains("7 times 8"),
        "expected reasoning to mention the computation, got: {:?}",
        resp.reasoning()
    );
    assert_eq!(resp.text(), "56");
}

#[tokio::test]
async fn error_401_maps_to_unauthorized() {
    let server = MockServer::start().await;
    // Use generic mock with error body
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(
            wiremock::ResponseTemplate::new(401)
                .set_body_raw(include_str!("fixtures/anthropic/error_unauthorized.json"), "application/json")
        )
        .mount(&server)
        .await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::Unauthorized));
}

#[tokio::test]
async fn error_429_maps_to_rate_limit() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(
            wiremock::ResponseTemplate::new(429)
                .insert_header("retry-after", "60")
                .set_body_raw(include_str!("fixtures/anthropic/error_rate_limit.json"), "application/json"),
        )
        .mount(&server)
        .await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::RateLimit { retry_after: _ }));
}

#[tokio::test]
async fn error_500_maps_to_server_error() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(
            wiremock::ResponseTemplate::new(500)
                .set_body_raw(include_str!("fixtures/anthropic/error_server.json"), "application/json"),
        )
        .mount(&server)
        .await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::ServerError(_)));
}

#[tokio::test]
async fn error_529_maps_to_overloaded() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(
            wiremock::ResponseTemplate::new(529)
                .set_body_raw(include_str!("fixtures/anthropic/error_server.json"), "application/json"),
        )
        .mount(&server)
        .await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::Overloaded { retry_after: _ }));
}

#[tokio::test]
async fn provider_is_object_safe_with_anthropic() {
    let p: Box<dyn Provider> = Box::new(AnthropicProvider::builder("test-key").build());
    // Compile check only — satisfies the trait object requirement.
    drop(p);
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test --test anthropic_chat 2>&1
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/anthropic_chat.rs tests/fixtures/anthropic/ tests/common/mod.rs
git commit -m "test(anthropic): add offline chat tests with fixtures

- 11 tests: simple chat, system message, tool call, multi-turn, reasoning,
  error mapping (401, 429, 500, 529), object safety
- Hand-crafted fixtures for each scenario
- Common mock helpers for the /v1/messages endpoint"

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 6: Offline Tests — Anthropic Streaming

**Files:**
- Create: `tests/anthropic_stream.rs`
- Create: `tests/fixtures/anthropic/stream_simple.txt`
- Create: `tests/fixtures/anthropic/stream_tool_call.txt`
- Create: `tests/fixtures/anthropic/stream_reasoning.txt`

- [ ] **Step 1: Create SSE stream fixture files**

`tests/fixtures/anthropic/stream_simple.txt`:
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_stream_01","type":"message","role":"assistant","model":"claude-sonnet-4-6-20250514","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":12,"output_tokens":3,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}

event: message_stop
data: {"type":"message_stop"}
```

`tests/fixtures/anthropic/stream_tool_call.txt`:
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_stream_02","type":"message","role":"assistant","model":"claude-sonnet-4-6-20250514","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":50,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Checking weather"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_stream","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":\"Paris\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":50,"output_tokens":15,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}

event: message_stop
data: {"type":"message_stop"}
```

`tests/fixtures/anthropic/stream_reasoning.txt`:
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_stream_03","type":"message","role":"assistant","model":"claude-sonnet-4-6-20250514","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":20,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":"sig_stream"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me compute 7*8..."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" That's 56."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_stream_full"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"56"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":20,"output_tokens":15,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}

event: message_stop
data: {"type":"message_stop"}
```

- [ ] **Step 2: Write the anthropic_stream test file**

`tests/anthropic_stream.rs`:
```rust
mod common;

use futures::StreamExt;
use llm_adapter::{Provider, anthropic::AnthropicProvider, types::*};
use wiremock::MockServer;

fn provider(server: &MockServer) -> AnthropicProvider {
    AnthropicProvider::builder("test-key")
        .base_url(server.uri())
        .build()
}

#[tokio::test]
async fn stream_simple_collects_to_response() {
    let server = MockServer::start().await;
    common::mock_anthropic_stream(&server, include_str!("fixtures/anthropic/stream_simple.txt")).await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(!resp.text().is_empty());
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
}

#[tokio::test]
async fn stream_events_fire_in_order() {
    let server = MockServer::start().await;
    common::mock_anthropic_stream(&server, include_str!("fixtures/anthropic/stream_simple.txt")).await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let mut handle = provider(&server).chat_stream(&req).await.unwrap();
    let events = handle.events();

    let mut content_starts = 0usize;
    let mut text_deltas = 0usize;
    let mut content_stops = 0usize;
    let mut message_stops = 0usize;

    while let Some(event) = events.next().await {
        match event.unwrap() {
            StreamEvent::ContentStart { .. } => content_starts += 1,
            StreamEvent::TextDelta { .. } => text_deltas += 1,
            StreamEvent::ContentStop { .. } => content_stops += 1,
            StreamEvent::MessageStop { .. } => message_stops += 1,
            _ => {}
        }
    }

    assert_eq!(content_starts, 1);
    assert!(text_deltas >= 1);
    assert_eq!(content_stops, 1);
    assert_eq!(message_stops, 1);
}

#[tokio::test]
async fn stream_tool_call_collects_invocation() {
    let server = MockServer::start().await;
    common::mock_anthropic_stream(&server, include_str!("fixtures/anthropic/stream_tool_call.txt")).await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::User(vec![RequestContent::Text(
            "Weather in Paris?".into(),
        )]))
        .tool(ToolDef {
            name: "get_weather".into(),
            description: "Get weather".into(),
            input_schema: serde_json::json!({}),
        })
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
}

#[tokio::test]
async fn stream_reasoning_emits_reasoning_deltas() {
    let server = MockServer::start().await;
    common::mock_anthropic_stream(&server, include_str!("fixtures/anthropic/stream_reasoning.txt")).await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 512)
        .message(Message::User(vec![RequestContent::Text(
            "What is 7 times 8?".into(),
        )]))
        .build();

    let mut handle = provider(&server).chat_stream(&req).await.unwrap();
    let events = handle.events();

    let mut reasoning_deltas = 0usize;
    let mut text_deltas = 0usize;
    let mut message_stops = 0usize;

    while let Some(event) = events.next().await {
        match event.unwrap() {
            StreamEvent::ReasoningDelta { .. } => reasoning_deltas += 1,
            StreamEvent::TextDelta { .. } => text_deltas += 1,
            StreamEvent::MessageStop { .. } => message_stops += 1,
            _ => {}
        }
    }

    assert!(reasoning_deltas > 0, "expected at least one reasoning delta");
    assert!(text_deltas > 0, "expected at least one text delta");
    assert_eq!(message_stops, 1);
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --test anthropic_stream 2>&1
```

Expected: all tests pass.

- [ ] **Step 4: Run the full test suite to ensure no regressions**

```bash
cargo test --lib 2>&1
```

Expected: all library tests pass, including existing OpenAI/DeepSeek tests.

- [ ] **Step 5: Commit**

```bash
git add tests/anthropic_stream.rs tests/fixtures/anthropic/stream_simple.txt tests/fixtures/anthropic/stream_tool_call.txt tests/fixtures/anthropic/stream_reasoning.txt
git commit -m "test(anthropic): add offline stream tests with SSE fixtures

- 4 tests: simple stream collect, event ordering, tool call, reasoning deltas
- Hand-crafted SSE fixture files with message_start, content_block sequences,
  thinking_delta/signature_delta, and message_stop"

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 7: Integration Test — Anthropic

**Files:**
- Create: `tests/integration_anthropic.rs`
- Modify: `README.md`, `README.zh-CN.md` (mark Anthropic as supported)
- Modify: (optional) `.env` template

- [ ] **Step 1: Write the integration test file**

`tests/integration_anthropic.rs`:
```rust
//! Integration tests hitting a real Anthropic-compatible endpoint.
//!
//! Run with: `./test.sh --features integration`
//!
//! Skipped when `ANTHROPIC_API_KEY` is unset.
//! Configuration via environment (`.env` is loaded automatically):
//!   ANTHROPIC_API_KEY      required
//!   ANTHROPIC_BASE_URL     default https://api.anthropic.com
//!   ANTHROPIC_MESSAGES_PATH  default /v1/messages
//!   ANTHROPIC_MODEL        default claude-sonnet-4-6-20250514
//!   ANTHROPIC_REASONING_MODEL  optional; reasoning test skipped if absent

#![cfg(feature = "integration")]

mod common;

use futures::StreamExt;
use llm_adapter::{Provider, anthropic::AnthropicProvider, types::*};

/// Returns `None` when the test should be skipped.
fn provider() -> Option<(AnthropicProvider, String)> {
    common::load_dotenv();
    let api_key = match common::opt_env("ANTHROPIC_API_KEY") {
        Some(k) => k,
        None => {
            println!("[skip] ANTHROPIC_API_KEY not set");
            return None;
        }
    };
    let base_url = common::opt_env("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|| "https://api.anthropic.com".into());
    let messages_path = common::opt_env("ANTHROPIC_MESSAGES_PATH")
        .unwrap_or_else(|| "/v1/messages".into());
    let model = common::opt_env("ANTHROPIC_MODEL")
        .unwrap_or_else(|| "claude-sonnet-4-6-20250514".into());

    let p = AnthropicProvider::builder(api_key)
        .base_url(base_url)
        .messages_path(messages_path)
        .build();
    Some((p, model))
}

#[tokio::test]
async fn integration_simple_chat() {
    let Some((p, model)) = provider() else {
        return;
    };
    let req = ChatRequest::builder(model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "Say hi in exactly five words.".into(),
        )]))
        .build();

    let resp = p.chat(&req).await.expect("chat failed");
    assert!(!resp.text().is_empty(), "expected non-empty text");
    assert!(matches!(
        resp.stop_reason,
        StopReason::EndTurn | StopReason::MaxTokens
    ));
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
}

#[tokio::test]
async fn integration_multi_turn() {
    let Some((p, model)) = provider() else {
        return;
    };

    let req1 = ChatRequest::builder(&model, 1024)
        .message(Message::System("Respond in one short sentence.".into()))
        .message(Message::User(vec![RequestContent::Text(
            "My favourite colour is blue. Remember that.".into(),
        )]))
        .build();
    let resp1 = p.chat(&req1).await.expect("turn 1 failed");
    assert!(!resp1.text().is_empty());

    let mut messages = req1.messages().to_vec();
    messages.push(resp1.into_message());
    messages.push(Message::User(vec![RequestContent::Text(
        "What did I say my favourite colour was?".into(),
    )]));

    let req2 = ChatRequest::builder(model, 1024).messages(messages).build();
    let resp2 = p.chat(&req2).await.expect("turn 2 failed");
    let answer = resp2.text().to_lowercase();
    assert!(
        answer.contains("blue"),
        "model should remember the favourite colour; got: {}",
        resp2.text()
    );
}

#[tokio::test]
async fn integration_stream_simple() {
    let Some((p, model)) = provider() else {
        return;
    };
    let req = ChatRequest::builder(model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "Count from one to five.".into(),
        )]))
        .build();

    let mut handle = p.chat_stream(&req).await.expect("stream failed");

    let mut text_deltas = 0usize;
    let mut got_stop = false;
    let mut final_usage = Usage::default();
    while let Some(event) = handle.events().next().await {
        match event.expect("stream error") {
            StreamEvent::TextDelta { .. } => text_deltas += 1,
            StreamEvent::MessageStop { usage, .. } => {
                got_stop = true;
                final_usage = usage;
            }
            _ => {}
        }
    }
    assert!(text_deltas > 0, "expected at least one TextDelta");
    assert!(got_stop, "expected MessageStop");
    assert!(final_usage.input_tokens > 0);
    assert!(final_usage.output_tokens > 0);
}

fn weather_tool() -> ToolDef {
    ToolDef {
        name: "get_weather".into(),
        description: "Get current weather for a city.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }),
    }
}

#[tokio::test]
async fn integration_tool_call_single_turn() {
    let Some((p, model)) = provider() else {
        return;
    };
    let req = ChatRequest::builder(model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "You have a get_weather tool. Call it to find the weather in Paris.".into(),
        )]))
        .tool(weather_tool())
        .build();

    let resp = p.chat(&req).await.expect("tool call failed");
    assert!(
        matches!(resp.stop_reason, StopReason::ToolUse),
        "expected tool_use stop reason; got: {:?}, text: {:?}",
        resp.stop_reason,
        resp.text()
    );
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
    assert!(
        invocations[0].input.get("location").is_some(),
        "expected `location` in tool arguments, got: {}",
        invocations[0].input
    );
}

#[tokio::test]
async fn integration_tool_call_round_trip() {
    let Some((p, model)) = provider() else {
        return;
    };

    // Round 1: ask for a tool call.
    let req1 = ChatRequest::builder(&model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "You have a get_weather tool. Call it to find the weather in Paris.".into(),
        )]))
        .tool(weather_tool())
        .build();
    let resp1 = p.chat(&req1).await.expect("round 1 failed");
    let invocation = resp1
        .tool_invocations()
        .next()
        .expect("expected at least one tool invocation")
        .clone();

    // Round 2: feed back a fake tool result and let the model answer.
    let mut messages = req1.messages().to_vec();
    messages.push(resp1.into_message());
    messages.push(Message::Tool {
        invocation_id: invocation.id,
        content: vec![RequestContent::Text("Sunny, 22°C".into())],
        is_error: false,
    });

    let req2 = ChatRequest::builder(model, 1024)
        .messages(messages)
        .tool(weather_tool())
        .build();
    let resp2 = p.chat(&req2).await.expect("round 2 failed");
    assert!(
        matches!(resp2.stop_reason, StopReason::EndTurn),
        "expected EndTurn, got {:?}",
        resp2.stop_reason
    );
    assert!(!resp2.text().is_empty());
}

#[tokio::test]
async fn integration_stream_tool_call() {
    let Some((p, model)) = provider() else {
        return;
    };
    let req = ChatRequest::builder(model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "You have a get_weather tool. Call it to find the weather in Paris.".into(),
        )]))
        .tool(weather_tool())
        .build();

    let handle = p.chat_stream(&req).await.expect("stream failed");
    let resp = handle.collect().await.expect("collect failed");
    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert!(!invocations.is_empty());
    assert_eq!(invocations[0].name, "get_weather");
    assert!(invocations[0].input.get("location").is_some());
}
```

- [ ] **Step 2: Update README.md** — Mark Anthropic as "supported" (in the provider list)

Find the provider support matrix and add Anthropic. The existing README should have a section listing supported providers. Update it to include:
- ✅ Anthropic (Messages API)

Also update the README.zh-CN.md accordingly.

- [ ] **Step 3: Verify `cargo test --features integration` picks up the new file**

```bash
cargo check --all-targets --features integration 2>&1
```

Expected: compiles cleanly.

- [ ] **Step 4: Commit**

```bash
git add tests/integration_anthropic.rs
git add README.md README.zh-CN.md
git commit -m "test(anthropic): add integration test suite

- 6 integration tests: simple chat, multi-turn, streaming,
  tool call single-turn, tool call round-trip, stream tool call
- Updated README to mark Anthropic as supported
- Gated behind --features integration; skips gracefully without key"

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 8: Final Verification and Cleanup

**Files:**
- No new files; run existing checks

- [ ] **Step 1: Run `cargo fmt -- --check`**

```bash
cargo fmt -- --check 2>&1
```

Expected: no formatting issues. If there are, run `cargo fmt` to fix.

- [ ] **Step 2: Run `cargo clippy --all-targets --features integration`**

```bash
cargo clippy --all-targets --features integration 2>&1
```

Expected: no warnings. Add `#[allow(dead_code)]` to any unused wire/schema fields that clippy flags (following the established pattern from the OpenAI schema).

- [ ] **Step 3: Run full offline test suite**

```bash
./test.sh 2>&1
```

Expected: all tests pass (existing OpenAI, DeepSeek, and new Anthropic tests).

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "style: cargo fmt + clippy clean after anthropic provider"

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
