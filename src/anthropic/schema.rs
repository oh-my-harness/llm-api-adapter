use serde::{Deserialize, Serialize};

// ── Request ──────────────────────────────────────────────────────────────────

/// Anthropic Messages API request.
#[derive(Debug, Serialize)]
pub(crate) struct AnthropicRequest {
    /// Model identifier (e.g., "claude-sonnet-4-20250514").
    pub model: String,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// List of messages in the conversation.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<AnthropicWireMessage>,
    /// System prompt (single string or list of content blocks).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicContentBlocks>,
    /// List of tools available to the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AnthropicTool>,
    /// Tool choice configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Sampling temperature (0.0 to 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter (0.0 to 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Only sample from the top K tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    /// Enable streaming responses.
    pub stream: bool,
    /// Thinking / extended thinking configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinkingConfig>,
    /// Metadata about the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,
    /// Whether to disable parallel tool use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
}

/// A message in the conversation (wire format: role + content).
#[derive(Debug, Serialize)]
pub(crate) struct AnthropicWireMessage {
    /// Role: "user" or "assistant".
    pub role: String,
    /// Message content (single string or array of content blocks).
    pub content: AnthropicContentBlocks,
}

/// Message content: either a plain text string or an array of content blocks.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum AnthropicContentBlocks {
    /// Simple text string.
    Text(String),
    /// Array of mixed content blocks.
    Blocks(Vec<AnthropicContentBlock>),
}

/// A single content block in a multipart message.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicContentBlock {
    /// Text content (optionally with cache control).
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    /// Image content.
    Image { source: AnthropicImageSource },
    /// Tool use request from the model.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Tool result from the user.
    ToolResult {
        tool_use_id: String,
        content: AnthropicContentBlocks,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    /// Thinking content (extended thinking).
    Thinking { thinking: String, signature: String },
    /// Redacted thinking content (extended thinking).
    RedactedThinking { data: String },
}

/// Image source metadata.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicImageSource {
    /// Source type (e.g., "base64", "url").
    #[serde(rename = "type")]
    pub source_type: String,
    /// Media type (e.g., "image/png", "image/jpeg").
    pub media_type: String,
    /// Base64-encoded image data (when source_type is "base64").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// Image URL (when source_type is "url").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Cache control marker for prompt caching.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicCacheControl {
    /// Cache control type (e.g., "ephemeral").
    pub r#type: String,
}

/// Extended thinking configuration.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicThinkingConfig {
    /// Thinking type (e.g., "enabled").
    pub r#type: String,
    /// Maximum tokens to spend on thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

/// Metadata about the request.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicMetadata {
    /// An external identifier for the user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// A tool available to the model (no "function" wrapper).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicTool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON Schema describing the tool's input.
    pub input_schema: serde_json::Value,
}

// ── Response ─────────────────────────────────────────────────────────────────

/// Anthropic Messages API response (non-streaming).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AnthropicResponse {
    /// Response ID (e.g., "msg_01...").
    pub id: String,
    /// Object type (always "message").
    #[serde(rename = "type")]
    pub response_type: String,
    /// Role (always "assistant").
    pub role: String,
    /// Model used.
    pub model: String,
    /// Response content blocks.
    pub content: Vec<AnthropicResponseBlock>,
    /// Reason for stopping.
    pub stop_reason: Option<String>,
    /// Stop sequence that caused the stop.
    pub stop_sequence: Option<String>,
    /// Token usage information.
    pub usage: AnthropicUsage,
}

/// A single content block in the response.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicResponseBlock {
    /// Text content.
    Text { text: String },
    /// Tool use request.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Thinking content (extended thinking).
    Thinking { thinking: String, signature: String },
    /// Redacted thinking content (extended thinking).
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
}

/// Token usage statistics.
#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct AnthropicUsage {
    /// Number of tokens in the input.
    pub input_tokens: u32,
    /// Number of tokens in the output.
    pub output_tokens: u32,
    /// Tokens from cache creation.
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Tokens read from cache.
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

// ── Streaming ─────────────────────────────────────────────────────────────────

/// An SSE event in a streaming response.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicStreamEvent {
    /// First event: message metadata.
    #[serde(rename = "message_start")]
    MessageStart {
        /// Event type (always "message_start").
        #[serde(rename = "type")]
        event_type: String,
        /// The initial message (with empty content).
        message: AnthropicResponse,
    },
    /// Content block begins.
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        /// Index of the content block.
        index: usize,
        /// The content block.
        content_block: AnthropicResponseBlock,
    },
    /// Content block delta update.
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        /// Index of the content block.
        index: usize,
        /// The delta payload.
        delta: AnthropicDelta,
    },
    /// Content block finished.
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        /// Index of the content block.
        index: usize,
    },
    /// Message-level delta (stop reason/sequence, usage).
    #[serde(rename = "message_delta")]
    MessageDelta {
        /// The delta payload.
        delta: AnthropicMessageDeltaPayload,
        /// Token usage (output tokens only).
        usage: AnthropicUsage,
    },
    /// Stream is complete.
    #[serde(rename = "message_stop")]
    MessageStop,
    /// Heartbeat.
    Ping,
}

/// Delta content in a streaming content block update.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicDelta {
    /// Text delta.
    #[serde(rename = "text_delta")]
    TextDelta {
        /// Incremental text content.
        text: String,
    },
    /// JSON tool use input delta.
    #[serde(rename = "input_json_delta")]
    InputJsonDelta {
        /// Incremental JSON string.
        partial_json: String,
    },
    /// Thinking content delta.
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        /// Incremental thinking content.
        thinking: String,
    },
    /// Signature delta for thinking.
    #[serde(rename = "signature_delta")]
    SignatureDelta {
        /// Incremental signature.
        signature: String,
    },
}

/// Message-level delta payload (stop info).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AnthropicMessageDeltaPayload {
    /// Reason for stopping.
    pub stop_reason: Option<String>,
    /// Stop sequence that caused the stop.
    pub stop_sequence: Option<String>,
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// Anthropic error response.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicErrorResponse {
    /// Object type (always "error").
    #[serde(rename = "type")]
    pub response_type: String,
    /// Error details.
    pub error: AnthropicErrorBody,
}

/// Error body.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicErrorBody {
    /// Error type (e.g., "invalid_request_error", "authentication_error").
    pub r#type: String,
    /// Error message.
    pub message: String,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_deserializes() {
        let json = serde_json::json!({
            "id": "msg_01abcd1234",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! I can help you with that."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01abc123",
                    "name": "get_weather",
                    "input": {
                        "location": "San Francisco, CA"
                    }
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 25
            }
        });

        let resp: AnthropicResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "msg_01abcd1234");
        assert_eq!(resp.response_type, "message");
        assert_eq!(resp.role, "assistant");
        assert_eq!(resp.model, "claude-sonnet-4-20250514");
        assert_eq!(resp.content.len(), 2);
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));
        assert!(resp.stop_sequence.is_none());
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 25);
        assert_eq!(resp.usage.cache_creation_input_tokens, 0);
        assert_eq!(resp.usage.cache_read_input_tokens, 0);

        // Check text block
        match &resp.content[0] {
            AnthropicResponseBlock::Text { text } => {
                assert_eq!(text, "Hello! I can help you with that.");
            }
            other => panic!("expected Text, got {other:?}"),
        }

        // Check tool_use block
        match &resp.content[1] {
            AnthropicResponseBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_01abc123");
                assert_eq!(name, "get_weather");
                assert_eq!(
                    input.get("location").and_then(|v| v.as_str()),
                    Some("San Francisco, CA")
                );
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn thinking_response_deserializes() {
        let json = serde_json::json!({
            "id": "msg_02xyz789",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me reason about this step by step...",
                    "signature": "EqoBCkgIARAU...abc123"
                },
                {
                    "type": "text",
                    "text": "After thinking, the answer is 42."
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 15,
                "output_tokens": 100,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 0
            }
        });

        let resp: AnthropicResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.content.len(), 2);
        assert_eq!(resp.usage.input_tokens, 15);
        assert_eq!(resp.usage.cache_creation_input_tokens, 10);

        // Check thinking block
        match &resp.content[0] {
            AnthropicResponseBlock::Thinking {
                thinking,
                signature,
            } => {
                assert!(thinking.contains("step by step"));
                assert_eq!(signature, "EqoBCkgIARAU...abc123");
            }
            other => panic!("expected Thinking, got {other:?}"),
        }

        // Check text block
        match &resp.content[1] {
            AnthropicResponseBlock::Text { text } => {
                assert_eq!(text, "After thinking, the answer is 42.");
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn stream_event_deserializes() {
        // content_block_delta with text_delta
        let json = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello"
            }
        });

        let event: AnthropicStreamEvent = serde_json::from_value(json).unwrap();
        match event {
            AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    AnthropicDelta::TextDelta { text } => {
                        assert_eq!(text, "Hello");
                    }
                    other => panic!("expected TextDelta, got {other:?}"),
                }
            }
            other => panic!("expected ContentBlockDelta, got {other:?}"),
        }

        // message_start
        let start_json = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_01xyz",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 0
                }
            }
        });
        let start_event: AnthropicStreamEvent = serde_json::from_value(start_json).unwrap();
        match start_event {
            AnthropicStreamEvent::MessageStart {
                event_type,
                message,
            } => {
                assert_eq!(event_type, "message_start");
                assert_eq!(message.id, "msg_01xyz");
            }
            other => panic!("expected MessageStart, got {other:?}"),
        }

        // message_stop
        let stop_json = serde_json::json!({ "type": "message_stop" });
        let stop_event: AnthropicStreamEvent = serde_json::from_value(stop_json).unwrap();
        assert!(matches!(stop_event, AnthropicStreamEvent::MessageStop));

        // ping
        let ping_json = serde_json::json!({ "type": "ping" });
        let ping_event: AnthropicStreamEvent = serde_json::from_value(ping_json).unwrap();
        assert!(matches!(ping_event, AnthropicStreamEvent::Ping));
    }

    #[test]
    fn content_blocks_text_and_blocks_roundtrip() {
        // Text variant
        let text_json = serde_json::json!("plain text content");
        let text: AnthropicContentBlocks = serde_json::from_value(text_json).unwrap();
        match text {
            AnthropicContentBlocks::Text(s) => assert_eq!(s, "plain text content"),
            other => panic!("expected Text, got {other:?}"),
        }

        // Serialize Text variant back
        let serialized = serde_json::to_value(&text).unwrap();
        assert_eq!(serialized, serde_json::json!("plain text content"));

        // Blocks variant
        let blocks_json = serde_json::json!([
            {
                "type": "text",
                "text": "Hello"
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgo="
                }
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01abc",
                "content": "result text"
            },
            {
                "type": "tool_use",
                "id": "toolu_02def",
                "name": "get_weather",
                "input": {}
            },
            {
                "type": "thinking",
                "thinking": "thinking content",
                "signature": "sig123"
            },
            {
                "type": "redacted_thinking",
                "data": "redacted"
            }
        ]);
        let blocks: AnthropicContentBlocks = serde_json::from_value(blocks_json).unwrap();
        match blocks {
            AnthropicContentBlocks::Blocks(list) => {
                assert_eq!(list.len(), 6, "expected 6 content blocks");
                // Text
                assert!(
                    matches!(&list[0], AnthropicContentBlock::Text { text, .. } if text == "Hello")
                );
                // Image
                match &list[1] {
                    AnthropicContentBlock::Image { source } => {
                        assert_eq!(source.source_type, "base64");
                        assert_eq!(source.media_type, "image/png");
                    }
                    other => panic!("expected Image, got {other:?}"),
                }
                // ToolResult
                match &list[2] {
                    AnthropicContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        assert_eq!(tool_use_id, "toolu_01abc");
                        assert!(is_error.is_none());
                        assert!(
                            matches!(content, AnthropicContentBlocks::Text(s) if s == "result text")
                        );
                    }
                    other => panic!("expected ToolResult, got {other:?}"),
                }
                // ToolUse
                match &list[3] {
                    AnthropicContentBlock::ToolUse { id, name, .. } => {
                        assert_eq!(id, "toolu_02def");
                        assert_eq!(name, "get_weather");
                    }
                    other => panic!("expected ToolUse, got {other:?}"),
                }
                // Thinking
                match &list[4] {
                    AnthropicContentBlock::Thinking {
                        thinking,
                        signature,
                    } => {
                        assert_eq!(thinking, "thinking content");
                        assert_eq!(signature, "sig123");
                    }
                    other => panic!("expected Thinking, got {other:?}"),
                }
                // RedactedThinking
                match &list[5] {
                    AnthropicContentBlock::RedactedThinking { data } => {
                        assert_eq!(data, "redacted");
                    }
                    other => panic!("expected RedactedThinking, got {other:?}"),
                }
            }
            other => panic!("expected Blocks, got {other:?}"),
        }

        // Serialize Blocks variant back
        let serialized_blocks = serde_json::to_value(&blocks).unwrap();
        let reparsed: AnthropicContentBlocks = serde_json::from_value(serialized_blocks).unwrap();
        assert!(matches!(reparsed, AnthropicContentBlocks::Blocks(_)));
    }
}
