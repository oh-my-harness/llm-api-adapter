use crate::LlmError;
use crate::types::{
    ChatRequest, ChatResponse, Message, RequestContent, ResponseContent, StopReason, ToolChoice,
    ToolInvocation, Usage,
};
use serde::Serialize;

use super::schema::{
    OaiContent, OaiContentPart, OaiFileData, OaiFunction, OaiImageUrl, OaiMessage, OaiRequest,
    OaiResponse, OaiTool, OaiToolCall, OaiToolCallFunction, OaiUsage,
};

use crate::types::ThinkingLevel;

/// Map a [`ThinkingLevel`] to the OpenAI `reasoning_effort` string.
///
/// Returns `None` for [`ThinkingLevel::Off`] (meaning: omit the field entirely,
/// so the model uses its default effort). Note: on OpenAI o-series / GPT-5.x
/// reasoning models, reasoning is built into the architecture and cannot be
/// truly "turned off" — setting `Off` only avoids specifying an effort level.
/// To fully disable reasoning, switch to a non-reasoning model (e.g. gpt-4o).
///
/// [`ThinkingLevel::Budget`] is interval-mapped to one of the 5 discrete levels.
pub(crate) fn thinking_to_oai_effort(spec: ThinkingLevel) -> Option<String> {
    match spec {
        ThinkingLevel::Off => None,
        ThinkingLevel::Minimal => Some("minimal".into()),
        ThinkingLevel::Low => Some("low".into()),
        ThinkingLevel::Medium => Some("medium".into()),
        ThinkingLevel::High => Some("high".into()),
        ThinkingLevel::XHigh => Some("xhigh".into()),
        ThinkingLevel::Budget(n) => Some(
            if n <= 2048 {
                "minimal"
            } else if n <= 4096 {
                "low"
            } else if n <= 8192 {
                "medium"
            } else if n <= 16384 {
                "high"
            } else {
                "xhigh"
            }
            .into(),
        ),
    }
}

/// Map a [`ThinkingLevel`] to the DeepSeek `reasoning_effort` string.
///
/// DeepSeek only supports `"high"` and `"max"` — per official docs, `low` and
/// `medium` are mapped to `"high"`, and `xhigh` is mapped to `"max"`.
/// Returns `None` for [`ThinkingLevel::Off`] (omit the field).
pub(crate) fn thinking_to_deepseek_effort(spec: ThinkingLevel) -> Option<String> {
    match spec {
        ThinkingLevel::Off => None,
        ThinkingLevel::XHigh => Some("max".into()),
        ThinkingLevel::Budget(n) => Some(if n > 8192 { "max" } else { "high" }.into()),
        // Minimal, Low, Medium, High all map to "high" per DeepSeek docs.
        _ => Some("high".into()),
    }
}

// ── OpenAIExt ────────────────────────────────────────────────────────────────

/// Provider-specific extensions for OpenAI requests.
///
/// These fields allow fine-tuning of OpenAI-specific behavior beyond the
/// canonical ChatRequest. All fields are optional.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OpenAIExt {
    /// Seed for reproducibility. Passed as `seed` to OpenAI API.
    pub seed: Option<u64>,
    /// Frequency penalty (-2.0 to 2.0). Passed as `frequency_penalty` to OpenAI API.
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (-2.0 to 2.0). Passed as `presence_penalty` to OpenAI API.
    pub presence_penalty: Option<f32>,
    /// Unique identifier for tracking requests. Passed as `user` to OpenAI API.
    pub user: Option<String>,
    /// Reasoning effort for o-series / GPT-5 models ("low", "medium", "high", etc.).
    /// Has no effect on models that do not support reasoning.
    pub reasoning_effort: Option<String>,
    /// Arbitrary extra fields merged into the request body at the top level.
    ///
    /// Useful for provider-specific extensions outside the OpenAI Chat
    /// Completions schema — e.g. DeepSeek's `thinking: {type: enabled|disabled}`
    /// to explicitly control thinking mode, or other gateway-specific fields.
    ///
    /// Must be a JSON object. Keys override any field already produced by
    /// the canonical conversion. Mirrors the OpenAI Python SDK's `extra_body`.
    pub extra_body: Option<serde_json::Value>,
    /// If true, parse DeepSeek reasoning_content field from responses.
    pub parse_reasoning_content: bool,
    /// If true, tolerate keepalive messages in stream (DeepSeek compatibility).
    pub tolerant_keepalive: bool,
}

impl OpenAIExt {
    /// Create a new builder for OpenAIExt.
    pub fn builder() -> OpenAIExtBuilder {
        OpenAIExtBuilder::default()
    }
}

/// Builder for OpenAIExt with fluent API.
#[derive(Debug, Default)]
pub struct OpenAIExtBuilder {
    seed: Option<u64>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    user: Option<String>,
    reasoning_effort: Option<String>,
    extra_body: Option<serde_json::Value>,
    parse_reasoning_content: bool,
    tolerant_keepalive: bool,
}

impl OpenAIExtBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        OpenAIExtBuilder::default()
    }

    /// Set the seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the frequency penalty.
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Set the presence penalty.
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Set the user identifier.
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set reasoning effort for o-series / GPT-5 models ("low", "medium", "high", etc.).
    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Provide arbitrary extra request-body fields (must be a JSON object).
    /// Merged into the request at the top level after canonical conversion.
    pub fn extra_body(mut self, body: serde_json::Value) -> Self {
        self.extra_body = Some(body);
        self
    }

    /// Enable parsing of DeepSeek reasoning_content.
    pub fn parse_reasoning_content(mut self) -> Self {
        self.parse_reasoning_content = true;
        self
    }

    /// Enable tolerant keepalive for stream compatibility.
    pub fn tolerant_keepalive(mut self) -> Self {
        self.tolerant_keepalive = true;
        self
    }

    /// Build the OpenAIExt instance.
    pub fn build(self) -> OpenAIExt {
        OpenAIExt {
            seed: self.seed,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            user: self.user,
            reasoning_effort: self.reasoning_effort,
            extra_body: self.extra_body,
            parse_reasoning_content: self.parse_reasoning_content,
            tolerant_keepalive: self.tolerant_keepalive,
        }
    }
}

/// A ChatRequest paired with OpenAI provider-specific extensions.
pub struct OpenAIRequest {
    /// The canonical chat request.
    pub base: ChatRequest,
    /// OpenAI-specific extensions.
    pub ext: OpenAIExt,
}

// ── Conversion: Canonical → OpenAI ──────────────────────────────────────────

/// Convert a canonical ResponseFormat to an OpenAI response_format value.
fn response_format_to_oai(fmt: &crate::types::ResponseFormat) -> serde_json::Value {
    match fmt {
        crate::types::ResponseFormat::JsonObject => {
            serde_json::json!({"type": "json_object"})
        }
        crate::types::ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => {
            let mut js = serde_json::json!({
                "name": name,
                "schema": schema,
            });
            if let Some(s) = strict {
                js["strict"] = serde_json::Value::Bool(*s);
            }
            serde_json::json!({"type": "json_schema", "json_schema": js})
        }
    }
}

/// Convert a canonical ChatRequest to an OpenAI OaiRequest.
///
/// # Arguments
/// * `req` - The canonical chat request
/// * `ext` - Optional OpenAI-specific extensions
/// * `stream` - Whether to enable streaming
///
/// # Returns
/// An OaiRequest ready to send to the OpenAI API, or an LlmError if conversion fails.
pub(crate) fn to_oai(
    req: &ChatRequest,
    ext: Option<&OpenAIExt>,
    scheme: super::ThinkingScheme,
    max_tokens_scheme: super::MaxTokensScheme,
    stream: bool,
    documents_inline: bool,
) -> Result<OaiRequest, LlmError> {
    let default_ext = OpenAIExt::default();
    let ext = ext.unwrap_or(&default_ext);

    // Convert messages
    let messages = req
        .messages()
        .iter()
        .map(|m| message_to_oai(m, documents_inline))
        .collect::<Result<Vec<_>, _>>()?;

    // Convert tools
    let tools = req
        .tools()
        .iter()
        .map(|t| OaiTool {
            r#type: "function".into(),
            function: OaiFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect();

    // Convert tool choice
    let tool_choice = if req.tools().is_empty() {
        None
    } else {
        Some(tool_choice_to_oai(req.tool_choice()))
    };
    let params = req.params();

    Ok(OaiRequest {
        model: req.model().into(),
        messages,
        tools,
        tool_choice,
        temperature: params.temperature(),
        max_tokens: (max_tokens_scheme == super::MaxTokensScheme::MaxTokens)
            .then_some(params.max_tokens()),
        max_completion_tokens: (max_tokens_scheme == super::MaxTokensScheme::MaxCompletionTokens)
            .then_some(params.max_tokens()),
        top_p: params.top_p(),
        stop: params.stop_sequences().to_vec(),
        stream,
        reasoning_effort: ext.reasoning_effort.clone().or_else(|| {
            if scheme == super::ThinkingScheme::ReasoningEffort {
                req.thinking().and_then(thinking_to_oai_effort)
            } else if scheme == super::ThinkingScheme::ThinkingToggle {
                req.thinking().and_then(thinking_to_deepseek_effort)
            } else {
                None
            }
        }),
        stream_options: if stream {
            Some(super::schema::OaiStreamOptions {
                include_usage: true,
            })
        } else {
            None
        },
        seed: ext.seed,
        frequency_penalty: ext.frequency_penalty,
        presence_penalty: ext.presence_penalty,
        user: ext.user.clone(),
        response_format: req.response_format().map(response_format_to_oai),
    })
}

/// Convert a canonical Message to an OpenAI OaiMessage.
fn message_to_oai(msg: &Message, documents_inline: bool) -> Result<OaiMessage, LlmError> {
    match msg {
        Message::System(text) => Ok(OaiMessage {
            role: "system".into(),
            content: Some(OaiContent::Text(text.clone())),
            tool_calls: vec![],
            tool_call_id: None,
            reasoning_content: None,
        }),
        Message::User(content) => {
            let oai_content = if content.len() == 1 {
                if let RequestContent::Text(t) = &content[0] {
                    OaiContent::Text(t.clone())
                } else {
                    OaiContent::Parts(
                        content
                            .iter()
                            .map(|c| req_content_to_part(c, documents_inline))
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                }
            } else {
                OaiContent::Parts(
                    content
                        .iter()
                        .map(|c| req_content_to_part(c, documents_inline))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            };

            Ok(OaiMessage {
                role: "user".into(),
                content: Some(oai_content),
                tool_calls: vec![],
                tool_call_id: None,
                reasoning_content: None,
            })
        }
        Message::Assistant(content) => {
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();
            let mut reasoning_text = None;

            for item in content {
                match item {
                    ResponseContent::Text(t) => text_parts.push(t.clone()),
                    ResponseContent::ToolInvocation(ti) => {
                        tool_calls.push(OaiToolCall {
                            id: ti.id.clone(),
                            r#type: "function".into(),
                            function: OaiToolCallFunction {
                                name: ti.name.clone(),
                                arguments: ti.input.to_string(),
                            },
                        });
                    }
                    ResponseContent::Reasoning { text: r, .. } => {
                        reasoning_text = Some(r.clone());
                    }
                }
            }

            let text_content = if text_parts.is_empty() {
                None
            } else {
                Some(OaiContent::Text(text_parts.join("")))
            };

            Ok(OaiMessage {
                role: "assistant".into(),
                content: text_content,
                tool_calls,
                tool_call_id: None,
                reasoning_content: reasoning_text,
            })
        }
        Message::Tool {
            invocation_id,
            content,
            is_error: _,
        } => {
            let oai_content = if content.len() == 1 {
                if let RequestContent::Text(t) = &content[0] {
                    OaiContent::Text(t.clone())
                } else {
                    OaiContent::Parts(
                        content
                            .iter()
                            .map(|c| req_content_to_part(c, documents_inline))
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                }
            } else {
                OaiContent::Parts(
                    content
                        .iter()
                        .map(|c| req_content_to_part(c, documents_inline))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            };

            Ok(OaiMessage {
                role: "tool".into(),
                content: Some(oai_content),
                tool_calls: vec![],
                tool_call_id: Some(invocation_id.clone()),
                reasoning_content: None,
            })
        }
    }
}

/// Convert a canonical RequestContent to an OpenAI OaiContentPart.
fn req_content_to_part(
    c: &RequestContent,
    documents_inline: bool,
) -> Result<OaiContentPart, LlmError> {
    match c {
        RequestContent::Text(t) => Ok(OaiContentPart::Text { text: t.clone() }),
        RequestContent::Image(img) => {
            use crate::types::ImageSource;
            let url = match img {
                ImageSource::Url { url: u } => u.clone(),
                ImageSource::Base64 { media_type, data } => {
                    format!("data:{};base64,{}", media_type, data)
                }
            };
            Ok(OaiContentPart::ImageUrl {
                image_url: OaiImageUrl { url },
            })
        }
        RequestContent::Document(doc) => {
            use crate::types::DocumentSource;
            match doc {
                // Zhipu-style: the gateway's own document part keyed by URL.
                DocumentSource::Url { url } if !documents_inline => Ok(OaiContentPart::FileUrl {
                    file_url: OaiImageUrl { url: url.clone() },
                }),
                // OpenAI-ecosystem standard (OpenAI official / OpenRouter):
                // {"type":"file","file":{"file_data":"data:<mt>;base64,<data>"}}.
                DocumentSource::Base64 { media_type, data } if documents_inline => {
                    Ok(OaiContentPart::File {
                        file: OaiFileData {
                            file_data: format!("data:{media_type};base64,{data}"),
                        },
                    })
                }
                DocumentSource::Url { url } => Ok(OaiContentPart::File {
                    file: OaiFileData {
                        file_data: url.clone(),
                    },
                }),
                DocumentSource::Base64 { media_type, .. } => {
                    Err(LlmError::InvalidRequest(format!(
                        "this endpoint accepts documents by URL only (file_url); \
                         inline base64 documents ({media_type}) require the documents_inline \
                         capability — enable it on the builder or host the file at a URL"
                    )))
                }
            }
        }
    }
}

/// Convert a canonical ToolChoice to an OpenAI tool_choice value.
fn tool_choice_to_oai(tc: &ToolChoice) -> serde_json::Value {
    match tc {
        ToolChoice::Auto => serde_json::Value::String("auto".into()),
        ToolChoice::None => serde_json::Value::String("none".into()),
        ToolChoice::Required => serde_json::Value::String("required".into()),
        ToolChoice::Specific(name) => {
            serde_json::json!({
                "type": "function",
                "function": { "name": name }
            })
        }
    }
}

// ── Conversion: OpenAI → Canonical ──────────────────────────────────────────

/// Convert an OpenAI OaiResponse to a canonical ChatResponse.
///
/// # Arguments
/// * `resp` - The OpenAI response
/// * `parse_reasoning` - Whether to parse DeepSeek reasoning_content
///
/// # Returns
/// A ChatResponse in canonical form, or an LlmError if conversion fails.
pub(crate) fn from_oai(resp: OaiResponse, parse_reasoning: bool) -> Result<ChatResponse, LlmError> {
    if resp.choices.is_empty() {
        return Err(LlmError::InvalidRequest(
            "OaiResponse has no choices".into(),
        ));
    }

    // Use the first choice (OpenAI typically returns one)
    let choice = &resp.choices[0];
    let content = oai_message_to_content(choice.message.clone(), parse_reasoning)?;
    let stop_reason = parse_stop_reason(choice.finish_reason.as_deref());
    let usage = match resp.usage {
        Some(usage) => oai_usage_to_canonical(usage, parse_reasoning)?,
        None => Usage::missing(),
    };

    Ok(ChatResponse {
        id: resp.id,
        model: resp.model,
        content,
        stop_reason,
        usage,
    })
}

/// Convert an OpenAI OaiMessage to canonical ResponseContent vec.
fn oai_message_to_content(
    msg: OaiMessage,
    parse_reasoning: bool,
) -> Result<Vec<ResponseContent>, LlmError> {
    let mut content = Vec::new();

    // Add reasoning if present and enabled
    if parse_reasoning && let Some(reasoning) = msg.reasoning_content {
        content.push(ResponseContent::Reasoning {
            text: reasoning,
            signature: None,
        });
    }

    // Add text content if present
    if let Some(oai_content) = msg.content {
        match oai_content {
            OaiContent::Text(text) => {
                if !text.is_empty() {
                    content.push(ResponseContent::Text(text));
                }
            }
            OaiContent::Parts(_parts) => {
                // For multipart responses, we only extract text; images are skipped
                // since ResponseContent doesn't include images for responses
            }
        }
    }

    // Add tool invocations
    for tool_call in msg.tool_calls.iter() {
        let input = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
            .unwrap_or_else(|_| serde_json::json!({"raw": tool_call.function.arguments.clone()}));

        content.push(ResponseContent::ToolInvocation(ToolInvocation {
            id: tool_call.id.clone(),
            name: tool_call.function.name.clone(),
            input,
        }));
    }

    Ok(content)
}

/// Parse an OpenAI finish_reason string to a canonical StopReason.
fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls") => StopReason::ToolUse,
        Some("content_filter") => StopReason::ContentFilter,
        Some(other) => StopReason::Other(other.into()),
        None => StopReason::Other("unknown".into()),
    }
}

/// Convert an OpenAI OaiUsage to a canonical Usage.
pub(crate) fn oai_usage_to_canonical(
    u: OaiUsage,
    _parse_reasoning: bool,
) -> Result<Usage, LlmError> {
    // Cached tokens: OpenAI uses prompt_tokens_details.cached_tokens,
    // DeepSeek uses prompt_cache_hit_tokens — take whichever is non-zero.
    let openai_cached = u
        .prompt_tokens_details
        .as_ref()
        .map(|details| details.cached_tokens);
    let deepseek_cached = u.prompt_cache_hit_tokens;
    let mut provider_invariants_valid = true;
    let cached_input_tokens = match (openai_cached, deepseek_cached) {
        (Some(openai), Some(deepseek)) if openai != 0 && deepseek != 0 && openai != deepseek => {
            provider_invariants_valid = false;
            openai.max(deepseek)
        }
        (Some(openai), Some(deepseek)) => openai.max(deepseek),
        (Some(openai), None) => openai,
        (None, Some(deepseek)) => deepseek,
        (None, None) => 0,
    };

    if let Some(cache_miss) = u.prompt_cache_miss_tokens {
        match cached_input_tokens.checked_add(cache_miss) {
            Some(classified) if classified == u.prompt_tokens => {}
            Some(_) | None => provider_invariants_valid = false,
        }
    }

    // Reasoning tokens: OpenAI o-series reports via completion_tokens_details.
    let reasoning_tokens = u
        .completion_tokens_details
        .as_ref()
        .map(|d| d.reasoning_tokens)
        .unwrap_or(0);

    let usage = Usage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        cached_input_tokens,
        cache_creation_input_tokens: 0,
        reasoning_tokens,
        provenance: crate::types::UsageProvenance::Missing,
    }
    .reported();
    Ok(if provider_invariants_valid {
        usage
    } else {
        usage.reported_invalid()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ImageSource, Message, RequestContent as RC, ResponseContent as RespC, ToolChoice,
        ToolInvocation as TI,
    };

    #[test]
    fn deepseek_usage_marks_hit_miss_total_mismatch_invalid() {
        let usage = OaiUsage {
            prompt_tokens: 1_000,
            completion_tokens: 500,
            prompt_cache_hit_tokens: Some(400),
            prompt_cache_miss_tokens: Some(599),
            ..Default::default()
        };

        let usage = oai_usage_to_canonical(usage, false).unwrap();
        assert_eq!(
            usage.provenance,
            crate::types::UsageProvenance::ReportedInvalid
        );
        assert!(usage.validate_report().is_err());
    }

    #[test]
    fn deepseek_usage_overflow_is_a_completed_invalid_report() {
        let usage = OaiUsage {
            prompt_tokens: u32::MAX,
            completion_tokens: 1,
            prompt_cache_hit_tokens: Some(u32::MAX),
            prompt_cache_miss_tokens: Some(1),
            ..Default::default()
        };

        let usage = oai_usage_to_canonical(usage, false)
            .expect("parsed provider usage must remain completed-call evidence");

        assert_eq!(
            usage.provenance,
            crate::types::UsageProvenance::ReportedInvalid
        );
        assert!(usage.validate_report().is_err());
    }

    #[test]
    fn response_format_json_object_serializes() {
        use crate::types::ResponseFormat;
        let req = ChatRequest::builder("gpt-4o", 512)
            .message(Message::User(vec![crate::types::RequestContent::Text(
                "hi".into(),
            )]))
            .response_format(ResponseFormat::JsonObject)
            .build();
        let oai = to_oai(
            &req,
            None,
            super::super::ThinkingScheme::None,
            super::super::MaxTokensScheme::MaxTokens,
            false,
            true,
        )
        .unwrap();
        let body = serde_json::to_value(&oai).unwrap();
        assert_eq!(body["response_format"]["type"], "json_object");
    }

    #[test]
    fn response_format_json_schema_serializes() {
        use crate::types::ResponseFormat;
        let schema = serde_json::json!({"type": "object"});
        let req = ChatRequest::builder("gpt-4o", 512)
            .message(Message::User(vec![crate::types::RequestContent::Text(
                "hi".into(),
            )]))
            .response_format(ResponseFormat::JsonSchema {
                name: "out".into(),
                schema: schema.clone(),
                strict: Some(true),
            })
            .build();
        let oai = to_oai(
            &req,
            None,
            super::super::ThinkingScheme::None,
            super::super::MaxTokensScheme::MaxTokens,
            false,
            true,
        )
        .unwrap();
        let body = serde_json::to_value(&oai).unwrap();
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["name"], "out");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
    }

    #[test]
    fn response_format_none_omitted() {
        let req = ChatRequest::builder("gpt-4o", 512)
            .message(Message::User(vec![crate::types::RequestContent::Text(
                "hi".into(),
            )]))
            .build();
        let oai = to_oai(
            &req,
            None,
            super::super::ThinkingScheme::None,
            super::super::MaxTokensScheme::MaxTokens,
            false,
            true,
        )
        .unwrap();
        let body = serde_json::to_value(&oai).unwrap();
        assert!(body.get("response_format").is_none() || body["response_format"].is_null());
    }

    #[test]
    fn oai_preserves_adjacent_tool_result_messages() {
        let req = ChatRequest::builder("gpt-4o", 512)
            .message(Message::User(vec![RC::Text("Use both tools.".into())]))
            .message(Message::Assistant(vec![
                RespC::ToolInvocation(TI {
                    id: "call_1".into(),
                    name: "first".into(),
                    input: serde_json::json!({}),
                }),
                RespC::ToolInvocation(TI {
                    id: "call_2".into(),
                    name: "second".into(),
                    input: serde_json::json!({"x": 1}),
                }),
            ]))
            .message(Message::Tool {
                invocation_id: "call_1".into(),
                content: vec![RC::Text("first result".into())],
                is_error: false,
            })
            .message(Message::Tool {
                invocation_id: "call_2".into(),
                content: vec![RC::Text("second result".into())],
                is_error: false,
            })
            .build();

        let oai = to_oai(
            &req,
            None,
            super::super::ThinkingScheme::None,
            super::super::MaxTokensScheme::MaxTokens,
            false,
            true,
        )
        .unwrap();

        assert_eq!(oai.messages.len(), 4);
        assert_eq!(oai.messages[2].role, "tool");
        assert_eq!(oai.messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(oai.messages[3].role, "tool");
        assert_eq!(oai.messages[3].tool_call_id.as_deref(), Some("call_2"));
        assert!(matches!(
            &oai.messages[2].content,
            Some(OaiContent::Text(text)) if text == "first result"
        ));
        assert!(matches!(
            &oai.messages[3].content,
            Some(OaiContent::Text(text)) if text == "second result"
        ));
    }

    #[test]
    fn oai_effort_mapping() {
        use crate::types::ThinkingLevel;
        assert_eq!(thinking_to_oai_effort(ThinkingLevel::Off), None);
        assert_eq!(
            thinking_to_oai_effort(ThinkingLevel::Minimal).as_deref(),
            Some("minimal")
        );
        assert_eq!(
            thinking_to_oai_effort(ThinkingLevel::Low).as_deref(),
            Some("low")
        );
        assert_eq!(
            thinking_to_oai_effort(ThinkingLevel::Medium).as_deref(),
            Some("medium")
        );
        assert_eq!(
            thinking_to_oai_effort(ThinkingLevel::High).as_deref(),
            Some("high")
        );
        assert_eq!(
            thinking_to_oai_effort(ThinkingLevel::XHigh).as_deref(),
            Some("xhigh")
        );
    }

    #[test]
    fn oai_effort_budget_boundaries() {
        use crate::types::ThinkingLevel;
        assert_eq!(
            thinking_to_oai_effort(ThinkingLevel::Budget(2048)).as_deref(),
            Some("minimal")
        );
        assert_eq!(
            thinking_to_oai_effort(ThinkingLevel::Budget(2049)).as_deref(),
            Some("low")
        );
        assert_eq!(
            thinking_to_oai_effort(ThinkingLevel::Budget(4097)).as_deref(),
            Some("medium")
        );
        assert_eq!(
            thinking_to_oai_effort(ThinkingLevel::Budget(8193)).as_deref(),
            Some("high")
        );
        assert_eq!(
            thinking_to_oai_effort(ThinkingLevel::Budget(16385)).as_deref(),
            Some("xhigh")
        );
    }

    #[test]
    fn tool_choice_to_oai_all_variants() {
        assert_eq!(
            tool_choice_to_oai(&ToolChoice::Auto),
            serde_json::Value::String("auto".into())
        );
        assert_eq!(
            tool_choice_to_oai(&ToolChoice::None),
            serde_json::Value::String("none".into())
        );
        assert_eq!(
            tool_choice_to_oai(&ToolChoice::Required),
            serde_json::Value::String("required".into())
        );
        let specific = tool_choice_to_oai(&ToolChoice::Specific("my_tool".into()));
        assert_eq!(specific["type"], "function");
        assert_eq!(specific["function"]["name"], "my_tool");
    }

    #[test]
    fn req_content_to_part_image_url() {
        let part = req_content_to_part(
            &RC::Image(ImageSource::Url {
                url: "https://example.com/img.png".into(),
            }),
            true,
        )
        .unwrap();
        match part {
            OaiContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/img.png");
            }
            other => panic!("expected ImageUrl, got {other:?}"),
        }
    }

    #[test]
    fn req_content_to_part_image_base64() {
        let part = req_content_to_part(
            &RC::Image(ImageSource::Base64 {
                media_type: "image/png".into(),
                data: "iVBORw0KGgo=".into(),
            }),
            true,
        )
        .unwrap();
        match part {
            OaiContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "data:image/png;base64,iVBORw0KGgo=");
            }
            other => panic!("expected ImageUrl, got {other:?}"),
        }
    }

    #[test]
    fn req_content_to_part_document_url() {
        let part = req_content_to_part(
            &RC::Document(crate::types::DocumentSource::Url {
                url: "https://cdn.example.com/doc.pdf".into(),
            }),
            false,
        )
        .unwrap();
        match part {
            OaiContentPart::FileUrl { ref file_url } => {
                assert_eq!(file_url.url, "https://cdn.example.com/doc.pdf");
                // Wire format: {"type":"file_url","file_url":{"url":...}}
                let body = serde_json::to_value(&part).unwrap();
                assert_eq!(body["type"], "file_url");
                assert_eq!(body["file_url"]["url"], "https://cdn.example.com/doc.pdf");
            }
            other => panic!("expected FileUrl, got {other:?}"),
        }
    }

    #[test]
    fn req_content_to_part_document_base64_inline_when_capable() {
        // OpenAI-ecosystem standard: {"type":"file","file":{file_data: data-URL}}.
        // Used by OpenAI official, OpenRouter, DeepInfra... — gated by the
        // documents_inline capability flag threaded through the request.
        let part = req_content_to_part(
            &RC::Document(crate::types::DocumentSource::Base64 {
                media_type: "application/pdf".into(),
                data: "JVBERi0=".into(),
            }),
            true,
        )
        .unwrap();
        let body = serde_json::to_value(&part).unwrap();
        assert_eq!(body["type"], "file");
        assert_eq!(
            body["file"]["file_data"],
            "data:application/pdf;base64,JVBERi0="
        );
    }

    #[test]
    fn req_content_to_part_document_url_inline_when_capable() {
        // On inline-capable (OpenAI-ecosystem) endpoints, URL documents go out
        // as the standard `file` part with a plain URL in file_data — not the
        // Zhipu-style `file_url` part.
        let part = req_content_to_part(
            &RC::Document(crate::types::DocumentSource::Url {
                url: "https://cdn.example.com/doc.pdf".into(),
            }),
            true,
        )
        .unwrap();
        let body = serde_json::to_value(&part).unwrap();
        assert_eq!(body["type"], "file");
        assert_eq!(body["file"]["file_data"], "https://cdn.example.com/doc.pdf");
    }

    #[test]
    fn req_content_to_part_document_base64_rejected() {
        // Zhipu-style endpoints accept URL documents only; base64 documents
        // must be a clear InvalidRequest, not silent.
        let err = req_content_to_part(
            &RC::Document(crate::types::DocumentSource::Base64 {
                media_type: "application/pdf".into(),
                data: "JVBERi0=".into(),
            }),
            false,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("URL"),
            "error must mention URL-only limitation: {msg}"
        );
        assert!(
            msg.contains("document"),
            "error must mention document: {msg}"
        );
    }
}
