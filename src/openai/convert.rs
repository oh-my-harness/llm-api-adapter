use crate::types::{
    ChatRequest, ChatResponse, Message, RequestContent, ResponseContent, StopReason, ToolChoice,
    ToolInvocation, Usage,
};
use crate::LlmError;
use serde::Serialize;

use super::schema::{
    OaiContent, OaiContentPart, OaiImageUrl, OaiMessage, OaiRequest, OaiResponse,
    OaiTool, OaiToolCall, OaiToolCallFunction, OaiUsage, OaiFunction,
};

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
    /// If true, parse DeepSeek reasoning_content field from responses.
    pub parse_reasoning_content: bool,
    /// If true, tolerate keepalive messages in stream (DeepSeek compatibility).
    pub tolerant_keepalive: bool,
}

/// Builder for OpenAIExt with fluent API.
#[derive(Debug, Default)]
pub struct OpenAIExtBuilder {
    seed: Option<u64>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    user: Option<String>,
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

/// Convert a canonical ChatRequest to an OpenAI OaiRequest.
///
/// # Arguments
/// * `req` - The canonical chat request
/// * `ext` - Optional OpenAI-specific extensions
/// * `stream` - Whether to enable streaming
///
/// # Returns
/// An OaiRequest ready to send to the OpenAI API, or an LlmError if conversion fails.
pub fn to_oai(
    req: &ChatRequest,
    ext: Option<&OpenAIExt>,
    stream: bool,
) -> Result<OaiRequest, LlmError> {
    let default_ext = OpenAIExt::default();
    let ext = ext.unwrap_or(&default_ext);

    // Convert messages
    let messages = req
        .messages()
        .iter()
        .map(|m| message_to_oai(m))
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
        max_tokens: params.max_tokens(),
        top_p: params.top_p(),
        stop: params.stop_sequences().to_vec(),
        stream,
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
    })
}

/// Convert a canonical Message to an OpenAI OaiMessage.
fn message_to_oai(msg: &Message) -> Result<OaiMessage, LlmError> {
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
                            .map(|c| req_content_to_part(c))
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                }
            } else {
                OaiContent::Parts(
                    content
                        .iter()
                        .map(|c| req_content_to_part(c))
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
                    ResponseContent::Reasoning(r) => {
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
                            .map(|c| req_content_to_part(c))
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                }
            } else {
                OaiContent::Parts(
                    content
                        .iter()
                        .map(|c| req_content_to_part(c))
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
fn req_content_to_part(c: &RequestContent) -> Result<OaiContentPart, LlmError> {
    match c {
        RequestContent::Text(t) => Ok(OaiContentPart::Text { text: t.clone() }),
        RequestContent::Image(img) => {
            use crate::types::ImageSource;
            let url = match img {
                ImageSource::Url(u) => u.clone(),
                ImageSource::Base64 { media_type, data } => {
                    format!("data:{};base64,{}", media_type, data)
                }
            };
            Ok(OaiContentPart::ImageUrl {
                image_url: OaiImageUrl { url },
            })
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
pub fn from_oai(resp: OaiResponse, parse_reasoning: bool) -> Result<ChatResponse, LlmError> {
    if resp.choices.is_empty() {
        return Err(LlmError::InvalidRequest(
            "OaiResponse has no choices".into(),
        ));
    }

    // Use the first choice (OpenAI typically returns one)
    let choice = &resp.choices[0];
    let content = oai_message_to_content(choice.message.clone(), parse_reasoning)?;
    let stop_reason = parse_stop_reason(choice.finish_reason.as_deref());
    let usage = oai_usage_to_canonical(
        resp.usage.unwrap_or_default(),
        parse_reasoning,
    );

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
    if parse_reasoning {
        if let Some(reasoning) = msg.reasoning_content {
            content.push(ResponseContent::Reasoning(reasoning));
        }
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
fn oai_usage_to_canonical(u: OaiUsage, _parse_reasoning: bool) -> Usage {
    Usage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        cached_input_tokens: u.prompt_cache_hit_tokens,
        cache_creation_input_tokens: 0,
        reasoning_tokens: 0,
    }
}
