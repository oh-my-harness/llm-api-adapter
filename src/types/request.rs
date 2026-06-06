use super::message::Message;

/// Controls the format of the model's response.
///
/// Pass to [`ChatRequestBuilder::response_format()`] to request structured output.
/// Currently only forwarded by `OpenAIProvider`; `AnthropicProvider` ignores it
/// (use the native path with tool-forcing for Anthropic structured output).
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    /// Force the model to emit valid JSON (no specific schema).
    JsonObject,
    /// Force the model to emit JSON conforming to the supplied JSON Schema.
    JsonSchema {
        /// Logical name for the schema (required by OpenAI wire format).
        name: String,
        /// JSON Schema object describing the expected output.
        schema: serde_json::Value,
        /// When `Some(true)`, the model is required to strictly follow the schema.
        strict: Option<bool>,
    },
}

/// Definition of a callable tool that the model can invoke.
///
/// `input_schema` is a JSON Schema object — use `serde_json::json!` to construct it:
///
/// ```ignore
/// ToolDef::new(
///     "get_weather",
///     "Get the current weather for a city",
///     serde_json::json!({
///         "type": "object",
///         "properties": {
///             "city": { "type": "string", "description": "City name" }
///         },
///         "required": ["city"]
///     }),
/// )
/// ```
///
/// Pass `ToolDef` values to [`ChatRequestBuilder::tool()`] or
/// [`ChatRequestBuilder::tools()`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    /// Unique tool name. Must match the identifier used in `input_schema`
    /// and should be short (snake_case is conventional).
    pub name: String,
    /// Human-readable description shown to the model. A good description
    /// helps the model decide when to call this tool.
    pub description: String,
    /// JSON Schema object describing accepted parameters.
    ///
    /// Should be an object with `"type": "object"`, `"properties"`, and
    /// optional `"required"` array. For tools with no parameters, use
    /// `serde_json::json!({"type": "object", "properties": {}})`.
    pub input_schema: serde_json::Value,
}

impl ToolDef {
    /// Create a new tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Controls how the model decides whether to call tools.
///
/// Use this to constrain the model's behaviour when tools are available.
/// The default is [`Auto`](ToolChoice::Auto), which lets the model decide
/// whether to call a tool or respond with text.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool (default).
    #[default]
    Auto,
    /// The model must **not** call any tool — respond with text only.
    None,
    /// The model **must** call at least one tool.
    Required,
    /// The model **must** call exactly the named tool.
    /// Only supported by some providers (OpenAI, Anthropic).
    Specific(String),
}

/// Sampling hyper-parameters shared by all providers.
///
/// Only includes parameters common to OpenAI, Anthropic, and DeepSeek:
/// `temperature`, `max_tokens`, `top_p`, and `stop_sequences`.
///
/// Provider-specific parameters (e.g. `top_k`, `seed`, `frequency_penalty`,
/// `presence_penalty`) belong in the provider's `*Ext` extension type
/// (e.g. [`OpenAIExt`](crate::openai::OpenAIExt)).
///
/// Construct via [`SamplingParams::builder()`].
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: u32,
    pub(crate) top_p: Option<f32>,
    pub(crate) stop_sequences: Vec<String>,
}

impl SamplingParams {
    /// Create a builder that sets `max_tokens` and default values for all other fields.
    pub fn builder(max_tokens: u32) -> SamplingParamsBuilder {
        SamplingParamsBuilder {
            temperature: None,
            max_tokens,
            top_p: None,
            stop_sequences: Vec::new(),
        }
    }

    /// Sampling temperature (0.0–2.0). Lower values make output more deterministic.
    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Maximum number of tokens to generate.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Nucleus sampling threshold (0.0–1.0). Only tokens with cumulative probability
    /// up to `top_p` are considered.
    pub fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    /// Sequences that cause the model to stop generating.
    pub fn stop_sequences(&self) -> &[String] {
        &self.stop_sequences
    }
}

/// Builder for [`SamplingParams`]. Created via [`SamplingParams::builder()`].
pub struct SamplingParamsBuilder {
    temperature: Option<f32>,
    max_tokens: u32,
    top_p: Option<f32>,
    stop_sequences: Vec<String>,
}

impl SamplingParamsBuilder {
    /// Set the sampling temperature (0.0–2.0). Default: `None` (provider default).
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the nucleus sampling threshold (0.0–1.0). Default: `None` (provider default).
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Add a stop sequence. The model stops generating when any of these sequences
    /// is encountered. Default: empty (no stop sequences).
    pub fn stop_sequence(mut self, s: impl Into<String>) -> Self {
        self.stop_sequences.push(s.into());
        self
    }

    /// Consume the builder and produce a [`SamplingParams`].
    pub fn build(self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences,
        }
    }
}

/// A fully-specified chat request, constructed via [`ChatRequest::builder()`].
///
/// This is the universal request type used by the [`Provider`] trait. All
/// provider-specific extensions are carried separately via a paired `*Ext`
/// type (e.g. [`OpenAIRequest`], [`AnthropicRequest`]).
///
/// [`Provider`]: crate::provider::Provider
/// [`OpenAIRequest`]: crate::openai::OpenAIRequest
/// [`AnthropicRequest`]: crate::anthropic::AnthropicRequest
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Vec<ToolDef>,
    pub(crate) tool_choice: ToolChoice,
    pub(crate) params: SamplingParams,
    pub(crate) response_format: Option<ResponseFormat>,
}

impl ChatRequest {
    /// Create a builder with the given model name and `max_tokens`.
    ///
    /// All other fields are optional and default to reasonable values.
    pub fn builder(model: impl Into<String>, max_tokens: u32) -> ChatRequestBuilder {
        ChatRequestBuilder {
            model: model.into(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            temperature: None,
            top_p: None,
            max_tokens,
            stop_sequences: Vec::new(),
            response_format: None,
        }
    }

    /// The model identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`).
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The conversation history in chronological order.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Tool definitions available to the model.
    pub fn tools(&self) -> &[ToolDef] {
        &self.tools
    }

    /// How the model should decide whether to call tools.
    pub fn tool_choice(&self) -> &ToolChoice {
        &self.tool_choice
    }

    /// Sampling parameters (temperature, max_tokens, top_p, stop_sequences).
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// The requested response format, if any.
    pub fn response_format(&self) -> Option<&ResponseFormat> {
        self.response_format.as_ref()
    }
}

/// Builder for [`ChatRequest`]. Created via [`ChatRequest::builder()`].
///
/// # Example
///
/// ```no_run
/// # use llm_adapter::types::*;
/// let request = ChatRequest::builder("gpt-4o", 1024)
///     .message(Message::User(vec![RequestContent::Text("Hello".into())]))
///     .temperature(0.7)
///     .build();
/// ```
pub struct ChatRequestBuilder {
    model: String,
    messages: Vec<Message>,
    tools: Vec<ToolDef>,
    tool_choice: ToolChoice,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: u32,
    stop_sequences: Vec<String>,
    response_format: Option<ResponseFormat>,
}

impl ChatRequestBuilder {
    /// Append a single message to the conversation.
    pub fn message(mut self, m: Message) -> Self {
        self.messages.push(m);
        self
    }

    /// Replace the entire conversation history.
    pub fn messages(mut self, ms: Vec<Message>) -> Self {
        self.messages = ms;
        self
    }

    /// Append a single tool definition.
    pub fn tool(mut self, t: ToolDef) -> Self {
        self.tools.push(t);
        self
    }

    /// Replace all tool definitions.
    pub fn tools(mut self, ts: Vec<ToolDef>) -> Self {
        self.tools = ts;
        self
    }

    /// Set how the model should decide whether to call tools.
    /// Default: [`ToolChoice::Auto`].
    pub fn tool_choice(mut self, c: ToolChoice) -> Self {
        self.tool_choice = c;
        self
    }

    /// Set the sampling temperature (0.0–2.0). Default: `None` (provider default).
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the nucleus sampling threshold (0.0–1.0). Default: `None` (provider default).
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Override `max_tokens` from the initial builder call (optional).
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    /// Add a stop sequence. Default: empty (no stop sequences).
    pub fn stop_sequence(mut self, s: impl Into<String>) -> Self {
        self.stop_sequences.push(s.into());
        self
    }

    /// Set the response format. See [`ResponseFormat`] for options.
    pub fn response_format(mut self, f: ResponseFormat) -> Self {
        self.response_format = Some(f);
        self
    }

    /// Consume the builder and produce a [`ChatRequest`].
    pub fn build(self) -> ChatRequest {
        ChatRequest {
            model: self.model,
            messages: self.messages,
            tools: self.tools,
            tool_choice: self.tool_choice,
            params: SamplingParams {
                temperature: self.temperature,
                max_tokens: self.max_tokens,
                top_p: self.top_p,
                stop_sequences: self.stop_sequences,
            },
            response_format: self.response_format,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_format_json_object_builder() {
        let req = ChatRequest::builder("gpt-4o", 512)
            .response_format(ResponseFormat::JsonObject)
            .build();
        assert_eq!(req.response_format(), Some(&ResponseFormat::JsonObject));
    }

    #[test]
    fn response_format_json_schema_builder() {
        let schema =
            serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let req = ChatRequest::builder("gpt-4o", 512)
            .response_format(ResponseFormat::JsonSchema {
                name: "person".into(),
                schema: schema.clone(),
                strict: Some(true),
            })
            .build();
        match req.response_format() {
            Some(ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            }) => {
                assert_eq!(name, "person");
                assert_eq!(
                    schema,
                    &serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}})
                );
                assert_eq!(*strict, Some(true));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn response_format_none_by_default() {
        let req = ChatRequest::builder("gpt-4o", 512).build();
        assert!(req.response_format().is_none());
    }
}
