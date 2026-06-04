use super::message::Message;

/// Definition of a callable tool.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    /// Unique tool name.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema object describing accepted parameters.
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

/// How the model should decide whether to call tools.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ToolChoice {
    /// Model decides (default when tools are provided).
    #[default]
    Auto,
    /// Model must not call any tool.
    None,
    /// Model must call at least one tool.
    Required,
    /// Model must call exactly the named tool.
    Specific(String),
}

/// Sampling hyper-parameters shared across all providers.
///
/// Construct via `SamplingParams::builder(max_tokens)`.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: u32,
    pub(crate) top_p: Option<f32>,
    pub(crate) stop_sequences: Vec<String>,
}

impl SamplingParams {
    pub fn builder(max_tokens: u32) -> SamplingParamsBuilder {
        SamplingParamsBuilder {
            temperature: None,
            max_tokens,
            top_p: None,
            stop_sequences: Vec::new(),
        }
    }

    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    pub fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    pub fn stop_sequences(&self) -> &[String] {
        &self.stop_sequences
    }
}

pub struct SamplingParamsBuilder {
    temperature: Option<f32>,
    max_tokens: u32,
    top_p: Option<f32>,
    stop_sequences: Vec<String>,
}

impl SamplingParamsBuilder {
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn stop_sequence(mut self, s: impl Into<String>) -> Self {
        self.stop_sequences.push(s.into());
        self
    }

    pub fn build(self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences,
        }
    }
}

/// A fully-specified chat request, constructed via `ChatRequest::builder`.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Vec<ToolDef>,
    pub(crate) tool_choice: ToolChoice,
    pub(crate) params: SamplingParams,
}

impl ChatRequest {
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
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn tools(&self) -> &[ToolDef] {
        &self.tools
    }

    pub fn tool_choice(&self) -> &ToolChoice {
        &self.tool_choice
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }
}

pub struct ChatRequestBuilder {
    model: String,
    messages: Vec<Message>,
    tools: Vec<ToolDef>,
    tool_choice: ToolChoice,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: u32,
    stop_sequences: Vec<String>,
}

impl ChatRequestBuilder {
    pub fn message(mut self, m: Message) -> Self {
        self.messages.push(m);
        self
    }

    pub fn messages(mut self, ms: Vec<Message>) -> Self {
        self.messages = ms;
        self
    }

    pub fn tool(mut self, t: ToolDef) -> Self {
        self.tools.push(t);
        self
    }

    pub fn tools(mut self, ts: Vec<ToolDef>) -> Self {
        self.tools = ts;
        self
    }

    pub fn tool_choice(mut self, c: ToolChoice) -> Self {
        self.tool_choice = c;
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn stop_sequence(mut self, s: impl Into<String>) -> Self {
        self.stop_sequences.push(s.into());
        self
    }

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
        }
    }
}
