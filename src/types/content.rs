/// Image data source for vision-capable requests.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageSource {
    /// Publicly accessible URL.
    Url(String),
    /// Inline base64-encoded image.
    Base64 { media_type: String, data: String },
}

/// Content that a user or tool produces as input to a request.
#[derive(Debug, Clone, PartialEq)]
pub enum RequestContent {
    /// Plain text.
    Text(String),
    /// Image (URL or base64).
    Image(ImageSource),
}

/// A tool invocation emitted by the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInvocation {
    /// Provider-assigned call ID, required for tool result routing.
    pub id: String,
    /// Name of the tool being called.
    pub name: String,
    /// Arguments the model produced; callers must validate this JSON.
    pub input: serde_json::Value,
}

/// Content that the model produces in a response.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseContent {
    /// Plain text output.
    Text(String),
    /// Tool the model wants to invoke.
    ToolInvocation(ToolInvocation),
    /// Chain-of-thought reasoning text.
    /// `signature` is Anthropic-only; `None` for all other providers.
    /// Pass responses through `AnthropicNativeResponse::into_message()` to enable multi-turn thinking.
    Reasoning {
        text: String,
        signature: Option<String>,
    },
}
