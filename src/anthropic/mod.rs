pub mod convert;
pub(crate) mod schema;
pub(crate) mod stream;

pub use convert::{
    AnthropicExt, AnthropicExtBuilder, AnthropicMetadata, AnthropicRequest, CacheHint,
    ThinkingConfig,
};

use crate::{error::LlmError, provider::Provider, stream_handle::StreamHandle, types::*};
use async_trait::async_trait;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_MESSAGES_PATH: &str = "/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// The native (non-streaming) Anthropic response, returned by [`AnthropicProvider::chat_native()`].
///
/// Use [`into_canonical()`](Self::into_canonical) to get a [`ChatResponse`] (thinking
/// signatures are stripped).  Use [`into_message()`](Self::into_message) to get a
/// [`Message::Assistant`] with signatures preserved, suitable for multi-turn extended
/// thinking conversations.
///
/// # Limitations
///
/// `redacted_thinking` blocks carry an encrypted `data` field on the wire. The non-streaming
/// path encodes it as `[redacted_thinking: {data}]` text inside a `Reasoning` block, but the
/// streaming path does not surface `data` at all — a streamed `redacted_thinking` arrives as
/// an empty `Reasoning { text: "", signature: None }` and cannot be round-tripped.
pub struct AnthropicNativeResponse {
    inner: ChatResponse,
}

impl AnthropicNativeResponse {
    pub(crate) fn from_wire(wire: schema::AnthropicResponse) -> Result<Self, LlmError> {
        Ok(Self {
            inner: convert::from_anthropic(wire)?,
        })
    }

    /// Convert to a `ChatResponse`, dropping thinking signatures.
    pub fn into_canonical(self) -> ChatResponse {
        let content = self
            .inner
            .content
            .into_iter()
            .map(|c| match c {
                ResponseContent::Reasoning { text, .. } => ResponseContent::Reasoning {
                    text,
                    signature: None,
                },
                other => other,
            })
            .collect();
        ChatResponse {
            content,
            ..self.inner
        }
    }

    /// Convert to `Message::Assistant`, preserving thinking signatures for multi-turn.
    ///
    /// # Preconditions
    ///
    /// The subsequent request must have `thinking` enabled via
    /// [`AnthropicExtBuilder::thinking`] (crate::anthropic::AnthropicExtBuilder::thinking).
    /// Sending a `Thinking` block to the Anthropic API when thinking is not enabled will
    /// result in a `400` error.
    pub fn into_message(self) -> Message {
        Message::Assistant(self.inner.content)
    }

    /// Access the underlying `ChatResponse` (signatures preserved as-is).
    pub fn as_response(&self) -> &ChatResponse {
        &self.inner
    }
}

/// Anthropic Messages API provider, constructed via [`AnthropicProvider::builder()`].
///
/// Supports the Anthropic Messages API endpoint. Use the [`Provider`] trait
/// methods for provider-agnostic usage, or the native-path methods
/// ([`chat_native()`](AnthropicProvider::chat_native),
/// [`chat_stream_native()`](AnthropicProvider::chat_stream_native)) when you
/// need Anthropic-specific extensions like `thinking`, `metadata`, `top_k`,
/// or `cache_hints`.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    messages_path: String,
    capabilities: crate::provider::ProviderCapabilities,
}

/// Builder for [`AnthropicProvider`] with fluent configuration API.
///
/// Created via [`AnthropicProvider::builder()`]. At minimum you must provide
/// the API key; all other settings have sensible defaults.
pub struct AnthropicProviderBuilder {
    api_key: String,
    base_url: String,
    messages_path: String,
    capabilities: crate::provider::ProviderCapabilities,
}

fn extract_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers.get(reqwest::header::RETRY_AFTER)?;
    let s = value.to_str().ok()?;
    // RFC 7231: seconds format (e.g. "30")
    if let Ok(secs) = s.trim().parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    // HTTP-date format: skip (would require extra dep; None is safe fallback)
    None
}

impl AnthropicProvider {
    /// Create a new builder for AnthropicProvider with the given API key.
    pub fn builder(api_key: impl Into<String>) -> AnthropicProviderBuilder {
        AnthropicProviderBuilder {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            messages_path: DEFAULT_MESSAGES_PATH.to_string(),
            capabilities: crate::provider::ProviderCapabilities {
                images: true,
                reasoning: true,
                json_schema: false,
                embeddings: false,
            },
        }
    }

    /// Send an HTTP request to the Anthropic Messages API.
    ///
    /// Handles error responses and maps them to canonical LlmError types.
    async fn send(
        &self,
        anthropic_req: &schema::AnthropicRequest,
        _stream: bool,
    ) -> Result<reqwest::Response, LlmError> {
        let url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.messages_path
        );
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
            let retry_after = extract_retry_after_header(response.headers());
            let error_text = response.text().await?;

            // Attempt to parse structured error response
            let _error_type = serde_json::from_str::<schema::AnthropicErrorResponse>(&error_text)
                .ok()
                .map(|e| e.error.r#type);

            return match status.as_u16() {
                401 => Err(LlmError::Unauthorized),
                403 => Err(LlmError::Forbidden),
                404 => Err(LlmError::InvalidRequest(format!("not_found: {error_text}"))),
                413 => Err(LlmError::InvalidRequest(format!(
                    "request_too_large: {error_text}"
                ))),
                429 => Err(LlmError::RateLimit { retry_after }),
                500 | 502 | 503 | 504 => Err(LlmError::ServerError(error_text)),
                529 => Err(LlmError::Overloaded { retry_after }),
                _ => Err(LlmError::InvalidRequest(error_text)),
            };
        }

        Ok(response)
    }

    /// Chat completion using the native Anthropic path (non-streaming).
    ///
    /// Accepts `AnthropicRequest` (a `ChatRequest` paired with `AnthropicExt`) so
    /// callers can opt into provider-specific extensions — top_k, thinking,
    /// metadata, cache_hints, etc.
    pub async fn chat_native(
        &self,
        req: &AnthropicRequest,
    ) -> Result<AnthropicNativeResponse, LlmError> {
        let wire_req = convert::to_anthropic(&req.base, Some(&req.ext), false)?;
        let response = self.send(&wire_req, false).await?;
        let wire_resp = response.json::<schema::AnthropicResponse>().await?;
        AnthropicNativeResponse::from_wire(wire_resp)
    }

    /// Chat completion using the native Anthropic path (streaming).
    ///
    /// Thinking-block signatures are preserved through the streaming path:
    /// `signature_delta` events are accumulated and surfaced via
    /// [`StreamEvent::ContentStop`](crate::types::StreamEvent::ContentStop), then folded
    /// back into `Reasoning { signature }` by [`StreamHandle::collect()`]. `redacted_thinking`
    /// blocks remain a limitation — their encrypted `data` is not surfaced on this path, so
    /// they cannot be round-tripped.
    pub async fn chat_stream_native(
        &self,
        req: &AnthropicRequest,
    ) -> Result<StreamHandle, LlmError> {
        let wire_req = convert::to_anthropic(&req.base, Some(&req.ext), true)?;
        let response = self.send(&wire_req, true).await?;
        let model = wire_req.model.clone();
        let stream = stream::parse_stream(response);
        Ok(StreamHandle::new(model, stream))
    }
}

impl AnthropicProviderBuilder {
    /// Set the base URL for the API (default: `"https://api.anthropic.com"`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Override the messages path appended to `base_url`
    /// (default: "/v1/messages"). Useful for compatible gateways
    /// like DeepSeek's `/anthropic/v1/messages` endpoint.
    pub fn messages_path(mut self, path: impl Into<String>) -> Self {
        self.messages_path = path.into();
        self
    }

    /// Build the AnthropicProvider instance.
    pub fn build(self) -> AnthropicProvider {
        AnthropicProvider {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build HTTP client"),
            api_key: self.api_key,
            base_url: self.base_url,
            messages_path: self.messages_path,
            capabilities: self.capabilities,
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn capabilities(&self) -> crate::provider::ProviderCapabilities {
        self.capabilities.clone()
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let anthropic_req = AnthropicRequest {
            base: req.clone(),
            ext: AnthropicExt::default(),
        };
        self.chat_native(&anthropic_req)
            .await
            .map(AnthropicNativeResponse::into_canonical)
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<StreamHandle, LlmError> {
        let anthropic_req = AnthropicRequest {
            base: req.clone(),
            ext: AnthropicExt::default(),
        };
        self.chat_stream_native(&anthropic_req).await
    }
}
