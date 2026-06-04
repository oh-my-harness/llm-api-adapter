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
    pub async fn chat_native(&self, req: &AnthropicRequest) -> Result<ChatResponse, LlmError> {
        let wire_req = convert::to_anthropic(&req.base, Some(&req.ext), false)?;
        let response = self.send(&wire_req, false).await?;
        let wire_resp = response.json::<schema::AnthropicResponse>().await?;
        convert::from_anthropic(wire_resp)
    }

    /// Chat completion using the native Anthropic path (streaming).
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
    /// Set the base URL for the API (default: "https://api.anthropic.com").
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
