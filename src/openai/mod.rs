pub mod convert;
pub(crate) mod schema;
pub(crate) mod stream;

pub use convert::{OpenAIExt, OpenAIExtBuilder, OpenAIRequest};

use crate::{error::LlmError, provider::Provider, stream_handle::StreamHandle, types::*};
use async_trait::async_trait;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.openai.com";
const DEFAULT_CHAT_PATH: &str = "/v1/chat/completions";

/// OpenAI (and OpenAI-compatible) provider, constructed via `OpenAIProvider::builder`.
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    chat_path: String,
    parse_reasoning_content: bool,
}

/// Builder for OpenAIProvider with fluent configuration API.
pub struct OpenAIProviderBuilder {
    api_key: String,
    base_url: String,
    chat_path: String,
    parse_reasoning_content: bool,
    tolerant_keepalive: bool,
}

impl OpenAIProvider {
    /// Create a new builder for OpenAIProvider with the given API key.
    pub fn builder(api_key: impl Into<String>) -> OpenAIProviderBuilder {
        OpenAIProviderBuilder {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            chat_path: DEFAULT_CHAT_PATH.to_string(),
            parse_reasoning_content: false,
            tolerant_keepalive: false,
        }
    }

    /// Send an HTTP request to the OpenAI API.
    ///
    /// Handles error responses, including provider-specific errors,
    /// and maps them to canonical LlmError types.
    async fn send(
        &self,
        oai_req: &schema::OaiRequest,
        extra_body: Option<&serde_json::Value>,
        _stream: bool,
    ) -> Result<reqwest::Response, LlmError> {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), self.chat_path);
        let body = merge_extra_body(oai_req, extra_body)?;

        let req = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .body(body);

        let response = req.send().await?;
        let status = response.status();

        if !status.is_success() {
            // Try to parse structured error response
            let error_text = response.text().await?;

            // Attempt to parse as OaiErrorResponse for details
            let _error_code = serde_json::from_str::<schema::OaiErrorResponse>(&error_text)
                .ok()
                .and_then(|e| e.error.code);

            return match status.as_u16() {
                401 => Err(LlmError::Unauthorized),
                403 => Err(LlmError::Forbidden),
                402 => {
                    // DeepSeek payment_required
                    Err(LlmError::Provider {
                        code: "payment_required".into(),
                        message: error_text,
                    })
                }
                429 => {
                    let retry_after = self.extract_retry_after(&error_text);
                    Err(LlmError::RateLimit { retry_after })
                }
                500 | 502 | 503 => Err(LlmError::ServerError(error_text)),
                529 => {
                    let retry_after = self.extract_retry_after(&error_text);
                    Err(LlmError::Overloaded { retry_after })
                }
                _ => Err(LlmError::InvalidRequest(error_text)),
            };
        }

        Ok(response)
    }

    /// Extract retry-after delay from error text (if present).
    fn extract_retry_after(&self, _error_text: &str) -> Option<Duration> {
        // TODO: Parse retry-after header from actual response headers
        // For now, return None
        None
    }

    /// Chat completion using the native OpenAI path (non-streaming).
    ///
    /// Accepts `OpenAIRequest` (a `ChatRequest` paired with `OpenAIExt`) so
    /// callers can opt into provider-specific extensions — seed, penalties,
    /// `reasoning_effort`, `extra_body`, etc. — without sacrificing the
    /// type-safe canonical request shape.
    pub async fn chat_native(&self, req: &OpenAIRequest) -> Result<ChatResponse, LlmError> {
        let oai_req = convert::to_oai(&req.base, Some(&req.ext), false)?;
        let response = self
            .send(&oai_req, req.ext.extra_body.as_ref(), false)
            .await?;
        let oai_resp = response.json::<schema::OaiResponse>().await?;
        convert::from_oai(oai_resp, req.ext.parse_reasoning_content)
    }

    /// Chat completion using the native OpenAI path (streaming).
    pub async fn chat_stream_native(&self, req: &OpenAIRequest) -> Result<StreamHandle, LlmError> {
        let oai_req = convert::to_oai(&req.base, Some(&req.ext), true)?;
        let response = self
            .send(&oai_req, req.ext.extra_body.as_ref(), true)
            .await?;
        let model = oai_req.model.clone();
        let stream = stream::parse_stream(response, req.ext.parse_reasoning_content);
        Ok(StreamHandle::new(model, stream))
    }
}

/// Serialise an OpenAI request body, merging any caller-supplied `extra_body`
/// fields at the top level. Extra-body keys override canonical-conversion
/// keys, so callers can e.g. set `thinking: {type: disabled}` or any other
/// non-standard field accepted by an OpenAI-compatible endpoint.
fn merge_extra_body(
    oai_req: &schema::OaiRequest,
    extra_body: Option<&serde_json::Value>,
) -> Result<String, LlmError> {
    let mut body = serde_json::to_value(oai_req)?;
    if let Some(extra) = extra_body {
        let extra_obj = extra.as_object().ok_or_else(|| {
            LlmError::InvalidRequest("OpenAIExt.extra_body must be a JSON object".into())
        })?;
        let body_obj = body
            .as_object_mut()
            .expect("OaiRequest serializes to object");
        for (k, v) in extra_obj {
            body_obj.insert(k.clone(), v.clone());
        }
    }
    Ok(serde_json::to_string(&body)?)
}

impl OpenAIProviderBuilder {
    /// Set the base URL for the API (default: "https://api.openai.com").
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Override the chat completions path appended to `base_url`
    /// (default: "/v1/chat/completions"). Useful for OpenAI-compatible
    /// gateways with non-standard paths, e.g. GitHub Models uses
    /// "/chat/completions" (no `/v1`).
    pub fn chat_path(mut self, path: impl Into<String>) -> Self {
        self.chat_path = path.into();
        self
    }

    /// Enable parsing of DeepSeek reasoning_content from responses.
    pub fn parse_reasoning_content(mut self, v: bool) -> Self {
        self.parse_reasoning_content = v;
        self
    }

    /// Enable tolerant keepalive messages in streams (DeepSeek compatibility).
    pub fn tolerant_keepalive(mut self, v: bool) -> Self {
        self.tolerant_keepalive = v;
        self
    }

    /// Build the OpenAIProvider instance.
    pub fn build(self) -> OpenAIProvider {
        OpenAIProvider {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build HTTP client"),
            api_key: self.api_key,
            base_url: self.base_url,
            chat_path: self.chat_path,
            parse_reasoning_content: self.parse_reasoning_content,
        }
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let oai_req = OpenAIRequest {
            base: req.clone(),
            ext: OpenAIExt {
                parse_reasoning_content: self.parse_reasoning_content,
                ..Default::default()
            },
        };
        self.chat_native(&oai_req).await
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<StreamHandle, LlmError> {
        let oai_req = OpenAIRequest {
            base: req.clone(),
            ext: OpenAIExt {
                parse_reasoning_content: self.parse_reasoning_content,
                ..Default::default()
            },
        };
        self.chat_stream_native(&oai_req).await
    }
}
