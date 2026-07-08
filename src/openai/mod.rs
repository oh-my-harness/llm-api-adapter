pub mod convert;
pub(crate) mod schema;
pub(crate) mod stream;

pub use convert::{OpenAIExt, OpenAIExtBuilder, OpenAIRequest};

use crate::{
    error::LlmError,
    provider::{EmbeddingProvider, Provider},
    stream_handle::StreamHandle,
    types::*,
};
use async_trait::async_trait;
use std::borrow::Cow;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.openai.com";
const DEFAULT_CHAT_PATH: &str = "/v1/chat/completions";
const DEFAULT_EMBEDDINGS_PATH: &str = "/v1/embeddings";

/// How [`ChatRequest::thinking`](crate::types::ChatRequest::thinking) (a
/// [`ThinkingLevel`]) is translated to
/// OpenAI-compatible wire format on this provider.
///
/// This is the **provider-side translation strategy** — distinct from
/// `ThinkingLevel` (the canonical intensity level set on `ChatRequest`).
/// `ThinkingLevel` says *what* intensity you want; `ThinkingScheme` says
/// *how* the OpenAI-compatible provider translates it to wire format.
///
/// Default: `None` (no translation — `ThinkingLevel` is ignored, only
/// `OpenAIExt.reasoning_effort` / `extra_body` take effect). Use
/// `deepseek::client()` factory for automatic `ThinkingToggle` configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ThinkingScheme {
    /// Do not translate `thinking` (legacy). Only `OpenAIExt` fields take effect.
    #[default]
    None,
    /// Standard OpenAI: `thinking` → top-level `reasoning_effort` string.
    ReasoningEffort,
    /// DeepSeek-style: `thinking` → `extra_body.thinking` object + top-level `reasoning_effort`.
    ThinkingToggle,
}

/// OpenAI (and OpenAI-compatible) provider implementation.
///
/// Supports the OpenAI Chat Completions API and any compatible endpoint
/// (e.g. DeepSeek, GitHub Models, Azure OpenAI-compatible gateways).
/// Construct via [`OpenAIProvider::builder()`].
///
/// Use the [`Provider`] trait methods ([`chat()`](Provider::chat),
/// [`chat_stream()`](Provider::chat_stream)) for provider-agnostic usage,
/// or the native-path methods ([`chat_native()`](OpenAIProvider::chat_native),
/// [`chat_stream_native()`](OpenAIProvider::chat_stream_native)) when you
/// need provider-specific extensions like `seed`, `frequency_penalty`,
/// or `extra_body`.
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    chat_path: String,
    embeddings_path: String,
    parse_reasoning_content: bool,
    tolerant_keepalive: bool,
    thinking_scheme: ThinkingScheme,
    capabilities: crate::provider::ProviderCapabilities,
}

/// Builder for [`OpenAIProvider`] with fluent configuration API.
///
/// Created via [`OpenAIProvider::builder()`]. At minimum you must provide
/// the API key; all other settings have sensible defaults.
pub struct OpenAIProviderBuilder {
    api_key: String,
    base_url: String,
    chat_path: String,
    embeddings_path: String,
    parse_reasoning_content: bool,
    tolerant_keepalive: bool,
    thinking_scheme: ThinkingScheme,
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

impl OpenAIProvider {
    /// Create a new builder for OpenAIProvider with the given API key.
    pub fn builder(api_key: impl Into<String>) -> OpenAIProviderBuilder {
        OpenAIProviderBuilder {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            chat_path: DEFAULT_CHAT_PATH.to_string(),
            embeddings_path: DEFAULT_EMBEDDINGS_PATH.to_string(),
            parse_reasoning_content: false,
            tolerant_keepalive: false,
            thinking_scheme: ThinkingScheme::None,
            capabilities: crate::provider::ProviderCapabilities {
                images: true,
                reasoning: true,
                json_schema: true,
                embeddings: true,
            },
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
            let retry_after = extract_retry_after_header(response.headers());
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
                429 => Err(LlmError::RateLimit { retry_after }),
                500 | 502 | 503 => Err(LlmError::ServerError(error_text)),
                529 => Err(LlmError::Overloaded { retry_after }),
                _ => Err(LlmError::InvalidRequest(error_text)),
            };
        }

        Ok(response)
    }

    /// Send an embeddings request to the OpenAI-compatible embeddings endpoint.
    async fn send_embeddings(
        &self,
        req: &schema::OaiEmbeddingRequest,
    ) -> Result<reqwest::Response, LlmError> {
        let url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.embeddings_path
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(req)
            .send()
            .await?;
        let status = response.status();

        if !status.is_success() {
            let retry_after = extract_retry_after_header(response.headers());
            let error_text = response.text().await?;

            return match status.as_u16() {
                401 => Err(LlmError::Unauthorized),
                403 => Err(LlmError::Forbidden),
                402 => Err(LlmError::Provider {
                    code: "payment_required".into(),
                    message: error_text,
                }),
                429 => Err(LlmError::RateLimit { retry_after }),
                500 | 502 | 503 => Err(LlmError::ServerError(error_text)),
                529 => Err(LlmError::Overloaded { retry_after }),
                _ => Err(LlmError::InvalidRequest(error_text)),
            };
        }

        Ok(response)
    }

    /// Chat completion using the native OpenAI path (non-streaming).
    ///
    /// Accepts `OpenAIRequest` (a `ChatRequest` paired with `OpenAIExt`) so
    /// callers can opt into provider-specific extensions — seed, penalties,
    /// `reasoning_effort`, `extra_body`, etc. — without sacrificing the
    /// type-safe canonical request shape.
    pub async fn chat_native(&self, req: &OpenAIRequest) -> Result<ChatResponse, LlmError> {
        let oai_req = convert::to_oai(&req.base, Some(&req.ext), self.thinking_scheme, false)?;
        let extra_body = self.resolve_extra_body(&req.ext.extra_body, req.base.thinking())?;
        let response = self.send(&oai_req, extra_body.as_deref(), false).await?;
        let oai_resp = response.json::<schema::OaiResponse>().await?;
        convert::from_oai(oai_resp, req.ext.parse_reasoning_content)
    }

    /// Chat completion using the native OpenAI path (streaming).
    ///
    /// Accepts `OpenAIRequest` (a `ChatRequest` paired with `OpenAIExt`) so
    /// callers can use provider-specific streaming extensions.
    pub async fn chat_stream_native(&self, req: &OpenAIRequest) -> Result<StreamHandle, LlmError> {
        let oai_req = convert::to_oai(&req.base, Some(&req.ext), self.thinking_scheme, true)?;
        let extra_body = self.resolve_extra_body(&req.ext.extra_body, req.base.thinking())?;
        let response = self.send(&oai_req, extra_body.as_deref(), true).await?;
        let model = oai_req.model.clone();
        let tolerant = req.ext.tolerant_keepalive || self.tolerant_keepalive;
        let stream = stream::parse_stream(response, req.ext.parse_reasoning_content, tolerant);
        Ok(StreamHandle::new(model, stream))
    }
    /// Generate embeddings using the native OpenAI-compatible embeddings API.
    pub async fn embed_native(
        &self,
        req: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        let oai_req = schema::OaiEmbeddingRequest::from(req);
        let response = self.send_embeddings(&oai_req).await?;
        let oai_resp = response.json::<schema::OaiEmbeddingResponse>().await?;
        Ok(oai_resp.into())
    }

    /// Build a DeepSeek-style `thinking` extra_body from `ChatRequest.thinking`.
    ///
    /// Returns `None` when scheme is not `ThinkingToggle` or `thinking` is
    /// `None`. Returns `Some({"thinking":{"type":"disabled"}})` for
    /// `ThinkingLevel::Off`.
    ///
    /// Only the `thinking` toggle object is produced here; the `reasoning_effort`
    /// string is emitted as a top-level field by `to_oai` (via the
    /// `OaiRequest.reasoning_effort` field), matching DeepSeek's wire format
    /// where `reasoning_effort` is top-level and `thinking` only carries `type`.
    fn build_thinking_body(
        &self,
        thinking: Option<crate::types::ThinkingLevel>,
    ) -> Option<serde_json::Value> {
        if self.thinking_scheme != ThinkingScheme::ThinkingToggle {
            return None;
        }
        let spec = thinking?;
        Some(match spec {
            crate::types::ThinkingLevel::Off => serde_json::json!({
                "thinking": {"type": "disabled"}
            }),
            _ => serde_json::json!({
                "thinking": {"type": "enabled"}
            }),
        })
    }

    /// Resolve the final `extra_body` to send, merging scheme-derived
    /// thinking body when present. Returns a borrow when no thinking body
    /// is produced (the common default-scheme path), avoiding a clone of
    /// the caller's `extra_body`. Returns `None` when there is nothing to
    /// send (no caller extra_body and no scheme-derived thinking body).
    fn resolve_extra_body<'a>(
        &self,
        caller_extra: &'a Option<serde_json::Value>,
        thinking: Option<crate::types::ThinkingLevel>,
    ) -> Result<Option<Cow<'a, serde_json::Value>>, LlmError> {
        let thinking_body = self.build_thinking_body(thinking);
        match thinking_body {
            None => Ok(caller_extra.as_ref().map(Cow::Borrowed)),
            Some(t) => merge_thinking_into_extra(caller_extra, t).map(|v| v.map(Cow::Owned)),
        }
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

/// Merge scheme-derived thinking body into caller's extra_body.
///
/// Caller's extra_body takes precedence (its `thinking` key wins).
/// Returns `Err(InvalidRequest)` if `caller_extra` is present but not a JSON
/// object (consistent with [`merge_extra_body`]).
fn merge_thinking_into_extra(
    caller_extra: &Option<serde_json::Value>,
    thinking_body: serde_json::Value,
) -> Result<Option<serde_json::Value>, LlmError> {
    match caller_extra {
        None => Ok(Some(thinking_body)),
        Some(caller) => {
            // If caller already has a `thinking` key, caller wins — return caller as-is.
            if caller.get("thinking").is_some() {
                return Ok(Some(caller.clone()));
            }
            // Otherwise merge thinking keys into caller. Caller must be an object.
            caller.as_object().ok_or_else(|| {
                LlmError::InvalidRequest("OpenAIExt.extra_body must be a JSON object".into())
            })?;
            let mut merged = caller.clone();
            if let Some(merged_obj) = merged.as_object_mut()
                && let Some(think_obj) = thinking_body.as_object()
            {
                for (k, v) in think_obj {
                    merged_obj.insert(k.clone(), v.clone());
                }
            }
            Ok(Some(merged))
        }
    }
}

impl OpenAIProviderBuilder {
    /// Set the base URL for the API (default: `"https://api.openai.com"`).
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

    /// Override the embeddings path appended to `base_url`
    /// (default: "/v1/embeddings").
    pub fn embeddings_path(mut self, path: impl Into<String>) -> Self {
        self.embeddings_path = path.into();
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

    /// Override provider capabilities (used by factory functions such as `deepseek::client`).
    pub(crate) fn capabilities(mut self, caps: crate::provider::ProviderCapabilities) -> Self {
        self.capabilities = caps;
        self
    }

    /// Configure how `ChatRequest.thinking` is translated to wire format.
    /// Default: `ThinkingScheme::None`.
    pub fn thinking_scheme(mut self, scheme: ThinkingScheme) -> Self {
        self.thinking_scheme = scheme;
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
            embeddings_path: self.embeddings_path,
            thinking_scheme: self.thinking_scheme,
            parse_reasoning_content: self.parse_reasoning_content,
            tolerant_keepalive: self.tolerant_keepalive,
            capabilities: self.capabilities,
        }
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn capabilities(&self) -> crate::provider::ProviderCapabilities {
        self.capabilities.clone()
    }

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

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    async fn embed(&self, req: &EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        self.embed_native(req).await
    }
}
