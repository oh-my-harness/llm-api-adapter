use crate::{error::LlmError, stream_handle::StreamHandle, types::*};
use async_trait::async_trait;

/// Runtime-queryable capability flags for a provider.
///
/// Use `provider.capabilities()` to discover what a provider supports before
/// constructing requests that depend on those features.
///
/// Marked `#[non_exhaustive]` so new capability flags can be added without
/// breaking existing code.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderCapabilities {
    /// Whether this provider accepts image content in user messages.
    pub images: bool,
    /// Whether this provider supports reasoning / extended thinking output.
    pub reasoning: bool,
    /// Whether this provider supports JSON schema structured output via the
    /// canonical `ResponseFormat::JsonSchema` field on `ChatRequest`.
    pub json_schema: bool,
    /// Whether this provider supports embedding requests.
    pub embeddings: bool,
    /// Whether this provider accepts document content (e.g. PDF) in messages.
    pub documents: bool,
    /// Whether this provider accepts **inline** base64 documents (the
    /// OpenAI-ecosystem `file` part with a data URL — OpenAI official,
    /// OpenRouter, DeepInfra…). When `false` (Zhipu-style), URL documents go
    /// out as the gateway's `file_url` part and inline documents are rejected
    /// at construction time.
    pub documents_inline: bool,
}

impl ProviderCapabilities {
    /// Create a new `ProviderCapabilities` instance.
    pub fn new(images: bool, reasoning: bool, json_schema: bool) -> Self {
        Self {
            images,
            reasoning,
            json_schema,
            embeddings: false,
            documents: false,
            documents_inline: false,
        }
    }

    /// Create a new `ProviderCapabilities` instance with embedding support.
    pub fn with_embeddings(
        images: bool,
        reasoning: bool,
        json_schema: bool,
        embeddings: bool,
    ) -> Self {
        Self {
            images,
            reasoning,
            json_schema,
            embeddings,
            documents: false,
            documents_inline: false,
        }
    }
}

impl ProviderCapabilities {
    /// Returns `true` if this provider accepts image content in user messages.
    pub fn supports_images(&self) -> bool {
        self.images
    }

    /// Returns `true` if this provider supports reasoning / extended thinking.
    pub fn supports_reasoning(&self) -> bool {
        self.reasoning
    }

    /// Returns `true` if this provider supports `ResponseFormat::JsonSchema`
    /// on the canonical `ChatRequest`.
    pub fn supports_json_schema(&self) -> bool {
        self.json_schema
    }

    /// Returns `true` if this provider supports embedding requests.
    pub fn supports_embeddings(&self) -> bool {
        self.embeddings
    }

    /// Returns `true` if this provider accepts document content in messages.
    pub fn supports_documents(&self) -> bool {
        self.documents
    }
}

/// Pre-dispatch guard: reject requests carrying document content when the
/// provider lacks the `documents` capability.
///
/// Called by each provider before wire conversion so the error surfaces at
/// request-construction time — no network request is issued. The message
/// names "documents capability disabled" so callers can distinguish this
/// provider-level restriction from source-level restrictions (e.g. an
/// OpenAI-compatible gateway that accepts URL documents only).
pub(crate) fn check_documents_capability(
    req: &ChatRequest,
    caps: &ProviderCapabilities,
) -> Result<(), LlmError> {
    if caps.supports_documents() {
        return Ok(());
    }
    let has_document = req.messages().iter().any(|m| match m {
        Message::User(content) | Message::Tool { content, .. } => content
            .iter()
            .any(|c| matches!(c, RequestContent::Document(_))),
        _ => false,
    });
    if has_document {
        return Err(LlmError::InvalidRequest(
            "this provider does not accept document content (documents capability disabled); \
             if this model actually supports documents, enable it on the builder with \
             .documents(true) (URL documents, Zhipu-style file_url) or .documents_inline(true) \
             (OpenAI-ecosystem file part, implies .documents(true))"
                .into(),
        ));
    }
    Ok(())
}

/// Unified interface for any LLM provider.
///
/// Implementors should be cheaply cloneable (hold an `Arc` internally) so
/// callers can pass `Box<dyn Provider>` across threads.
///
/// # Universal vs. native paths
///
/// - **Universal path** (this trait): accepts [`ChatRequest`] and returns
///   canonical [`ChatResponse`] / [`StreamHandle`]. Provider-specific
///   features are not directly accessible.
/// - **Native path**: the concrete provider type exposes additional methods
///   (e.g. `OpenAIProvider::chat_native()`) that accept a provider-specific
///   `*Request` type (e.g. [`OpenAIRequest`]) carrying extensions like
///   `seed`, `frequency_penalty`, `extra_body`, etc.
///
/// Use the universal path for provider-agnostic code and the native path
/// when you need provider-specific knobs.
///
/// [`OpenAIRequest`]: crate::openai::OpenAIRequest
#[async_trait]
pub trait Provider: Send + Sync {
    /// Return the static capability flags for this provider.
    ///
    /// Capabilities are determined at construction time (they depend on the
    /// provider endpoint, not on a specific model).  Use them to guard
    /// features before sending a request:
    ///
    /// ```ignore
    /// if provider.capabilities().supports_json_schema() {
    ///     req_builder.response_format(ResponseFormat::JsonSchema { ... });
    /// }
    /// ```
    fn capabilities(&self) -> ProviderCapabilities;

    /// Send a request and wait for the complete (non-streaming) response.
    ///
    /// Prefer this method when you need the full response at once, including
    /// the message `id` and `model` fields.
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;

    /// Send a request and receive a streaming handle for incremental output.
    ///
    /// Use [`StreamHandle::events()`] for per-token UX or
    /// [`StreamHandle::collect()`] to fold the stream into a [`ChatResponse`].
    /// Note that `id` and `model` may be empty when collected from streaming.
    async fn chat_stream(&self, req: &ChatRequest) -> Result<StreamHandle, LlmError>;
}

/// Unified interface for providers that can generate embeddings.
///
/// This trait is separate from [`Provider`] so existing chat-only providers and
/// trait objects remain focused on chat/streaming behavior.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embeddings for the request input batch.
    async fn embed(&self, req: &EmbeddingRequest) -> Result<EmbeddingResponse, LlmError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documents_capability_defaults_false_and_field_flips_it() {
        let mut caps = ProviderCapabilities::new(true, true, true);
        assert!(!caps.supports_documents());
        caps.documents = true;
        assert!(caps.supports_documents());
    }
}
