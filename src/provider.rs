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
}

impl ProviderCapabilities {
    /// Create a new `ProviderCapabilities` instance.
    pub fn new(images: bool, reasoning: bool, json_schema: bool) -> Self {
        Self {
            images,
            reasoning,
            json_schema,
        }
    }

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
