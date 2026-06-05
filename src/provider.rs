use crate::{error::LlmError, stream_handle::StreamHandle, types::*};
use async_trait::async_trait;

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
