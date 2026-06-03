use async_trait::async_trait;
use crate::{error::LlmError, stream_handle::StreamHandle, types::*};

/// Unified interface for any LLM provider.
///
/// Implementors should be cheaply cloneable (hold an `Arc` internally) so
/// callers can pass `Box<dyn Provider>` across threads.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Send a request and wait for the complete response.
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;

    /// Send a request and receive a streaming handle for incremental output.
    async fn chat_stream(&self, req: &ChatRequest) -> Result<StreamHandle, LlmError>;
}
