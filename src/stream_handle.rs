use crate::{error::LlmError, types::*};
use futures::{Stream, StreamExt};
use std::collections::HashMap;

type BoxStream = futures::stream::BoxStream<'static, Result<StreamEvent, LlmError>>;

/// Streaming response handle returned by [`crate::provider::Provider::chat_stream`].
///
/// Provides two usage modes:
///
/// - **Incremental**: call [`events()`](StreamHandle::events) to process
///   each [`StreamEvent`] as it arrives (e.g. for a per-token UX).
/// - **Collect**: call [`collect()`](StreamHandle::collect) to fold all
///   events into a [`ChatResponse`] (note: `id` and `model` fields may be
///   empty when collected from streaming — use
///   [`crate::provider::Provider::chat`] if you need them).
pub struct StreamHandle {
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) inner: BoxStream,
}

impl StreamHandle {
    pub(crate) fn new(model: impl Into<String>, inner: BoxStream) -> Self {
        Self {
            id: String::new(),
            model: model.into(),
            inner,
        }
    }

    /// Create a `StreamHandle` from a raw stream — for testing and adapters only.
    pub fn from_raw_stream(model: impl Into<String>, inner: BoxStream) -> Self {
        Self::new(model, inner)
    }

    /// Returns the model name reported by the provider for this stream.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Raw event stream for incremental processing.
    ///
    /// Yields [`StreamEvent`] values as the provider sends them. Use this
    /// for real-time per-token UX.
    ///
    /// # Errors
    ///
    /// Each item is `Result<StreamEvent, LlmError>` — an `Err` terminates
    /// the stream.
    pub fn events(&mut self) -> &mut impl Stream<Item = Result<StreamEvent, LlmError>> {
        &mut self.inner
    }

    /// Read the entire stream and fold all events into a [`ChatResponse`].
    ///
    /// Convenience method that handles aggregation of deltas, assembles
    /// content blocks in index order, and preserves the final stop reason
    /// and usage.
    ///
    /// # Caveats
    ///
    /// - The `id` field on the resulting `ChatResponse` will be empty
    ///   (streaming responses from most providers do not include an ID).
    /// - The `model` field is populated from the initial request, not
    ///   from the provider's stream response.
    /// - `Reasoning.signature` is populated from `ContentStop.signature` events;
    ///   only Anthropic streaming responses provide this field.
    pub async fn collect(mut self) -> Result<ChatResponse, LlmError> {
        let mut text_blocks: HashMap<usize, String> = HashMap::new();
        let mut reasoning_blocks: HashMap<usize, String> = HashMap::new();
        // Anthropic reasoning block signatures, keyed by block index.
        let mut reasoning_signatures: HashMap<usize, String> = HashMap::new();
        // index → (id, name)
        let mut tool_starts: HashMap<usize, (String, String)> = HashMap::new();
        let mut tool_args: HashMap<usize, String> = HashMap::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut usage = Usage::default();

        while let Some(event) = self.inner.next().await {
            match event? {
                StreamEvent::ContentStart { index, kind } => match kind {
                    ContentKind::Text => {
                        text_blocks.insert(index, String::new());
                    }
                    ContentKind::Reasoning => {
                        reasoning_blocks.insert(index, String::new());
                    }
                    ContentKind::ToolInvocation { id, name } => {
                        tool_starts.insert(index, (id, name));
                        tool_args.insert(index, String::new());
                    }
                },
                StreamEvent::TextDelta { index, text } => {
                    text_blocks.entry(index).or_default().push_str(&text);
                }
                StreamEvent::ReasoningDelta { index, text } => {
                    reasoning_blocks.entry(index).or_default().push_str(&text);
                }
                StreamEvent::ToolDelta { index, arguments } => {
                    tool_args.entry(index).or_default().push_str(&arguments);
                }
                StreamEvent::ContentStop { index, signature } => {
                    if let Some(sig) = signature {
                        reasoning_signatures.insert(index, sig);
                    }
                }
                StreamEvent::MessageStop {
                    stop_reason: sr,
                    usage: u,
                } => {
                    stop_reason = sr;
                    usage = u;
                }
            }
        }

        // Assemble content blocks in index order
        let mut all: Vec<(usize, ResponseContent)> = Vec::new();
        for (idx, text) in text_blocks {
            all.push((idx, ResponseContent::Text(text)));
        }
        for (idx, text) in reasoning_blocks {
            let sig = reasoning_signatures.remove(&idx);
            all.push((
                idx,
                ResponseContent::Reasoning {
                    text,
                    signature: sig,
                },
            ));
        }
        for (idx, (tool_id, name)) in tool_starts {
            let args = tool_args.remove(&idx).unwrap_or_default();
            let input = serde_json::from_str(&args).unwrap_or_else(|_| {
                tracing::warn!(args = %args, "tool delta args not valid JSON; falling back to string value");
                serde_json::Value::String(args)
            });
            all.push((
                idx,
                ResponseContent::ToolInvocation(ToolInvocation {
                    id: tool_id,
                    name,
                    input,
                }),
            ));
        }
        all.sort_by_key(|(idx, _)| *idx);
        let content = all.into_iter().map(|(_, c)| c).collect();

        Ok(ChatResponse {
            id: self.id,
            model: self.model,
            content,
            stop_reason,
            usage,
        })
    }
}
