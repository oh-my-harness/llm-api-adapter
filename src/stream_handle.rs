use crate::{error::LlmError, types::*};
use futures::{Stream, StreamExt, future::ready};
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
    pub(crate) status_code: Option<u16>,
    pub(crate) latency_ms: u64,
}

impl StreamHandle {
    pub(crate) fn new(model: impl Into<String>, inner: BoxStream) -> Self {
        Self::with_meta(model, inner, None, 0)
    }

    /// Create a `StreamHandle` with HTTP response metadata.
    pub(crate) fn with_meta(
        model: impl Into<String>,
        inner: BoxStream,
        status_code: Option<u16>,
        latency_ms: u64,
    ) -> Self {
        Self {
            id: String::new(),
            model: model.into(),
            inner: first_terminal_stream(inner),
            status_code,
            latency_ms,
        }
    }

    /// Create a `StreamHandle` from a raw stream — for testing and adapters only.
    pub fn from_raw_stream(model: impl Into<String>, inner: BoxStream) -> Self {
        Self::new(model, inner)
    }

    /// Returns the HTTP status code from the provider response, if available.
    pub fn status_code(&self) -> Option<u16> {
        self.status_code
    }

    /// Returns the request latency in milliseconds (time from send to first
    /// response byte).
    pub fn latency_ms(&self) -> u64 {
        self.latency_ms
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
        let mut received_chunks: u32 = 0;
        let mut message_stop_received = false;
        while let Some(event) = self.inner.next().await {
            received_chunks = received_chunks.saturating_add(1);
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
                    message_stop_received = true;
                    break;
                }
            }
        }
        if !message_stop_received {
            return Err(LlmError::StreamIncomplete {
                received_chunks,
                finish_reason: None,
            });
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

fn first_terminal_stream(inner: BoxStream) -> BoxStream {
    inner
        .scan(false, |terminated, event| {
            if *terminated {
                return ready(None);
            }
            if event.is_err() || matches!(&event, Ok(StreamEvent::MessageStop { .. })) {
                *terminated = true;
            }
            ready(Some(event))
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn terminal(usage: Usage) -> Result<StreamEvent, LlmError> {
        Ok(StreamEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            usage,
        })
    }

    #[tokio::test]
    async fn collect_keeps_the_first_terminal_usage() {
        let invalid = Usage::default().reported_invalid();
        let valid = Usage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        }
        .reported();
        let handle = StreamHandle::from_raw_stream(
            "model",
            Box::pin(stream::iter([terminal(invalid.clone()), terminal(valid)])),
        );

        let response = handle.collect().await.expect("first terminal completes");

        assert_eq!(response.usage, invalid);
    }

    #[tokio::test]
    async fn collect_never_polls_the_transport_after_message_stop() {
        let polls = Arc::new(AtomicUsize::new(0));
        let observed = polls.clone();
        let source = stream::poll_fn(move |_| match observed.fetch_add(1, Ordering::SeqCst) {
            0 => std::task::Poll::Ready(Some(terminal(Usage::default().reported()))),
            _ => panic!("transport was polled after MessageStop"),
        });
        let handle = StreamHandle::from_raw_stream("model", Box::pin(source));

        let response = handle
            .collect()
            .await
            .expect("terminal completes immediately");

        assert_eq!(response.usage.provenance, UsageProvenance::ReportedValid);
        assert_eq!(polls.load(Ordering::SeqCst), 1);
    }
}
