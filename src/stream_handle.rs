use crate::{error::LlmError, types::*};
use futures::{Stream, StreamExt};
use std::collections::HashMap;

type BoxStream = futures::stream::BoxStream<'static, Result<StreamEvent, LlmError>>;

/// Handle returned by `Provider::chat_stream`.
///
/// Use `events()` for incremental processing or `collect()` to fold the full
/// stream into a `ChatResponse` (id and model will be empty when collected
/// from a streamed response — use `chat()` if you need them).
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

    /// Raw event stream for incremental processing.
    pub fn events(&mut self) -> &mut impl Stream<Item = Result<StreamEvent, LlmError>> {
        &mut self.inner
    }

    /// Fold all events into a complete `ChatResponse`.
    pub async fn collect(mut self) -> Result<ChatResponse, LlmError> {
        let mut text_blocks: HashMap<usize, String> = HashMap::new();
        let mut reasoning_blocks: HashMap<usize, String> = HashMap::new();
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
                StreamEvent::ContentStop { .. } => {}
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
            all.push((
                idx,
                ResponseContent::Reasoning {
                    text,
                    signature: None,
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
