use crate::types::{ContentKind, StreamEvent, StopReason};
use crate::LlmError;
use super::schema::{OaiChunk, OaiDelta};
use std::collections::HashMap;

/// Tracks streaming state across chunks for a single message.
///
/// This struct maintains indices for each type of content block (text, reasoning, tools)
/// and manages the lifecycle of block events (ContentStart, deltas, ContentStop).
#[derive(Debug)]
pub struct OaiStreamState {
    /// Index of the reasoning block if started, None otherwise.
    reasoning_index: Option<usize>,
    /// Index of the text block if started, None otherwise.
    text_index: Option<usize>,
    /// Maps OAI tool index (from tool_calls array) to canonical block index.
    tool_index_map: HashMap<usize, usize>,
    /// Allocator for canonical block indices.
    next_index: usize,
    /// Blocks that have had ContentStart emitted.
    open_blocks: Vec<usize>,
    /// Pending stop reason to emit in MessageStop.
    pending_stop: Option<String>,
}

impl OaiStreamState {
    /// Create a new stream state.
    pub fn new() -> Self {
        OaiStreamState {
            reasoning_index: None,
            text_index: None,
            tool_index_map: HashMap::new(),
            next_index: 0,
            open_blocks: Vec::new(),
            pending_stop: None,
        }
    }

    /// Allocate a new canonical block index.
    fn allocate_index(&mut self) -> usize {
        let idx = self.next_index;
        self.next_index += 1;
        idx
    }
}

/// Parse a single chunk and emit corresponding StreamEvents.
///
/// # Arguments
/// * `state` - Mutable reference to the stream state
/// * `chunk` - The OaiChunk to process
/// * `parse_reasoning` - Whether to parse reasoning_content from deltas
///
/// # Returns
/// A vector of StreamEvent results. Errors are included in the result vector.
pub fn process_chunk(
    state: &mut OaiStreamState,
    chunk: OaiChunk,
    parse_reasoning: bool,
) -> Vec<Result<StreamEvent, LlmError>> {
    let mut events = Vec::new();

    // Store the finish_reason if present (for MessageStop event)
    if !chunk.choices.is_empty() {
        if let Some(ref finish_reason) = chunk.choices[0].finish_reason {
            state.pending_stop = Some(finish_reason.clone());
        }
    }

    // If choices is empty, this is a usage-only chunk; close all blocks and emit MessageStop
    if chunk.choices.is_empty() {
        // Close all open blocks
        for &idx in &state.open_blocks {
            events.push(Ok(StreamEvent::ContentStop { index: idx }));
        }
        state.open_blocks.clear();

        // Emit MessageStop if we have usage
        if let Some(usage) = chunk.usage {
            let stop_reason = parse_stop_reason(state.pending_stop.as_deref());
            events.push(Ok(StreamEvent::MessageStop {
                stop_reason,
                usage: crate::types::Usage {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                    cached_input_tokens: usage.prompt_cache_hit_tokens,
                    cache_creation_input_tokens: 0,
                    reasoning_tokens: 0,
                },
            }));
        }

        return events;
    }

    // Process each choice's delta
    for choice in chunk.choices.iter() {
        events.extend(process_delta(state, &choice.delta, parse_reasoning));
    }

    events
}

/// Process a single delta update.
fn process_delta(
    state: &mut OaiStreamState,
    delta: &OaiDelta,
    parse_reasoning: bool,
) -> Vec<Result<StreamEvent, LlmError>> {
    let mut events = Vec::new();

    // Process reasoning_content if enabled and present
    if parse_reasoning {
        if let Some(ref reasoning) = delta.reasoning_content {
            // If text_index is set, we need to close it first (reasoning comes before text in streaming)
            if state.text_index.is_some() {
                // This shouldn't happen in normal flow, but handle it
                let text_idx = state.text_index.unwrap();
                events.push(Ok(StreamEvent::ContentStop { index: text_idx }));
                state.open_blocks.retain(|&x| x != text_idx);
                state.text_index = None;
            }

            // Allocate reasoning block if needed
            if state.reasoning_index.is_none() {
                let idx = state.allocate_index();
                state.reasoning_index = Some(idx);
                state.open_blocks.push(idx);
                events.push(Ok(StreamEvent::ContentStart {
                    index: idx,
                    kind: ContentKind::Reasoning,
                }));
            }

            // Emit reasoning delta
            if let Some(idx) = state.reasoning_index {
                events.push(Ok(StreamEvent::ReasoningDelta {
                    index: idx,
                    text: reasoning.clone(),
                }));
            }
        }
    }

    // Process text content
    if let Some(ref content) = delta.content {
        // Close reasoning block if it was open and we now have text
        if let Some(reasoning_idx) = state.reasoning_index {
            events.push(Ok(StreamEvent::ContentStop {
                index: reasoning_idx,
            }));
            state.open_blocks.retain(|&x| x != reasoning_idx);
            state.reasoning_index = None;
        }

        // Allocate text block if needed
        if state.text_index.is_none() {
            let idx = state.allocate_index();
            state.text_index = Some(idx);
            state.open_blocks.push(idx);
            events.push(Ok(StreamEvent::ContentStart {
                index: idx,
                kind: ContentKind::Text,
            }));
        }

        // Emit text delta
        if let Some(idx) = state.text_index {
            events.push(Ok(StreamEvent::TextDelta {
                index: idx,
                text: content.clone(),
            }));
        }
    }

    // Process tool calls
    for tool_call in &delta.tool_calls {
        let oai_index = tool_call.index;

        // Get or allocate canonical index for this tool
        let canonical_index = if let Some(&idx) = state.tool_index_map.get(&oai_index) {
            idx
        } else {
            let idx = state.allocate_index();
            state.tool_index_map.insert(oai_index, idx);
            idx
        };

        // If this is the first delta for this tool (has id and name), emit ContentStart
        if let Some(ref id) = tool_call.id {
            if let Some(ref func) = tool_call.function {
                if let Some(ref name) = func.name {
                    // Only emit ContentStart once per tool
                    if !state.open_blocks.contains(&canonical_index) {
                        state.open_blocks.push(canonical_index);
                        events.push(Ok(StreamEvent::ContentStart {
                            index: canonical_index,
                            kind: ContentKind::ToolInvocation {
                                id: id.clone(),
                                name: name.clone(),
                            },
                        }));
                    }
                }
            }
        }

        // Emit tool delta if arguments are present
        if let Some(ref func) = tool_call.function {
            if let Some(ref args) = func.arguments {
                events.push(Ok(StreamEvent::ToolDelta {
                    index: canonical_index,
                    arguments: args.clone(),
                }));
            }
        }
    }

    events
}

/// Convert OpenAI finish_reason to canonical StopReason.
fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls") => StopReason::ToolUse,
        Some("content_filter") => StopReason::ContentFilter,
        Some(other) => StopReason::Other(other.into()),
        None => StopReason::Other("unknown".into()),
    }
}

/// Parse an HTTP stream response into a stream of StreamEvents.
///
/// This function takes a reqwest Response and converts the Server-Sent Events (SSE)
/// stream into a stream of canonical StreamEvent objects. It handles the [DONE]
/// marker that signals end of stream.
///
/// # Arguments
/// * `response` - The HTTP response with SSE stream
/// * `parse_reasoning` - Whether to parse reasoning_content from deltas
///
/// # Returns
/// A boxed stream of StreamEvent results
pub fn parse_stream(
    response: reqwest::Response,
    parse_reasoning: bool,
) -> futures::stream::BoxStream<'static, Result<StreamEvent, LlmError>> {
    use futures::stream::StreamExt;
    use eventsource_stream::Eventsource;

    let stream = response.bytes_stream();
    let event_stream = stream.eventsource();

    let mut state = OaiStreamState::new();

    let mapped = event_stream
        .flat_map(move |event_result| {
            let mut results = Vec::new();

            match event_result {
                Ok(event) => {
                    // Handle the [DONE] marker
                    if event.data == "[DONE]" {
                        return futures::stream::iter(results).boxed();
                    }

                    // Parse the JSON chunk
                    match serde_json::from_str::<OaiChunk>(&event.data) {
                        Ok(chunk) => {
                            // Process the chunk and emit events
                            let events = process_chunk(&mut state, chunk, parse_reasoning);
                            results = events;
                        }
                        Err(e) => {
                            results.push(Err(LlmError::Decode(e)));
                        }
                    }
                }
                Err(e) => {
                    results.push(Err(LlmError::Stream(e.to_string())));
                }
            }

            futures::stream::iter(results).boxed()
        })
        .boxed();

    mapped
}
