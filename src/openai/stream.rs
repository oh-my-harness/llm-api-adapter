use super::schema::{OaiChunk, OaiDelta};
use crate::LlmError;
use crate::stream_handle::{BoxStream, box_stream};
use crate::types::{ContentKind, StopReason, StreamEvent};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

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
    /// Whether MessageStop has already been emitted (prevents duplicate on [DONE] fallback).
    message_stop_sent: bool,
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
            message_stop_sent: false,
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

    // Capture finish_reason from any choice that provides one (for MessageStop).
    for choice in chunk.choices.iter() {
        if let Some(ref finish_reason) = choice.finish_reason {
            state.pending_stop = Some(finish_reason.clone());
        }
    }

    // Process deltas (no-op when choices is empty, e.g. a usage-only chunk).
    for choice in chunk.choices.iter() {
        events.extend(process_delta(state, &choice.delta, parse_reasoning));
    }

    // Emit MessageStop when this chunk is terminal. DeepSeek packs
    // finish_reason and usage into the same chunk; OpenAI sends a separate
    // usage-only chunk (empty choices) after stream_options.include_usage.
    // SiliconFlow includes usage in EVERY chunk (progressive accounting),
    // so usage alone must not terminate the stream: require finish_reason
    // to have been seen, or an empty-choices usage-only chunk (which only
    // follows a finish chunk on OpenAI-compatible gateways).
    if let Some(usage) = chunk.usage
        && (state.pending_stop.is_some() || chunk.choices.is_empty())
    {
        for &idx in &state.open_blocks {
            events.push(Ok(StreamEvent::ContentStop {
                index: idx,
                signature: None,
            }));
        }
        state.open_blocks.clear();

        let stop_reason = parse_stop_reason(state.pending_stop.as_deref());
        match super::convert::oai_usage_to_canonical(usage, parse_reasoning) {
            Ok(usage) => events.push(Ok(StreamEvent::MessageStop { stop_reason, usage })),
            Err(error) => events.push(Err(error)),
        }
        state.message_stop_sent = true;
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

    // Process reasoning_content if enabled and present.
    // Do NOT close the reasoning block when content is also present in the
    // same delta — some providers (vLLM, GLM) interleave reasoning and text
    // in the same chunk. Both blocks can coexist as open_blocks entries.
    if parse_reasoning && let Some(ref reasoning) = delta.reasoning_content {
        // Allocate reasoning block if needed.
        if state.reasoning_index.is_none() {
            let idx = state.allocate_index();
            state.reasoning_index = Some(idx);
            state.open_blocks.push(idx);
            events.push(Ok(StreamEvent::ContentStart {
                index: idx,
                kind: ContentKind::Reasoning,
            }));
        }

        if let Some(idx) = state.reasoning_index {
            events.push(Ok(StreamEvent::ReasoningDelta {
                index: idx,
                text: reasoning.clone(),
            }));
        }
    }

    // Process text content.
    // Do NOT close the reasoning block when content appears — they can
    // coexist. The reasoning block will be closed during MessageStop or
    // when the stream ends.
    if let Some(ref content) = delta.content {
        // Allocate text block if needed.
        if state.text_index.is_none() {
            let idx = state.allocate_index();
            state.text_index = Some(idx);
            state.open_blocks.push(idx);
            events.push(Ok(StreamEvent::ContentStart {
                index: idx,
                kind: ContentKind::Text,
            }));
        }

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
        if let Some(ref id) = tool_call.id
            && let Some(ref func) = tool_call.function
            && let Some(ref name) = func.name
        {
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

        // Emit tool delta if arguments are present
        if let Some(ref func) = tool_call.function
            && let Some(ref args) = func.arguments
        {
            events.push(Ok(StreamEvent::ToolDelta {
                index: canonical_index,
                arguments: args.clone(),
            }));
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
/// * `tolerant_keepalive` - Whether to silently skip non-JSON data lines (e.g. DeepSeek keepalives)
///
/// # Returns
/// A boxed stream of StreamEvent results
pub fn parse_stream(
    response: reqwest::Response,
    parse_reasoning: bool,
    tolerant_keepalive: bool,
) -> BoxStream {
    use eventsource_stream::Eventsource;
    use futures::stream::{StreamExt, once};

    let stream = response.bytes_stream();
    let event_stream = stream.eventsource();

    let mut state = OaiStreamState::new();

    let chunk_count = Arc::new(AtomicU32::new(0));
    let stop_received = Arc::new(AtomicBool::new(false));
    let finish_reason_seen = Arc::new(Mutex::new(None::<String>));

    box_stream(
        event_stream
            .flat_map({
                let chunk_count = chunk_count.clone();
                let finish_reason_seen = finish_reason_seen.clone();
                let stop_received = stop_received.clone();
                move |event_result| {
                    let mut results = Vec::new();

                    match event_result {
                        Ok(event) => {
                            chunk_count.fetch_add(1, Ordering::SeqCst);

                            // Handle the [DONE] marker
                            if event.data == "[DONE]" {
                                if !state.message_stop_sent {
                                    // Path A tightening: no finish_reason and no usage → error.
                                    if state.pending_stop.is_none() {
                                        for &idx in &state.open_blocks {
                                            results.push(Ok(StreamEvent::ContentStop {
                                                index: idx,
                                                signature: None,
                                            }));
                                        }
                                        state.open_blocks.clear();
                                        results.push(Err(LlmError::StreamIncomplete {
                                            received_chunks: chunk_count.load(Ordering::SeqCst),
                                            finish_reason: None,
                                        }));
                                        state.message_stop_sent = true;
                                        stop_received.store(true, Ordering::SeqCst);
                                        return box_stream(futures::stream::iter(results));
                                    }
                                    // The response completed, but the provider did
                                    // not supply authoritative usage.
                                    let mut fallback = Vec::new();
                                    for &idx in &state.open_blocks {
                                        fallback.push(Ok(StreamEvent::ContentStop {
                                            index: idx,
                                            signature: None,
                                        }));
                                    }
                                    state.open_blocks.clear();
                                    let stop_reason =
                                        parse_stop_reason(state.pending_stop.as_deref());
                                    fallback.push(Ok(StreamEvent::MessageStop {
                                        stop_reason,
                                        usage: crate::types::Usage::missing(),
                                    }));
                                    state.message_stop_sent = true;
                                    return box_stream(futures::stream::iter(fallback));
                                }
                                return box_stream(futures::stream::iter(results));
                            }

                            // Parse the JSON chunk
                            match serde_json::from_str::<OaiChunk>(&event.data) {
                                Ok(chunk) => {
                                    let events = process_chunk(&mut state, chunk, parse_reasoning);
                                    if let Some(ref fr) = state.pending_stop {
                                        *finish_reason_seen.lock().unwrap() = Some(fr.clone());
                                    }
                                    results = events;
                                }
                                Err(e) => {
                                    if tolerant_keepalive {
                                        let _ = e;
                                    } else {
                                        results.push(Err(LlmError::Decode(e)));
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            // Issue #11: a real connection drop surfaces as an
                            // SSE-level Err. `collect` short-circuits on the first
                            // Err, so the `.chain(once(...))` tail below would
                            // never be polled — emit StreamIncomplete here to
                            // preserve received_chunks/finish_reason for callers
                            // matching on that variant. The underlying transport
                            // error is logged for diagnostics.
                            tracing::warn!(
                                error = %e,
                                received_chunks = chunk_count.load(Ordering::SeqCst),
                                "SSE stream errored mid-flight; reporting StreamIncomplete"
                            );
                            results.push(Err(LlmError::StreamIncomplete {
                                received_chunks: chunk_count.load(Ordering::SeqCst),
                                finish_reason: finish_reason_seen.lock().unwrap().clone(),
                            }));
                            stop_received.store(true, Ordering::SeqCst);
                        }
                    }

                    box_stream(futures::stream::iter(results))
                }
            })
            .inspect({
                let stop_received = stop_received.clone();
                move |ev| {
                    if let Ok(StreamEvent::MessageStop { .. }) = ev {
                        stop_received.store(true, Ordering::SeqCst);
                    }
                }
            })
            .chain(
                once(async move {
                    if stop_received.load(Ordering::SeqCst) {
                        None
                    } else {
                        Some(Err(LlmError::StreamIncomplete {
                            received_chunks: chunk_count.load(Ordering::SeqCst),
                            finish_reason: finish_reason_seen.lock().unwrap().clone(),
                        }))
                    }
                })
                .filter_map(|opt| async move { opt }),
            ),
    )
}
