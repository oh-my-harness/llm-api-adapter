use super::schema::{OaiChunk, OaiDelta};
use crate::LlmError;
use crate::StreamEndKind;
use crate::provider::Instant;
use crate::stream_diag::{ResponseMeta, classify_event_stream_error, timing_detail};
use crate::stream_handle::{BoxStream, box_stream};
use crate::types::{ContentKind, StopReason, StreamEvent};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Byte-level body-read clock for one streamed request.
///
/// Only the adapter can observe this: SSE comment/heartbeat bytes reset the
/// underlying read timeout but produce no runtime-visible `StreamEvent`, so a
/// runtime idle watchdog measures a different (usually larger) idle window.
///
/// Uses `AtomicU32` millisecond offsets (supported on wasm32) instead of
/// storing `Instant` values. `u32::MAX` is the "no read yet" sentinel and
/// `u32::MAX - 1` is the saturation marker: once the elapsed millisecond count
/// would reach `CAP_MS`, the snapshot reports every timing field as
/// unavailable instead of silently truncating (which would keep printing a
/// growing idle value after ~49.7 days).
#[derive(Debug)]
struct BodyReadClock {
    request_start: Instant,
    last_read_ms: AtomicU32,
}

impl BodyReadClock {
    const NO_READ: u32 = u32::MAX;
    /// `u32::MAX - 1` milliseconds: the last offset the clock can represent.
    const CAP_MS: u64 = (u32::MAX - 1) as u64;

    fn new(request_start: Instant) -> Self {
        Self {
            request_start,
            last_read_ms: AtomicU32::new(Self::NO_READ),
        }
    }

    /// Record a successfully read body chunk.
    fn mark_read(&self) {
        let elapsed_ms = Self::elapsed_ms(&self.request_start);
        let stored = elapsed_ms.map_or(Self::CAP_MS as u32, |ms| ms.min(Self::CAP_MS) as u32);
        self.last_read_ms.store(stored, Ordering::Relaxed);
    }

    /// Milliseconds since the request was issued, `None` once saturated.
    fn elapsed_ms(request_start: &Instant) -> Option<u64> {
        // `as_millis()` is a u128; never truncate silently into a u32/u64.
        let elapsed = u64::try_from(request_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        (elapsed < Self::CAP_MS).then_some(elapsed)
    }

    /// `(elapsed_ms, last_read_ms, idle_body_ms)` at failure time.
    ///
    /// With no body byte ever read, `last_read_ms` and `idle_body_ms` are
    /// `None`: the elapsed time includes the response-header wait and must not
    /// be mis-attributed to body idleness. A saturated clock yields
    /// `(None, None, None)`.
    fn snapshot(&self) -> (Option<u64>, Option<u64>, Option<u64>) {
        let Some(elapsed_ms) = Self::elapsed_ms(&self.request_start) else {
            return (None, None, None);
        };
        match self.last_read_ms.load(Ordering::Relaxed) {
            Self::NO_READ => (Some(elapsed_ms), None, None),
            // A saturated read offset makes the whole triple untrustworthy.
            ms if ms == Self::CAP_MS as u32 => (None, None, None),
            last => {
                let last = u64::from(last);
                (
                    Some(elapsed_ms),
                    Some(last),
                    Some(elapsed_ms.saturating_sub(last)),
                )
            }
        }
    }

    /// Compact redacted timing fragment for an incomplete-stream detail.
    fn detail(&self) -> String {
        let (elapsed_ms, last_read_ms, idle_body_ms) = self.snapshot();
        timing_detail(elapsed_ms, last_read_ms, idle_body_ms)
    }
}

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
/// * `request_start` - When the HTTP request was issued, used to report
///   `elapsed_ms`. Header latency is included.
/// * `read_timeout` - Effective client body-read idle timeout, recorded in the
///   detail so a failure can be compared against the configured limit. `None`
///   when the platform's HTTP client does not implement one (wasm); the field
///   is then omitted from the detail.
///
/// # Returns
/// A boxed stream of StreamEvent results
pub fn parse_stream(
    response: reqwest::Response,
    parse_reasoning: bool,
    tolerant_keepalive: bool,
    request_start: Instant,
    read_timeout: Option<Duration>,
) -> BoxStream {
    use eventsource_stream::Eventsource;
    use futures::stream::{StreamExt, once};

    let meta = ResponseMeta::capture_with_read_timeout(&response, read_timeout);
    // Record the instant of every successful body read *before* SSE decoding:
    // heartbeat/comment frames reset the transport read timeout but produce no
    // StreamEvent, so this is the only layer that can measure read idle time.
    let read_clock = Arc::new(BodyReadClock::new(request_start));
    let stream = response.bytes_stream().inspect({
        let read_clock = read_clock.clone();
        move |chunk| {
            if chunk.is_ok() {
                read_clock.mark_read();
            }
        }
    });
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
                let meta = meta.clone();
                let read_clock = read_clock.clone();
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
                                            kind: StreamEndKind::DoneWithoutFinishReason,
                                            detail: Some(format!(
                                                "{} {} saw_done=true usage=absent",
                                                meta.prefix(),
                                                read_clock.detail()
                                            )),
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
                                    if let Some(fr) = &state.pending_stop {
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
                            // matching on that variant. Classify the error now:
                            // persisted Display text must distinguish upstream
                            // protocol omissions from transport failures.
                            let (kind, cause) = classify_event_stream_error(&e);
                            let detail =
                                format!("{} {} {}", meta.prefix(), read_clock.detail(), cause);
                            tracing::warn!(
                                kind = %kind,
                                detail = %detail,
                                received_chunks = chunk_count.load(Ordering::SeqCst),
                                "SSE stream errored mid-flight; reporting StreamIncomplete"
                            );
                            results.push(Err(LlmError::StreamIncomplete {
                                received_chunks: chunk_count.load(Ordering::SeqCst),
                                finish_reason: finish_reason_seen.lock().unwrap().clone(),
                                kind,
                                detail: Some(detail),
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
                    let read_clock = read_clock.clone();
                    if stop_received.load(Ordering::SeqCst) {
                        None
                    } else {
                        Some(Err(LlmError::StreamIncomplete {
                            received_chunks: chunk_count.load(Ordering::SeqCst),
                            finish_reason: finish_reason_seen.lock().unwrap().clone(),
                            kind: StreamEndKind::EofWithoutDone,
                            detail: Some(format!(
                                "{} {} saw_done=false",
                                meta.prefix(),
                                read_clock.detail()
                            )),
                        }))
                    }
                })
                .filter_map(|opt| async move { opt }),
            ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturated_read_offset_reports_unavailable_timing() {
        let clock = BodyReadClock {
            request_start: Instant::now(),
            last_read_ms: AtomicU32::new(BodyReadClock::CAP_MS as u32),
        };
        assert_eq!(clock.snapshot(), (None, None, None));
        assert_eq!(
            clock.detail(),
            "elapsed_ms=unavailable last_read_ms=none idle_body_ms=unavailable"
        );
    }

    #[test]
    fn no_read_never_reports_a_body_idle_window() {
        let clock = BodyReadClock::new(Instant::now());
        let (elapsed, last, idle) = clock.snapshot();
        assert!(
            elapsed.is_some(),
            "elapsed is still valid before saturation"
        );
        assert_eq!(last, None);
        assert_eq!(idle, None, "unread body time is not body idle time");
        assert!(clock.detail().contains("idle_body_ms=unavailable"));
    }
}
