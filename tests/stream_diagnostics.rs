//! End-to-end tests for abnormally terminated SSE streams.
//!
//! These use a raw TCP listener so the server can close the body at an exact
//! byte boundary. wiremock cannot express "headers + one chunk, then reset".

use std::time::Duration;

use llm_adapter::{
    LlmError, Provider, StreamEndKind,
    openai::OpenAIProvider,
    types::{ChatRequest, Message, RequestContent},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve exactly one HTTP request with the given raw response bytes, then
/// close the socket abruptly.
async fn serve_raw(response: Vec<u8>) -> String {
    serve_raw_then_hold(response, Duration::ZERO).await
}

/// Serve one request, write the raw response bytes, keep the socket open for
/// `hold`, then close it. A non-zero hold lets the client observe a read-idle
/// gap (or hit its own read timeout) before the truncated body is detected.
async fn serve_raw_then_hold(response: Vec<u8>, hold: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 4096];
        let _ = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut buf)).await;
        socket.write_all(&response).await.unwrap();
        socket.flush().await.unwrap();
        if !hold.is_zero() {
            tokio::time::sleep(hold).await;
        }
        // Drop without a graceful shutdown so hyper sees a truncated body.
    });
    format!("http://{addr}")
}

/// Serve one request: response headers, then `heartbeats` SSE comment frames
/// spaced by `interval` (the first is written immediately), then hold `hold`
/// and drop the socket without a terminating chunk.
///
/// Comments produce body bytes but no SSE `data:` event, so this isolates the
/// adapter's byte-level clock from event-level counting.
async fn serve_heartbeats(interval: Duration, heartbeats: usize, hold: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 4096];
        let _ = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut buf)).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
            )
            .await
            .unwrap();
        socket.flush().await.unwrap();
        for i in 0..heartbeats {
            if i > 0 {
                tokio::time::sleep(interval).await;
            }
            let frame = sse_chunk(": keep-alive\n\n");
            socket.write_all(frame.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
        }
        if !hold.is_zero() {
            tokio::time::sleep(hold).await;
        }
    });
    format!("http://{addr}")
}

/// Read a whitespace-delimited `key=value` field from a detail string.
fn detail_field<'a>(detail: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    detail
        .split_whitespace()
        .find_map(|t| t.strip_prefix(&prefix))
}

fn sse_chunk(payload: &str) -> String {
    format!("{:x}\r\n{payload}\r\n", payload.len())
}

fn bypass_process_proxy() {
    // The developer shell exports http_proxy; a local raw listener must be
    // reached directly or the proxy's 502 masks the stream behaviour.
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: tests in this binary only read these variables through
        // reqwest's client construction, which happens after this call.
        unsafe {
            std::env::set_var("NO_PROXY", "127.0.0.1,localhost");
            std::env::set_var("no_proxy", "127.0.0.1,localhost");
            std::env::remove_var("HTTP_PROXY");
            std::env::remove_var("http_proxy");
            std::env::remove_var("HTTPS_PROXY");
            std::env::remove_var("https_proxy");
        }
    });
}

async fn collect_stream_error(base_url: String) -> LlmError {
    collect_stream_error_with_read_timeout(base_url, None).await
}

async fn collect_stream_error_with_read_timeout(
    base_url: String,
    read_timeout: Option<Duration>,
) -> LlmError {
    bypass_process_proxy();
    let mut builder = OpenAIProvider::builder("test-key").base_url(base_url);
    if let Some(timeout) = read_timeout {
        builder = builder.read_timeout(timeout);
    }
    let provider = builder.build();
    let request = ChatRequest::builder("test-model", 64)
        .message(Message::User(vec![RequestContent::Text("hi".into())]))
        .build();
    provider
        .chat_stream(&request)
        .await
        .expect("HTTP 200 must open a stream")
        .collect()
        .await
        .expect_err("stream must not complete normally")
}

/// A body that ends mid-chunk must be reported as `transport_error`, with the
/// client-visible cause chain preserved for operators.
#[tokio::test]
async fn truncated_chunked_body_is_classified_as_transport_error() {
    let request: Vec<u8> = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nX-Request-Id: req-test-123\r\n\r\n30\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\r\n".to_vec();
    // The chunk declares 0x30 bytes; the terminating 0-length chunk is never
    // sent and the socket closes immediately after the payload.

    let error = collect_stream_error(serve_raw(request).await).await;
    let LlmError::StreamIncomplete {
        kind,
        detail,
        received_chunks,
        ..
    } = error
    else {
        panic!("expected StreamIncomplete, got {error:?}");
    };
    assert_eq!(kind, StreamEndKind::TransportError);
    assert_eq!(received_chunks, 1);
    let detail = detail.expect("transport failures must carry a detail");
    assert!(
        detail.contains("http=200"),
        "detail must record the HTTP status: {detail}"
    );
    assert!(
        detail.contains("rid=req-test-123"),
        "detail must record a whitelisted request id: {detail}"
    );
    assert!(
        detail.contains("cause=transport") && detail.contains("flags=["),
        "detail must classify the transport failure: {detail}"
    );
    // Timing is measured at the byte read, and the effective read timeout is
    // persisted so an operator can compare the gap against the configured limit.
    assert_eq!(
        detail_field(&detail, "read_timeout_ms"),
        Some("60000"),
        "default read timeout must be recorded: {detail}"
    );
    assert!(
        detail_field(&detail, "elapsed_ms").is_some()
            && detail_field(&detail, "last_read_ms").is_some()
            && detail_field(&detail, "idle_body_ms").is_some(),
        "detail must carry byte-level timing: {detail}"
    );
    // No credentials, URLs or payload may appear in the added fields either.
    assert!(!detail.contains("test-key"), "{detail}");
}

/// A silent gap between the last successful body read and the socket close is
/// only observable at the adapter's byte layer: the runtime sees no SSE event
/// during the gap.
#[tokio::test]
async fn body_read_idle_is_measured_at_byte_level() {
    let body = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{}",
        sse_chunk("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"),
    );
    // One chunk arrives immediately, then the server stays silent for 150ms
    // before closing without the terminating chunk.
    let base_url = serve_raw_then_hold(body.into_bytes(), Duration::from_millis(150)).await;

    let error = collect_stream_error(base_url).await;
    let LlmError::StreamIncomplete { kind, detail, .. } = error else {
        panic!("expected StreamIncomplete, got {error:?}");
    };
    assert_eq!(kind, StreamEndKind::TransportError);
    let detail = detail.expect("transport failures must carry a detail");

    let elapsed: u64 = detail_field(&detail, "elapsed_ms")
        .expect("elapsed_ms")
        .parse()
        .expect("numeric elapsed_ms");
    let last_read: u64 = detail_field(&detail, "last_read_ms")
        .expect("last_read_ms")
        .parse()
        .expect("numeric last_read_ms");
    let idle: u64 = detail_field(&detail, "idle_body_ms")
        .expect("idle_body_ms")
        .parse()
        .expect("numeric idle_body_ms");

    assert!(
        elapsed >= idle,
        "elapsed must include the idle gap: {detail}"
    );
    assert_eq!(
        last_read + idle,
        elapsed,
        "last_read_ms + idle_body_ms must equal elapsed_ms: {detail}"
    );
    assert!(
        idle >= 120,
        "the 150ms silent gap must be reported as read idle time: {detail}"
    );
    // Redaction is unchanged: timing adds only numbers.
    assert!(!detail.contains("test-key"), "{detail}");
}

/// A body-read timeout must persist the effective read timeout and must not be
/// mislabelled as a decode failure (reqwest wraps timeouts as decode-class
/// errors).
#[tokio::test]
async fn read_timeout_is_recorded_and_not_mislabelled_as_decode() {
    let body = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{}",
        sse_chunk("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"),
    );
    // Hold the socket far beyond the configured read timeout.
    let base_url = serve_raw_then_hold(body.into_bytes(), Duration::from_secs(2)).await;

    let error =
        collect_stream_error_with_read_timeout(base_url, Some(Duration::from_millis(100))).await;
    let LlmError::StreamIncomplete { kind, detail, .. } = error else {
        panic!("expected StreamIncomplete, got {error:?}");
    };
    assert_eq!(kind, StreamEndKind::TransportError);
    let detail = detail.expect("transport failures must carry a detail");

    assert_eq!(
        detail_field(&detail, "read_timeout_ms"),
        Some("100"),
        "configured read timeout must be recorded: {detail}"
    );
    let flags = detail_field(&detail, "flags").expect("flags");
    assert!(
        flags.contains("timeout"),
        "timeouts must be labelled: {detail}"
    );
    assert!(
        !flags.contains("decode"),
        "a body read timeout must not be labelled decode: {detail}"
    );

    let elapsed: u64 = detail_field(&detail, "elapsed_ms")
        .expect("elapsed_ms")
        .parse()
        .expect("numeric elapsed_ms");
    let idle: u64 = detail_field(&detail, "idle_body_ms")
        .expect("idle_body_ms")
        .parse()
        .expect("numeric idle_body_ms");
    assert!(
        idle >= 80 && elapsed >= idle,
        "the read timeout gap must be visible: {detail}"
    );
}

/// Heartbeat-only streams prove the clock measures *body byte* reads: three
/// SSE comments arrive 60ms apart while no `data:` event is ever produced, so
/// `received_chunks` stays 0 yet `last_read_ms` follows the comments.
#[tokio::test]
async fn heartbeat_only_body_is_measured_at_the_byte_layer() {
    let base_url = serve_heartbeats(Duration::from_millis(60), 3, Duration::from_millis(40)).await;

    let error = collect_stream_error(base_url).await;
    let LlmError::StreamIncomplete {
        kind,
        detail,
        received_chunks,
        ..
    } = error
    else {
        panic!("expected StreamIncomplete, got {error:?}");
    };
    // Closing without the terminating chunk surfaces as a transport error;
    // accept a clean EOF too so the assertion documents the classification
    // rather than pinning hyper's exact framing behaviour.
    assert!(
        matches!(
            kind,
            StreamEndKind::TransportError | StreamEndKind::EofWithoutDone
        ),
        "unexpected kind {kind:?} for a heartbeat-only body"
    );
    let detail = detail.expect("incomplete streams must carry a detail");
    assert_eq!(
        received_chunks, 0,
        "comments are not data events and must not be counted: {detail}"
    );

    let elapsed: u64 = detail_field(&detail, "elapsed_ms")
        .expect("elapsed_ms")
        .parse()
        .expect("numeric elapsed_ms");
    let last_read: u64 = detail_field(&detail, "last_read_ms")
        .expect("last_read_ms")
        .parse()
        .expect("numeric last_read_ms");
    let idle: u64 = detail_field(&detail, "idle_body_ms")
        .expect("idle_body_ms")
        .parse()
        .expect("numeric idle_body_ms");

    assert!(
        last_read >= 100,
        "the last comment arrives ~120ms in, so the byte-level clock must see it: {detail}"
    );
    assert!(
        idle < elapsed,
        "body idle must not swallow the whole elapsed window: {detail}"
    );
    assert!(
        idle < 60,
        "only the final 40ms hold is idle, not the preceding heartbeats: {detail}"
    );
}

/// Headers with a body that never yields a single byte: the elapsed time
/// belongs to waiting (and eventually timing out), not to body idleness.
#[tokio::test]
async fn zero_byte_body_reports_no_body_read_instead_of_idle() {
    let headers_only =
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            .to_vec();
    let base_url = serve_raw_then_hold(headers_only, Duration::from_secs(2)).await;

    let error =
        collect_stream_error_with_read_timeout(base_url, Some(Duration::from_millis(100))).await;
    let LlmError::StreamIncomplete { kind, detail, .. } = error else {
        panic!("expected StreamIncomplete, got {error:?}");
    };
    assert_eq!(kind, StreamEndKind::TransportError);
    let detail = detail.expect("transport failures must carry a detail");

    assert_eq!(
        detail_field(&detail, "read_timeout_ms"),
        Some("100"),
        "the effective read timeout must still be recorded: {detail}"
    );
    assert_eq!(
        detail_field(&detail, "last_read_ms"),
        Some("none"),
        "no body byte was ever read: {detail}"
    );
    assert_eq!(
        detail_field(&detail, "idle_body_ms"),
        Some("unavailable"),
        "a zero-byte body has no body-idle window: {detail}"
    );
    let elapsed: u64 = detail_field(&detail, "elapsed_ms")
        .expect("elapsed_ms")
        .parse()
        .expect("numeric elapsed_ms");
    assert!(
        elapsed >= 80,
        "the failed header-wait/body-wait must remain visible as elapsed: {detail}"
    );
}

/// `[DONE]` without `finish_reason` is an upstream protocol omission, not a
/// transport failure, and must be labelled as such.
#[tokio::test]
async fn done_without_finish_reason_is_classified_separately() {
    let body = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{}{}0\r\n\r\n",
        sse_chunk("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"),
        sse_chunk("data: [DONE]\n\n"),
    );

    let error = collect_stream_error(serve_raw(body.into_bytes()).await).await;
    let LlmError::StreamIncomplete {
        kind,
        detail,
        finish_reason,
        ..
    } = error
    else {
        panic!("expected StreamIncomplete, got {error:?}");
    };
    assert_eq!(kind, StreamEndKind::DoneWithoutFinishReason);
    assert!(finish_reason.is_none());
    let detail = detail.expect("protocol omissions must carry a detail");
    assert!(detail.contains("saw_done=true"), "{detail}");
    assert!(!detail.contains("cause=transport"), "{detail}");
}

/// A clean connection-close-delimited body without `[DONE]` is distinguishable
/// from both a transport error and a provider protocol omission.
#[tokio::test]
async fn clean_eof_without_done_is_classified_as_eof() {
    let body = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{}",
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
    );

    let error = collect_stream_error(serve_raw(body.into_bytes()).await).await;
    let LlmError::StreamIncomplete { kind, detail, .. } = error else {
        panic!("expected StreamIncomplete, got {error:?}");
    };
    assert_eq!(kind, StreamEndKind::EofWithoutDone);
    let detail = detail.expect("EOF must carry a detail");
    assert!(detail.contains("saw_done=false"), "{detail}");
}

/// The persisted `Display` text must include the classification; the runtime
/// only stores `to_string()`, so a variant without it would reproduce the
/// original un-diagnosable log line.
#[test]
fn display_carries_kind_and_redacted_detail() {
    let error = LlmError::StreamIncomplete {
        received_chunks: 1165,
        finish_reason: None,
        kind: StreamEndKind::TransportError,
        detail: Some("http=200 cause=transport flags=[body,decode]".into()),
    };
    let text = error.to_string();
    assert!(
        text.contains("stream incomplete (transport_error)"),
        "{text}"
    );
    assert!(text.contains("1165 chunks"), "{text}");
    assert!(text.contains("detail=http=200"), "{text}");
}
