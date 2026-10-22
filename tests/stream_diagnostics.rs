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
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 4096];
        let _ = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut buf)).await;
        socket.write_all(&response).await.unwrap();
        socket.flush().await.unwrap();
        // Drop without a graceful shutdown so hyper sees a truncated body.
    });
    format!("http://{addr}")
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
    bypass_process_proxy();
    let provider = OpenAIProvider::builder("test-key")
        .base_url(base_url)
        .build();
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
