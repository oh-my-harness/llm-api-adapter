mod common;

use futures::StreamExt;
use llm_adapter::{LlmError, Provider, anthropic::AnthropicProvider, types::*};
use wiremock::MockServer;

fn provider(server: &MockServer) -> AnthropicProvider {
    AnthropicProvider::builder("test-key")
        .base_url(server.uri())
        .build()
}

#[tokio::test]
async fn stream_simple_collects_to_response() {
    let server = MockServer::start().await;
    common::mock_anthropic_stream(
        &server,
        include_str!("fixtures/anthropic/stream_simple.txt"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text(
            "Count from one to three.".into(),
        )]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
    assert_eq!(resp.usage.provenance, UsageProvenance::ReportedValid);
}

#[tokio::test]
async fn stream_without_message_start_marks_usage_missing() {
    let server = MockServer::start().await;
    common::mock_anthropic_stream(
        &server,
        include_str!("fixtures/anthropic/stream_missing_message_start.txt"),
    )
    .await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 256)
        .message(Message::User(vec![RequestContent::Text("Say hi".into())]))
        .build();
    let response = provider(&server)
        .chat_stream(&req)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_eq!(response.text(), "hi");
    assert_eq!(response.usage.provenance, UsageProvenance::Missing);
}

#[tokio::test]
async fn stream_events_fire_in_order() {
    let server = MockServer::start().await;
    common::mock_anthropic_stream(
        &server,
        include_str!("fixtures/anthropic/stream_simple.txt"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text(
            "Count from one to three.".into(),
        )]))
        .build();

    let mut handle = provider(&server).chat_stream(&req).await.unwrap();
    let events = handle.events();

    let mut content_starts = 0usize;
    let mut content_stops = 0usize;
    let mut message_stops = 0usize;

    while let Some(event) = events.next().await {
        match event.unwrap() {
            StreamEvent::ContentStart { .. } => content_starts += 1,
            StreamEvent::ContentStop { .. } => content_stops += 1,
            StreamEvent::MessageStop { .. } => message_stops += 1,
            _ => {}
        }
    }

    assert!(content_starts >= 1, "expected at least one ContentStart");
    assert!(content_stops >= 1, "expected at least one ContentStop");
    assert_eq!(message_stops, 1);
}

#[tokio::test]
async fn stream_tool_call_collects_invocation() {
    let server = MockServer::start().await;
    common::mock_anthropic_stream(
        &server,
        include_str!("fixtures/anthropic/stream_tool_call.txt"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text(
            "What is the weather in London?".into(),
        )]))
        .tool(ToolDef::new(
            "get_weather",
            "Get weather",
            serde_json::json!({}),
        ))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
}

#[tokio::test]
async fn stream_signature_preserved_in_reasoning_block() {
    let server = MockServer::start().await;
    common::mock_anthropic_stream(
        &server,
        include_str!("fixtures/anthropic/stream_simple.txt"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text(
            "Count from one to three.".into(),
        )]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    // The fixture has a thinking block with signature_delta.
    // Verify the signature propagated through to the response.
    let reasoning_blocks: Vec<_> = resp
        .content
        .iter()
        .filter_map(|c| {
            if let ResponseContent::Reasoning { text: _, signature } = c {
                Some(signature.clone())
            } else {
                None
            }
        })
        .collect();

    assert!(
        !reasoning_blocks.is_empty(),
        "expected at least one reasoning block"
    );
    assert_eq!(
        reasoning_blocks[0],
        Some("862aad5394484928855530328e82e357".to_string()),
        "Reasoning block signature should be preserved from signature_delta in stream"
    );
}

#[tokio::test]
async fn anthropic_stream_dropped_mid_stream_returns_error() {
    let server = MockServer::start().await;
    common::mock_anthropic_stream(
        &server,
        include_str!("fixtures/anthropic/stream_dropped_mid.txt"),
    )
    .await;

    let req = ChatRequest::builder("claude-test", 256)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let err = handle.collect().await.expect_err("must return Err");

    match err {
        LlmError::StreamIncomplete {
            received_chunks,
            finish_reason,
        } => {
            assert!(received_chunks > 0, "received_chunks should be > 0");
            assert!(finish_reason.is_none(), "finish_reason must be None");
        }
        other => panic!("expected StreamIncomplete, got {other:?}"),
    }
}

#[tokio::test]
async fn anthropic_stream_dropped_after_message_delta_surfaces_finish_reason() {
    // Regression for issue #11: when the stream drops *after* a `message_delta`
    // (which carries `stop_reason`) but *before* `message_stop`, Layer 1 must
    // surface `finish_reason: Some("end_turn")` — the "model finished but stream
    // termination was lost" case described in `LlmError::StreamIncomplete`'s
    // docstring. Without the `pending_stop_reason` capture in `parse_stream`,
    // this would be `None`.
    let server = MockServer::start().await;
    common::mock_anthropic_stream(
        &server,
        include_str!("fixtures/anthropic/stream_dropped_after_delta.txt"),
    )
    .await;

    let req = ChatRequest::builder("claude-test", 256)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let err = handle.collect().await.expect_err("must return Err");

    match err {
        LlmError::StreamIncomplete {
            received_chunks,
            finish_reason,
        } => {
            assert!(
                received_chunks > 0,
                "received_chunks should be > 0, got {received_chunks}"
            );
            assert_eq!(
                finish_reason.as_deref(),
                Some("end_turn"),
                "finish_reason must be captured from message_delta, got {finish_reason:?}"
            );
        }
        other => panic!("expected StreamIncomplete, got {other:?}"),
    }
}
