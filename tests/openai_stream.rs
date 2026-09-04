mod common;

use futures::StreamExt;
use llm_adapter::{
    LlmError, Provider,
    openai::{OpenAIExt, OpenAIProvider, OpenAIRequest},
    types::*,
};
use wiremock::MockServer;

fn provider(server: &MockServer) -> OpenAIProvider {
    OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .build()
}

#[tokio::test]
async fn stream_simple_collects_to_response() {
    let server = MockServer::start().await;
    common::mock_stream(&server, include_str!("fixtures/openai/stream_simple.txt")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(!resp.text().is_empty());
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
}

#[tokio::test]
async fn stream_events_fire_in_order() {
    let server = MockServer::start().await;
    common::mock_stream(&server, include_str!("fixtures/openai/stream_simple.txt")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let mut handle = provider(&server).chat_stream(&req).await.unwrap();
    let events = handle.events();

    let mut content_starts = 0usize;
    let mut text_deltas = 0usize;
    let mut content_stops = 0usize;
    let mut message_stops = 0usize;

    while let Some(event) = events.next().await {
        match event.unwrap() {
            StreamEvent::ContentStart { .. } => content_starts += 1,
            StreamEvent::TextDelta { .. } => text_deltas += 1,
            StreamEvent::ContentStop { .. } => content_stops += 1,
            StreamEvent::MessageStop { .. } => message_stops += 1,
            _ => {}
        }
    }

    assert_eq!(content_starts, 1);
    assert!(text_deltas >= 1);
    assert_eq!(content_stops, 1);
    assert_eq!(message_stops, 1);
}

#[tokio::test]
async fn stream_tool_call_collects_invocation() {
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/openai/stream_tool_call.txt"),
    )
    .await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text(
            "Weather in Paris?".into(),
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
    assert_eq!(invocations[0].input["location"], "Paris");
}

#[tokio::test]
async fn stream_done_without_usage_emits_fallback_message_stop() {
    let server = MockServer::start().await;
    common::mock_stream(&server, include_str!("fixtures/openai/stream_no_usage.txt")).await;

    let req = ChatRequest::builder("test-model", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let mut handle = provider(&server).chat_stream(&req).await.unwrap();
    let events = handle.events();

    let mut message_stops = 0usize;
    let mut content_stops = 0usize;

    while let Some(event) = events.next().await {
        match event.unwrap() {
            StreamEvent::ContentStop { .. } => content_stops += 1,
            StreamEvent::MessageStop { stop_reason, usage } => {
                message_stops += 1;
                assert!(matches!(stop_reason, StopReason::EndTurn));
                assert_eq!(usage.provenance, UsageProvenance::Missing);
            }
            _ => {}
        }
    }

    assert_eq!(
        content_stops, 1,
        "ContentStop should fire for the text block"
    );
    assert_eq!(
        message_stops, 1,
        "fallback MessageStop must fire on [DONE] without usage"
    );
}

#[tokio::test]
async fn stream_interleaved_reasoning_content_produces_single_blocks() {
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/openai/stream_interleaved_reasoning.txt"),
    )
    .await;

    let req = ChatRequest::builder("test-model", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let handle = provider(&server)
        .chat_stream_native(&OpenAIRequest {
            base: req,
            ext: OpenAIExt {
                parse_reasoning_content: true,
                ..Default::default()
            },
        })
        .await
        .unwrap();
    let resp = handle.collect().await.unwrap();

    // Should have exactly 1 reasoning block and 1 text block, not fragmented.
    let reasoning: Vec<_> = resp
        .content
        .iter()
        .filter(|c| matches!(c, ResponseContent::Reasoning { .. }))
        .collect();
    let text: Vec<_> = resp
        .content
        .iter()
        .filter(|c| matches!(c, ResponseContent::Text(_)))
        .collect();

    assert_eq!(
        reasoning.len(),
        1,
        "should have 1 reasoning block, got {}: {:?}",
        reasoning.len(),
        reasoning
    );
    assert_eq!(
        text.len(),
        1,
        "should have 1 text block, got {}: {:?}",
        text.len(),
        text
    );

    // Verify content is correctly accumulated
    if let ResponseContent::Reasoning { text, .. } = &reasoning[0] {
        assert_eq!(text, "Thinking more");
    }
    if let ResponseContent::Text(t) = &text[0] {
        assert_eq!(t, "Answer here");
    }
}
#[tokio::test]
async fn stream_stop_reason_length_maps_to_max_tokens() {
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/openai/stream_max_tokens.txt"),
    )
    .await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text(
            "Write a very long story.".into(),
        )]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(matches!(resp.stop_reason, StopReason::MaxTokens));
}

#[tokio::test]
async fn stream_done_without_finish_reason_or_usage_returns_error() {
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/openai/stream_truncated_no_finish.txt"),
    )
    .await;

    let req = ChatRequest::builder("test-model", 512)
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
            assert!(
                finish_reason.is_none(),
                "finish_reason must be None, got {finish_reason:?}"
            );
        }
        other => panic!("expected StreamIncomplete, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_dropped_mid_stream_returns_error() {
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/openai/stream_dropped_mid.txt"),
    )
    .await;

    let req = ChatRequest::builder("test-model", 512)
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
async fn stream_done_with_finish_reason_but_no_usage_still_fallbacks() {
    let server = MockServer::start().await;
    common::mock_stream(&server, include_str!("fixtures/openai/stream_no_usage.txt")).await;

    let req = ChatRequest::builder("test-model", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.expect("must return Ok");

    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert_eq!(resp.usage.provenance, UsageProvenance::Missing);
}

#[tokio::test]
async fn stream_handle_collect_surfaces_stream_incomplete_on_err() {
    // Regression for issue #11: when the underlying SSE stream yields an Err
    // mid-flight (connection drop), `collect` must surface `StreamIncomplete`
    // with a non-zero `received_chunks`, not `LlmError::Stream` and not a
    // silent success. Exercises the collect-layer short-circuit path that the
    // `.chain(once(...))` provider tail cannot reach.
    use futures::stream;
    use llm_adapter::StreamHandle;

    let events: Vec<Result<StreamEvent, LlmError>> = vec![
        Ok(StreamEvent::ContentStart {
            index: 0,
            kind: ContentKind::Text,
        }),
        Ok(StreamEvent::TextDelta {
            index: 0,
            text: "partial".into(),
        }),
        // Simulate a connection drop: provider's flat_map Err arm would emit
        // StreamIncomplete; here we inject it directly to test collect's
        // handling of an in-band Err after partial data.
        Err(LlmError::StreamIncomplete {
            received_chunks: 2,
            finish_reason: None,
        }),
    ];

    let boxed: futures::stream::BoxStream<'static, Result<StreamEvent, LlmError>> =
        stream::iter(events).boxed();
    let handle = StreamHandle::from_raw_stream("test-model", boxed);

    let err = handle.collect().await.expect_err("must return Err");
    match err {
        LlmError::StreamIncomplete {
            received_chunks,
            finish_reason,
        } => {
            assert_eq!(
                received_chunks, 2,
                "collect must preserve provider-supplied count"
            );
            assert!(finish_reason.is_none());
        }
        other => panic!("expected StreamIncomplete, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_handle_collect_reports_real_count_on_clean_eof_without_stop() {
    // Layer 2: stream ends cleanly (no Err) but never emitted MessageStop.
    // collect must return StreamIncomplete with the real number of events
    // consumed, not a hardcoded 0.
    use futures::stream;
    use llm_adapter::StreamHandle;

    let events: Vec<Result<StreamEvent, LlmError>> = vec![
        Ok(StreamEvent::ContentStart {
            index: 0,
            kind: ContentKind::Text,
        }),
        Ok(StreamEvent::TextDelta {
            index: 0,
            text: "a".into(),
        }),
        Ok(StreamEvent::TextDelta {
            index: 0,
            text: "b".into(),
        }),
    ];

    let boxed: futures::stream::BoxStream<'static, Result<StreamEvent, LlmError>> =
        stream::iter(events).boxed();
    let handle = StreamHandle::from_raw_stream("test-model", boxed);

    let err = handle.collect().await.expect_err("must return Err");
    match err {
        LlmError::StreamIncomplete {
            received_chunks, ..
        } => {
            assert_eq!(
                received_chunks, 3,
                "Layer 2 must report real count, got {received_chunks}"
            );
        }
        other => panic!("expected StreamIncomplete, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_progressive_usage_does_not_terminate_early() {
    // Regression: SiliconFlow includes `usage` in EVERY chunk (progressive
    // accounting). MessageStop must fire only when finish_reason has been
    // seen — not on the first chunk that merely carries usage. Before the
    // fix, the stream terminated after chunk 1 with empty text, stop
    // "unknown", and output_tokens 0.
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/openai/stream_progressive_usage.txt"),
    )
    .await;

    let req = ChatRequest::builder("zai-org/GLM-5.2", 512)
        .message(Message::User(vec![RequestContent::Text(
            "Reply with exactly: WASM_OK".into(),
        )]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert_eq!(resp.text(), "WASM_OK");
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    // Final usage (46), not chunk-1 usage (0).
    assert_eq!(resp.usage.output_tokens, 46);
    assert_eq!(resp.usage.reasoning_tokens, 43);
}
