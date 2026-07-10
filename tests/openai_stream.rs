mod common;

use futures::StreamExt;
use llm_adapter::{
    Provider,
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
            StreamEvent::MessageStop { stop_reason, .. } => {
                message_stops += 1;
                assert!(matches!(stop_reason, StopReason::EndTurn));
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
