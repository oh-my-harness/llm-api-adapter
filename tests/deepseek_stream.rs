mod common;

use futures::StreamExt;
use llm_adapter::types::StreamEvent;
use llm_adapter::{Provider, openai::OpenAIProvider, types::*};
use wiremock::MockServer;

fn provider(server: &MockServer) -> OpenAIProvider {
    OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .parse_reasoning_content(true)
        .build()
}

#[tokio::test]
async fn deepseek_stream_reasoning_then_text() {
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/deepseek/stream_reasoning.txt"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text(
            "What is the answer?".into(),
        )]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(resp.reasoning().is_some());
    assert!(!resp.reasoning().unwrap().is_empty());
    assert!(!resp.text().is_empty());
    // Reasoning block has lower index than text block
    let reasoning_idx = resp
        .content
        .iter()
        .position(|c| matches!(c, ResponseContent::Reasoning { .. }))
        .unwrap();
    let text_idx = resp
        .content
        .iter()
        .position(|c| matches!(c, ResponseContent::Text(_)))
        .unwrap();
    assert!(reasoning_idx < text_idx);
}

#[tokio::test]
async fn deepseek_stream_events_fire_in_order() {
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/deepseek/stream_reasoning.txt"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text(
            "What is the answer?".into(),
        )]))
        .build();

    let mut handle = provider(&server).chat_stream(&req).await.unwrap();
    let events = handle.events();

    let mut content_starts = 0usize;
    let mut text_deltas = 0usize;
    let mut reasoning_deltas = 0usize;
    let mut content_stops = 0usize;
    let mut message_stops = 0usize;

    while let Some(event) = events.next().await {
        match event.unwrap() {
            StreamEvent::ContentStart { .. } => content_starts += 1,
            StreamEvent::TextDelta { .. } => text_deltas += 1,
            StreamEvent::ReasoningDelta { .. } => reasoning_deltas += 1,
            StreamEvent::ContentStop { .. } => content_stops += 1,
            StreamEvent::MessageStop { .. } => message_stops += 1,
            StreamEvent::ToolDelta { .. } => {}
        }
    }

    // The stream_reasoning fixture has reasoning then text: at least 2 blocks
    assert!(
        content_starts >= 2,
        "expected at least 2 ContentStart events"
    );
    assert!(reasoning_deltas >= 1, "expected at least 1 ReasoningDelta");
    assert!(text_deltas >= 1, "expected at least 1 TextDelta");
    assert_eq!(
        content_stops, content_starts,
        "ContentStop count should match ContentStart count"
    );
    assert_eq!(message_stops, 1, "expected exactly 1 MessageStop");
}

#[tokio::test]
async fn deepseek_stream_tool_call_collects_invocation() {
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/deepseek/stream_tool_call.txt"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text(
            "Weather in Paris?".into(),
        )]))
        .tool(ToolDef::new(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object"}),
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
