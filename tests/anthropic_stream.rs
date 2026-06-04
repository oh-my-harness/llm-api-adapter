mod common;

use futures::StreamExt;
use llm_adapter::{Provider, anthropic::AnthropicProvider, types::*};
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
        .tool(ToolDef {
            name: "get_weather".into(),
            description: "Get weather".into(),
            input_schema: serde_json::json!({}),
        })
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
}
