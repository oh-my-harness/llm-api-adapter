mod common;

use llm_adapter::{
    openai::OpenAIProvider,
    types::*,
    Provider,
};
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
    common::mock_stream(&server, include_str!("fixtures/deepseek/stream_reasoning.txt")).await;

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text("What is the answer?".into())]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(resp.reasoning().is_some());
    let reasoning = resp.reasoning().unwrap();
    assert!(reasoning.contains("Let me"));
    assert_eq!(resp.text(), "Answer: 42");
    // Reasoning block has lower index than text block
    let reasoning_idx = resp.content.iter().position(|c| matches!(c, ResponseContent::Reasoning(_))).unwrap();
    let text_idx = resp.content.iter().position(|c| matches!(c, ResponseContent::Text(_))).unwrap();
    assert!(reasoning_idx < text_idx);
}
