mod common;

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
