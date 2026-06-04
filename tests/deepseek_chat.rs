mod common;

use llm_adapter::{Provider, openai::OpenAIProvider, types::*};
use wiremock::MockServer;

fn provider(server: &MockServer) -> OpenAIProvider {
    // deepseek::client() returns OpenAIProvider; rebuild it with the mock base URL
    OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .parse_reasoning_content(true)
        .build()
}

#[tokio::test]
async fn deepseek_simple_chat() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/deepseek/chat_simple.json")).await;

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text("Hello".into())]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();
    assert!(!resp.text().is_empty());
    // Recorded with thinking explicitly disabled — no reasoning_content.
    assert!(resp.reasoning().is_none());
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
}

#[tokio::test]
async fn deepseek_reasoning_response() {
    let server = MockServer::start().await;
    common::mock_chat(
        &server,
        include_str!("fixtures/deepseek/chat_reasoning.json"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text(
            "What is the answer?".into(),
        )]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();

    assert!(resp.reasoning().is_some());
    assert!(!resp.reasoning().unwrap().is_empty());
    assert!(!resp.text().is_empty());
    // Reasoning block precedes text block.
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
    // DeepSeek V4 reports reasoning_tokens in completion_tokens_details.
    assert!(resp.usage.reasoning_tokens > 0);
}
