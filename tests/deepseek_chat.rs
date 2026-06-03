mod common;

use llm_adapter::{
    openai::OpenAIProvider,
    types::*,
    Provider,
};
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
    assert_eq!(resp.text(), "Hello from DeepSeek!");
    assert_eq!(resp.usage.cached_input_tokens, 0);
}

#[tokio::test]
async fn deepseek_reasoning_response() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/deepseek/chat_reasoning.json")).await;

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text("What is the answer?".into())]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();

    // Reasoning should appear as ResponseContent::Reasoning before the text
    assert!(resp.reasoning().is_some());
    assert!(resp.reasoning().unwrap().contains("think"));
    assert_eq!(resp.text(), "The answer is 42.");
    // Reasoning block comes first (index 0), text second (index 1)
    assert!(matches!(&resp.content[0], ResponseContent::Reasoning(_)));
    assert!(matches!(&resp.content[1], ResponseContent::Text(_)));
}
