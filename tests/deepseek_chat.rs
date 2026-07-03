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

#[tokio::test]
async fn deepseek_thinking_enabled_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(
        &server,
        include_str!("fixtures/deepseek/chat_reasoning.json"),
    )
    .await;

    let prov = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .parse_reasoning_content(true)
        .thinking_scheme(llm_adapter::openai::ThinkingScheme::ThinkingToggle)
        .build();

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text("Think.".into())]))
        .thinking(ThinkingLevel::High)
        .build();

    prov.chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["thinking"]["type"], "enabled");
    // reasoning_effort is a top-level field (not inside thinking object)
    assert_eq!(body["reasoning_effort"], "high");
}

#[tokio::test]
async fn deepseek_thinking_disabled_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/deepseek/chat_simple.json")).await;

    let prov = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .thinking_scheme(llm_adapter::openai::ThinkingScheme::ThinkingToggle)
        .build();

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .thinking(ThinkingLevel::Off)
        .build();

    prov.chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["thinking"]["type"], "disabled");
}

#[tokio::test]
async fn deepseek_extra_body_overrides_thinking_scheme() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/deepseek/chat_simple.json")).await;

    let prov = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .thinking_scheme(llm_adapter::openai::ThinkingScheme::ThinkingToggle)
        .build();

    let ext = llm_adapter::openai::OpenAIExt::builder()
        .extra_body(serde_json::json!({"thinking": {"type": "disabled"}}))
        .build();
    let oai_req = llm_adapter::openai::OpenAIRequest {
        base: ChatRequest::builder("deepseek-v4-flash", 512)
            .message(Message::User(vec![RequestContent::Text("Hi".into())]))
            .thinking(ThinkingLevel::High)
            .build(),
        ext,
    };

    prov.chat_native(&oai_req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    // Caller's extra_body thinking wins over scheme-derived thinking.
    assert_eq!(body["thinking"]["type"], "disabled");
}
