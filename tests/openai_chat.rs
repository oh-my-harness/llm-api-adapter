mod common;

use llm_adapter::{
    openai::OpenAIProvider,
    types::*,
    LlmError, Provider,
};
use wiremock::MockServer;

fn provider(server: &MockServer) -> OpenAIProvider {
    OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .build()
}

#[tokio::test]
async fn simple_chat_returns_text() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_simple.json")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hello".into())]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();

    assert_eq!(resp.id, "chatcmpl-test123");
    assert_eq!(resp.model, "gpt-4o");
    assert!(!resp.text().is_empty());
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert_eq!(resp.usage.input_tokens, 10);
    assert_eq!(resp.usage.output_tokens, 9);
    assert_eq!(resp.usage.cached_input_tokens, 5);
}

#[tokio::test]
async fn chat_returns_tool_call() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_tool_call.json")).await;

    let tool = ToolDef {
        name: "get_weather".into(),
        description: "Get the weather".into(),
        input_schema: serde_json::json!({"type":"object","properties":{"location":{"type":"string"}}}),
    };
    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("What's the weather in Paris?".into())]))
        .tool(tool)
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();

    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
    assert_eq!(invocations[0].id, "call_abc123");
    assert_eq!(invocations[0].input["location"], "Paris");
}

#[tokio::test]
async fn multi_turn_tool_call() {
    let server = MockServer::start().await;

    // First call returns tool call, second returns final answer
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_raw(include_str!("fixtures/openai/chat_tool_call.json"), "application/json"),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_raw(include_str!("fixtures/openai/chat_after_tool.json"), "application/json"),
        )
        .mount(&server)
        .await;

    let p = provider(&server);

    // Round 1: get tool call
    let mut messages = vec![
        Message::User(vec![RequestContent::Text("What's the weather in Paris?".into())]),
    ];
    let req1 = ChatRequest::builder("gpt-4o", 512)
        .messages(messages.clone())
        .tool(ToolDef {
            name: "get_weather".into(),
            description: "Get weather".into(),
            input_schema: serde_json::json!({}),
        })
        .build();
    let resp1 = p.chat(&req1).await.unwrap();
    assert!(matches!(resp1.stop_reason, StopReason::ToolUse));

    let invocation_id = resp1.tool_invocations().next().unwrap().id.clone();
    messages.push(resp1.into_message());
    messages.push(Message::Tool {
        invocation_id,
        content: vec![RequestContent::Text("Sunny, 22°C".into())],
        is_error: false,
    });

    // Round 2: final answer
    let req2 = ChatRequest::builder("gpt-4o", 512).messages(messages).build();
    let resp2 = p.chat(&req2).await.unwrap();
    assert!(matches!(resp2.stop_reason, StopReason::EndTurn));
    assert!(resp2.text().contains("22"));
}

#[tokio::test]
async fn error_401_maps_to_unauthorized() {
    let server = MockServer::start().await;
    common::mock_error(&server, 401, include_str!("fixtures/openai/error_unauthorized.json")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::Unauthorized));
}

#[tokio::test]
async fn error_429_maps_to_rate_limit() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(429)
                .insert_header("retry-after", "60")
                .set_body_raw(include_str!("fixtures/openai/error_rate_limit.json"), "application/json"),
        )
        .mount(&server)
        .await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::RateLimit { retry_after: _ }));
}

#[tokio::test]
async fn error_500_maps_to_server_error() {
    let server = MockServer::start().await;
    common::mock_error(&server, 500, include_str!("fixtures/openai/error_server.json")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::ServerError(_)));
}
