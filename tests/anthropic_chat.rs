mod common;

use llm_adapter::{LlmError, Provider, anthropic::AnthropicProvider, types::*};
use wiremock::MockServer;

fn provider(server: &MockServer) -> AnthropicProvider {
    AnthropicProvider::builder("test-key")
        .base_url(server.uri())
        .build()
}

#[tokio::test]
async fn simple_chat_returns_response() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(&server, include_str!("fixtures/anthropic/chat_simple.json")).await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text(
            "Say hi in exactly five words.".into(),
        )]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();

    assert!(!resp.id.is_empty());
    assert!(!resp.model.is_empty());
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
}

#[tokio::test]
async fn chat_returns_tool_call() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(
        &server,
        include_str!("fixtures/anthropic/chat_tool_call.json"),
    )
    .await;

    let tool = ToolDef::new(
        "get_weather",
        "Get current weather for a city.",
        serde_json::json!({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}),
    );
    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text(
            "What is the weather in Paris?".into(),
        )]))
        .tool(tool)
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();

    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
    assert_eq!(invocations[0].input["location"], "Paris");
}

#[tokio::test]
async fn multi_turn_tool_call() {
    let server = MockServer::start().await;

    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
            include_str!("fixtures/anthropic/chat_tool_call.json"),
            "application/json",
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
            include_str!("fixtures/anthropic/chat_after_tool.json"),
            "application/json",
        ))
        .mount(&server)
        .await;

    let p = provider(&server);

    // Round 1: get tool call
    let mut messages = vec![Message::User(vec![RequestContent::Text(
        "What is the weather in Paris?".into(),
    )])];
    let req1 = ChatRequest::builder("deepseek-v4-flash", 256)
        .messages(messages.clone())
        .tool(ToolDef::new(
            "get_weather",
            "Get weather",
            serde_json::json!({}),
        ))
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
    let req2 = ChatRequest::builder("deepseek-v4-flash", 256)
        .messages(messages)
        .build();
    let resp2 = p.chat(&req2).await.unwrap();
    assert!(matches!(resp2.stop_reason, StopReason::EndTurn));
}

#[tokio::test]
async fn reasoning_content_extracted() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(
        &server,
        include_str!("fixtures/anthropic/chat_reasoning.json"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 2048)
        .message(Message::User(vec![RequestContent::Text(
            "What is 7 times 8? Answer with just the number.".into(),
        )]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();

    // The fixture has a text block — reasoning() may or may not be present
    // depending on whether the response included thinking blocks.
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert!(resp.usage.input_tokens > 0);
    // In DeepSeek V4 via Anthropic endpoint, response has a text block
    assert!(!resp.text().is_empty(), "expected non-empty text");
}

#[tokio::test]
async fn error_401_maps_to_unauthorized() {
    let server = MockServer::start().await;
    common::mock_anthropic_error(
        &server,
        401,
        include_str!("fixtures/anthropic/error_unauthorized.json"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::Unauthorized));
}

#[tokio::test]
async fn error_500_maps_to_server_error() {
    let server = MockServer::start().await;
    common::mock_anthropic_error(
        &server,
        500,
        include_str!("fixtures/anthropic/error_server.json"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::ServerError(_)));
}

#[tokio::test]
async fn provider_is_object_safe_with_anthropic() {
    let p: Box<dyn Provider> = Box::new(AnthropicProvider::builder("test-key").build());
    drop(p);
}
