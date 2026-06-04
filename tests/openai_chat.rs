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

    assert!(!resp.id.is_empty());
    assert!(!resp.model.is_empty());
    assert!(!resp.text().is_empty());
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
    // Simple scenario records with thinking disabled — no reasoning content.
    assert!(resp.reasoning().is_none());
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
    assert!(!invocations[0].id.is_empty());
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
    assert!(!resp2.text().is_empty());
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

#[tokio::test]
async fn custom_chat_path_for_compatible_gateways() {
    // GitHub Models-style: /chat/completions (no /v1 prefix).
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_raw(
                include_str!("fixtures/openai/chat_simple.json"),
                "application/json",
            ),
        )
        .mount(&server)
        .await;

    let p = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .chat_path("/chat/completions")
        .build();

    let req = ChatRequest::builder("gpt-4o-mini", 64)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let resp = p.chat(&req).await.unwrap();
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn reasoning_tokens_populated_from_completion_details() {
    // Regression test: OpenAI o-series populates
    // `usage.completion_tokens_details.reasoning_tokens`. Before commit
    // 6760922 this was hardcoded to 0 in oai_usage_to_canonical.
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_reasoning.json")).await;

    let p = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .parse_reasoning_content(true)
        .build();

    let req = ChatRequest::builder("gpt-4o", 1024)
        .message(Message::User(vec![RequestContent::Text("What is 7 times 8?".into())]))
        .build();

    let resp = p.chat(&req).await.unwrap();

    assert!(resp.usage.reasoning_tokens > 0, "reasoning_tokens should be parsed from completion_tokens_details");
    assert!(resp.reasoning().is_some(), "reasoning content should be parsed when parse_reasoning_content is enabled");
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn extra_body_is_merged_into_request() {
    // Verify OpenAIExt.extra_body adds top-level fields to the outgoing JSON.
    // Useful for provider-specific extensions like DeepSeek's `thinking` field.
    use llm_adapter::openai::{OpenAIExt, OpenAIRequest};

    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .and(wiremock::matchers::body_partial_json(serde_json::json!({
            "thinking": {"type": "disabled"},
            "custom_flag": true
        })))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_raw(
                include_str!("fixtures/openai/chat_simple.json"),
                "application/json",
            ),
        )
        .mount(&server)
        .await;

    let p = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .build();

    let ext = OpenAIExt::builder()
        .extra_body(serde_json::json!({
            "thinking": {"type": "disabled"},
            "custom_flag": true,
        }))
        .build();
    let req = OpenAIRequest {
        base: ChatRequest::builder("gpt-4o", 64)
            .message(Message::User(vec![RequestContent::Text("Hi".into())]))
            .build(),
        ext,
    };

    let resp = p.chat_native(&req).await.unwrap();
    // The mock only matches when the request body contains both extra_body keys
    // at the top level — reaching this assertion proves the merge worked.
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn extra_body_rejects_non_object() {
    use llm_adapter::openai::{OpenAIExt, OpenAIRequest};

    let server = MockServer::start().await;
    // No mock mounted; we expect to fail before HTTP.
    let p = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .build();

    let ext = OpenAIExt::builder()
        .extra_body(serde_json::json!("not an object"))
        .build();
    let req = OpenAIRequest {
        base: ChatRequest::builder("gpt-4o", 64)
            .message(Message::User(vec![RequestContent::Text("Hi".into())]))
            .build(),
        ext,
    };

    let err = p.chat_native(&req).await.unwrap_err();
    assert!(
        matches!(err, LlmError::InvalidRequest(ref m) if m.contains("extra_body")),
        "expected InvalidRequest about extra_body, got: {:?}", err
    );
}
