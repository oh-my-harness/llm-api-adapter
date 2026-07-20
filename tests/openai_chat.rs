mod common;

use llm_adapter::{LlmError, Provider, openai::OpenAIProvider, types::*};
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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

    let tool = ToolDef::new(
        "get_weather",
        "Get the weather",
        serde_json::json!({"type":"object","properties":{"location":{"type":"string"}}}),
    );
    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text(
            "What's the weather in Paris?".into(),
        )]))
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
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
            include_str!("fixtures/openai/chat_tool_call.json"),
            "application/json",
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
            include_str!("fixtures/openai/chat_after_tool.json"),
            "application/json",
        ))
        .mount(&server)
        .await;

    let p = provider(&server);

    // Round 1: get tool call
    let mut messages = vec![Message::User(vec![RequestContent::Text(
        "What's the weather in Paris?".into(),
    )])];
    let req1 = ChatRequest::builder("gpt-4o", 512)
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
    let req2 = ChatRequest::builder("gpt-4o", 512)
        .messages(messages)
        .build();
    let resp2 = p.chat(&req2).await.unwrap();
    assert!(matches!(resp2.stop_reason, StopReason::EndTurn));
    assert!(!resp2.text().is_empty());
}

#[tokio::test]
async fn error_401_maps_to_unauthorized() {
    let server = MockServer::start().await;
    common::mock_error(
        &server,
        401,
        include_str!("fixtures/openai/error_unauthorized.json"),
    )
    .await;

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
                .set_body_raw(
                    include_str!("fixtures/openai/error_rate_limit.json"),
                    "application/json",
                ),
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
    common::mock_error(
        &server,
        500,
        include_str!("fixtures/openai/error_server.json"),
    )
    .await;

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
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
            include_str!("fixtures/openai/chat_simple.json"),
            "application/json",
        ))
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
        .message(Message::User(vec![RequestContent::Text(
            "What is 7 times 8?".into(),
        )]))
        .build();

    let resp = p.chat(&req).await.unwrap();

    assert!(
        resp.usage.reasoning_tokens > 0,
        "reasoning_tokens should be parsed from completion_tokens_details"
    );
    assert!(
        resp.reasoning().is_some(),
        "reasoning content should be parsed when parse_reasoning_content is enabled"
    );
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
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
            include_str!("fixtures/openai/chat_simple.json"),
            "application/json",
        ))
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
        "expected InvalidRequest about extra_body, got: {:?}",
        err
    );
}

#[tokio::test]
async fn thinking_reasoning_effort_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_reasoning.json")).await;

    let prov = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .thinking_scheme(llm_adapter::openai::ThinkingScheme::ReasoningEffort)
        .build();

    let req = ChatRequest::builder("o3-mini", 512)
        .message(Message::User(vec![RequestContent::Text("Think.".into())]))
        .thinking(ThinkingLevel::Medium)
        .build();

    prov.chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    assert!(!received.is_empty());
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["reasoning_effort"], "medium");
}

#[tokio::test]
async fn thinking_off_omits_reasoning_effort() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_simple.json")).await;

    let prov = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .thinking_scheme(llm_adapter::openai::ThinkingScheme::ReasoningEffort)
        .build();

    let req = ChatRequest::builder("o3-mini", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .thinking(ThinkingLevel::Off)
        .build();

    prov.chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert!(body.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn thinking_toggle_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_reasoning.json")).await;

    let prov = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .thinking_scheme(llm_adapter::openai::ThinkingScheme::ThinkingToggle)
        .build();

    let req = ChatRequest::builder("deepseek-v4-pro", 512)
        .message(Message::User(vec![RequestContent::Text("Think.".into())]))
        .thinking(ThinkingLevel::High)
        .build();

    prov.chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    assert!(!received.is_empty());
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["thinking"]["type"], "enabled");
    // reasoning_effort is top-level, not inside thinking object
    assert_eq!(body["reasoning_effort"], "high");
}

#[tokio::test]
async fn thinking_toggle_off_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_simple.json")).await;

    let prov = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .thinking_scheme(llm_adapter::openai::ThinkingScheme::ThinkingToggle)
        .build();

    let req = ChatRequest::builder("deepseek-v4-pro", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .thinking(ThinkingLevel::Off)
        .build();

    prov.chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["thinking"]["type"], "disabled");
}
#[tokio::test]
async fn error_403_maps_to_forbidden() {
    let server = MockServer::start().await;
    common::mock_error(
        &server,
        403,
        include_str!("fixtures/openai/error_forbidden.json"),
    )
    .await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::Forbidden));
}
#[tokio::test]
async fn error_402_maps_to_payment_required() {
    let server = MockServer::start().await;
    common::mock_error(
        &server,
        402,
        include_str!("fixtures/openai/error_payment.json"),
    )
    .await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    match err {
        LlmError::Provider { code, .. } => {
            assert_eq!(code, "payment_required");
        }
        other => panic!("expected Provider error, got {other:?}"),
    }
}
#[tokio::test]
async fn error_529_maps_to_overloaded_with_retry_after() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(529)
                .insert_header("retry-after", "30")
                .set_body_raw(
                    include_str!("fixtures/openai/error_overloaded.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    match err {
        LlmError::Overloaded { retry_after } => {
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(30)));
        }
        other => panic!("expected Overloaded, got {other:?}"),
    }
}

#[tokio::test]
async fn error_429_extracts_retry_after_header() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(429)
                .insert_header("retry-after", "120")
                .set_body_raw(
                    include_str!("fixtures/openai/error_rate_limit.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    match err {
        LlmError::RateLimit { retry_after } => {
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(120)));
        }
        other => panic!("expected RateLimit, got {other:?}"),
    }
}
#[tokio::test]
async fn stop_reason_length_maps_to_max_tokens() {
    let server = MockServer::start().await;
    common::mock_chat(
        &server,
        include_str!("fixtures/openai/chat_max_tokens.json"),
    )
    .await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text(
            "Write a very long story.".into(),
        )]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();
    assert!(matches!(resp.stop_reason, StopReason::MaxTokens));
}
#[tokio::test]
async fn stop_reason_content_filter_maps_correctly() {
    let server = MockServer::start().await;
    common::mock_chat(
        &server,
        include_str!("fixtures/openai/chat_content_filter.json"),
    )
    .await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text(
            "Something inappropriate".into(),
        )]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();
    assert!(matches!(resp.stop_reason, StopReason::ContentFilter));
}

#[tokio::test]
async fn image_url_appears_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_simple.json")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![
            RequestContent::Text("What is in this image?".into()),
            RequestContent::Image(ImageSource::Url("https://example.com/cat.png".into())),
        ]))
        .build();

    provider(&server).chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    let content = &body["messages"][0]["content"];
    assert!(content.is_array(), "content should be array for multipart");
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "What is in this image?");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(
        content[1]["image_url"]["url"],
        "https://example.com/cat.png"
    );
}

#[tokio::test]
async fn image_base64_appears_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_simple.json")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Image(
            ImageSource::Base64 {
                media_type: "image/png".into(),
                data: "iVBORw0KGgo=".into(),
            },
        )]))
        .build();

    provider(&server).chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    let content = &body["messages"][0]["content"];
    assert!(content.is_array(), "single image should still be array");
    assert_eq!(content[0]["type"], "image_url");
    assert_eq!(
        content[0]["image_url"]["url"],
        "data:image/png;base64,iVBORw0KGgo="
    );
}

#[tokio::test]
async fn tool_choice_required_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_simple.json")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .tool(ToolDef::new(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object"}),
        ))
        .tool_choice(ToolChoice::Required)
        .build();

    provider(&server).chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["tool_choice"], "required");
}

#[tokio::test]
async fn tool_choice_none_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_simple.json")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .tool(ToolDef::new(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object"}),
        ))
        .tool_choice(ToolChoice::None)
        .build();

    provider(&server).chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["tool_choice"], "none");
}

#[tokio::test]
async fn tool_choice_specific_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_simple.json")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .tool(ToolDef::new(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object"}),
        ))
        .tool_choice(ToolChoice::Specific("get_weather".into()))
        .build();

    provider(&server).chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["tool_choice"]["type"], "function");
    assert_eq!(body["tool_choice"]["function"]["name"], "get_weather");
}

#[tokio::test]
async fn read_timeout_aborts_slow_non_stream_response() {
    let server = MockServer::start().await;
    // Delay the response 500ms; read_timeout is 100ms → must time out.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(500))
                .set_body_raw(
                    include_str!("fixtures/openai/chat_simple.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;

    let provider = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .read_timeout(Duration::from_millis(100))
        .build();

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hello".into())]))
        .build();

    let result = provider.chat(&req).await;
    assert!(result.is_err(), "expected timeout error, got {:?}", result);
    let err = result.unwrap_err();
    assert!(
        matches!(err, LlmError::Timeout),
        "expected LlmError::Timeout, got {err:?}"
    );
}

#[tokio::test]
async fn request_timeout_aborts_slow_non_stream_response() {
    let server = MockServer::start().await;
    // Delay the response 500ms; request_timeout (total) is 100ms → must time out.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(500))
                .set_body_raw(
                    include_str!("fixtures/openai/chat_simple.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;

    let provider = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        // read_timeout is generous so only request_timeout can fire.
        .read_timeout(Duration::from_secs(60))
        .request_timeout(Duration::from_millis(100))
        .build();

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hello".into())]))
        .build();

    let result = provider.chat(&req).await;
    assert!(result.is_err(), "expected timeout error, got {:?}", result);
    let err = result.unwrap_err();
    assert!(
        matches!(err, LlmError::Timeout),
        "expected LlmError::Timeout, got {err:?}"
    );
}

#[tokio::test]
async fn builder_timeout_methods_chain_without_panic() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_simple.json")).await;

    // All three methods chainable; build does not panic.
    let _provider = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(30))
        .request_timeout(Duration::from_secs(120))
        .build();

    // Defaults (no explicit timeout) also build cleanly.
    let _default_provider = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .build();
}
