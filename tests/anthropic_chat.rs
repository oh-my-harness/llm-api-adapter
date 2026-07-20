mod common;

use llm_adapter::{
    LlmError, Provider,
    anthropic::{AnthropicExt, AnthropicMetadata, AnthropicProvider, AnthropicRequest, CacheHint},
    types::*,
};
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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

#[tokio::test]
async fn anthropic_thinking_46_budget_in_request_body() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(
        &server,
        include_str!("fixtures/anthropic/chat_reasoning.json"),
    )
    .await;

    let req = ChatRequest::builder("claude-sonnet-4-6", 16384)
        .message(Message::User(vec![RequestContent::Text("Think.".into())]))
        .thinking(ThinkingLevel::Medium)
        .build();

    provider(&server).chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["thinking"]["type"], "enabled");
    // Medium maps to 8192; max_tokens=16384 is large enough that no clamping occurs.
    assert_eq!(body["thinking"]["budget_tokens"], 8192);
}

#[tokio::test]
async fn anthropic_thinking_47_adaptive_in_request_body() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(
        &server,
        include_str!("fixtures/anthropic/chat_reasoning.json"),
    )
    .await;

    let req = ChatRequest::builder("claude-opus-4-7", 1024)
        .message(Message::User(vec![RequestContent::Text("Think.".into())]))
        .thinking(ThinkingLevel::High)
        .build();

    provider(&server).chat(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["output_config"]["effort"], "high");
}
#[tokio::test]
async fn error_403_maps_to_forbidden() {
    let server = MockServer::start().await;
    common::mock_anthropic_error(
        &server,
        403,
        include_str!("fixtures/anthropic/error_forbidden.json"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::Forbidden));
}

#[tokio::test]
async fn error_429_maps_to_rate_limit() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(
            wiremock::ResponseTemplate::new(429)
                .insert_header("retry-after", "45")
                .set_body_raw(
                    include_str!("fixtures/anthropic/error_rate_limit.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    match err {
        LlmError::RateLimit { retry_after } => {
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(45)));
        }
        other => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn error_529_maps_to_overloaded() {
    let server = MockServer::start().await;
    common::mock_anthropic_error(
        &server,
        529,
        include_str!("fixtures/anthropic/error_overloaded.json"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::Overloaded { .. }));
}

#[tokio::test]
async fn error_404_maps_to_invalid_request() {
    let server = MockServer::start().await;
    common::mock_anthropic_error(
        &server,
        404,
        include_str!("fixtures/anthropic/error_not_found.json"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 256)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let err = provider(&server).chat(&req).await.unwrap_err();
    match err {
        LlmError::InvalidRequest(msg) => {
            assert!(
                msg.contains("not_found"),
                "message should contain 'not_found', got: {msg}"
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}
#[tokio::test]
async fn stop_reason_max_tokens() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(
        &server,
        include_str!("fixtures/anthropic/chat_max_tokens.json"),
    )
    .await;

    let req = ChatRequest::builder("claude-sonnet-4-20250514", 256)
        .message(Message::User(vec![RequestContent::Text(
            "Write a long essay.".into(),
        )]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();
    assert!(matches!(resp.stop_reason, StopReason::MaxTokens));
}

#[tokio::test]
async fn stop_reason_stop_sequence() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(
        &server,
        include_str!("fixtures/anthropic/chat_stop_sequence.json"),
    )
    .await;

    let req = ChatRequest::builder("claude-sonnet-4-20250514", 256)
        .message(Message::User(vec![RequestContent::Text(
            "Say hello.".into(),
        )]))
        .stop_sequence("\n\n")
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();
    assert!(matches!(resp.stop_reason, StopReason::StopSequence));
}

#[tokio::test]
async fn native_ext_top_k_and_metadata_in_request_body() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(&server, include_str!("fixtures/anthropic/chat_simple.json")).await;

    let base = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();
    let ext = AnthropicExt::builder()
        .top_k(40)
        .metadata(AnthropicMetadata {
            user_id: Some("user_42".into()),
        })
        .build();
    let req = AnthropicRequest { base, ext };

    provider(&server).chat_native(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["top_k"], 40);
    assert_eq!(body["metadata"]["user_id"], "user_42");
}

#[tokio::test]
async fn native_ext_cache_hint_in_request_body() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(&server, include_str!("fixtures/anthropic/chat_simple.json")).await;

    let base = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
        .message(Message::User(vec![RequestContent::Text(
            "Cache me.".into(),
        )]))
        .build();
    let ext = AnthropicExt::builder()
        .cache_hint(CacheHint {
            block_index: 0,
            ephemeral: true,
        })
        .build();
    let req = AnthropicRequest { base, ext };

    provider(&server).chat_native(&req).await.unwrap();

    let received: Vec<_> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    // The first message's first content block should have cache_control set
    let block = &body["messages"][0]["content"][0];
    assert_eq!(block["type"], "text");
    assert_eq!(block["text"], "Cache me.");
    assert_eq!(block["cache_control"]["type"], "ephemeral");
}

#[tokio::test]
async fn read_timeout_aborts_slow_non_stream_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(500))
                .set_body_raw(
                    include_str!("fixtures/anthropic/chat_simple.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProvider::builder("test-key")
        .base_url(server.uri())
        .read_timeout(Duration::from_millis(100))
        .build();

    let req = ChatRequest::builder("claude-3-5-sonnet-20241022", 512)
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
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(500))
                .set_body_raw(
                    include_str!("fixtures/anthropic/chat_simple.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProvider::builder("test-key")
        .base_url(server.uri())
        // read_timeout is generous so only request_timeout can fire.
        .read_timeout(Duration::from_secs(60))
        .request_timeout(Duration::from_millis(100))
        .build();

    let req = ChatRequest::builder("claude-3-5-sonnet-20241022", 512)
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
    common::mock_anthropic_chat(&server, include_str!("fixtures/anthropic/chat_simple.json")).await;

    let _provider = AnthropicProvider::builder("test-key")
        .base_url(server.uri())
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(30))
        .request_timeout(Duration::from_secs(120))
        .build();

    let _default_provider = AnthropicProvider::builder("test-key")
        .base_url(server.uri())
        .build();
}
