use std::time::Duration;

use llm_adapter::openai::{
    OpenAIProvider,
    responses::{
        AttemptState, OpenAIResponsesCachePolicy, OpenAIResponsesConversationState,
        OpenAIResponsesReasoning, OpenAIResponsesReasoningContext, OpenAIResponsesReasoningEffort,
        OpenAIResponsesReasoningMode, OpenAIResponsesResponseControls, OpenAIResponsesServiceTier,
        OpenAIResponsesText, OpenAIResponsesTextVerbosity, OpenAIResponsesTokenProjection,
        OpenAIResponsesToolChoice, OpenAIResponsesTruncation, PreparedOpenAIResponse,
        ResponsesTerminalStatus,
    },
};
use llm_adapter::{UsageProvenance, openai::responses::OpenAIResponsesError};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn projection() -> OpenAIResponsesTokenProjection {
    OpenAIResponsesTokenProjection::builder(
        "gpt-5.6-terra",
        "Return one strict JSON policy mutation.",
        json!([{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Repair the public fixture."}]
        }]),
    )
    .reasoning(OpenAIResponsesReasoning::new(
        OpenAIResponsesReasoningEffort::Medium,
        OpenAIResponsesReasoningContext::CurrentTurn,
        OpenAIResponsesReasoningMode::Standard,
    ))
    .text(OpenAIResponsesText::json_schema(
        "policy_mutation",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["op", "path", "value"],
            "properties": {
                "op": {"const": "replace"},
                "path": {"type": "string"},
                "value": {}
            }
        }),
        true,
        OpenAIResponsesTextVerbosity::Medium,
    ))
    .tool_choice(OpenAIResponsesToolChoice::None)
    .cache_policy(OpenAIResponsesCachePolicy::Explicit)
    .conversation_state(OpenAIResponsesConversationState::Fresh)
    .truncation(OpenAIResponsesTruncation::Disabled)
    .build()
    .expect("canonical projection")
}

fn prepared(stream: bool) -> PreparedOpenAIResponse {
    let controls = OpenAIResponsesResponseControls::builder(1_200)
        .stream(stream)
        .store(false)
        .background(false)
        .service_tier(OpenAIResponsesServiceTier::Default)
        .build()
        .expect("bounded controls");
    PreparedOpenAIResponse::new(projection(), controls).expect("prepared response")
}

fn provider(server: &MockServer) -> OpenAIProvider {
    OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .build()
}

async fn mount_json(server: &MockServer, path_value: &str, body: &'static str, request_id: &str) {
    Mock::given(method("POST"))
        .and(path(path_value))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("x-request-id", request_id)
                .set_body_raw(body, "application/json"),
        )
        .mount(server)
        .await;
}

async fn mount_sse(server: &MockServer, body: &'static str, request_id: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("x-request-id", request_id)
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(server)
        .await;
}

fn completed_response_value() -> Value {
    serde_json::from_str(include_str!("fixtures/openai/responses_completed.json"))
        .expect("valid completed response fixture")
}

async fn mount_json_value(server: &MockServer, path_value: &str, body: &Value, request_id: &str) {
    Mock::given(method("POST"))
        .and(path(path_value))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("x-request-id", request_id)
                .set_body_json(body),
        )
        .mount(server)
        .await;
}

#[tokio::test]
async fn count_and_response_are_derived_from_one_immutable_projection() {
    let server = MockServer::start().await;
    mount_json(
        &server,
        "/v1/responses/input_tokens",
        include_str!("fixtures/openai/responses_input_tokens.json"),
        "req_count_1",
    )
    .await;
    mount_json(
        &server,
        "/v1/responses",
        include_str!("fixtures/openai/responses_completed.json"),
        "req_response_1",
    )
    .await;

    let call = prepared(false);
    let projection_digest = call.projection_digest().to_owned();
    let count = provider(&server)
        .count_response_input_tokens(&call)
        .await
        .expect("count receipt");
    let response = provider(&server)
        .create_response(&call)
        .await
        .expect("response receipt");

    assert_eq!(count.input_tokens(), 321);
    assert_eq!(count.schema_version(), response.receipt().schema_version());
    assert_eq!(count.projection_digest(), projection_digest);
    assert_eq!(count.request_id(), "req_count_1");
    assert_eq!(response.receipt().projection_digest(), projection_digest);
    assert_eq!(response.receipt().request_id(), "req_response_1");

    let requests = server.received_requests().await.expect("request log");
    assert_eq!(requests.len(), 2);
    let count_body: Value = serde_json::from_slice(&requests[0].body).expect("count body");
    let response_body: Value = serde_json::from_slice(&requests[1].body).expect("response body");

    for field in [
        "model",
        "instructions",
        "input",
        "reasoning",
        "text",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "prompt_cache_options",
        "truncation",
    ] {
        assert_eq!(count_body[field], response_body[field], "field {field}");
    }
    assert!(count_body.get("max_output_tokens").is_none());
    assert!(count_body.get("stream").is_none());
    assert_eq!(response_body["max_output_tokens"], 1_200);
    assert_eq!(response_body["stream"], false);
    assert_eq!(response_body["store"], false);
    assert_eq!(response_body["background"], false);
    assert_eq!(response_body["service_tier"], "default");
    assert!(!call.controls().store());
    assert!(!call.controls().background());
    assert_eq!(
        call.controls().service_tier(),
        OpenAIResponsesServiceTier::Default
    );
    assert!(call.controls().include().is_empty());
    assert!(!call.controls().has_implicit_retry());
    assert_eq!(response_body["tools"], json!([]));
    assert_eq!(response_body["tool_choice"], "none");
    assert!(response_body.get("max_tool_calls").is_none());
    assert!(response_body.get("prompt_cache_key").is_none());
    assert!(response_body.get("previous_response_id").is_none());
    assert!(response_body.get("conversation").is_none());
}

#[tokio::test]
async fn tampered_projection_is_rejected_before_http() {
    let server = MockServer::start().await;
    let call = prepared(false);
    let mut wire = serde_json::to_value(&call).expect("serialize prepared call");
    wire["projection"]["instructions"] = json!("tampered instruction");
    let forged: PreparedOpenAIResponse = serde_json::from_value(wire).expect("typed forged call");

    let error = provider(&server)
        .create_response(&forged)
        .await
        .expect_err("digest mismatch must fail");

    assert!(matches!(
        error,
        OpenAIResponsesError::InvalidPreparedCall(_)
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn tampered_response_controls_are_rejected_before_http() {
    let server = MockServer::start().await;
    let call = prepared(false);
    let mut wire = serde_json::to_value(&call).expect("serialize prepared call");
    wire["controls"]["max_output_tokens"] = json!(9_999);
    let forged: PreparedOpenAIResponse = serde_json::from_value(wire).expect("typed forged call");

    let error = provider(&server)
        .create_response(&forged)
        .await
        .expect_err("control digest mismatch must fail");

    assert!(matches!(
        error,
        OpenAIResponsesError::InvalidPreparedCall(_)
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn non_stream_receipt_binds_identity_and_authoritative_usage() {
    let server = MockServer::start().await;
    mount_json(
        &server,
        "/v1/responses",
        include_str!("fixtures/openai/responses_completed.json"),
        "req_response_2",
    )
    .await;

    let result = provider(&server)
        .create_response(&prepared(false))
        .await
        .expect("response");
    let receipt = result.receipt();

    assert_eq!(
        result.output_text(),
        r#"{"op":"replace","path":"/fallback","value":"safe"}"#
    );
    assert_eq!(receipt.provider_identity().provider(), "openai");
    assert_eq!(receipt.provider_identity().base_url(), server.uri());
    assert_eq!(receipt.requested_model(), "gpt-5.6-terra");
    assert_eq!(receipt.returned_model(), "gpt-5.6-terra-2026-07-15");
    assert_eq!(receipt.response_id(), "resp_fixture_completed");
    assert_eq!(
        receipt.terminal_status(),
        ResponsesTerminalStatus::Completed
    );
    assert_eq!(receipt.usage().provenance, UsageProvenance::ReportedValid);
    assert_eq!(receipt.usage().input_tokens, 1_000);
    assert_eq!(receipt.usage().cached_input_tokens, 400);
    assert_eq!(receipt.usage().cache_creation_input_tokens, 100);
    assert_eq!(receipt.usage().output_tokens, 500);
    assert_eq!(receipt.usage().reasoning_tokens, 200);
    assert!(receipt.observed_tool_events().is_empty());
}

#[tokio::test]
async fn reported_zero_missing_and_invalid_usage_remain_distinct() {
    for (fixture, expected) in [
        (
            include_str!("fixtures/openai/responses_zero_usage.json"),
            UsageProvenance::ReportedValid,
        ),
        (
            include_str!("fixtures/openai/responses_missing_usage.json"),
            UsageProvenance::Missing,
        ),
        (
            include_str!("fixtures/openai/responses_invalid_usage.json"),
            UsageProvenance::ReportedInvalid,
        ),
    ] {
        let server = MockServer::start().await;
        mount_json(&server, "/v1/responses", fixture, "req_usage").await;
        let result = provider(&server)
            .create_response(&prepared(false))
            .await
            .expect("terminal response remains observable");
        assert_eq!(result.receipt().usage().provenance, expected);
    }
}

#[tokio::test]
async fn malformed_usage_detail_is_reported_invalid_not_coerced_to_zero() {
    let server = MockServer::start().await;
    mount_json(
        &server,
        "/v1/responses",
        include_str!("fixtures/openai/responses_malformed_usage_detail.json"),
        "req_bad_detail",
    )
    .await;

    let result = provider(&server)
        .create_response(&prepared(false))
        .await
        .expect("completed call retains invalid usage evidence");

    assert_eq!(
        result.receipt().usage().provenance,
        UsageProvenance::ReportedInvalid
    );
}

#[tokio::test]
async fn streaming_terminal_is_accounted_once_and_does_not_wait_for_done() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        include_str!("fixtures/openai/responses_stream_completed.txt"),
        "req_stream_1",
    )
    .await;

    let result = provider(&server)
        .create_response(&prepared(true))
        .await
        .expect("stream terminal");

    assert_eq!(result.receipt().terminal_sequence_number(), Some(3));
    assert_eq!(result.receipt().usage().input_tokens, 1_000);
    assert_eq!(
        result.receipt().usage().provenance,
        UsageProvenance::ReportedValid
    );
    assert_eq!(
        result.output_text(),
        r#"{"op":"replace","path":"/fallback","value":"safe"}"#
    );
}

#[tokio::test]
async fn first_stream_terminal_wins_even_when_later_terminal_is_valid() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        include_str!("fixtures/openai/responses_stream_invalid_then_valid.txt"),
        "req_stream_2",
    )
    .await;

    let result = provider(&server)
        .create_response(&prepared(true))
        .await
        .expect("first terminal is preserved");

    assert_eq!(result.receipt().response_id(), "resp_stream_invalid");
    assert_eq!(result.receipt().terminal_sequence_number(), Some(1));
    assert_eq!(
        result.receipt().usage().provenance,
        UsageProvenance::ReportedInvalid
    );
}

#[tokio::test]
async fn stream_records_tool_observation_even_with_end_turn_terminal() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        include_str!("fixtures/openai/responses_stream_tool_event.txt"),
        "req_stream_tool",
    )
    .await;

    let result = provider(&server)
        .create_response(&prepared(true))
        .await
        .expect("tool-bearing terminal remains evidence");
    let tools = result.receipt().observed_tool_events();

    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].item_type(), "function_call");
    assert_eq!(tools[0].call_id(), Some("call_fixture_1"));
    assert_eq!(tools[0].name(), Some("forbidden_tool"));
}

#[tokio::test]
async fn failed_stream_retains_tool_observation_in_attempt_receipt() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        include_str!("fixtures/openai/responses_stream_tool_then_error.txt"),
        "req_stream_tool_error",
    )
    .await;

    let error = provider(&server)
        .create_response(&prepared(true))
        .await
        .expect_err("stream error must fail");

    let tools = error.attempt().observed_tool_events();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].item_type(), "function_call");
    assert_eq!(tools[0].call_id(), Some("call_before_error"));
}

#[tokio::test]
async fn failed_stream_retains_tool_delta_without_an_output_item() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        include_str!("fixtures/openai/responses_stream_tool_delta_then_error.txt"),
        "req_stream_tool_delta_error",
    )
    .await;

    let error = provider(&server)
        .create_response(&prepared(true))
        .await
        .expect_err("stream error must fail");

    let tools = error.attempt().observed_tool_events();
    assert_eq!(tools.len(), 1);
    assert_eq!(
        tools[0].item_type(),
        "response.function_call_arguments.delta"
    );
    assert_eq!(tools[0].item_id(), Some("fc_delta_only"));
}

#[tokio::test]
async fn missing_request_id_fails_closed_but_preserves_terminal_accounting() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            include_str!("fixtures/openai/responses_completed.json"),
            "application/json",
        ))
        .mount(&server)
        .await;

    let error = provider(&server)
        .create_response(&prepared(false))
        .await
        .expect_err("trusted receipt requires x-request-id");

    assert_eq!(error.attempt().state(), AttemptState::TerminalObserved);
    assert_eq!(
        error
            .attempt()
            .terminal_usage()
            .expect("known terminal usage")
            .provenance,
        UsageProvenance::ReportedValid
    );
    assert_eq!(
        error.attempt().response_id(),
        Some("resp_fixture_completed")
    );
}

#[tokio::test]
async fn timeout_is_indeterminate_and_never_retried() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(250))
                .append_header("x-request-id", "req_too_late")
                .set_body_raw(
                    include_str!("fixtures/openai/responses_completed.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;
    let provider = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .request_timeout(Duration::from_millis(25))
        .build();

    let error = provider
        .create_response(&prepared(false))
        .await
        .expect_err("request must time out");

    assert_eq!(error.attempt().state(), AttemptState::OutcomeUnknown);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn response_create_never_follows_redirects() {
    for status in [301, 302, 307, 308] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(status).append_header("location", "/redirected-responses"),
            )
            .mount(&server)
            .await;
        mount_json(
            &server,
            "/redirected-responses",
            include_str!("fixtures/openai/responses_completed.json"),
            "req_redirect_target",
        )
        .await;

        let error = provider(&server)
            .create_response(&prepared(false))
            .await
            .expect_err("Responses transport must reject redirects");

        assert_eq!(error.attempt().state(), AttemptState::HttpRejected);
        assert_eq!(error.attempt().status_code(), Some(status));
        let requests = server.received_requests().await.expect("request log");
        assert_eq!(
            requests.len(),
            1,
            "status {status} caused an implicit second HTTP attempt"
        );
        assert_eq!(requests[0].url.path(), "/v1/responses");
    }
}

#[tokio::test]
async fn response_create_never_follows_redirect_chains() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(307).append_header("location", "/redirect-middle"))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/redirect-middle"))
        .respond_with(ResponseTemplate::new(308).append_header("location", "/redirect-terminal"))
        .mount(&server)
        .await;
    mount_json(
        &server,
        "/redirect-terminal",
        include_str!("fixtures/openai/responses_completed.json"),
        "req_redirect_chain",
    )
    .await;

    provider(&server)
        .create_response(&prepared(false))
        .await
        .expect_err("Responses transport must reject the first redirect");

    let requests = server.received_requests().await.expect("request log");
    assert_eq!(requests.len(), 1, "redirect chain must not be traversed");
    assert_eq!(requests[0].url.path(), "/v1/responses");
}

#[tokio::test]
async fn input_token_count_never_follows_redirects() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses/input_tokens"))
        .respond_with(ResponseTemplate::new(307).append_header("location", "/redirected-count"))
        .mount(&server)
        .await;
    mount_json(
        &server,
        "/redirected-count",
        include_str!("fixtures/openai/responses_input_tokens.json"),
        "req_redirected_count",
    )
    .await;

    let error = provider(&server)
        .count_response_input_tokens(&prepared(false))
        .await
        .expect_err("count transport must reject redirects");

    assert_eq!(error.attempt().state(), AttemptState::HttpRejected);
    assert_eq!(error.attempt().status_code(), Some(307));
    let requests = server.received_requests().await.expect("request log");
    assert_eq!(requests.len(), 1, "count endpoint was retried by redirect");
    assert_eq!(requests[0].url.path(), "/v1/responses/input_tokens");
}

#[tokio::test]
async fn malformed_non_stream_output_preserves_terminal_accounting() {
    let server = MockServer::start().await;
    let mut response = completed_response_value();
    response["output"] = json!({"not": "an array"});
    mount_json_value(&server, "/v1/responses", &response, "req_bad_output").await;

    let error = provider(&server)
        .create_response(&prepared(false))
        .await
        .expect_err("malformed terminal output must fail");

    assert_eq!(error.attempt().state(), AttemptState::TerminalObserved);
    assert_eq!(
        error.attempt().response_id(),
        Some("resp_fixture_completed")
    );
    assert_eq!(
        error.attempt().returned_model(),
        Some("gpt-5.6-terra-2026-07-15")
    );
    assert_eq!(
        error
            .attempt()
            .terminal_usage()
            .expect("known terminal usage")
            .provenance,
        UsageProvenance::ReportedValid
    );
}

#[tokio::test]
async fn malformed_stream_output_preserves_terminal_accounting() {
    let server = MockServer::start().await;
    let mut response = completed_response_value();
    response["output"] = json!({"not": "an array"});
    let body = format!(
        "event: response.completed\ndata: {}\n\n",
        json!({
            "type": "response.completed",
            "sequence_number": 7,
            "response": response,
        })
    );
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("x-request-id", "req_bad_stream_output")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let error = provider(&server)
        .create_response(&prepared(true))
        .await
        .expect_err("malformed stream terminal output must fail");

    assert_eq!(error.attempt().state(), AttemptState::TerminalObserved);
    assert_eq!(
        error.attempt().response_id(),
        Some("resp_fixture_completed")
    );
    assert_eq!(
        error
            .attempt()
            .terminal_usage()
            .expect("known terminal usage")
            .provenance,
        UsageProvenance::ReportedValid
    );
}

#[tokio::test]
async fn completed_terminal_with_provider_error_is_rejected() {
    let server = MockServer::start().await;
    let mut response = completed_response_value();
    response["error"] = json!({
        "code": "provider_internal_error",
        "message": "the response failed after completion"
    });
    mount_json_value(
        &server,
        "/v1/responses",
        &response,
        "req_contradictory_terminal",
    )
    .await;

    let error = provider(&server)
        .create_response(&prepared(false))
        .await
        .expect_err("completed response cannot also contain a provider error");

    assert_eq!(error.attempt().state(), AttemptState::TerminalObserved);
    assert_eq!(
        error.attempt().response_id(),
        Some("resp_fixture_completed")
    );
    assert_eq!(
        error
            .attempt()
            .terminal_usage()
            .expect("known terminal usage")
            .provenance,
        UsageProvenance::ReportedValid
    );
}
