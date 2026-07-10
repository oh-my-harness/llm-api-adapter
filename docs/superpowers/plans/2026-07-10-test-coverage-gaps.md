# Test Coverage Gap Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six highest-priority test coverage gaps identified in the llm-api-adapter test audit: HTTP error code mapping, StopReason mapping, OpenAI image conversion, OpenAI tool_choice mapping, DeepSeek stream parity, and native-path ext field verification.

**Architecture:** All tests are offline wiremock-based integration tests or inline `#[cfg(test)]` unit tests, following existing patterns in `tests/` and `src/**/*.rs`. No production code changes — only test additions and fixture creation.

**Tech Stack:** Rust 2024, wiremock 0.6, serde_json, tokio::test, existing `tests/common/mod.rs` helpers.

## Global Constraints

- Run all tests via `./test.sh` (sets `NO_PROXY=127.0.0.1,localhost` and loads `.env`).
- No `--features integration` — all tests must be offline.
- Follow existing fixture naming: `tests/fixtures/<provider>/<type>_<scenario>.{json,txt}`.
- Follow existing test file naming: `tests/<provider>_<type>.rs`.
- Use `mod common;` in test crates that need shared helpers.
- Inline unit tests go in the `#[cfg(test)] mod tests` block of the source file being tested.
- Each test crate uses `#[allow(dead_code)]` on common helpers it doesn't call — this is already handled in `tests/common/mod.rs`.
- Fixture JSON should be hand-crafted (not recorded) to keep it minimal and targeted.
- No new dependencies.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `tests/openai_chat.rs` | Add 403, 402, 529, Retry-After, StopReason (length/stop/content_filter), image content, tool_choice tests |
| `tests/anthropic_chat.rs` | Add 403, 429, 529, 404, Retry-After, StopReason (max_tokens/stop_sequence), native ext fields tests |
| `tests/deepseek_stream.rs` | Add stream events order, stream tool call, stream no-usage fallback tests |
| `tests/openai_stream.rs` | Add StopReason via stream test (length) |
| `tests/fixtures/openai/error_forbidden.json` | 403 error body |
| `tests/fixtures/openai/error_payment.json` | 402 error body |
| `tests/fixtures/openai/error_overloaded.json` | 529 error body |
| `tests/fixtures/openai/error_rate_limit_retry_after.json` | Not needed — reuse `error_rate_limit.json` with custom mock |
| `tests/fixtures/openai/chat_max_tokens.json` | Response with `finish_reason: "length"` |
| `tests/fixtures/openai/chat_content_filter.json` | Response with `finish_reason: "content_filter"` |
| `tests/fixtures/openai/chat_stop_sequence.json` | Response with `finish_reason: "stop"` (already have this in chat_simple, but a dedicated fixture is clearer) |
| `tests/fixtures/anthropic/error_forbidden.json` | 403 error body |
| `tests/fixtures/anthropic/error_overloaded.json` | 529 error body |
| `tests/fixtures/anthropic/error_not_found.json` | 404 error body |
| `tests/fixtures/anthropic/chat_max_tokens.json` | Response with `stop_reason: "max_tokens"` |
| `tests/fixtures/anthropic/chat_stop_sequence.json` | Response with `stop_reason: "stop_sequence"` |
| `src/openai/convert.rs` | Add unit tests for `tool_choice_to_oai` and `req_content_to_part` (image) |
| `src/anthropic/convert.rs` | Add unit test for `into_canonical` signature stripping |

---

### Task 1: OpenAI HTTP error code mapping — 403 Forbidden

**Files:**
- Create: `tests/fixtures/openai/error_forbidden.json`
- Modify: `tests/openai_chat.rs` (append new test)

**Interfaces:**
- Consumes: `common::mock_error(server, status, body)`, `OpenAIProvider::builder()`, `Provider::chat()`
- Produces: None

- [ ] **Step 1: Create the fixture**

```json
{"error": {"message": "You do not have access to this model", "type": "permission_error", "code": "forbidden"}}
```

Save to `tests/fixtures/openai/error_forbidden.json`.

- [ ] **Step 2: Write the failing test**

Append to `tests/openai_chat.rs`:

```rust
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
```

- [ ] **Step 3: Run test to verify it passes**

Run: `./test.sh --test openai_chat -- error_403_maps_to_forbidden`
Expected: PASS (the mapping already exists in `src/openai/mod.rs:149`)

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/openai/error_forbidden.json tests/openai_chat.rs
git commit -m "test(openai): 403 maps to Forbidden"
```

---

### Task 2: OpenAI HTTP error code mapping — 402 Payment Required

**Files:**
- Create: `tests/fixtures/openai/error_payment.json`
- Modify: `tests/openai_chat.rs`

**Interfaces:**
- Consumes: `common::mock_error`, `Provider::chat`
- Produces: None

- [ ] **Step 1: Create the fixture**

```json
{"error": {"message": "Insufficient balance", "type": "billing_error", "code": "payment_required"}}
```

Save to `tests/fixtures/openai/error_payment.json`.

- [ ] **Step 2: Write the test**

Append to `tests/openai_chat.rs`:

```rust
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
```

- [ ] **Step 3: Run test to verify it passes**

Run: `./test.sh --test openai_chat -- error_402_maps_to_payment_required`
Expected: PASS (mapping at `src/openai/mod.rs:150-156`)

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/openai/error_payment.json tests/openai_chat.rs
git commit -m "test(openai): 402 maps to Provider{payment_required}"
```

---

### Task 3: OpenAI HTTP error code mapping — 529 Overloaded + Retry-After

**Files:**
- Create: `tests/fixtures/openai/error_overloaded.json`
- Modify: `tests/openai_chat.rs`

**Interfaces:**
- Consumes: `MockServer`, `wiremock::Mock`, `Provider::chat`
- Produces: None

- [ ] **Step 1: Create the fixture**

```json
{"error": {"message": "The server is overloaded", "type": "server_error", "code": "overloaded"}}
```

Save to `tests/fixtures/openai/error_overloaded.json`.

- [ ] **Step 2: Write the test**

Append to `tests/openai_chat.rs`:

```rust
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
```

- [ ] **Step 3: Run test to verify it passes**

Run: `./test.sh --test openai_chat -- error_529_maps_to_overloaded_with_retry_after`
Expected: PASS (mapping at `src/openai/mod.rs:159`)

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/openai/error_overloaded.json tests/openai_chat.rs
git commit -m "test(openai): 529 maps to Overloaded with retry_after"
```

---

### Task 4: OpenAI Retry-After header on 429 RateLimit

**Files:**
- Modify: `tests/openai_chat.rs`

**Interfaces:**
- Consumes: `MockServer`, `wiremock::Mock`, `Provider::chat`
- Produces: None

- [ ] **Step 1: Write the test**

Append to `tests/openai_chat.rs`:

```rust
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `./test.sh --test openai_chat -- error_429_extracts_retry_after_header`
Expected: PASS (the existing `error_429_maps_to_rate_limit` test only checks the variant; this one checks the `retry_after` value)

- [ ] **Step 3: Commit**

```bash
git add tests/openai_chat.rs
git commit -m "test(openai): 429 extracts retry_after header value"
```

---

### Task 5: OpenAI StopReason — MaxTokens (length)

**Files:**
- Create: `tests/fixtures/openai/chat_max_tokens.json`
- Modify: `tests/openai_chat.rs`

**Interfaces:**
- Consumes: `common::mock_chat`, `Provider::chat`
- Produces: None

- [ ] **Step 1: Create the fixture**

```json
{"id":"max-tokens-001","object":"chat.completion","created":1780566139,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"This is a partial response that was cut"},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":512,"total_tokens":522}}
```

Save to `tests/fixtures/openai/chat_max_tokens.json`.

- [ ] **Step 2: Write the test**

Append to `tests/openai_chat.rs`:

```rust
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
```

- [ ] **Step 3: Run test to verify it passes**

Run: `./test.sh --test openai_chat -- stop_reason_length_maps_to_max_tokens`
Expected: PASS (mapping at `src/openai/convert.rs:530`)

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/openai/chat_max_tokens.json tests/openai_chat.rs
git commit -m "test(openai): finish_reason length maps to MaxTokens"
```

---

### Task 6: OpenAI StopReason — ContentFilter

**Files:**
- Create: `tests/fixtures/openai/chat_content_filter.json`
- Modify: `tests/openai_chat.rs`

**Interfaces:**
- Consumes: `common::mock_chat`, `Provider::chat`
- Produces: None

- [ ] **Step 1: Create the fixture**

```json
{"id":"filter-001","object":"chat.completion","created":1780566139,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":""},"logprobs":null,"finish_reason":"content_filter"}],"usage":{"prompt_tokens":5,"completion_tokens":0,"total_tokens":5}}
```

Save to `tests/fixtures/openai/chat_content_filter.json`.

- [ ] **Step 2: Write the test**

Append to `tests/openai_chat.rs`:

```rust
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
```

- [ ] **Step 3: Run test to verify it passes**

Run: `./test.sh --test openai_chat -- stop_reason_content_filter_maps_correctly`
Expected: PASS (mapping at `src/openai/convert.rs:532`)

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/openai/chat_content_filter.json tests/openai_chat.rs
git commit -m "test(openai): finish_reason content_filter maps to ContentFilter"
```

---

### Task 7: OpenAI image content — URL and Base64 in request body

**Files:**
- Create: `tests/fixtures/openai/chat_simple.json` (reuse existing)
- Modify: `tests/openai_chat.rs`

**Interfaces:**
- Consumes: `common::mock_chat`, `Provider::chat`, `ImageSource`, `RequestContent::Image`
- Produces: None

- [ ] **Step 1: Write the test for image URL**

Append to `tests/openai_chat.rs`:

```rust
#[tokio::test]
async fn image_url_appears_in_request_body() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/openai/chat_simple.json")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![
            RequestContent::Text("What is in this image?".into()),
            RequestContent::Image(ImageSource::Url(
                "https://example.com/cat.png".into(),
            )),
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `./test.sh --test openai_chat -- image_url_appears_in_request_body`
Expected: PASS (conversion at `src/openai/convert.rs:416-432`)

- [ ] **Step 3: Write the test for image base64**

Append to `tests/openai_chat.rs`:

```rust
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./test.sh --test openai_chat -- image_base64_appears_in_request_body`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/openai_chat.rs
git commit -m "test(openai): image URL and base64 appear in request body"
```

---

### Task 8: OpenAI tool_choice — None, Required, Specific in request body

**Files:**
- Modify: `tests/openai_chat.rs`

**Interfaces:**
- Consumes: `common::mock_chat`, `Provider::chat`, `ToolDef`, `ToolChoice`
- Produces: None

- [ ] **Step 1: Write the test**

Append to `tests/openai_chat.rs`:

```rust
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `./test.sh --test openai_chat -- tool_choice_required_in_request_body tool_choice_none_in_request_body tool_choice_specific_in_request_body`
Expected: PASS (mapping at `src/openai/convert.rs:434-447`)

- [ ] **Step 3: Commit**

```bash
git add tests/openai_chat.rs
git commit -m "test(openai): tool_choice None/Required/Specific in request body"
```

---

### Task 9: OpenAI convert unit tests — tool_choice_to_oai and req_content_to_part

**Files:**
- Modify: `src/openai/convert.rs` (append to `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `tool_choice_to_oai`, `req_content_to_part`, `ToolChoice`, `RequestContent`, `ImageSource`, `OaiContentPart`
- Produces: None

- [ ] **Step 1: Write the unit tests**

Append to the `mod tests` block in `src/openai/convert.rs` (after the existing `oai_effort_budget_boundaries` test):

```rust
    #[test]
    fn tool_choice_to_oai_all_variants() {
        assert_eq!(
            tool_choice_to_oai(&ToolChoice::Auto),
            serde_json::Value::String("auto".into())
        );
        assert_eq!(
            tool_choice_to_oai(&ToolChoice::None),
            serde_json::Value::String("none".into())
        );
        assert_eq!(
            tool_choice_to_oai(&ToolChoice::Required),
            serde_json::Value::String("required".into())
        );
        let specific = tool_choice_to_oai(&ToolChoice::Specific("my_tool".into()));
        assert_eq!(specific["type"], "function");
        assert_eq!(specific["function"]["name"], "my_tool");
    }

    #[test]
    fn req_content_to_part_image_url() {
        let part = req_content_to_part(&RC::Image(ImageSource::Url(
            "https://example.com/img.png".into(),
        )))
        .unwrap();
        match part {
            OaiContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/img.png");
            }
            other => panic!("expected ImageUrl, got {other:?}"),
        }
    }

    #[test]
    fn req_content_to_part_image_base64() {
        let part = req_content_to_part(&RC::Image(ImageSource::Base64 {
            media_type: "image/png".into(),
            data: "iVBORw0KGgo=".into(),
        }))
        .unwrap();
        match part {
            OaiContentPart::ImageUrl { image_url } => {
                assert_eq!(
                    image_url.url,
                    "data:image/png;base64,iVBORw0KGgo="
                );
            }
            other => panic!("expected ImageUrl, got {other:?}"),
        }
    }
```

Note: You need to add `ImageSource` to the `use crate::types::` import at the top of the test module if not already present. The existing import is:

```rust
    use crate::types::{
        Message, RequestContent as RC, ResponseContent as RespC, ToolInvocation as TI,
    };
```

Change to:

```rust
    use crate::types::{
        ImageSource, Message, RequestContent as RC, ResponseContent as RespC,
        ToolInvocation as TI,
    };
```

Also add `ToolChoice` to the import (it's in `crate::types`):

```rust
    use crate::types::{
        ImageSource, Message, RequestContent as RC, ResponseContent as RespC,
        ToolChoice, ToolInvocation as TI,
    };
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `./test.sh --lib -- openai::convert::tests::tool_choice_to_oai_all_variants openai::convert::tests::req_content_to_part_image_url openai::convert::tests::req_content_to_part_image_base64`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/openai/convert.rs
git commit -m "test(openai): unit tests for tool_choice_to_oai and req_content_to_part"
```

---

### Task 10: Anthropic HTTP error code mapping — 403, 429, 529, 404

**Files:**
- Create: `tests/fixtures/anthropic/error_forbidden.json`
- Create: `tests/fixtures/anthropic/error_overloaded.json`
- Create: `tests/fixtures/anthropic/error_not_found.json`
- Modify: `tests/anthropic_chat.rs`

**Interfaces:**
- Consumes: `common::mock_anthropic_error`, `Provider::chat`
- Produces: None

- [ ] **Step 1: Create fixtures**

`tests/fixtures/anthropic/error_forbidden.json`:
```json
{"type":"error","error":{"type":"forbidden","message":"Access denied"}}
```

`tests/fixtures/anthropic/error_overloaded.json`:
```json
{"type":"error","error":{"type":"overloaded_error","message":"Server overloaded"}}
```

`tests/fixtures/anthropic/error_not_found.json`:
```json
{"type":"error","error":{"type":"not_found_error","message":"Resource not found"}}
```

- [ ] **Step 2: Write the tests**

Append to `tests/anthropic_chat.rs`:

```rust
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
            assert!(msg.contains("not_found"), "message should contain 'not_found', got: {msg}");
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `./test.sh --test anthropic_chat -- error_403_maps_to_forbidden error_429_maps_to_rate_limit error_529_maps_to_overloaded error_404_maps_to_invalid_request`
Expected: PASS (mappings at `src/anthropic/mod.rs:169-180`)

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/anthropic/error_forbidden.json tests/fixtures/anthropic/error_overloaded.json tests/fixtures/anthropic/error_not_found.json tests/anthropic_chat.rs
git commit -m "test(anthropic): 403/429/529/404 error code mappings"
```

---

### Task 11: Anthropic StopReason — MaxTokens and StopSequence

**Files:**
- Create: `tests/fixtures/anthropic/chat_max_tokens.json`
- Create: `tests/fixtures/anthropic/chat_stop_sequence.json`
- Modify: `tests/anthropic_chat.rs`

**Interfaces:**
- Consumes: `common::mock_anthropic_chat`, `Provider::chat`
- Produces: None

- [ ] **Step 1: Create fixtures**

`tests/fixtures/anthropic/chat_max_tokens.json`:
```json
{"id":"msg_max_001","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"This response was truncated because"}],"stop_reason":"max_tokens","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":256,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}
```

`tests/fixtures/anthropic/chat_stop_sequence.json`:
```json
{"id":"msg_stop_001","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"Hello world"}],"stop_reason":"stop_sequence","stop_sequence":"\n\n","usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}
```

- [ ] **Step 2: Write the tests**

Append to `tests/anthropic_chat.rs`:

```rust
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
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `./test.sh --test anthropic_chat -- stop_reason_max_tokens stop_reason_stop_sequence`
Expected: PASS (mapping at `src/anthropic/convert.rs:778-779`)

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/anthropic/chat_max_tokens.json tests/fixtures/anthropic/chat_stop_sequence.json tests/anthropic_chat.rs
git commit -m "test(anthropic): stop_reason max_tokens and stop_sequence mappings"
```

---

### Task 12: Anthropic native path — top_k, metadata, cache_hints in request body

**Files:**
- Modify: `tests/anthropic_chat.rs`

**Interfaces:**
- Consumes: `common::mock_anthropic_chat`, `AnthropicProvider::chat_native`, `AnthropicRequest`, `AnthropicExt`, `AnthropicExtBuilder`, `AnthropicMetadata`, `CacheHint`, `ThinkingConfig`
- Produces: None

- [ ] **Step 1: Write the test for top_k and metadata**

Append to `tests/anthropic_chat.rs`. First, update the existing import at line 3 to include the native-path types:

```rust
// Change line 3 from:
//   use llm_adapter::{LlmError, Provider, anthropic::AnthropicProvider, types::*};
// To:
use llm_adapter::{
    LlmError, Provider,
    anthropic::{AnthropicExt, AnthropicMetadata, AnthropicProvider, AnthropicRequest, CacheHint},
    types::*,
};
```

Then append the test. This test uses the native path with `AnthropicExt` fields set, then inspects the wiremock received request body:

```rust
#[tokio::test]
async fn native_ext_top_k_and_metadata_in_request_body() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(
        &server,
        include_str!("fixtures/anthropic/chat_simple.json"),
    )
    .await;

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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `./test.sh --test anthropic_chat -- native_ext_top_k_and_metadata_in_request_body`
Expected: PASS

- [ ] **Step 3: Write the test for cache_hints**

Append to `tests/anthropic_chat.rs`:

```rust
#[tokio::test]
async fn native_ext_cache_hint_in_request_body() {
    let server = MockServer::start().await;
    common::mock_anthropic_chat(
        &server,
        include_str!("fixtures/anthropic/chat_simple.json"),
    )
    .await;

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./test.sh --test anthropic_chat -- native_ext_cache_hint_in_request_body`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/anthropic_chat.rs
git commit -m "test(anthropic): native ext top_k, metadata, cache_hints in request body"
```

---

### Task 13: AnthropicNativeResponse::into_canonical strips signature

**Files:**
- Modify: `src/anthropic/convert.rs` (append to `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `AnthropicNativeResponse::from_wire`, `AnthropicNativeResponse::into_canonical`, `AnthropicResponse`, `AnthropicResponseBlock::Thinking`
- Produces: None

- [ ] **Step 1: Write the unit test**

Append to the `mod tests` block in `src/anthropic/convert.rs`, after the existing `anthropic_native_response_into_message_preserves_signature` test:

```rust
    #[test]
    fn anthropic_native_response_into_canonical_strips_signature() {
        use crate::anthropic::AnthropicNativeResponse;
        let resp = AnthropicResponse {
            id: "msg_sig2".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![
                AnthropicResponseBlock::Thinking {
                    thinking: "Let me think...".into(),
                    signature: "sig_xyz".into(),
                },
                AnthropicResponseBlock::Text {
                    text: "Answer.".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        };
        let native = AnthropicNativeResponse::from_wire(resp).unwrap();
        let canon = native.into_canonical();
        match &canon.content[0] {
            ResponseContent::Reasoning { text, signature } => {
                assert_eq!(text, "Let me think...");
                assert!(signature.is_none(), "into_canonical must strip signature");
            }
            other => panic!("expected Reasoning, got {other:?}"),
        }
    }
```

- [ ] **Step 2: Run test to verify it passes**

Run: `./test.sh --lib -- anthropic::convert::tests::anthropic_native_response_into_canonical_strips_signature`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/anthropic/convert.rs
git commit -m "test(anthropic): into_canonical strips thinking signature"
```

---

### Task 14: DeepSeek stream — events fire in order

**Files:**
- Modify: `tests/deepseek_stream.rs`

**Interfaces:**
- Consumes: `common::mock_stream`, `Provider::chat_stream`, `StreamHandle::events`, `StreamEvent`, `ContentKind`
- Produces: None

- [ ] **Step 1: Write the test**

Append to `tests/deepseek_stream.rs`:

```rust
use futures::StreamExt;
use llm_adapter::types::StreamEvent;

#[tokio::test]
async fn deepseek_stream_events_fire_in_order() {
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

    let mut handle = provider(&server).chat_stream(&req).await.unwrap();
    let events = handle.events();

    let mut content_starts = 0usize;
    let mut text_deltas = 0usize;
    let mut reasoning_deltas = 0usize;
    let mut content_stops = 0usize;
    let mut message_stops = 0usize;

    while let Some(event) = events.next().await {
        match event.unwrap() {
            StreamEvent::ContentStart { .. } => content_starts += 1,
            StreamEvent::TextDelta { .. } => text_deltas += 1,
            StreamEvent::ReasoningDelta { .. } => reasoning_deltas += 1,
            StreamEvent::ContentStop { .. } => content_stops += 1,
            StreamEvent::MessageStop { .. } => message_stops += 1,
        }
    }

    // The stream_reasoning fixture has reasoning then text: at least 2 blocks
    assert!(content_starts >= 2, "expected at least 2 ContentStart events");
    assert!(reasoning_deltas >= 1, "expected at least 1 ReasoningDelta");
    assert!(text_deltas >= 1, "expected at least 1 TextDelta");
    assert_eq!(content_stops, content_starts, "ContentStop count should match ContentStart count");
    assert_eq!(message_stops, 1, "expected exactly 1 MessageStop");
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `./test.sh --test deepseek_stream -- deepseek_stream_events_fire_in_order`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/deepseek_stream.rs
git commit -m "test(deepseek): stream events fire in order"
```

---

### Task 15: DeepSeek stream — tool call collects invocation

**Files:**
- Create: `tests/fixtures/deepseek/stream_tool_call.txt`
- Modify: `tests/deepseek_stream.rs`

**Interfaces:**
- Consumes: `common::mock_stream`, `Provider::chat_stream`, `StreamHandle::collect`, `ToolDef`
- Produces: None

- [ ] **Step 1: Create the fixture**

This is an OpenAI-compatible SSE stream with a tool call. Save to `tests/fixtures/deepseek/stream_tool_call.txt`:

```
data: {"id":"chatcmpl-ds-tc","object":"chat.completion.chunk","created":1780566200,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_ds01","type":"function","function":{"name":"get_weather","arguments":""}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-ds-tc","object":"chat.completion.chunk","created":1780566200,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-ds-tc","object":"chat.completion.chunk","created":1780566200,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-ds-tc","object":"chat.completion.chunk","created":1780566200,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":15,"completion_tokens":10,"total_tokens":25}}

data: [DONE]

```

- [ ] **Step 2: Write the test**

Append to `tests/deepseek_stream.rs`:

```rust
#[tokio::test]
async fn deepseek_stream_tool_call_collects_invocation() {
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/deepseek/stream_tool_call.txt"),
    )
    .await;

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text(
            "Weather in Paris?".into(),
        )]))
        .tool(ToolDef::new(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object"}),
        ))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
    assert_eq!(invocations[0].input["location"], "Paris");
}
```

- [ ] **Step 3: Run test to verify it passes**

Run: `./test.sh --test deepseek_stream -- deepseek_stream_tool_call_collects_invocation`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/deepseek/stream_tool_call.txt tests/deepseek_stream.rs
git commit -m "test(deepseek): stream tool call collects invocation"
```

---

### Task 16: OpenAI stream — StopReason MaxTokens via stream

**Files:**
- Create: `tests/fixtures/openai/stream_max_tokens.txt`
- Modify: `tests/openai_stream.rs`

**Interfaces:**
- Consumes: `common::mock_stream`, `Provider::chat_stream`, `StreamHandle::collect`
- Produces: None

- [ ] **Step 1: Create the fixture**

Save to `tests/fixtures/openai/stream_max_tokens.txt`:

```
data: {"id":"stream-max-001","object":"chat.completion.chunk","created":1780566200,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"stream-max-001","object":"chat.completion.chunk","created":1780566200,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Partial text"},"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"stream-max-001","object":"chat.completion.chunk","created":1780566200,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":""},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":512,"total_tokens":517}}

data: [DONE]

```

- [ ] **Step 2: Write the test**

Append to `tests/openai_stream.rs`:

```rust
#[tokio::test]
async fn stream_stop_reason_length_maps_to_max_tokens() {
    let server = MockServer::start().await;
    common::mock_stream(
        &server,
        include_str!("fixtures/openai/stream_max_tokens.txt"),
    )
    .await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text(
            "Write a very long story.".into(),
        )]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(matches!(resp.stop_reason, StopReason::MaxTokens));
}
```

- [ ] **Step 3: Run test to verify it passes**

Run: `./test.sh --test openai_stream -- stream_stop_reason_length_maps_to_max_tokens`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/openai/stream_max_tokens.txt tests/openai_stream.rs
git commit -m "test(openai): stream finish_reason length maps to MaxTokens"
```

---

### Task 17: Run full test suite and verify all pass

**Files:**
- None (verification only)

- [ ] **Step 1: Run full offline test suite**

Run: `./test.sh`
Expected: All tests pass, 0 failures. The test count should increase from ~123 to ~145+.

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --all-targets --features integration 2>&1 | tail -5`
Expected: No warnings on new test code

- [ ] **Step 3: Run fmt check**

Run: `cargo fmt -- --check`
Expected: No formatting issues

- [ ] **Step 4: Commit any formatting fixes if needed**

```bash
git add -A
git commit -m "test: formatting fixes from coverage gap remediation"
```
