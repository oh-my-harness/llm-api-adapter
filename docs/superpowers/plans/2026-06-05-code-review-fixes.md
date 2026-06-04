# Code Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all confirmed findings from the 2026-06-04 code review — 2 P1 bugs and 7 P2 improvements.

**Architecture:** Fixes are applied in dependency order: isolated schema tweaks first (P2-7, P2-6), then infrastructure (P2-4, P2-3), then logic fixes (P2-5, P2-1, P2-2), then the two complex P1 bugs (tolerant_keepalive wiring, Anthropic thinking signature round-trip).

**Tech Stack:** Rust, reqwest 0.12, serde_json, tokio, tracing (new dep added in Task 4)

---

## File Map

| File | Change |
|------|--------|
| `src/anthropic/schema.rs` | `media_type: String` → `Option<String>` |
| `src/types/request.rs` | Add `#[non_exhaustive]` to `ToolDef` |
| `src/openai/mod.rs` | Client timeout, extract_retry_after refactor, tolerant_keepalive field + build() |
| `src/anthropic/mod.rs` | Client timeout, extract_retry_after refactor, add public `AnthropicResponse`, change `chat_native()` return type |
| `Cargo.toml` | Add `tracing = "0.1"` dep |
| `src/stream_handle.rs` | Add `tracing::warn!` for ToolDelta JSON fallback; update `Reasoning` construction |
| `src/anthropic/convert.rs` | Delete first pass of `apply_cache_hints`; filter empty content blocks; update `Reasoning` construction + match arms; update `assistant_content_to_wire` to handle signature |
| `src/openai/convert.rs` | Update `Reasoning` match arm and construction |
| `src/openai/stream.rs` | Update `parse_stream` signature to accept `tolerant_keepalive`; skip non-JSON on keepalive |
| `src/types/content.rs` | `Reasoning(String)` → `Reasoning { text: String, signature: Option<String> }` |
| `src/types/response.rs` | Update `reasoning()` helper method |
| `src/lib.rs` | Update test that constructs `Reasoning` |
| `tests/deepseek_chat.rs` | Update `matches!` pattern for Reasoning |
| `tests/deepseek_stream.rs` | Update `matches!` pattern for Reasoning |

---

## Task 1: P2-7 — `AnthropicImageSource::media_type` → `Option<String>`

**Files:**
- Modify: `src/anthropic/schema.rs:100-113`
- Modify: `src/anthropic/convert.rs:376-405`

The `media_type` field is currently `String` (required), so URL-type images send `"media_type": ""` to the wire. Changing to `Option<String>` with `skip_serializing_if = "Option::is_none"` omits it for URL images.

- [ ] **Step 1: Write the failing test**

Add in `src/anthropic/schema.rs` tests block (already has `content_blocks_text_and_blocks_roundtrip`), add a new test:

```rust
#[test]
fn image_source_url_omits_media_type() {
    let source = AnthropicImageSource {
        source_type: "url".into(),
        media_type: None,
        data: None,
        url: Some("https://example.com/img.png".into()),
    };
    let json = serde_json::to_value(&source).unwrap();
    assert!(!json.as_object().unwrap().contains_key("media_type"));
    assert_eq!(json["url"], "https://example.com/img.png");
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test --lib anthropic::schema::tests::image_source_url_omits_media_type 2>&1 | tail -20
```

Expected: compile error — `AnthropicImageSource` has `media_type: String` (required).

- [ ] **Step 3: Update `AnthropicImageSource` in `src/anthropic/schema.rs`**

Change lines 101-113 from:

```rust
pub(crate) struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}
```

To:

```rust
pub(crate) struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}
```

- [ ] **Step 4: Update image conversion in `src/anthropic/convert.rs:376-395`**

Change:

```rust
let (source_type, media_type, data, url) = match img {
    crate::types::ImageSource::Url(u) => {
        ("url".into(), String::new(), None, Some(u.clone()))
    }
    crate::types::ImageSource::Base64 { media_type, data } => (
        "base64".into(),
        media_type.clone(),
        Some(data.clone()),
        None,
    ),
};
blocks.push(AnthropicContentBlock::Image {
    source: AnthropicImageSource {
        source_type,
        media_type,
        data,
        url,
    },
});
```

To:

```rust
let (source_type, media_type, data, url) = match img {
    crate::types::ImageSource::Url(u) => {
        ("url".into(), None, None, Some(u.clone()))
    }
    crate::types::ImageSource::Base64 { media_type, data } => (
        "base64".into(),
        Some(media_type.clone()),
        Some(data.clone()),
        None,
    ),
};
blocks.push(AnthropicContentBlock::Image {
    source: AnthropicImageSource {
        source_type,
        media_type,
        data,
        url,
    },
});
```

- [ ] **Step 5: Update the existing test in `src/anthropic/convert.rs` that checks `source.media_type`**

The test `image_conversion_base64` (near line 996-1019) checks:
```rust
assert_eq!(source.media_type, "image/png");
```

Change to:
```rust
assert_eq!(source.media_type.as_deref(), Some("image/png"));
```

The test `image_conversion_url` checks the url source — it doesn't assert on `media_type` directly, so no change needed there.

Also update the roundtrip test in `src/anthropic/schema.rs` (`content_blocks_text_and_blocks_roundtrip`):
```rust
// Image check (line ~578-582)
AnthropicContentBlock::Image { source } => {
    assert_eq!(source.source_type, "base64");
    assert_eq!(source.media_type, "image/png");  // change this line
```
To:
```rust
    assert_eq!(source.media_type.as_deref(), Some("image/png"));
```

- [ ] **Step 6: Run tests to verify all pass**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/anthropic/schema.rs src/anthropic/convert.rs
git commit -m "fix(anthropic): media_type Option<String> to skip serialization for URL images

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: P2-6 — `#[non_exhaustive]` on `ToolDef`

**Files:**
- Modify: `src/types/request.rs:4-12`

`ToolDef` is the only public constructible type not following the builder pattern. `#[non_exhaustive]` preserves forward-compatibility without requiring a full builder.

- [ ] **Step 1: Add `#[non_exhaustive]` to `ToolDef`**

In `src/types/request.rs`, change:

```rust
/// Definition of a callable tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
```

To:

```rust
/// Definition of a callable tool.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
```

- [ ] **Step 2: Run tests**

```bash
cargo test --lib 2>&1 | tail -10
```

Expected: all pass. `#[non_exhaustive]` only affects external crates doing struct literal construction — internal code still works.

- [ ] **Step 3: Commit**

```bash
git add src/types/request.rs
git commit -m "fix(types): add #[non_exhaustive] to ToolDef for forward-compatibility

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: P2-4 — Default 120s timeout for reqwest Clients

**Files:**
- Modify: `src/openai/mod.rs:188-196`
- Modify: `src/anthropic/mod.rs:147-154`

Without a timeout, a hung provider server causes an infinite wait. 120s is a generous ceiling that covers even the slowest thinking-mode responses.

- [ ] **Step 1: Update `OpenAIProviderBuilder::build()` in `src/openai/mod.rs`**

Change:

```rust
pub fn build(self) -> OpenAIProvider {
    OpenAIProvider {
        client: reqwest::Client::new(),
        api_key: self.api_key,
        base_url: self.base_url,
        chat_path: self.chat_path,
        parse_reasoning_content: self.parse_reasoning_content,
    }
}
```

To:

```rust
pub fn build(self) -> OpenAIProvider {
    OpenAIProvider {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client"),
        api_key: self.api_key,
        base_url: self.base_url,
        chat_path: self.chat_path,
        parse_reasoning_content: self.parse_reasoning_content,
    }
}
```

(`Duration` is already imported via `use std::time::Duration;` at the top of the file.)

- [ ] **Step 2: Update `AnthropicProviderBuilder::build()` in `src/anthropic/mod.rs`**

Change:

```rust
pub fn build(self) -> AnthropicProvider {
    AnthropicProvider {
        client: reqwest::Client::new(),
        api_key: self.api_key,
        base_url: self.base_url,
        messages_path: self.messages_path,
    }
}
```

To:

```rust
pub fn build(self) -> AnthropicProvider {
    AnthropicProvider {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client"),
        api_key: self.api_key,
        base_url: self.base_url,
        messages_path: self.messages_path,
    }
}
```

(`Duration` is already imported via `use std::time::Duration;`.)

- [ ] **Step 3: Run tests**

```bash
cargo test --lib 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/openai/mod.rs src/anthropic/mod.rs
git commit -m "fix: add 120s default timeout to reqwest clients

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: P2-3 — tracing::warn! for ToolDelta JSON parse fallback

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/stream_handle.rs:86`

When `StreamHandle::collect()` encounters non-JSON tool arguments, it silently falls back to `Value::String`. This is invisible to callers. Adding a `tracing::warn!` makes it observable.

- [ ] **Step 1: Add `tracing` to `Cargo.toml`**

In the `[dependencies]` section of `Cargo.toml`, add:

```toml
tracing = "0.1"
```

- [ ] **Step 2: Update `stream_handle.rs` ToolDelta fallback at line 86**

Change:

```rust
let input = serde_json::from_str(&args).unwrap_or(serde_json::Value::String(args));
```

To:

```rust
let input = serde_json::from_str(&args).unwrap_or_else(|_| {
    tracing::warn!(args = %args, "tool delta args not valid JSON; falling back to string value");
    serde_json::Value::String(args)
});
```

Add `use` import at the top of `src/stream_handle.rs` (no change needed — `tracing::warn!` is a macro accessed via its path).

- [ ] **Step 3: Run tests**

```bash
cargo test --lib 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml src/stream_handle.rs
git commit -m "fix(stream): warn on non-JSON tool delta args fallback

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: P2-5 — Filter empty content blocks in `from_anthropic`

**Files:**
- Modify: `src/anthropic/convert.rs:524-530`

Currently `from_anthropic` uses `.map()`, so empty `Text("")` and `Thinking("")` blocks are pushed into content unchanged. Change to `.filter_map()` with an emptiness guard.

- [ ] **Step 1: Write the failing test**

In `src/anthropic/convert.rs` tests block, add:

```rust
#[test]
fn from_anthropic_filters_empty_text_blocks() {
    let resp = AnthropicResponse {
        id: "msg_07".into(),
        response_type: "message".into(),
        role: "assistant".into(),
        model: "claude-sonnet-4-20250514".into(),
        content: vec![
            AnthropicResponseBlock::Text { text: "".into() },
            AnthropicResponseBlock::Text { text: "Hello".into() },
        ],
        stop_reason: Some("end_turn".into()),
        stop_sequence: None,
        usage: AnthropicUsage::default(),
    };
    let canon = from_anthropic(resp).unwrap();
    assert_eq!(canon.content.len(), 1);
    assert_eq!(canon.text(), "Hello");
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test --lib anthropic::convert::tests::from_anthropic_filters_empty_text_blocks 2>&1 | tail -10
```

Expected: FAIL — currently returns 2 content blocks.

- [ ] **Step 3: Change `from_anthropic` to use `filter_map`**

In `src/anthropic/convert.rs`, change `from_anthropic` (around line 524):

```rust
pub(crate) fn from_anthropic(resp: AnthropicResponse) -> Result<ChatResponse, LlmError> {
    let content: Vec<ResponseContent> = resp
        .content
        .into_iter()
        .map(anthropic_response_block_to_canonical)
        .collect();
```

To:

```rust
pub(crate) fn from_anthropic(resp: AnthropicResponse) -> Result<ChatResponse, LlmError> {
    let content: Vec<ResponseContent> = resp
        .content
        .into_iter()
        .filter_map(anthropic_response_block_to_canonical)
        .collect();
```

- [ ] **Step 4: Update `anthropic_response_block_to_canonical` to return `Option<ResponseContent>`**

Change the function signature and add emptiness guards. Keep `Reasoning(...)` with the old tuple syntax for now — Task 9 will update it to the struct form:

```rust
fn anthropic_response_block_to_canonical(block: AnthropicResponseBlock) -> Option<ResponseContent> {
    match block {
        AnthropicResponseBlock::Text { text } if !text.is_empty() => Some(ResponseContent::Text(text)),
        AnthropicResponseBlock::Text { .. } => None,
        AnthropicResponseBlock::ToolUse { id, name, input } => {
            Some(ResponseContent::ToolInvocation(ToolInvocation { id, name, input }))
        }
        AnthropicResponseBlock::Thinking { thinking, signature: _ } if !thinking.is_empty() => {
            Some(ResponseContent::Reasoning(thinking))
        }
        AnthropicResponseBlock::Thinking { .. } => None,
        AnthropicResponseBlock::RedactedThinking { data } => {
            Some(ResponseContent::Reasoning(format!("[redacted_thinking: {data}]")))
        }
    }
}
```

- [ ] **Step 5: Run tests to verify all pass**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/anthropic/convert.rs
git commit -m "fix(anthropic): filter empty text/thinking blocks from responses

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: P2-1 — Delete dead first pass in `apply_cache_hints`

**Files:**
- Modify: `src/anthropic/convert.rs:438-491`

The first loop (lines 443-461) iterates messages and blocks but contains only a comment — no actual operations. It sets `flat_index` but then `flat_index` is re-initialized to 0 before the second pass anyway.

- [ ] **Step 1: Delete the first loop**

In `src/anthropic/convert.rs`, delete the entire first loop including the comment and the first `flat_index` declaration. The function should go from:

```rust
fn apply_cache_hints(
    mut messages: Vec<super::schema::AnthropicWireMessage>,
    hints: &[CacheHint],
) -> Vec<super::schema::AnthropicWireMessage> {
    // Build a flat index across all content blocks in all messages.
    let mut flat_index = 0usize;
    for msg in &mut messages {
        match &msg.content {
            AnthropicContentBlocks::Blocks(blocks) => {
                for _block in blocks {
                    for hint in hints {
                        if hint.block_index == flat_index && hint.ephemeral {
                            // We need to mutate the block — we can't borrow mutably here
                            // since blocks is behind a shared ref. Collect the info and do it in a second pass.
                        }
                    }
                    flat_index += 1;
                }
            }
            AnthropicContentBlocks::Text(_) => {
                flat_index += 1;
            }
        }
    }

    // Second pass: actually apply hints.
    let mut flat_index = 0usize;
```

To:

```rust
fn apply_cache_hints(
    mut messages: Vec<super::schema::AnthropicWireMessage>,
    hints: &[CacheHint],
) -> Vec<super::schema::AnthropicWireMessage> {
    let mut flat_index = 0usize;
```

The rest of the function (the second pass loop) stays unchanged.

- [ ] **Step 2: Run tests**

```bash
cargo test --lib 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add src/anthropic/convert.rs
git commit -m "refactor(anthropic): remove dead first pass in apply_cache_hints

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: P2-2 — Fix `extract_retry_after` to read HTTP headers

**Files:**
- Modify: `src/openai/mod.rs:44-107`
- Modify: `src/anthropic/mod.rs:46-104`

Currently `response.text().await?` consumes the response body before headers are accessible. The `Retry-After` header is in HTTP headers, not the body. Refactor `send()` to capture `Retry-After` from headers before consuming the body.

- [ ] **Step 1: Add free function `extract_retry_after_header` in `src/openai/mod.rs`**

Add this function above `impl OpenAIProvider` (or at the bottom of the file, outside any `impl` block):

```rust
/// Parse `Retry-After` header value to a Duration.
///
/// Supports both RFC 7231 formats:
/// - Seconds: `Retry-After: 30`
/// - HTTP-date: `Retry-After: Wed, 21 Oct 2015 07:28:00 GMT`  (returns None — date parsing not worth the dep)
fn extract_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers.get(reqwest::header::RETRY_AFTER)?;
    let s = value.to_str().ok()?;
    // Try seconds format first.
    if let Ok(secs) = s.trim().parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    // HTTP-date format: skip (would require chrono/httpdate dep; None is safe fallback).
    None
}
```

- [ ] **Step 2: Refactor `OpenAIProvider::send()` to capture headers before reading body**

In `src/openai/mod.rs`, change the error handling block in `send()`:

```rust
if !status.is_success() {
    // Try to parse structured error response
    let error_text = response.text().await?;
    ...
    return match status.as_u16() {
        ...
        429 => {
            let retry_after = self.extract_retry_after(&error_text);
            Err(LlmError::RateLimit { retry_after })
        }
        ...
        529 => {
            let retry_after = self.extract_retry_after(&error_text);
            Err(LlmError::Overloaded { retry_after })
        }
```

To:

```rust
if !status.is_success() {
    let retry_after = extract_retry_after_header(response.headers());
    let error_text = response.text().await?;
    ...
    return match status.as_u16() {
        ...
        429 => Err(LlmError::RateLimit { retry_after }),
        ...
        529 => Err(LlmError::Overloaded { retry_after }),
```

- [ ] **Step 3: Remove the now-unused `extract_retry_after` method from `OpenAIProvider`**

Delete the `extract_retry_after` method (lines 102-107):

```rust
/// Extract retry-after delay from error text (if present).
fn extract_retry_after(&self, _error_text: &str) -> Option<Duration> {
    // TODO: Parse retry-after header from actual response headers
    // For now, return None
    None
}
```

- [ ] **Step 4: Do the same refactor for `AnthropicProvider` in `src/anthropic/mod.rs`**

Add `extract_retry_after_header` free function (same implementation as OpenAI — put at module level):

```rust
fn extract_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers.get(reqwest::header::RETRY_AFTER)?;
    let s = value.to_str().ok()?;
    if let Ok(secs) = s.trim().parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    None
}
```

Refactor `AnthropicProvider::send()` error block:

```rust
// Before (extract_retry_after called with error text):
429 => {
    let retry_after = Self::extract_retry_after(&error_text);
    Err(LlmError::RateLimit { retry_after })
}
...
529 => {
    let retry_after = Self::extract_retry_after(&error_text);
    Err(LlmError::Overloaded { retry_after })
}
```

To:

```rust
// After (retry_after captured before body consumed):
// (at top of error block, before `let error_text = ...`)
let retry_after = extract_retry_after_header(response.headers());
let error_text = response.text().await?;
...
429 => Err(LlmError::RateLimit { retry_after }),
...
529 => Err(LlmError::Overloaded { retry_after }),
```

Remove the old `extract_retry_after` associated function from `AnthropicProvider`:

```rust
fn extract_retry_after(_error_text: &str) -> Option<Duration> {
    // TODO: Parse retry-after header from actual response headers
    None
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test --lib 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/openai/mod.rs src/anthropic/mod.rs
git commit -m "fix: extract Retry-After from HTTP headers before consuming response body

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 8: P1-1 — Wire `tolerant_keepalive` through to stream parser

**Files:**
- Modify: `src/openai/mod.rs:15-21` (struct), `188-196` (build), `125-133` (chat_stream_native)
- Modify: `src/openai/stream.rs:234` (parse_stream signature + non-JSON skip)

DeepSeek may send non-JSON SSE data lines (keepalive messages) during thinking-mode streaming. `tolerant_keepalive` exists in the builder and `OpenAIExt`, but was never stored in `OpenAIProvider` or passed to `parse_stream`. This fix wires it through.

- [ ] **Step 1: Add `tolerant_keepalive` field to `OpenAIProvider` struct**

In `src/openai/mod.rs`, change:

```rust
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    chat_path: String,
    parse_reasoning_content: bool,
}
```

To:

```rust
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    chat_path: String,
    parse_reasoning_content: bool,
    tolerant_keepalive: bool,
}
```

- [ ] **Step 2: Transfer `tolerant_keepalive` in `build()`**

In `OpenAIProviderBuilder::build()`, add `tolerant_keepalive: self.tolerant_keepalive` to the `OpenAIProvider` struct literal:

```rust
pub fn build(self) -> OpenAIProvider {
    OpenAIProvider {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client"),
        api_key: self.api_key,
        base_url: self.base_url,
        chat_path: self.chat_path,
        parse_reasoning_content: self.parse_reasoning_content,
        tolerant_keepalive: self.tolerant_keepalive,
    }
}
```

- [ ] **Step 3: Pass `tolerant_keepalive` from `chat_stream_native` to `parse_stream`**

In `chat_stream_native` (around line 125-133), change:

```rust
let stream = stream::parse_stream(response, req.ext.parse_reasoning_content);
```

To:

```rust
let tolerant = req.ext.tolerant_keepalive || self.tolerant_keepalive;
let stream = stream::parse_stream(response, req.ext.parse_reasoning_content, tolerant);
```

Also update the `Provider::chat_stream` implementation (which calls `chat_stream_native`). Inside `chat_stream`, the `OpenAIRequest` is built with default ext, so `tolerant_keepalive` in the ext is `false` — the provider-level field is used via the `tolerant` variable above. This is already handled by the `||` expression above.

- [ ] **Step 4: Update `parse_stream` signature and logic in `src/openai/stream.rs`**

Change the function signature:

```rust
pub fn parse_stream(
    response: reqwest::Response,
    parse_reasoning: bool,
    tolerant_keepalive: bool,
) -> futures::stream::BoxStream<'static, Result<StreamEvent, LlmError>> {
```

Inside the closure, change the JSON parse error branch:

```rust
Err(e) => {
    results.push(Err(LlmError::Decode(e)));
}
```

To:

```rust
Err(e) => {
    if tolerant_keepalive {
        // Non-JSON data line (e.g. DeepSeek keepalive); skip silently.
    } else {
        results.push(Err(LlmError::Decode(e)));
    }
}
```

- [ ] **Step 5: Write a unit test for tolerant_keepalive behavior**

In `src/openai/stream.rs`, inside the tests module (or add one if absent), add:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StreamEvent;

    #[test]
    fn process_chunk_handles_basic_text() {
        let mut state = OaiStreamState::new();
        let chunk = super::super::schema::OaiChunk {
            id: "c1".into(),
            model: "gpt-4o".into(),
            choices: vec![super::super::schema::OaiStreamChoice {
                delta: super::super::schema::OaiDelta {
                    content: Some("hello".into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let events = process_chunk(&mut state, chunk, false);
        assert_eq!(events.len(), 2); // ContentStart + TextDelta
    }
}
```

(This is a smoke-test; the actual tolerant_keepalive path is tested via the parse_stream integration path which requires mocking HTTP, so defer that to integration tests.)

- [ ] **Step 6: Run tests**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/openai/mod.rs src/openai/stream.rs
git commit -m "fix(openai): wire tolerant_keepalive through to parse_stream

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 9: P1-2 — Preserve Anthropic thinking signature for multi-turn

This is the most complex change. It involves:
1. Extending `ResponseContent::Reasoning` to carry an optional signature
2. Updating all code that constructs or matches on `Reasoning`
3. Adding a public `AnthropicResponse` type
4. Changing `chat_native()` return type
5. Updating `assistant_content_to_wire` to emit proper `Thinking` blocks

**Files:**
- Modify: `src/types/content.rs:38`
- Modify: `src/types/response.rs:82-90`
- Modify: `src/lib.rs:37`
- Modify: `src/stream_handle.rs:82`
- Modify: `src/openai/convert.rs:276, 407`
- Modify: `src/anthropic/convert.rs:340, 544-561` (assistant_content_to_wire, anthropic_response_block_to_canonical)
- Modify: `src/anthropic/convert.rs:795-798, 931-934` (tests)
- Modify: `src/anthropic/mod.rs` (add AnthropicResponse, change chat_native return type)
- Modify: `tests/deepseek_chat.rs:55`
- Modify: `tests/deepseek_stream.rs:38`

### Sub-task 9a: Extend `ResponseContent::Reasoning` with optional signature

- [ ] **Step 1: Write failing test in `src/lib.rs`**

Add to `src/lib.rs` tests:

```rust
#[test]
fn reasoning_content_carries_signature() {
    use types::ResponseContent;
    let with_sig = ResponseContent::Reasoning {
        text: "thinking".into(),
        signature: Some("sig123".into()),
    };
    let without_sig = ResponseContent::Reasoning {
        text: "thinking".into(),
        signature: None,
    };
    // Ensure both construct without error
    assert!(matches!(with_sig, ResponseContent::Reasoning { .. }));
    assert!(matches!(without_sig, ResponseContent::Reasoning { .. }));
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test --lib lib::tests::reasoning_content_carries_signature 2>&1 | tail -10
```

Expected: compile error — `Reasoning` is a tuple variant, not a struct variant.

- [ ] **Step 3: Change `ResponseContent::Reasoning` in `src/types/content.rs`**

Change line 38:

```rust
    /// Chain-of-thought reasoning text (read-only; not all providers support multi-turn with this).
    Reasoning(String),
```

To:

```rust
    /// Chain-of-thought reasoning text.
    /// `signature` is Anthropic-only; `None` for all other providers.
    /// Pass via `AnthropicResponse::into_message()` to enable multi-turn thinking.
    Reasoning { text: String, signature: Option<String> },
```

- [ ] **Step 4: Update `ChatResponse::reasoning()` in `src/types/response.rs`**

Change lines 82-90:

```rust
pub fn reasoning(&self) -> Option<&str> {
    self.content.iter().find_map(|c| {
        if let ResponseContent::Reasoning(r) = c {
            Some(r.as_str())
        } else {
            None
        }
    })
}
```

To:

```rust
pub fn reasoning(&self) -> Option<&str> {
    self.content.iter().find_map(|c| {
        if let ResponseContent::Reasoning { text, .. } = c {
            Some(text.as_str())
        } else {
            None
        }
    })
}
```

- [ ] **Step 5: Update `src/lib.rs` existing test construction**

Change line 37:

```rust
let _reasoning = ResponseContent::Reasoning("my reasoning".into());
```

To:

```rust
let _reasoning = ResponseContent::Reasoning { text: "my reasoning".into(), signature: None };
```

- [ ] **Step 6: Update `StreamHandle::collect()` in `src/stream_handle.rs:82`**

Change:

```rust
for (idx, text) in reasoning_blocks {
    all.push((idx, ResponseContent::Reasoning(text)));
}
```

To:

```rust
for (idx, text) in reasoning_blocks {
    all.push((idx, ResponseContent::Reasoning { text, signature: None }));
}
```

- [ ] **Step 7: Update OpenAI `from_oai` in `src/openai/convert.rs:407`**

Change:

```rust
content.push(ResponseContent::Reasoning(reasoning));
```

To:

```rust
content.push(ResponseContent::Reasoning { text: reasoning, signature: None });
```

- [ ] **Step 8: Update OpenAI `assistant_content_to_oai` match arm in `src/openai/convert.rs:276`**

Change:

```rust
ResponseContent::Reasoning(r) => {
    reasoning_text = Some(r.clone());
}
```

To:

```rust
ResponseContent::Reasoning { text: r, .. } => {
    reasoning_text = Some(r.clone());
}
```

- [ ] **Step 9: Update Anthropic `assistant_content_to_wire` in `src/anthropic/convert.rs:340`**

Change the current fallback (which encodes reasoning as text):

```rust
ResponseContent::Reasoning(r) => {
    // Anthropic does not accept reasoning in requests; encode as text
    // for round-trip compatibility.
    blocks.push(AnthropicContentBlock::Text {
        text: r.clone(),
        cache_control: None,
    });
}
```

To (encode as proper `Thinking` block when signature present; fall back to text otherwise):

```rust
ResponseContent::Reasoning { text, signature: Some(sig) } => {
    // Re-encode as Thinking block with its signature for multi-turn compatibility.
    blocks.push(AnthropicContentBlock::Thinking {
        thinking: text.clone(),
        signature: sig.clone(),
    });
}
ResponseContent::Reasoning { text, signature: None } => {
    // No signature available; encode as text (single-turn or non-native path).
    blocks.push(AnthropicContentBlock::Text {
        text: text.clone(),
        cache_control: None,
    });
}
```

- [ ] **Step 10: Update `anthropic_response_block_to_canonical` in `src/anthropic/convert.rs:544-561`**

Change (from Task 5, which had `Option<ResponseContent>` return):

```rust
AnthropicResponseBlock::Thinking { thinking, signature: _ } if !thinking.is_empty() => {
    Some(ResponseContent::Reasoning(thinking))
}
AnthropicResponseBlock::Thinking { .. } => None,
AnthropicResponseBlock::RedactedThinking { data } => {
    Some(ResponseContent::Reasoning(format!("[redacted_thinking: {data}]")))
}
```

To:

```rust
AnthropicResponseBlock::Thinking { thinking, signature } if !thinking.is_empty() => {
    let sig = if signature.is_empty() { None } else { Some(signature) };
    Some(ResponseContent::Reasoning { text: thinking, signature: sig })
}
AnthropicResponseBlock::Thinking { .. } => None,
AnthropicResponseBlock::RedactedThinking { data } => {
    Some(ResponseContent::Reasoning {
        text: format!("[redacted_thinking: {data}]"),
        signature: None,
    })
}
```

- [ ] **Step 11: Update tests in `src/anthropic/convert.rs` that match on `Reasoning`**

Near line 795-798, change:

```rust
match &canon.content[0] {
    ResponseContent::Reasoning(r) => assert_eq!(r, "Let me reason here..."),
    other => panic!("expected Reasoning, got {other:?}"),
}
```

To:

```rust
match &canon.content[0] {
    ResponseContent::Reasoning { text, signature } => {
        assert_eq!(text, "Let me reason here...");
        assert_eq!(signature.as_deref(), Some("sig_abc123"));
    }
    other => panic!("expected Reasoning, got {other:?}"),
}
```

Near line 931-934, change:

```rust
match &canon.content[0] {
    ResponseContent::Reasoning(r) => {
        assert_eq!(r, "[redacted_thinking: sensitive]");
    }
    other => panic!("expected Reasoning, got {other:?}"),
}
```

To:

```rust
match &canon.content[0] {
    ResponseContent::Reasoning { text, signature } => {
        assert_eq!(text, "[redacted_thinking: sensitive]");
        assert!(signature.is_none());
    }
    other => panic!("expected Reasoning, got {other:?}"),
}
```

- [ ] **Step 12: Update integration test patterns in `tests/deepseek_chat.rs` and `tests/deepseek_stream.rs`**

In `tests/deepseek_chat.rs:55`, change:

```rust
.position(|c| matches!(c, ResponseContent::Reasoning(_)))
```

To:

```rust
.position(|c| matches!(c, ResponseContent::Reasoning { .. }))
```

In `tests/deepseek_stream.rs:38`, change:

```rust
.position(|c| matches!(c, ResponseContent::Reasoning(_)))
```

To:

```rust
.position(|c| matches!(c, ResponseContent::Reasoning { .. }))
```

Also check `tests/integration_deepseek.rs:289` — same pattern, same change.

- [ ] **Step 13: Run tests to verify all compile and pass**

```bash
cargo test --lib 2>&1 | tail -30
```

Then:

```bash
./test.sh 2>&1 | tail -30
```

Expected: all pass.

### Sub-task 9b: Add public `AnthropicResponse` and change `chat_native()` return type

- [ ] **Step 14: Write failing test**

In `src/anthropic/mod.rs` (or `src/anthropic/convert.rs` tests), add a test that calls `chat_native()` and uses the new type:

```rust
// This is better as a doc-test style; add to the anthropic mod.rs or convert.rs unit tests.
// We'll verify via compile + type-checking.
```

Since `chat_native` makes real HTTP calls, we rely on the existing fixture tests to verify behavior. The key correctness property to test is `into_message()` preserving signatures. Add in `src/anthropic/convert.rs` tests:

```rust
#[test]
fn anthropic_response_into_message_preserves_signature() {
    use crate::anthropic::AnthropicNativeResponse;
    let resp = AnthropicResponse {
        id: "msg_02".into(),
        response_type: "message".into(),
        role: "assistant".into(),
        model: "claude-sonnet-4-20250514".into(),
        content: vec![
            AnthropicResponseBlock::Thinking {
                thinking: "Let me think...".into(),
                signature: "sig_abc".into(),
            },
            AnthropicResponseBlock::Text { text: "Answer.".into() },
        ],
        stop_reason: Some("end_turn".into()),
        stop_sequence: None,
        usage: AnthropicUsage::default(),
    };
    let native = AnthropicNativeResponse::from_wire(resp).unwrap();
    let msg = native.into_message();
    match &msg {
        crate::types::Message::Assistant(content) => {
            match &content[0] {
                ResponseContent::Reasoning { text, signature } => {
                    assert_eq!(text, "Let me think...");
                    assert_eq!(signature.as_deref(), Some("sig_abc"));
                }
                other => panic!("expected Reasoning, got {other:?}"),
            }
        }
        other => panic!("expected Assistant, got {other:?}"),
    }
}
```

- [ ] **Step 15: Declare `AnthropicNativeResponse` in `src/anthropic/mod.rs`**

Add a new public struct above `impl AnthropicProvider`:

```rust
/// The Anthropic-native non-streaming response, returned by `AnthropicProvider::chat_native()`.
///
/// Use `into_canonical()` to get a `ChatResponse` (thinking signatures dropped).
/// Use `into_message()` to get a `Message::Assistant` with signatures preserved,
/// suitable for multi-turn extended thinking conversations.
pub struct AnthropicNativeResponse {
    inner: crate::types::ChatResponse,
}

impl AnthropicNativeResponse {
    /// Build from the internal wire response (called only by `chat_native`).
    pub(crate) fn from_wire(
        wire: super::anthropic::schema::AnthropicResponse,
    ) -> Result<Self, LlmError> {
        Ok(Self {
            inner: convert::from_anthropic(wire)?,
        })
    }

    /// Convert to a `ChatResponse`, dropping thinking signatures.
    pub fn into_canonical(self) -> crate::types::ChatResponse {
        use crate::types::ResponseContent;
        let content = self
            .inner
            .content
            .into_iter()
            .map(|c| match c {
                ResponseContent::Reasoning { text, .. } => {
                    ResponseContent::Reasoning { text, signature: None }
                }
                other => other,
            })
            .collect();
        crate::types::ChatResponse {
            content,
            ..self.inner
        }
    }

    /// Convert to `Message::Assistant`, preserving thinking signatures for multi-turn.
    pub fn into_message(self) -> crate::types::Message {
        crate::types::Message::Assistant(self.inner.content)
    }

    /// Access the underlying `ChatResponse` (signatures preserved as-is).
    pub fn as_response(&self) -> &crate::types::ChatResponse {
        &self.inner
    }
}
```

Note: The `from_wire` method requires `schema::AnthropicResponse`. Since `schema` is `pub(crate)`, this is fine. But the path `super::anthropic::schema::AnthropicResponse` is wrong from within `src/anthropic/mod.rs` — it should be `schema::AnthropicResponse`. Fix the path in the actual implementation.

The full correct implementation in `src/anthropic/mod.rs`:

```rust
pub struct AnthropicNativeResponse {
    inner: crate::types::ChatResponse,
}

impl AnthropicNativeResponse {
    pub(crate) fn from_wire(wire: schema::AnthropicResponse) -> Result<Self, LlmError> {
        Ok(Self {
            inner: convert::from_anthropic(wire)?,
        })
    }

    pub fn into_canonical(self) -> crate::types::ChatResponse {
        use crate::types::ResponseContent;
        let content = self
            .inner
            .content
            .into_iter()
            .map(|c| match c {
                ResponseContent::Reasoning { text, .. } => {
                    ResponseContent::Reasoning { text, signature: None }
                }
                other => other,
            })
            .collect();
        crate::types::ChatResponse {
            content,
            ..self.inner
        }
    }

    pub fn into_message(self) -> crate::types::Message {
        crate::types::Message::Assistant(self.inner.content)
    }

    pub fn as_response(&self) -> &crate::types::ChatResponse {
        &self.inner
    }
}
```

- [ ] **Step 16: Change `chat_native()` to return `Result<AnthropicNativeResponse, LlmError>`**

In `src/anthropic/mod.rs`, change:

```rust
pub async fn chat_native(&self, req: &AnthropicRequest) -> Result<ChatResponse, LlmError> {
    let wire_req = convert::to_anthropic(&req.base, Some(&req.ext), false)?;
    let response = self.send(&wire_req, false).await?;
    let wire_resp = response.json::<schema::AnthropicResponse>().await?;
    convert::from_anthropic(wire_resp)
}
```

To:

```rust
pub async fn chat_native(
    &self,
    req: &AnthropicRequest,
) -> Result<AnthropicNativeResponse, LlmError> {
    let wire_req = convert::to_anthropic(&req.base, Some(&req.ext), false)?;
    let response = self.send(&wire_req, false).await?;
    let wire_resp = response.json::<schema::AnthropicResponse>().await?;
    AnthropicNativeResponse::from_wire(wire_resp)
}
```

- [ ] **Step 17: Update `Provider::chat` implementation to call `into_canonical()`**

In the `impl Provider for AnthropicProvider`, `chat()` calls `self.chat_native(...).await`:

```rust
async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
    let anthropic_req = AnthropicRequest {
        base: req.clone(),
        ext: AnthropicExt::default(),
    };
    self.chat_native(&anthropic_req).await
}
```

Change to:

```rust
async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
    let anthropic_req = AnthropicRequest {
        base: req.clone(),
        ext: AnthropicExt::default(),
    };
    self.chat_native(&anthropic_req).await.map(AnthropicNativeResponse::into_canonical)
}
```

- [ ] **Step 18: Export `AnthropicNativeResponse` from `src/anthropic/mod.rs`**

The type is defined in `src/anthropic/mod.rs` and already `pub`. Since `src/lib.rs` does `pub mod anthropic;`, callers can access it as `llm_adapter::anthropic::AnthropicNativeResponse`. No additional re-export needed.

- [ ] **Step 19: Run all tests**

```bash
cargo test --lib 2>&1 | tail -30
./test.sh 2>&1 | tail -30
```

Expected: all pass.

- [ ] **Step 20: Commit**

```bash
git add src/types/content.rs src/types/response.rs src/lib.rs \
        src/stream_handle.rs src/openai/convert.rs \
        src/anthropic/convert.rs src/anthropic/mod.rs \
        tests/deepseek_chat.rs tests/deepseek_stream.rs tests/integration_deepseek.rs
git commit -m "feat(anthropic): preserve thinking signature for multi-turn extended thinking

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final Verification

- [ ] **Run the full test suite**

```bash
./test.sh 2>&1
```

Expected: all offline tests pass.

- [ ] **Check clippy**

```bash
cargo clippy --all-targets 2>&1 | grep "^error" | head -20
```

Expected: zero errors.

- [ ] **Check format**

```bash
cargo fmt -- --check 2>&1
```

Expected: no diff.

- [ ] **Run examples to verify nothing is broken**

```bash
# These require API keys; run if .env is populated
cargo build --examples 2>&1 | tail -5
```

Expected: compile success.
