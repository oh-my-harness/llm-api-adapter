# Structured Output + Provider Capabilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add canonical `ResponseFormat` to `ChatRequest` for JSON schema/object output, and add a `ProviderCapabilities` API so callers can query what a given provider supports at runtime.

**Architecture:**
- `ResponseFormat` is a new enum in `src/types/request.rs`, plumbed through `ChatRequest`/builder and serialized only in the OpenAI wire path (`OaiRequest`). Anthropic ignores it at canonical layer (users go native).
- `ProviderCapabilities` is a plain struct in `src/provider.rs`, returned by a new `fn capabilities(&self) -> ProviderCapabilities` method on the `Provider` trait. Each concrete provider stores capabilities as a field, configured via builder.

**Tech Stack:** Rust 2024 edition, serde_json for wire serialization, existing `wiremock` / unit-test patterns.

---

## File Map

| File | Change |
|------|--------|
| `src/types/request.rs` | Add `ResponseFormat` enum; add `response_format` field to `ChatRequest` + builder |
| `src/types/mod.rs` | Re-export `ResponseFormat` |
| `src/openai/schema.rs` | Add `response_format: Option<serde_json::Value>` to `OaiRequest` |
| `src/openai/convert.rs` | Map `Option<ResponseFormat>` → wire JSON in `to_oai()` |
| `src/provider.rs` | Add `ProviderCapabilities` struct + `fn capabilities()` to `Provider` trait |
| `src/lib.rs` | Re-export `ProviderCapabilities` |
| `src/openai/mod.rs` | Add `capabilities` field to `OpenAIProvider` + builder, implement trait method |
| `src/anthropic/mod.rs` | Add `capabilities` field to `AnthropicProvider` + builder, implement trait method |
| `src/deepseek.rs` | Set DeepSeek-specific capabilities in `client()` |
| `tests/integration_deepseek.rs` | Add image support integration test |

---

## Part 1 — ResponseFormat

### Task 1: Add `ResponseFormat` to canonical types

**Files:**
- Modify: `src/types/request.rs`
- Modify: `src/types/mod.rs`

- [ ] **Step 1: Write the failing unit test in `src/types/request.rs`**

Add at the bottom of `src/types/request.rs` (inside a `#[cfg(test)] mod tests { ... }` block):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_format_json_object_builder() {
        let req = ChatRequest::builder("gpt-4o", 512)
            .response_format(ResponseFormat::JsonObject)
            .build();
        assert_eq!(req.response_format(), Some(&ResponseFormat::JsonObject));
    }

    #[test]
    fn response_format_json_schema_builder() {
        let schema = serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let req = ChatRequest::builder("gpt-4o", 512)
            .response_format(ResponseFormat::JsonSchema {
                name: "person".into(),
                schema: schema.clone(),
                strict: Some(true),
            })
            .build();
        match req.response_format() {
            Some(ResponseFormat::JsonSchema { name, strict, .. }) => {
                assert_eq!(name, "person");
                assert_eq!(*strict, Some(true));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn response_format_none_by_default() {
        let req = ChatRequest::builder("gpt-4o", 512).build();
        assert!(req.response_format().is_none());
    }
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cargo test response_format 2>&1 | head -20
```
Expected: compilation error — `ResponseFormat` not yet defined.

- [ ] **Step 3: Add `ResponseFormat` enum and wire it into `ChatRequest`**

In `src/types/request.rs`, directly after the `use super::message::Message;` line at the top, add:

```rust
/// Controls the format of the model's response.
///
/// Pass to [`ChatRequestBuilder::response_format()`] to request structured output.
/// Currently only forwarded by `OpenAIProvider`; `AnthropicProvider` ignores it
/// (use the native path with tool-forcing for Anthropic structured output).
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    /// Force the model to emit valid JSON (no specific schema).
    JsonObject,
    /// Force the model to emit JSON conforming to the supplied JSON Schema.
    JsonSchema {
        /// Logical name for the schema (required by OpenAI wire format).
        name: String,
        /// JSON Schema object describing the expected output.
        schema: serde_json::Value,
        /// When `Some(true)`, the model is required to strictly follow the schema.
        strict: Option<bool>,
    },
}
```

Then add `response_format: Option<ResponseFormat>` field to `ChatRequest`:

```rust
pub struct ChatRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Vec<ToolDef>,
    pub(crate) tool_choice: ToolChoice,
    pub(crate) params: SamplingParams,
    pub(crate) response_format: Option<ResponseFormat>,  // NEW
}
```

Add accessor to `ChatRequest` impl:

```rust
/// The requested response format, if any.
pub fn response_format(&self) -> Option<&ResponseFormat> {
    self.response_format.as_ref()
}
```

Add field to `ChatRequestBuilder`:

```rust
pub struct ChatRequestBuilder {
    model: String,
    messages: Vec<Message>,
    tools: Vec<ToolDef>,
    tool_choice: ToolChoice,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: u32,
    stop_sequences: Vec<String>,
    response_format: Option<ResponseFormat>,  // NEW
}
```

Add builder method to `ChatRequestBuilder` impl:

```rust
/// Set the response format. See [`ResponseFormat`] for options.
pub fn response_format(mut self, f: ResponseFormat) -> Self {
    self.response_format = Some(f);
    self
}
```

Update `ChatRequestBuilder::build()` to pass the new field:

```rust
pub fn build(self) -> ChatRequest {
    ChatRequest {
        model: self.model,
        messages: self.messages,
        tools: self.tools,
        tool_choice: self.tool_choice,
        params: SamplingParams {
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences,
        },
        response_format: self.response_format,  // NEW
    }
}
```

Update `ChatRequest::builder()` to zero-initialize the new field:

```rust
pub fn builder(model: impl Into<String>, max_tokens: u32) -> ChatRequestBuilder {
    ChatRequestBuilder {
        model: model.into(),
        messages: Vec::new(),
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        temperature: None,
        top_p: None,
        max_tokens,
        stop_sequences: Vec::new(),
        response_format: None,  // NEW
    }
}
```

- [ ] **Step 4: Verify `ResponseFormat` is already re-exported from `src/types/mod.rs`**

`src/types/mod.rs` uses `pub use request::*;` — a glob re-export that automatically includes all `pub` items from `request.rs`. Since `ResponseFormat` is `pub`, **no change to `mod.rs` is needed**. Just confirm the file still looks like:

```rust
pub use request::*;
// (plus other glob re-exports)
```

Actually look at the file — check whether it uses glob or explicit re-exports.

- [ ] **Step 5: Run failing tests to see them pass**

```bash
cargo test response_format 2>&1
```
Expected: 3 tests pass.

- [ ] **Step 6: Verify full build still compiles**

```bash
cargo check --all-targets 2>&1
```
Expected: no errors. There may be "field `response_format` is never read" warnings if the Anthropic path doesn't use it yet — that is fine.

- [ ] **Step 7: Commit**

```bash
git add src/types/request.rs src/types/mod.rs
git commit -m "feat: add ResponseFormat enum to ChatRequest canonical type"
```

---

### Task 2: Wire `ResponseFormat` into OpenAI conversion

**Files:**
- Modify: `src/openai/schema.rs`
- Modify: `src/openai/convert.rs`

- [ ] **Step 1: Write the failing unit test**

Add to the existing `#[cfg(test)]` block in `src/openai/convert.rs` (find `mod tests {` and add inside it):

```rust
#[test]
fn response_format_json_object_serializes() {
    use crate::types::ResponseFormat;
    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![crate::types::RequestContent::Text("hi".into())]))
        .response_format(ResponseFormat::JsonObject)
        .build();
    let oai = to_oai(&req, None, false).unwrap();
    let body = serde_json::to_value(&oai).unwrap();
    assert_eq!(body["response_format"]["type"], "json_object");
}

#[test]
fn response_format_json_schema_serializes() {
    use crate::types::ResponseFormat;
    let schema = serde_json::json!({"type": "object"});
    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![crate::types::RequestContent::Text("hi".into())]))
        .response_format(ResponseFormat::JsonSchema {
            name: "out".into(),
            schema: schema.clone(),
            strict: Some(true),
        })
        .build();
    let oai = to_oai(&req, None, false).unwrap();
    let body = serde_json::to_value(&oai).unwrap();
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["name"], "out");
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    assert_eq!(body["response_format"]["json_schema"]["schema"]["type"], "object");
}

#[test]
fn response_format_none_omitted() {
    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![crate::types::RequestContent::Text("hi".into())]))
        .build();
    let oai = to_oai(&req, None, false).unwrap();
    let body = serde_json::to_value(&oai).unwrap();
    assert!(body.get("response_format").is_none() || body["response_format"].is_null());
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cargo test response_format_json 2>&1 | head -30
```
Expected: compile error — `OaiRequest` doesn't have `response_format` yet.

- [ ] **Step 3: Add `response_format` field to `OaiRequest` in `src/openai/schema.rs`**

Locate `pub(crate) struct OaiRequest` and add the field after `reasoning_effort`:

```rust
/// Response format for structured output.
/// `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {...}}`
#[serde(skip_serializing_if = "Option::is_none")]
pub response_format: Option<serde_json::Value>,
```

- [ ] **Step 4: Add helper function and populate field in `to_oai()` in `src/openai/convert.rs`**

Add a private helper function near the top of the conversion section (after the imports):

```rust
fn response_format_to_oai(fmt: &crate::types::ResponseFormat) -> serde_json::Value {
    match fmt {
        crate::types::ResponseFormat::JsonObject => {
            serde_json::json!({"type": "json_object"})
        }
        crate::types::ResponseFormat::JsonSchema { name, schema, strict } => {
            let mut js = serde_json::json!({
                "name": name,
                "schema": schema,
            });
            if let Some(s) = strict {
                js["strict"] = serde_json::Value::Bool(*s);
            }
            serde_json::json!({"type": "json_schema", "json_schema": js})
        }
    }
}
```

In `to_oai()`, update the `Ok(OaiRequest { ... })` construction to include:

```rust
Ok(OaiRequest {
    model: req.model().into(),
    messages,
    tools,
    tool_choice,
    temperature: params.temperature(),
    max_tokens: params.max_tokens(),
    top_p: params.top_p(),
    stop: params.stop_sequences().to_vec(),
    stream,
    stream_options: if stream {
        Some(super::schema::OaiStreamOptions { include_usage: true })
    } else {
        None
    },
    seed: ext.seed,
    frequency_penalty: ext.frequency_penalty,
    presence_penalty: ext.presence_penalty,
    user: ext.user.clone(),
    reasoning_effort: ext.reasoning_effort.clone(),
    response_format: req.response_format().map(response_format_to_oai),  // NEW
})
```

- [ ] **Step 5: Run tests**

```bash
cargo test response_format 2>&1
```
Expected: all 6 response_format tests pass (3 in types, 3 in openai convert).

- [ ] **Step 6: Run full test suite**

```bash
./test.sh 2>&1
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/openai/schema.rs src/openai/convert.rs
git commit -m "feat: serialize ResponseFormat into OpenAI wire response_format field"
```

---

## Part 2 — ProviderCapabilities

### Task 3: Define `ProviderCapabilities` and add to `Provider` trait

**Files:**
- Modify: `src/provider.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing unit test**

Add at the bottom of `src/lib.rs` in the existing `mod tests`:

```rust
#[test]
fn provider_capabilities_openai() {
    let p = openai::OpenAIProvider::builder("key").build();
    let caps = p.capabilities();
    assert!(caps.supports_images());
    assert!(caps.supports_reasoning());
    assert!(caps.supports_json_schema());
}

#[test]
fn provider_capabilities_anthropic() {
    let p = anthropic::AnthropicProvider::builder("key").build();
    let caps = p.capabilities();
    assert!(caps.supports_images());
    assert!(caps.supports_reasoning());
    assert!(!caps.supports_json_schema());
}

#[test]
fn provider_capabilities_deepseek() {
    use openai::OpenAIProvider;
    let p: OpenAIProvider = deepseek::client("key");
    let caps = p.capabilities();
    assert!(!caps.supports_images(), "DeepSeek V3/Flash does not support images");
    assert!(caps.supports_reasoning());
    assert!(caps.supports_json_schema());
}

#[test]
fn provider_capabilities_box_dyn_provider() {
    // Verify trait object works — capabilities() must be object-safe.
    let p: Box<dyn provider::Provider> = Box::new(openai::OpenAIProvider::builder("key").build());
    let caps = p.capabilities();
    assert!(caps.supports_images());
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cargo test provider_capabilities 2>&1 | head -20
```
Expected: compile error — `capabilities()` not yet defined on `Provider`.

- [ ] **Step 3: Add `ProviderCapabilities` and update `Provider` trait in `src/provider.rs`**

Replace the entire content of `src/provider.rs` with:

```rust
use crate::{error::LlmError, stream_handle::StreamHandle, types::*};
use async_trait::async_trait;

/// Runtime-queryable capability flags for a provider.
///
/// Use `provider.capabilities()` to discover what a provider supports before
/// constructing requests that depend on those features.
///
/// Marked `#[non_exhaustive]` so new capability flags can be added without
/// breaking existing code.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderCapabilities {
    /// Whether this provider accepts image content in user messages.
    pub images: bool,
    /// Whether this provider supports reasoning / extended thinking output.
    pub reasoning: bool,
    /// Whether this provider supports JSON schema structured output via the
    /// canonical `ResponseFormat::JsonSchema` field on `ChatRequest`.
    pub json_schema: bool,
}

impl ProviderCapabilities {
    /// Returns `true` if this provider accepts image content in user messages.
    pub fn supports_images(&self) -> bool {
        self.images
    }

    /// Returns `true` if this provider supports reasoning / extended thinking.
    pub fn supports_reasoning(&self) -> bool {
        self.reasoning
    }

    /// Returns `true` if this provider supports `ResponseFormat::JsonSchema`
    /// on the canonical `ChatRequest`.
    pub fn supports_json_schema(&self) -> bool {
        self.json_schema
    }
}

/// Unified interface for any LLM provider.
///
/// Implementors should be cheaply cloneable (hold an `Arc` internally) so
/// callers can pass `Box<dyn Provider>` across threads.
///
/// # Universal vs. native paths
///
/// - **Universal path** (this trait): accepts [`ChatRequest`] and returns
///   canonical [`ChatResponse`] / [`StreamHandle`]. Provider-specific
///   features are not directly accessible.
/// - **Native path**: the concrete provider type exposes additional methods
///   (e.g. `OpenAIProvider::chat_native()`) that accept a provider-specific
///   `*Request` type (e.g. [`OpenAIRequest`]) carrying extensions like
///   `seed`, `frequency_penalty`, `extra_body`, etc.
///
/// Use the universal path for provider-agnostic code and the native path
/// when you need provider-specific knobs.
///
/// [`OpenAIRequest`]: crate::openai::OpenAIRequest
#[async_trait]
pub trait Provider: Send + Sync {
    /// Return the static capability flags for this provider.
    ///
    /// Capabilities are determined at construction time (they depend on the
    /// provider endpoint, not on a specific model).  Use them to guard
    /// features before sending a request:
    ///
    /// ```ignore
    /// if provider.capabilities().supports_json_schema() {
    ///     req_builder.response_format(ResponseFormat::JsonSchema { ... });
    /// }
    /// ```
    fn capabilities(&self) -> ProviderCapabilities;

    /// Send a request and wait for the complete (non-streaming) response.
    ///
    /// Prefer this method when you need the full response at once, including
    /// the message `id` and `model` fields.
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;

    /// Send a request and receive a streaming handle for incremental output.
    ///
    /// Use [`StreamHandle::events()`] for per-token UX or
    /// [`StreamHandle::collect()`] to fold the stream into a [`ChatResponse`].
    /// Note that `id` and `model` may be empty when collected from streaming.
    async fn chat_stream(&self, req: &ChatRequest) -> Result<StreamHandle, LlmError>;
}
```

- [ ] **Step 4: Re-export `ProviderCapabilities` from `src/lib.rs`**

In `src/lib.rs`, update the existing `pub use provider::Provider;` line to:

```rust
pub use provider::{Provider, ProviderCapabilities};
```

- [ ] **Step 5: Run to verify only "not implemented" errors remain**

```bash
cargo check --all-targets 2>&1 | grep error
```
Expected: errors about missing `capabilities()` impl on `OpenAIProvider` and `AnthropicProvider` — not about missing types.

- [ ] **Step 6: Commit (partial — types defined, impls coming next)**

```bash
git add src/provider.rs src/lib.rs
git commit -m "feat: add ProviderCapabilities struct and capabilities() to Provider trait"
```

---

### Task 4: Implement `capabilities()` for `OpenAIProvider`

**Files:**
- Modify: `src/openai/mod.rs`

- [ ] **Step 1: Add `capabilities` field to `OpenAIProvider` and `OpenAIProviderBuilder`**

In `src/openai/mod.rs`, update `OpenAIProvider` struct to add:

```rust
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    chat_path: String,
    parse_reasoning_content: bool,
    tolerant_keepalive: bool,
    capabilities: crate::provider::ProviderCapabilities,  // NEW
}
```

Update `OpenAIProviderBuilder` similarly:

```rust
pub struct OpenAIProviderBuilder {
    api_key: String,
    base_url: String,
    chat_path: String,
    parse_reasoning_content: bool,
    tolerant_keepalive: bool,
    capabilities: crate::provider::ProviderCapabilities,  // NEW
}
```

- [ ] **Step 2: Set default capabilities in `OpenAIProvider::builder()`**

Update the `OpenAIProvider::builder()` factory to initialize `capabilities`:

```rust
pub fn builder(api_key: impl Into<String>) -> OpenAIProviderBuilder {
    OpenAIProviderBuilder {
        api_key: api_key.into(),
        base_url: DEFAULT_BASE_URL.to_string(),
        chat_path: DEFAULT_CHAT_PATH.to_string(),
        parse_reasoning_content: false,
        tolerant_keepalive: false,
        capabilities: crate::provider::ProviderCapabilities {
            images: true,
            reasoning: true,
            json_schema: true,
        },
    }
}
```

- [ ] **Step 3: Add builder method for overriding capabilities**

In `impl OpenAIProviderBuilder`, add:

```rust
/// Override provider capabilities (used by factory functions such as `deepseek::client`).
pub(crate) fn capabilities(mut self, caps: crate::provider::ProviderCapabilities) -> Self {
    self.capabilities = caps;
    self
}
```

(`pub(crate)` is intentional — callers use factory functions, not raw capability overrides.)

- [ ] **Step 4: Pass capabilities through in `OpenAIProviderBuilder::build()`**

Update `build()`:

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
        capabilities: self.capabilities,  // NEW
    }
}
```

- [ ] **Step 5: Implement `capabilities()` in the `Provider` impl**

In `impl Provider for OpenAIProvider`, add:

```rust
fn capabilities(&self) -> crate::provider::ProviderCapabilities {
    self.capabilities.clone()
}
```

- [ ] **Step 6: Run the capabilities tests**

```bash
cargo test provider_capabilities_openai provider_capabilities_deepseek 2>&1
```
Expected: `openai` test passes; `deepseek` test still fails because `deepseek::client()` hasn't been updated yet.

- [ ] **Step 7: Commit**

```bash
git add src/openai/mod.rs
git commit -m "feat: add configurable ProviderCapabilities to OpenAIProvider"
```

---

### Task 5: Update `deepseek::client()` with DeepSeek-specific capabilities

**Files:**
- Modify: `src/deepseek.rs`

- [ ] **Step 1: Run failing test to confirm the state**

```bash
cargo test provider_capabilities_deepseek 2>&1
```
Expected: FAIL — `supports_images()` returns `true` but should be `false`.

- [ ] **Step 2: Update `deepseek::client()` to set DeepSeek capabilities**

Replace the entire content of `src/deepseek.rs`:

```rust
use crate::{openai::OpenAIProvider, provider::ProviderCapabilities};

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";

/// Create an [`OpenAIProvider`] pre-configured for the DeepSeek API.
///
/// DeepSeek uses the OpenAI Chat Completions schema at a different base URL,
/// with `reasoning_content` parsing and tolerant keepalive enabled for
/// thinking-mode models.
///
/// Capabilities: reasoning ✓, json_schema ✓, images ✗
/// (DeepSeek V3/V4 do not accept image input; see integration test
/// `integration_image_input_returns_error` for verification.)
///
/// # Example
///
/// ```no_run
/// # use llm_adapter::deepseek;
/// let provider = deepseek::client("sk-your-api-key");
/// ```
pub fn client(api_key: impl Into<String>) -> OpenAIProvider {
    OpenAIProvider::builder(api_key)
        .base_url(DEEPSEEK_BASE_URL)
        .parse_reasoning_content(true)
        .tolerant_keepalive(true)
        .capabilities(ProviderCapabilities {
            images: false,
            reasoning: true,
            json_schema: true,
        })
        .build()
}
```

- [ ] **Step 3: Run capabilities tests**

```bash
cargo test provider_capabilities 2>&1
```
Expected: `openai`, `deepseek`, and `box_dyn_provider` pass; `anthropic` still fails (next task).

- [ ] **Step 4: Commit**

```bash
git add src/deepseek.rs
git commit -m "feat: set DeepSeek capabilities (images: false) in deepseek::client()"
```

---

### Task 6: Implement `capabilities()` for `AnthropicProvider`

**Files:**
- Modify: `src/anthropic/mod.rs`

- [ ] **Step 1: Run failing test to confirm the state**

```bash
cargo test provider_capabilities_anthropic 2>&1
```
Expected: compile error or panic — `capabilities()` not implemented for `AnthropicProvider`.

- [ ] **Step 2: Add `capabilities` field to `AnthropicProvider` and builder**

In `src/anthropic/mod.rs`, update `AnthropicProvider`:

```rust
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    messages_path: String,
    capabilities: crate::provider::ProviderCapabilities,  // NEW
}
```

Update `AnthropicProviderBuilder`:

```rust
pub struct AnthropicProviderBuilder {
    api_key: String,
    base_url: String,
    messages_path: String,
    capabilities: crate::provider::ProviderCapabilities,  // NEW
}
```

- [ ] **Step 3: Initialize default capabilities in `AnthropicProvider::builder()`**

Update the factory:

```rust
pub fn builder(api_key: impl Into<String>) -> AnthropicProviderBuilder {
    AnthropicProviderBuilder {
        api_key: api_key.into(),
        base_url: DEFAULT_BASE_URL.to_string(),
        messages_path: DEFAULT_MESSAGES_PATH.to_string(),
        capabilities: crate::provider::ProviderCapabilities {
            images: true,
            reasoning: true,
            json_schema: false,   // Anthropic structured output not in canonical layer
        },
    }
}
```

- [ ] **Step 4: Pass capabilities through `AnthropicProviderBuilder::build()`**

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
        capabilities: self.capabilities,  // NEW
    }
}
```

- [ ] **Step 5: Implement `capabilities()` in the `Provider` impl**

In `impl Provider for AnthropicProvider`:

```rust
fn capabilities(&self) -> crate::provider::ProviderCapabilities {
    self.capabilities.clone()
}
```

- [ ] **Step 6: Run all capabilities tests**

```bash
cargo test provider_capabilities 2>&1
```
Expected: all 4 tests pass.

- [ ] **Step 7: Run full test suite**

```bash
./test.sh 2>&1
```
Expected: all tests pass (zero regressions).

- [ ] **Step 8: Commit**

```bash
git add src/anthropic/mod.rs
git commit -m "feat: add ProviderCapabilities to AnthropicProvider (images+reasoning: true, json_schema: false)"
```

---

### Task 7: Integration test — DeepSeek image support verification

**Files:**
- Modify: `tests/integration_deepseek.rs`

- [ ] **Step 1: Write the integration test**

Add at the bottom of `tests/integration_deepseek.rs` (before the final closing brace, keeping `#![cfg(feature = "integration")]` at top):

```rust
#[tokio::test]
async fn integration_image_input_returns_error() {
    // DeepSeek V3/V4 do not support image input. Sending an image_url
    // message should result in an error (4xx from the API).
    // If this test starts *passing* (no error), update deepseek::client()
    // capabilities to set images: true.
    let Some((p, model)) = provider() else {
        return;
    };

    let req = ChatRequest::builder(model, 256)
        .message(Message::User(vec![
            RequestContent::Text("What is in this image?".into()),
            RequestContent::Image(
                crate::types::ImageSource::Url("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg".into())
            ),
        ]))
        .build();

    let result = p.chat(&req).await;
    assert!(
        result.is_err(),
        "Expected error for image input on DeepSeek (images not supported); got: {:?}",
        result
    );
}
```

- [ ] **Step 2: Run locally (skip without API key)**

```bash
./test.sh 2>&1 | grep -E "image|SKIP|skip"
```
Expected: test is skipped if `DEEPSEEK_API_KEY` is not set (same skip pattern as other integration tests).

- [ ] **Step 3: Verify unit tests still pass**

```bash
./test.sh 2>&1
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/integration_deepseek.rs
git commit -m "test: add integration test to verify DeepSeek rejects image input"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|------------|------|
| `ChatRequest` supports `response_format` (`json_object`/`json_schema`) | Task 1 |
| OpenAI wire serialization of `response_format` | Task 2 |
| Anthropic note — no canonical layer, use native path | Noted in `ResponseFormat` doc comment |
| `ProviderCapabilities` type with `supports_images/reasoning/json_schema` | Task 3 |
| Each provider implements capabilities | Tasks 4–6 |
| DeepSeek V4 Flash image support verification (integration test) | Task 7 |

### No placeholders

All steps contain concrete code. No "TBD" or "similar to above."

### Type consistency

- `ResponseFormat` is defined in Task 1, used by name in Tasks 1 and 2.
- `ProviderCapabilities` is defined in Task 3, used by name in Tasks 4, 5, 6.
- `response_format_to_oai()` helper defined and called in Task 2.
- `capabilities` field type is `crate::provider::ProviderCapabilities` — consistent across `OpenAIProvider`, `AnthropicProvider`.
- `builder.capabilities(caps)` method is `pub(crate)` and called from `deepseek.rs` (same crate) — consistent.
