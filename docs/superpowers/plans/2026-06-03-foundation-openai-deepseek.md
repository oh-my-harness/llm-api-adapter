# Foundation + OpenAI + DeepSeek Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working Rust library with canonical types, Provider trait, OpenAI provider, and DeepSeek factory — fully offline-testable via wiremock fixtures.

**Architecture:** Canonical layer (types + trait) sits beneath provider-specific wire conversion. OpenAI uses Chat Completions schema. DeepSeek reuses OpenAI provider with `parse_reasoning_content: true`. All types constructed via builder pattern (private fields, public getters).

**Tech Stack:** Rust 2024, reqwest 0.12, async-trait 0.1, serde/serde_json 1, eventsource-stream 0.2, futures 0.3, thiserror 2, wiremock 0.6 (dev), tokio 1

**Note:** Plan 2 (Anthropic provider) follows as a separate plan once this is complete.

---

### Task 1: Project setup

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs` (skeleton)

- [ ] **Step 1: Write the failing compile check**

```rust
// src/lib.rs
pub fn _check() {}
```

Run: `cargo check`
Expected: FAIL — Cargo.toml missing

- [ ] **Step 2: Create Cargo.toml**

```toml
[package]
name = "llm_adapter"
version = "0.1.0"
edition = "2024"

[dependencies]
async-trait = "0.1"
eventsource-stream = "0.2"
futures = "0.3"
reqwest = { version = "0.12", features = ["json", "stream"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tokio = { version = "1", features = ["rt-multi-thread"] }

[dev-dependencies]
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
wiremock = "0.6"
```

- [ ] **Step 3: Verify compile**

Run: `cargo check`
Expected: Compiles cleanly

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml src/lib.rs
git commit -m "chore: initialize Rust project with dependencies"
```

---

### Task 2: LlmError

**Files:**
- Create: `src/error.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

Add to `src/lib.rs`:
```rust
mod error;
pub use error::LlmError;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn llm_error_display() {
        let e = LlmError::Unauthorized;
        assert_eq!(e.to_string(), "unauthorized");
        let e2 = LlmError::InvalidRequest("bad param".into());
        assert!(e2.to_string().contains("bad param"));
    }
}
```

Run: `cargo test`
Expected: FAIL — `error` module missing

- [ ] **Step 2: Create src/error.rs**

```rust
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Request was malformed or contained invalid parameters.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// API key missing or invalid.
    #[error("unauthorized")]
    Unauthorized,

    /// Key does not have permission for this operation.
    #[error("forbidden")]
    Forbidden,

    /// Rate limit hit; caller should back off.
    #[error("rate limited (retry after {retry_after:?})")]
    RateLimit { retry_after: Option<Duration> },

    /// Provider is overloaded; retry after delay.
    #[error("server overloaded (retry after {retry_after:?})")]
    Overloaded { retry_after: Option<Duration> },

    /// Unrecoverable server-side error.
    #[error("server error: {0}")]
    ServerError(String),

    /// Request or response timed out.
    #[error("request timed out")]
    Timeout,

    /// Error within an active SSE stream.
    #[error("stream error: {0}")]
    Stream(String),

    /// HTTP transport failure.
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    /// JSON serialisation or deserialisation failed.
    #[error("decode error: {0}")]
    Decode(#[from] serde_json::Error),

    /// Provider returned a structured error object.
    #[error("provider error {code}: {message}")]
    Provider { code: String, message: String },
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/error.rs src/lib.rs
git commit -m "feat: add LlmError type"
```

---

### Task 3: Canonical content types

**Files:**
- Create: `src/types/mod.rs`
- Create: `src/types/content.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

Add to `src/lib.rs`:
```rust
mod types;
```

Create `src/types/mod.rs` (empty for now):
```rust
mod content;
pub use content::*;
```

Create `src/types/content.rs`:
```rust
// empty
```

Add to the test module in `src/lib.rs`:
```rust
#[test]
fn content_construction() {
    use types::{RequestContent, ResponseContent, ImageSource, ToolInvocation};
    let _text = RequestContent::Text("hello".into());
    let _img = RequestContent::Image(ImageSource::Url("https://example.com/img.png".into()));
    let _resp = ResponseContent::Text("world".into());
    let _tool = ResponseContent::ToolInvocation(ToolInvocation {
        id: "call_1".into(),
        name: "fn".into(),
        input: serde_json::json!({"key": "val"}),
    });
    let _reasoning = ResponseContent::Reasoning("my reasoning".into());
}
```

Run: `cargo test`
Expected: FAIL — types not defined

- [ ] **Step 2: Implement src/types/content.rs**

```rust
/// Image data source for vision-capable requests.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageSource {
    /// Publicly accessible URL.
    Url(String),
    /// Inline base64-encoded image.
    Base64 { media_type: String, data: String },
}

/// Content that a user or tool produces as input to a request.
#[derive(Debug, Clone, PartialEq)]
pub enum RequestContent {
    /// Plain text.
    Text(String),
    /// Image (URL or base64).
    Image(ImageSource),
}

/// A tool invocation emitted by the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInvocation {
    /// Provider-assigned call ID, required for tool result routing.
    pub id: String,
    /// Name of the tool being called.
    pub name: String,
    /// Arguments the model produced; callers must validate this JSON.
    pub input: serde_json::Value,
}

/// Content that the model produces in a response.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseContent {
    /// Plain text output.
    Text(String),
    /// Tool the model wants to invoke.
    ToolInvocation(ToolInvocation),
    /// Chain-of-thought reasoning text (read-only; not all providers support multi-turn with this).
    Reasoning(String),
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/types/mod.rs src/types/content.rs src/lib.rs
git commit -m "feat: add canonical content types"
```

---

### Task 4: Message ADT

**Files:**
- Create: `src/types/message.rs`
- Modify: `src/types/mod.rs`

- [ ] **Step 1: Write the failing test**

Add to the test module in `src/lib.rs`:
```rust
#[test]
fn message_variants() {
    use types::{Message, RequestContent, ResponseContent, ToolInvocation};
    let _sys = Message::System("Be helpful".into());
    let _user = Message::User(vec![RequestContent::Text("Hi".into())]);
    let _asst = Message::Assistant(vec![ResponseContent::Text("Hello!".into())]);
    let _tool = Message::Tool {
        invocation_id: "call_1".into(),
        content: vec![RequestContent::Text("22°C".into())],
        is_error: false,
    };
}
```

Run: `cargo test`
Expected: FAIL — `Message` not found

- [ ] **Step 2: Create src/types/message.rs**

```rust
use super::content::{RequestContent, ResponseContent};

/// A single message in a conversation, with role encoded as an enum variant.
///
/// Reasoning content can only appear inside `Assistant` — any other placement
/// is a compile-time error.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// System-level instruction (not all providers support this role).
    System(String),
    /// Human turn; may contain text and images.
    User(Vec<RequestContent>),
    /// Model turn; may contain text, tool invocations, and reasoning.
    Assistant(Vec<ResponseContent>),
    /// Tool result returned to the model after it issued a `ToolInvocation`.
    Tool {
        /// Matches `ToolInvocation.id` from the preceding `Assistant` message.
        invocation_id: String,
        /// Result content (text or images).
        content: Vec<RequestContent>,
        /// Set to `true` when the tool execution failed.
        is_error: bool,
    },
}
```

- [ ] **Step 3: Add to src/types/mod.rs**

```rust
mod content;
mod message;
pub use content::*;
pub use message::*;
```

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/types/message.rs src/types/mod.rs
git commit -m "feat: add Message ADT"
```

---

### Task 5: ChatRequest and SamplingParams with builders

**Files:**
- Create: `src/types/request.rs`
- Modify: `src/types/mod.rs`

- [ ] **Step 1: Write the failing test**

Add to the test module in `src/lib.rs`:
```rust
#[test]
fn chat_request_builder() {
    use types::{ChatRequest, Message, RequestContent, ToolChoice};
    let req = ChatRequest::builder("gpt-4o", 1024)
        .message(Message::System("You are helpful.".into()))
        .message(Message::User(vec![RequestContent::Text("Hello".into())]))
        .temperature(0.7)
        .build();

    assert_eq!(req.model(), "gpt-4o");
    assert_eq!(req.params().max_tokens, 1024);
    assert_eq!(req.params().temperature, Some(0.7));
    assert_eq!(req.messages().len(), 2);
    assert!(matches!(req.tool_choice(), ToolChoice::Auto));
}
```

Run: `cargo test`
Expected: FAIL — `ChatRequest` not found

- [ ] **Step 2: Create src/types/request.rs**

```rust
use super::message::Message;

/// Definition of a callable tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    /// Unique tool name.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema object describing accepted parameters.
    pub input_schema: serde_json::Value,
}

/// How the model should decide whether to call tools.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ToolChoice {
    /// Model decides (default when tools are provided).
    #[default]
    Auto,
    /// Model must not call any tool.
    None,
    /// Model must call at least one tool.
    Required,
    /// Model must call exactly the named tool.
    Specific(String),
}

/// Sampling hyper-parameters shared across all providers.
///
/// Construct via `SamplingParams::builder(max_tokens)`.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: u32,
    pub(crate) top_p: Option<f32>,
    pub(crate) stop_sequences: Vec<String>,
}

impl SamplingParams {
    pub fn builder(max_tokens: u32) -> SamplingParamsBuilder {
        SamplingParamsBuilder {
            temperature: None,
            max_tokens,
            top_p: None,
            stop_sequences: Vec::new(),
        }
    }

    pub fn temperature(&self) -> Option<f32> { self.temperature }
    pub fn max_tokens(&self) -> u32 { self.max_tokens }
    pub fn top_p(&self) -> Option<f32> { self.top_p }
    pub fn stop_sequences(&self) -> &[String] { &self.stop_sequences }
}

pub struct SamplingParamsBuilder {
    temperature: Option<f32>,
    max_tokens: u32,
    top_p: Option<f32>,
    stop_sequences: Vec<String>,
}

impl SamplingParamsBuilder {
    pub fn temperature(mut self, t: f32) -> Self { self.temperature = Some(t); self }
    pub fn top_p(mut self, p: f32) -> Self { self.top_p = Some(p); self }
    pub fn stop_sequence(mut self, s: impl Into<String>) -> Self {
        self.stop_sequences.push(s.into()); self
    }
    pub fn build(self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences,
        }
    }
}

/// A fully-specified chat request, constructed via `ChatRequest::builder`.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Vec<ToolDef>,
    pub(crate) tool_choice: ToolChoice,
    pub(crate) params: SamplingParams,
}

impl ChatRequest {
    pub fn builder(model: impl Into<String>, max_tokens: u32) -> ChatRequestBuilder {
        ChatRequestBuilder {
            model: model.into(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            params: SamplingParams::builder(max_tokens),
        }
    }

    pub fn model(&self) -> &str { &self.model }
    pub fn messages(&self) -> &[Message] { &self.messages }
    pub fn tools(&self) -> &[ToolDef] { &self.tools }
    pub fn tool_choice(&self) -> &ToolChoice { &self.tool_choice }
    pub fn params(&self) -> &SamplingParams { &self.params }
}

pub struct ChatRequestBuilder {
    model: String,
    messages: Vec<Message>,
    tools: Vec<ToolDef>,
    tool_choice: ToolChoice,
    params: SamplingParams,
}

impl ChatRequestBuilder {
    pub fn message(mut self, m: Message) -> Self { self.messages.push(m); self }
    pub fn messages(mut self, ms: Vec<Message>) -> Self { self.messages = ms; self }
    pub fn tool(mut self, t: ToolDef) -> Self { self.tools.push(t); self }
    pub fn tool_choice(mut self, c: ToolChoice) -> Self { self.tool_choice = c; self }
    pub fn temperature(mut self, t: f32) -> Self { self.params.temperature = Some(t); self }
    pub fn top_p(mut self, p: f32) -> Self { self.params.top_p = Some(p); self }
    pub fn max_tokens(mut self, n: u32) -> Self { self.params.max_tokens = n; self }
    pub fn stop_sequence(mut self, s: impl Into<String>) -> Self {
        self.params.stop_sequences.push(s.into()); self
    }
    pub fn build(self) -> ChatRequest {
        ChatRequest {
            model: self.model,
            messages: self.messages,
            tools: self.tools,
            tool_choice: self.tool_choice,
            params: self.params,
        }
    }
}
```

- [ ] **Step 3: Add to src/types/mod.rs**

```rust
mod content;
mod message;
mod request;
pub use content::*;
pub use message::*;
pub use request::*;
```

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/types/request.rs src/types/mod.rs
git commit -m "feat: add ChatRequest and SamplingParams with builders"
```

---

### Task 6: ChatResponse and streaming types

**Files:**
- Create: `src/types/response.rs`
- Create: `src/types/stream.rs`
- Modify: `src/types/mod.rs`

- [ ] **Step 1: Write the failing tests**

Add to test module in `src/lib.rs`:
```rust
#[test]
fn chat_response_helpers() {
    use types::{ChatResponse, ResponseContent, StopReason, ToolInvocation, Usage};
    let resp = ChatResponse {
        id: "msg_1".into(),
        model: "gpt-4o".into(),
        content: vec![
            ResponseContent::Text("Hello ".into()),
            ResponseContent::Text("world".into()),
        ],
        stop_reason: StopReason::EndTurn,
        usage: Usage::default(),
    };
    assert_eq!(resp.text(), "Hello world");
    assert!(resp.reasoning().is_none());
    assert_eq!(resp.tool_invocations().count(), 0);
}

#[test]
fn stream_event_construction() {
    use types::{StreamEvent, ContentKind, StopReason, Usage};
    let _start = StreamEvent::ContentStart { index: 0, kind: ContentKind::Text };
    let _delta = StreamEvent::TextDelta { index: 0, text: "hi".into() };
    let _stop = StreamEvent::ContentStop { index: 0 };
    let _msg_stop = StreamEvent::MessageStop {
        stop_reason: StopReason::EndTurn,
        usage: Usage::default(),
    };
}
```

Run: `cargo test`
Expected: FAIL — types not found

- [ ] **Step 2: Create src/types/response.rs**

```rust
use super::content::{ResponseContent, ToolInvocation};

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// Natural end of the model's response.
    EndTurn,
    /// Hit the `max_tokens` limit.
    MaxTokens,
    /// A stop sequence was matched.
    StopSequence,
    /// Model wants to call a tool.
    ToolUse,
    /// Content policy triggered.
    ContentFilter,
    /// Any other provider-specific stop reason.
    Other(String),
}

/// Token usage for a single request/response pair.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Usage {
    /// Tokens in the prompt (input).
    pub input_tokens: u32,
    /// Tokens generated (output).
    pub output_tokens: u32,
    /// Prompt tokens served from provider cache (OpenAI/DeepSeek).
    pub cached_input_tokens: u32,
    /// Tokens written to Anthropic prompt cache; always 0 for other providers.
    pub cache_creation_input_tokens: u32,
    /// Reasoning tokens consumed (DeepSeek thinking mode); 0 for other providers.
    pub reasoning_tokens: u32,
}

/// A complete (non-streaming) response from the model.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Provider-assigned message ID.
    pub id: String,
    /// Model that generated this response.
    pub model: String,
    /// Ordered content blocks (text, tool invocations, reasoning).
    pub content: Vec<ResponseContent>,
    /// Reason the model stopped generating.
    pub stop_reason: StopReason,
    /// Token usage for this exchange.
    pub usage: Usage,
}

impl ChatResponse {
    /// Convert into a `Message::Assistant` for multi-turn continuations.
    pub fn into_message(self) -> super::message::Message {
        super::message::Message::Assistant(self.content)
    }

    /// Concatenate all `Text` blocks in order.
    pub fn text(&self) -> String {
        self.content.iter()
            .filter_map(|c| if let ResponseContent::Text(t) = c { Some(t.as_str()) } else { None })
            .collect()
    }

    /// Iterate over all `ToolInvocation` blocks.
    pub fn tool_invocations(&self) -> impl Iterator<Item = &ToolInvocation> {
        self.content.iter().filter_map(|c| {
            if let ResponseContent::ToolInvocation(ti) = c { Some(ti) } else { None }
        })
    }

    /// Return the first reasoning block text, if any.
    pub fn reasoning(&self) -> Option<&str> {
        self.content.iter().find_map(|c| {
            if let ResponseContent::Reasoning(r) = c { Some(r.as_str()) } else { None }
        })
    }
}
```

- [ ] **Step 3: Create src/types/stream.rs**

```rust
use super::response::{StopReason, Usage};

/// Identifies what type of content a stream block carries.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentKind {
    /// Plain text output.
    Text,
    /// Chain-of-thought reasoning.
    Reasoning,
    /// A tool call — id and name are known at block start, arguments arrive as deltas.
    ToolInvocation {
        /// Provider-assigned call ID.
        id: String,
        /// Name of the tool being called.
        name: String,
    },
}

/// A single event in a streaming response.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// A new content block has started.
    ContentStart { index: usize, kind: ContentKind },
    /// Incremental text for a text block.
    TextDelta { index: usize, text: String },
    /// Incremental reasoning text for a reasoning block.
    ReasoningDelta { index: usize, text: String },
    /// Incremental JSON fragment for a tool-call arguments block.
    ToolDelta { index: usize, arguments: String },
    /// A content block has finished.
    ContentStop { index: usize },
    /// The full message has finished; carries final usage.
    MessageStop { stop_reason: StopReason, usage: Usage },
}
```

- [ ] **Step 4: Update src/types/mod.rs**

```rust
mod content;
mod message;
mod request;
mod response;
mod stream;
pub use content::*;
pub use message::*;
pub use request::*;
pub use response::*;
pub use stream::*;
```

- [ ] **Step 5: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/types/response.rs src/types/stream.rs src/types/mod.rs
git commit -m "feat: add ChatResponse, StopReason, Usage, and StreamEvent types"
```

---

### Task 7: Provider trait and StreamHandle

**Files:**
- Create: `src/provider.rs`
- Create: `src/stream_handle.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

Add to test module in `src/lib.rs`:
```rust
#[test]
fn provider_is_object_safe() {
    use provider::Provider;
    // Verify we can box the trait
    fn _accepts(_p: Box<dyn Provider>) {}
}
```

Run: `cargo test`
Expected: FAIL — `provider` module missing

- [ ] **Step 2: Create src/provider.rs**

```rust
use async_trait::async_trait;
use crate::{error::LlmError, stream_handle::StreamHandle, types::*};

/// Unified interface for any LLM provider.
///
/// Implementors should be cheaply cloneable (hold an `Arc` internally) so
/// callers can pass `Box<dyn Provider>` across threads.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Send a request and wait for the complete response.
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;

    /// Send a request and receive a streaming handle for incremental output.
    async fn chat_stream(&self, req: &ChatRequest) -> Result<StreamHandle, LlmError>;
}
```

- [ ] **Step 3: Create src/stream_handle.rs**

```rust
use std::collections::HashMap;
use futures::{Stream, StreamExt};
use crate::{
    error::LlmError,
    types::*,
};

type BoxStream = futures::stream::BoxStream<'static, Result<StreamEvent, LlmError>>;

/// Handle returned by `Provider::chat_stream`.
///
/// Use `events()` for incremental processing or `collect()` to fold the full
/// stream into a `ChatResponse` (id and model will be empty when collected
/// from a streamed response — use `chat()` if you need them).
pub struct StreamHandle {
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) inner: BoxStream,
}

impl StreamHandle {
    pub(crate) fn new(model: impl Into<String>, inner: BoxStream) -> Self {
        Self { id: String::new(), model: model.into(), inner }
    }

    /// Raw event stream for incremental processing.
    pub fn events(&mut self) -> &mut impl Stream<Item = Result<StreamEvent, LlmError>> {
        &mut self.inner
    }

    /// Fold all events into a complete `ChatResponse`.
    pub async fn collect(mut self) -> Result<ChatResponse, LlmError> {
        let mut text_blocks: HashMap<usize, String> = HashMap::new();
        let mut reasoning_blocks: HashMap<usize, String> = HashMap::new();
        // index → (id, name)
        let mut tool_starts: HashMap<usize, (String, String)> = HashMap::new();
        let mut tool_args: HashMap<usize, String> = HashMap::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut usage = Usage::default();

        while let Some(event) = self.inner.next().await {
            match event? {
                StreamEvent::ContentStart { index, kind } => match kind {
                    ContentKind::Text => { text_blocks.insert(index, String::new()); }
                    ContentKind::Reasoning => { reasoning_blocks.insert(index, String::new()); }
                    ContentKind::ToolInvocation { id, name } => {
                        tool_starts.insert(index, (id, name));
                        tool_args.insert(index, String::new());
                    }
                },
                StreamEvent::TextDelta { index, text } => {
                    text_blocks.entry(index).or_default().push_str(&text);
                }
                StreamEvent::ReasoningDelta { index, text } => {
                    reasoning_blocks.entry(index).or_default().push_str(&text);
                }
                StreamEvent::ToolDelta { index, arguments } => {
                    tool_args.entry(index).or_default().push_str(&arguments);
                }
                StreamEvent::ContentStop { .. } => {}
                StreamEvent::MessageStop { stop_reason: sr, usage: u } => {
                    stop_reason = sr;
                    usage = u;
                }
            }
        }

        // Assemble content blocks in index order
        let mut all: Vec<(usize, ResponseContent)> = Vec::new();
        for (idx, text) in text_blocks {
            all.push((idx, ResponseContent::Text(text)));
        }
        for (idx, text) in reasoning_blocks {
            all.push((idx, ResponseContent::Reasoning(text)));
        }
        for (idx, (tool_id, name)) in tool_starts {
            let args = tool_args.remove(&idx).unwrap_or_default();
            let input = serde_json::from_str(&args)
                .unwrap_or(serde_json::Value::String(args));
            all.push((idx, ResponseContent::ToolInvocation(ToolInvocation { id: tool_id, name, input })));
        }
        all.sort_by_key(|(idx, _)| *idx);
        let content = all.into_iter().map(|(_, c)| c).collect();

        Ok(ChatResponse { id: self.id, model: self.model, content, stop_reason, usage })
    }
}
```

- [ ] **Step 4: Update src/lib.rs**

```rust
pub mod error;
pub mod provider;
pub mod stream_handle;
pub mod types;

pub use error::LlmError;
pub use provider::Provider;
pub use stream_handle::StreamHandle;
pub use types::*;
```

- [ ] **Step 5: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/provider.rs src/stream_handle.rs src/lib.rs
git commit -m "feat: add Provider trait and StreamHandle"
```

---

### Task 8: OpenAI wire types (schema.rs)

**Files:**
- Create: `src/openai/mod.rs` (empty skeleton)
- Create: `src/openai/schema.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Verify a schema round-trip fails**

Add to `src/lib.rs`: `pub mod openai;`

Create `src/openai/mod.rs`:
```rust
pub(crate) mod schema;
```

Create `src/openai/schema.rs` (empty for now):
```rust
```

Add to test module:
```rust
#[test]
fn oai_response_deserializes() {
    use openai::schema::OaiResponse;
    let json = r#"{
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    }"#;
    let resp: OaiResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.id, "chatcmpl-1");
    assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello"));
}
```

Run: `cargo test`
Expected: FAIL — `OaiResponse` not defined

- [ ] **Step 2: Implement src/openai/schema.rs**

```rust
use serde::{Deserialize, Serialize};

// ── Request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(crate) struct OaiRequest {
    pub model: String,
    pub messages: Vec<OaiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OaiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<OaiStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OaiStreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OaiMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OaiContent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OaiToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// DeepSeek reasoning_content field (present when parse_reasoning_content is enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum OaiContent {
    Text(String),
    Parts(Vec<OaiContentPart>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OaiContentPart {
    Text { text: String },
    ImageUrl { image_url: OaiImageUrl },
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OaiImageUrl {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OaiTool {
    pub r#type: String,
    pub function: OaiFunction,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OaiFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct OaiToolCall {
    pub id: String,
    pub r#type: String,
    pub function: OaiToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct OaiToolCallFunction {
    pub name: String,
    /// JSON-encoded string (not a parsed object).
    pub arguments: String,
}

// ── Response ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(crate) struct OaiResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<OaiChoice>,
    #[serde(default)]
    pub usage: Option<OaiUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiChoice {
    pub message: OaiMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct OaiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<OaiPromptTokensDetails>,
    /// DeepSeek cache hit tokens.
    #[serde(default)]
    pub prompt_cache_hit_tokens: u32,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct OaiPromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

// ── Streaming ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(crate) struct OaiChunk {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<OaiStreamChoice>,
    #[serde(default)]
    pub usage: Option<OaiUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamChoice {
    pub delta: OaiDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct OaiDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<OaiStreamToolCall>,
    /// DeepSeek reasoning_content delta.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamToolCall {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<OaiStreamFunction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiStreamFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(crate) struct OaiErrorResponse {
    pub error: OaiErrorBody,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OaiErrorBody {
    pub message: String,
    pub r#type: String,
    #[serde(default)]
    pub code: Option<String>,
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/openai/mod.rs src/openai/schema.rs src/lib.rs
git commit -m "feat: add OpenAI wire types"
```

---

### Task 9: OpenAI conversion (convert.rs)

**Files:**
- Create: `src/openai/convert.rs`
- Modify: `src/openai/mod.rs`

- [ ] **Step 1: Write the failing test**

Add to test module in `src/lib.rs`:
```rust
#[test]
fn convert_simple_request_to_oai() {
    use openai::convert::to_oai;
    use types::{ChatRequest, Message, RequestContent};
    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::System("You are helpful.".into()))
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .temperature(0.5)
        .build();
    let oai = to_oai(&req, None, false).unwrap();
    assert_eq!(oai.model, "gpt-4o");
    assert_eq!(oai.max_tokens, 512);
    assert_eq!(oai.temperature, Some(0.5));
    assert_eq!(oai.messages.len(), 2);
    assert_eq!(oai.messages[0].role, "system");
    assert_eq!(oai.messages[1].role, "user");
}

#[test]
fn convert_oai_response_to_canonical() {
    use openai::{convert::from_oai, schema::{OaiResponse, OaiChoice, OaiMessage, OaiUsage}};
    use types::StopReason;
    let resp = OaiResponse {
        id: "chatcmpl-1".into(),
        model: "gpt-4o".into(),
        choices: vec![OaiChoice {
            message: OaiMessage {
                role: "assistant".into(),
                content: Some(openai::schema::OaiContent::Text("Hello!".into())),
                tool_calls: vec![],
                tool_call_id: None,
                reasoning_content: None,
            },
            finish_reason: Some("stop".into()),
        }],
        usage: Some(OaiUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            ..Default::default()
        }),
    };
    let canon = from_oai(resp, false).unwrap();
    assert_eq!(canon.id, "chatcmpl-1");
    assert_eq!(canon.text(), "Hello!");
    assert!(matches!(canon.stop_reason, StopReason::EndTurn));
    assert_eq!(canon.usage.input_tokens, 10);
    assert_eq!(canon.usage.output_tokens, 5);
}
```

Run: `cargo test`
Expected: FAIL — `convert` module missing

- [ ] **Step 2: Create src/openai/convert.rs**

```rust
use crate::{
    error::LlmError,
    types::*,
};
use super::schema::*;

pub(crate) fn to_oai(
    req: &ChatRequest,
    ext: Option<&OpenAIExt>,
    stream: bool,
) -> Result<OaiRequest, LlmError> {
    let messages = req.messages().iter()
        .map(message_to_oai)
        .collect::<Result<Vec<_>, _>>()?;

    let tools: Vec<OaiTool> = req.tools().iter().map(|t| OaiTool {
        r#type: "function".into(),
        function: OaiFunction {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.input_schema.clone(),
        },
    }).collect();

    let tool_choice = if tools.is_empty() {
        None
    } else {
        Some(tool_choice_to_oai(req.tool_choice()))
    };

    let mut oai = OaiRequest {
        model: req.model().into(),
        messages,
        tools,
        tool_choice,
        temperature: req.params().temperature(),
        max_tokens: req.params().max_tokens(),
        top_p: req.params().top_p(),
        stop: req.params().stop_sequences().to_vec(),
        stream,
        stream_options: if stream {
            Some(OaiStreamOptions { include_usage: true })
        } else {
            None
        },
        seed: None,
        frequency_penalty: None,
        presence_penalty: None,
        user: None,
    };

    if let Some(ext) = ext {
        oai.seed = ext.seed;
        oai.frequency_penalty = ext.frequency_penalty;
        oai.presence_penalty = ext.presence_penalty;
        oai.user = ext.user.clone();
    }

    Ok(oai)
}

fn message_to_oai(msg: &Message) -> Result<OaiMessage, LlmError> {
    Ok(match msg {
        Message::System(text) => OaiMessage {
            role: "system".into(),
            content: Some(OaiContent::Text(text.clone())),
            tool_calls: vec![],
            tool_call_id: None,
            reasoning_content: None,
        },
        Message::User(parts) => {
            let content = if parts.len() == 1 {
                if let RequestContent::Text(t) = &parts[0] {
                    OaiContent::Text(t.clone())
                } else {
                    OaiContent::Parts(parts.iter().map(req_content_to_part).collect::<Result<_, _>>()?)
                }
            } else {
                OaiContent::Parts(parts.iter().map(req_content_to_part).collect::<Result<_, _>>()?)
            };
            OaiMessage {
                role: "user".into(),
                content: Some(content),
                tool_calls: vec![],
                tool_call_id: None,
                reasoning_content: None,
            }
        },
        Message::Assistant(parts) => {
            let mut text_bufs: Vec<&str> = Vec::new();
            let mut tool_calls: Vec<OaiToolCall> = Vec::new();
            let mut reasoning: Option<String> = None;

            for p in parts {
                match p {
                    ResponseContent::Text(t) => text_bufs.push(t),
                    ResponseContent::ToolInvocation(ti) => tool_calls.push(OaiToolCall {
                        id: ti.id.clone(),
                        r#type: "function".into(),
                        function: OaiToolCallFunction {
                            name: ti.name.clone(),
                            arguments: ti.input.to_string(),
                        },
                    }),
                    ResponseContent::Reasoning(r) => reasoning = Some(r.clone()),
                }
            }

            let content = if text_bufs.is_empty() {
                None
            } else {
                Some(OaiContent::Text(text_bufs.join("")))
            };

            OaiMessage {
                role: "assistant".into(),
                content,
                tool_calls,
                tool_call_id: None,
                reasoning_content: reasoning,
            }
        },
        Message::Tool { invocation_id, content, .. } => {
            let text = content.iter()
                .filter_map(|c| if let RequestContent::Text(t) = c { Some(t.as_str()) } else { None })
                .collect::<Vec<_>>()
                .join("\n");
            OaiMessage {
                role: "tool".into(),
                content: Some(OaiContent::Text(text)),
                tool_calls: vec![],
                tool_call_id: Some(invocation_id.clone()),
                reasoning_content: None,
            }
        },
    })
}

fn req_content_to_part(c: &RequestContent) -> Result<OaiContentPart, LlmError> {
    Ok(match c {
        RequestContent::Text(t) => OaiContentPart::Text { text: t.clone() },
        RequestContent::Image(src) => {
            use crate::types::ImageSource;
            let url = match src {
                ImageSource::Url(u) => u.clone(),
                ImageSource::Base64 { media_type, data } => {
                    format!("data:{};base64,{}", media_type, data)
                }
            };
            OaiContentPart::ImageUrl { image_url: OaiImageUrl { url } }
        }
    })
}

fn tool_choice_to_oai(tc: &ToolChoice) -> serde_json::Value {
    match tc {
        ToolChoice::Auto => serde_json::json!("auto"),
        ToolChoice::None => serde_json::json!("none"),
        ToolChoice::Required => serde_json::json!("required"),
        ToolChoice::Specific(name) => serde_json::json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

pub(crate) fn from_oai(
    resp: OaiResponse,
    parse_reasoning: bool,
) -> Result<ChatResponse, LlmError> {
    let choice = resp.choices.into_iter().next()
        .ok_or_else(|| LlmError::ServerError("response contained no choices".into()))?;

    let stop_reason = parse_stop_reason(choice.finish_reason.as_deref());
    let usage = resp.usage.map(|u| oai_usage_to_canonical(u, parse_reasoning))
        .unwrap_or_default();

    let content = oai_message_to_content(choice.message, parse_reasoning)?;

    Ok(ChatResponse { id: resp.id, model: resp.model, content, stop_reason, usage })
}

fn oai_message_to_content(
    msg: OaiMessage,
    parse_reasoning: bool,
) -> Result<Vec<ResponseContent>, LlmError> {
    let mut result = Vec::new();

    if parse_reasoning {
        if let Some(rc) = msg.reasoning_content {
            if !rc.is_empty() {
                result.push(ResponseContent::Reasoning(rc));
            }
        }
    }

    match msg.content {
        Some(OaiContent::Text(t)) if !t.is_empty() => result.push(ResponseContent::Text(t)),
        Some(OaiContent::Parts(parts)) => {
            for p in parts {
                if let OaiContentPart::Text { text } = p {
                    result.push(ResponseContent::Text(text));
                }
            }
        }
        _ => {}
    }

    for tc in msg.tool_calls {
        let input = serde_json::from_str(&tc.function.arguments)
            .unwrap_or(serde_json::Value::String(tc.function.arguments));
        result.push(ResponseContent::ToolInvocation(ToolInvocation {
            id: tc.id,
            name: tc.function.name,
            input,
        }));
    }

    Ok(result)
}

pub(crate) fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls") => StopReason::ToolUse,
        Some("content_filter") => StopReason::ContentFilter,
        Some(other) => StopReason::Other(other.into()),
        None => StopReason::EndTurn,
    }
}

pub(crate) fn oai_usage_to_canonical(u: OaiUsage, _parse_reasoning: bool) -> Usage {
    let cached = u.prompt_tokens_details
        .map(|d| d.cached_tokens)
        .unwrap_or(0)
        .max(u.prompt_cache_hit_tokens);
    Usage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        cached_input_tokens: cached,
        cache_creation_input_tokens: 0,
        reasoning_tokens: 0,
    }
}

// ── OpenAI native extension types ────────────────────────────────────────────

/// OpenAI-specific request extensions, constructed via `OpenAIExt::builder`.
#[derive(Debug, Clone)]
pub struct OpenAIExt {
    pub(crate) seed: Option<u64>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) user: Option<String>,
    /// Enable DeepSeek `reasoning_content` field parsing.
    pub(crate) parse_reasoning_content: bool,
    /// Tolerate SSE keep-alive lines (DeepSeek thinking mode).
    pub(crate) tolerant_keepalive: bool,
}

impl OpenAIExt {
    pub fn builder() -> OpenAIExtBuilder { OpenAIExtBuilder::default() }
}

#[derive(Default)]
pub struct OpenAIExtBuilder {
    seed: Option<u64>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    user: Option<String>,
    parse_reasoning_content: bool,
    tolerant_keepalive: bool,
}

impl OpenAIExtBuilder {
    pub fn seed(mut self, s: u64) -> Self { self.seed = Some(s); self }
    pub fn frequency_penalty(mut self, f: f32) -> Self { self.frequency_penalty = Some(f); self }
    pub fn presence_penalty(mut self, p: f32) -> Self { self.presence_penalty = Some(p); self }
    pub fn user(mut self, u: impl Into<String>) -> Self { self.user = Some(u.into()); self }
    pub fn parse_reasoning_content(mut self, v: bool) -> Self { self.parse_reasoning_content = v; self }
    pub fn tolerant_keepalive(mut self, v: bool) -> Self { self.tolerant_keepalive = v; self }
    pub fn build(self) -> OpenAIExt {
        OpenAIExt {
            seed: self.seed,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            user: self.user,
            parse_reasoning_content: self.parse_reasoning_content,
            tolerant_keepalive: self.tolerant_keepalive,
        }
    }
}

/// A `ChatRequest` with OpenAI-specific extensions.
pub struct OpenAIRequest {
    pub base: ChatRequest,
    pub ext: OpenAIExt,
}
```

- [ ] **Step 3: Update src/openai/mod.rs**

```rust
pub(crate) mod schema;
pub(crate) mod convert;
pub use convert::{OpenAIExt, OpenAIExtBuilder, OpenAIRequest};
```

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openai/convert.rs src/openai/mod.rs
git commit -m "feat: add OpenAI request/response conversion"
```

---

### Task 10: OpenAI stream parser

**Files:**
- Create: `src/openai/stream.rs`
- Modify: `src/openai/mod.rs`

- [ ] **Step 1: Write the failing test**

Add to test module in `src/lib.rs`:
```rust
#[test]
fn oai_stream_parser_emits_events() {
    use openai::stream::process_chunk;
    use openai::schema::{OaiChunk, OaiStreamChoice, OaiDelta};
    use types::{StreamEvent, ContentKind};

    let mut state = openai::stream::OaiStreamState::new();

    let chunk = OaiChunk {
        id: "c1".into(),
        model: "gpt-4o".into(),
        choices: vec![OaiStreamChoice {
            delta: OaiDelta { content: Some("Hello".into()), ..Default::default() },
            finish_reason: None,
        }],
        usage: None,
    };
    let events = process_chunk(&mut state, chunk, false);
    // First content delta should emit ContentStart then TextDelta
    assert_eq!(events.len(), 2);
    assert!(matches!(&events[0], Ok(StreamEvent::ContentStart { index: 0, kind: ContentKind::Text })));
    assert!(matches!(&events[1], Ok(StreamEvent::TextDelta { index: 0, .. })));
}
```

Run: `cargo test`
Expected: FAIL — `stream` module missing

- [ ] **Step 2: Create src/openai/stream.rs**

```rust
use std::collections::HashMap;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt, TryStreamExt};
use crate::{error::LlmError, stream_handle::StreamHandle, types::*};
use super::{convert::{oai_usage_to_canonical, parse_stop_reason}, schema::*};

pub(crate) struct OaiStreamState {
    reasoning_index: Option<usize>,
    text_index: Option<usize>,
    // OAI tool_calls[i].index → canonical ContentBlock index
    tool_index_map: HashMap<usize, usize>,
    next_index: usize,
    // Blocks opened (ContentStart emitted) but not yet closed
    open_blocks: Vec<usize>,
    // finish_reason from a choices chunk, consumed when usage chunk arrives
    pending_stop: Option<String>,
}

impl OaiStreamState {
    pub(crate) fn new() -> Self {
        Self {
            reasoning_index: None,
            text_index: None,
            tool_index_map: HashMap::new(),
            next_index: 0,
            open_blocks: Vec::new(),
            pending_stop: None,
        }
    }

    fn alloc(&mut self) -> usize {
        let i = self.next_index;
        self.next_index += 1;
        i
    }

    fn close_block(&mut self, idx: usize, out: &mut Vec<Result<StreamEvent, LlmError>>) {
        if let Some(pos) = self.open_blocks.iter().position(|&x| x == idx) {
            self.open_blocks.remove(pos);
            out.push(Ok(StreamEvent::ContentStop { index: idx }));
        }
    }
}

pub(crate) fn process_chunk(
    state: &mut OaiStreamState,
    chunk: OaiChunk,
    parse_reasoning: bool,
) -> Vec<Result<StreamEvent, LlmError>> {
    let mut out: Vec<Result<StreamEvent, LlmError>> = Vec::new();

    // Usage-only chunk (choices is empty): close open blocks + emit MessageStop
    if chunk.choices.is_empty() {
        if let Some(usage) = chunk.usage {
            let open: Vec<usize> = state.open_blocks.drain(..).collect();
            for idx in open {
                out.push(Ok(StreamEvent::ContentStop { index: idx }));
            }
            let stop_reason = parse_stop_reason(state.pending_stop.as_deref());
            state.pending_stop = None;
            out.push(Ok(StreamEvent::MessageStop {
                stop_reason,
                usage: oai_usage_to_canonical(usage, parse_reasoning),
            }));
        }
        return out;
    }

    for choice in chunk.choices {
        let delta = choice.delta;

        // Reasoning (DeepSeek only)
        if parse_reasoning {
            if let Some(rc) = delta.reasoning_content {
                if !rc.is_empty() {
                    if state.reasoning_index.is_none() {
                        let idx = state.alloc();
                        state.reasoning_index = Some(idx);
                        state.open_blocks.push(idx);
                        out.push(Ok(StreamEvent::ContentStart { index: idx, kind: ContentKind::Reasoning }));
                    }
                    out.push(Ok(StreamEvent::ReasoningDelta {
                        index: state.reasoning_index.unwrap(),
                        text: rc,
                    }));
                }
            }
        }

        // Text content
        if let Some(content) = delta.content {
            if !content.is_empty() {
                // Close reasoning block when text starts
                if let Some(ri) = state.reasoning_index {
                    state.close_block(ri, &mut out);
                }
                if state.text_index.is_none() {
                    let idx = state.alloc();
                    state.text_index = Some(idx);
                    state.open_blocks.push(idx);
                    out.push(Ok(StreamEvent::ContentStart { index: idx, kind: ContentKind::Text }));
                }
                out.push(Ok(StreamEvent::TextDelta {
                    index: state.text_index.unwrap(),
                    text: content,
                }));
            }
        }

        // Tool calls
        for tc in delta.tool_calls {
            let oai_idx = tc.index;
            if !state.tool_index_map.contains_key(&oai_idx) {
                let id = tc.id.clone().unwrap_or_default();
                let name = tc.function.as_ref()
                    .and_then(|f| f.name.clone())
                    .unwrap_or_default();
                let canonical_idx = state.alloc();
                state.tool_index_map.insert(oai_idx, canonical_idx);
                state.open_blocks.push(canonical_idx);
                out.push(Ok(StreamEvent::ContentStart {
                    index: canonical_idx,
                    kind: ContentKind::ToolInvocation { id, name },
                }));
            }

            if let Some(func) = tc.function {
                if let Some(args) = func.arguments {
                    if !args.is_empty() {
                        let canonical_idx = *state.tool_index_map.get(&oai_idx).unwrap();
                        out.push(Ok(StreamEvent::ToolDelta { index: canonical_idx, arguments: args }));
                    }
                }
            }
        }

        // Finish reason
        if let Some(fr) = choice.finish_reason {
            state.pending_stop = Some(fr);
        }
    }

    out
}

pub(crate) fn parse_stream(
    response: reqwest::Response,
    parse_reasoning: bool,
) -> futures::stream::BoxStream<'static, Result<StreamEvent, LlmError>> {
    let byte_stream = response.bytes_stream()
        .map_err(LlmError::Network);

    struct State {
        sse: futures::stream::BoxStream<
            'static,
            Result<
                eventsource_stream::Event,
                eventsource_stream::EventStreamError<LlmError>,
            >,
        >,
        parser: OaiStreamState,
        pending: std::collections::VecDeque<Result<StreamEvent, LlmError>>,
        done: bool,
        parse_reasoning: bool,
    }

    let state = State {
        sse: byte_stream.eventsource().boxed(),
        parser: OaiStreamState::new(),
        pending: std::collections::VecDeque::new(),
        done: false,
        parse_reasoning,
    };

    futures::stream::unfold(state, |mut st| async move {
        loop {
            if let Some(event) = st.pending.pop_front() {
                return Some((event, st));
            }
            if st.done {
                return None;
            }
            match st.sse.next().await {
                None => return None,
                Some(Err(e)) => {
                    return Some((Err(LlmError::Stream(e.to_string())), st));
                }
                Some(Ok(ev)) => {
                    if ev.data == "[DONE]" {
                        st.done = true;
                        return None;
                    }
                    match serde_json::from_str::<OaiChunk>(&ev.data) {
                        Err(e) => return Some((Err(LlmError::Decode(e)), st)),
                        Ok(chunk) => {
                            let events = process_chunk(&mut st.parser, chunk, st.parse_reasoning);
                            st.pending.extend(events);
                        }
                    }
                }
            }
        }
    })
    .boxed()
}
```

- [ ] **Step 3: Update src/openai/mod.rs**

```rust
pub(crate) mod convert;
pub(crate) mod schema;
pub(crate) mod stream;
pub use convert::{OpenAIExt, OpenAIExtBuilder, OpenAIRequest};
```

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openai/stream.rs src/openai/mod.rs
git commit -m "feat: add OpenAI SSE stream parser"
```

---

### Task 11: OpenAIProvider

**Files:**
- Modify: `src/openai/mod.rs` (add provider struct and impl)

- [ ] **Step 1: Write the failing test (compile-only)**

Add to test module in `src/lib.rs`:
```rust
#[test]
fn openai_provider_builds() {
    let _p = openai::OpenAIProvider::builder("test-key").build();
}
```

Run: `cargo test`
Expected: FAIL — `OpenAIProvider` not found

- [ ] **Step 2: Add OpenAIProvider to src/openai/mod.rs**

Replace/extend the file:

```rust
pub(crate) mod convert;
pub(crate) mod schema;
pub(crate) mod stream;

pub use convert::{OpenAIExt, OpenAIExtBuilder, OpenAIRequest};

use std::time::Duration;
use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use crate::{
    error::LlmError,
    provider::Provider,
    stream_handle::StreamHandle,
    types::*,
};
use convert::{from_oai, to_oai};
use schema::OaiErrorResponse;

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// OpenAI (and OpenAI-compatible) provider, constructed via `OpenAIProvider::builder`.
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    parse_reasoning_content: bool,
}

pub struct OpenAIProviderBuilder {
    api_key: String,
    base_url: String,
    parse_reasoning_content: bool,
    tolerant_keepalive: bool,
}

impl OpenAIProvider {
    pub fn builder(api_key: impl Into<String>) -> OpenAIProviderBuilder {
        OpenAIProviderBuilder {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.into(),
            parse_reasoning_content: false,
            tolerant_keepalive: false,
        }
    }

    async fn send(
        &self,
        oai_req: &schema::OaiRequest,
        stream: bool,
    ) -> Result<reqwest::Response, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json")
            .json(oai_req)
            .send()
            .await?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        // Parse error body
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs);

        let body = response.bytes().await?;
        let err_msg = serde_json::from_slice::<OaiErrorResponse>(&body)
            .map(|e| e.error.message)
            .unwrap_or_else(|_| String::from_utf8_lossy(&body).into_owned());

        Err(match status.as_u16() {
            401 => LlmError::Unauthorized,
            403 => LlmError::Forbidden,
            402 => LlmError::Provider { code: "payment_required".into(), message: err_msg },
            429 => LlmError::RateLimit { retry_after },
            500 | 502 | 503 => LlmError::ServerError(err_msg),
            529 => LlmError::Overloaded { retry_after },
            _ => LlmError::InvalidRequest(err_msg),
        })
    }

    /// Send a request with provider-specific extensions.
    pub async fn chat_native(
        &self,
        req: &OpenAIRequest,
    ) -> Result<ChatResponse, LlmError> {
        let oai_req = to_oai(&req.base, Some(&req.ext), false)?;
        let response = self.send(&oai_req, false).await?;
        let body: schema::OaiResponse = response.json().await?;
        let parse_reasoning = self.parse_reasoning_content || req.ext.parse_reasoning_content;
        from_oai(body, parse_reasoning)
    }

    /// Stream a request with provider-specific extensions.
    pub async fn chat_stream_native(
        &self,
        req: &OpenAIRequest,
    ) -> Result<StreamHandle, LlmError> {
        let oai_req = to_oai(&req.base, Some(&req.ext), true)?;
        let response = self.send(&oai_req, true).await?;
        let parse_reasoning = self.parse_reasoning_content || req.ext.parse_reasoning_content;
        let s = stream::parse_stream(response, parse_reasoning);
        Ok(StreamHandle::new(req.base.model(), s))
    }
}

impl OpenAIProviderBuilder {
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into(); self
    }
    pub fn parse_reasoning_content(mut self, v: bool) -> Self {
        self.parse_reasoning_content = v; self
    }
    pub fn tolerant_keepalive(mut self, v: bool) -> Self {
        self.tolerant_keepalive = v; self
    }
    pub fn build(self) -> OpenAIProvider {
        OpenAIProvider {
            client: reqwest::Client::new(),
            api_key: self.api_key,
            base_url: self.base_url,
            parse_reasoning_content: self.parse_reasoning_content,
        }
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let oai_req = to_oai(req, None, false)?;
        let response = self.send(&oai_req, false).await?;
        let body: schema::OaiResponse = response.json().await?;
        from_oai(body, self.parse_reasoning_content)
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<StreamHandle, LlmError> {
        let oai_req = to_oai(req, None, true)?;
        let response = self.send(&oai_req, true).await?;
        let s = stream::parse_stream(response, self.parse_reasoning_content);
        Ok(StreamHandle::new(req.model(), s))
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo check && cargo test`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/openai/mod.rs
git commit -m "feat: add OpenAIProvider with chat and chat_stream"
```

---

### Task 12: DeepSeek factory and public API surface

**Files:**
- Create: `src/deepseek.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

Add to test module in `src/lib.rs`:
```rust
#[test]
fn deepseek_factory_returns_openai_provider() {
    use openai::OpenAIProvider;
    let _p: OpenAIProvider = deepseek::client("test-key");
}
```

Run: `cargo test`
Expected: FAIL — `deepseek` module missing

- [ ] **Step 2: Create src/deepseek.rs**

```rust
use crate::openai::OpenAIProvider;

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";

/// Create an `OpenAIProvider` pre-configured for the DeepSeek API.
///
/// DeepSeek uses the OpenAI Chat Completions schema at a different base URL,
/// with `reasoning_content` field support enabled for thinking-mode models.
pub fn client(api_key: impl Into<String>) -> OpenAIProvider {
    OpenAIProvider::builder(api_key)
        .base_url(DEEPSEEK_BASE_URL)
        .parse_reasoning_content(true)
        .tolerant_keepalive(true)
        .build()
}
```

- [ ] **Step 3: Add deepseek module and finalise lib.rs**

```rust
pub mod deepseek;
pub mod error;
pub mod openai;
pub mod provider;
pub mod stream_handle;
pub mod types;

pub use error::LlmError;
pub use provider::Provider;
pub use stream_handle::StreamHandle;
pub use types::*;
```

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepseek.rs src/lib.rs
git commit -m "feat: add DeepSeek factory function"
```

---

### Task 13: Test fixtures

**Files:**
- Create: `tests/fixtures/openai/chat_simple.json`
- Create: `tests/fixtures/openai/chat_tool_call.json`
- Create: `tests/fixtures/openai/chat_after_tool.json`
- Create: `tests/fixtures/openai/error_rate_limit.json`
- Create: `tests/fixtures/openai/error_server.json`
- Create: `tests/fixtures/openai/error_unauthorized.json`
- Create: `tests/fixtures/openai/stream_simple.txt`
- Create: `tests/fixtures/openai/stream_tool_call.txt`
- Create: `tests/fixtures/deepseek/chat_simple.json`
- Create: `tests/fixtures/deepseek/chat_reasoning.json`
- Create: `tests/fixtures/deepseek/stream_reasoning.txt`

- [ ] **Step 1: Create fixture directories**

```bash
mkdir -p tests/fixtures/openai tests/fixtures/deepseek tests/common
```

- [ ] **Step 2: Create tests/fixtures/openai/chat_simple.json**

```json
{
  "id": "chatcmpl-test123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 9,
    "total_tokens": 19,
    "prompt_tokens_details": {"cached_tokens": 5}
  }
}
```

- [ ] **Step 3: Create tests/fixtures/openai/chat_tool_call.json**

```json
{
  "id": "chatcmpl-tool123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
}
```

- [ ] **Step 4: Create tests/fixtures/openai/chat_after_tool.json**

```json
{
  "id": "chatcmpl-after123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The weather in Paris is sunny and 22°C."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95}
}
```

- [ ] **Step 5: Create error fixtures**

`tests/fixtures/openai/error_rate_limit.json`:
```json
{"error": {"message": "Rate limit exceeded. Please retry after 60 seconds.", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}
```

`tests/fixtures/openai/error_server.json`:
```json
{"error": {"message": "Internal server error", "type": "server_error", "code": null}}
```

`tests/fixtures/openai/error_unauthorized.json`:
```json
{"error": {"message": "Incorrect API key provided", "type": "authentication_error", "code": "invalid_api_key"}}
```

- [ ] **Step 6: Create tests/fixtures/openai/stream_simple.txt**

```
data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":0}}}

data: [DONE]

```

- [ ] **Step 7: Create tests/fixtures/openai/stream_tool_call.txt**

```
data: {"id":"chatcmpl-tc1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":null},"finish_reason":null}]}

data: {"id":"chatcmpl-tc1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-tc1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-tc1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"Paris\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-tc1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"chatcmpl-tc1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":50,"completion_tokens":15,"total_tokens":65}}

data: [DONE]

```

- [ ] **Step 8: Create DeepSeek fixtures**

`tests/fixtures/deepseek/chat_simple.json`:
```json
{
  "id": "chatcmpl-ds-simple",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "deepseek-v4-flash",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello from DeepSeek!"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 5}
}
```

`tests/fixtures/deepseek/chat_reasoning.json`:
```json
{
  "id": "chatcmpl-ds-reason",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "deepseek-v4-flash",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The answer is 42.",
      "reasoning_content": "Let me think step by step. Based on my reasoning, the answer is 42."
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 15}
}
```

`tests/fixtures/deepseek/stream_reasoning.txt`:
```
data: {"id":"chatcmpl-ds-str","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me"},"finish_reason":null}]}

data: {"id":"chatcmpl-ds-str","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"reasoning_content":" think..."},"finish_reason":null}]}

data: {"id":"chatcmpl-ds-str","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"content":"Answer: 42"},"finish_reason":null}]}

data: {"id":"chatcmpl-ds-str","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-ds-str","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4-flash","choices":[],"usage":{"prompt_tokens":15,"completion_tokens":8,"total_tokens":23,"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":15}}

data: [DONE]

```

- [ ] **Step 9: Commit**

```bash
git add tests/
git commit -m "test: add wiremock fixture files for OpenAI and DeepSeek"
```

---

### Task 14: Test common utilities

**Files:**
- Create: `tests/common/mod.rs`

- [ ] **Step 1: Create tests/common/mod.rs**

```rust
use wiremock::{Mock, MockServer, ResponseTemplate};
use wiremock::matchers::{method, path};

/// Mount a mock that replies to POST /v1/chat/completions with the given JSON body.
pub async fn mock_chat(server: &MockServer, body: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "application/json"),
        )
        .mount(server)
        .await;
}

/// Mount a mock that replies to POST /v1/chat/completions with an SSE stream.
pub async fn mock_stream(server: &MockServer, body: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "text/event-stream")
                .append_header("cache-control", "no-cache"),
        )
        .mount(server)
        .await;
}

/// Mount a mock that replies with an HTTP error status and JSON error body.
pub async fn mock_error(server: &MockServer, status: u16, body: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(status)
                .set_body_raw(body, "application/json"),
        )
        .mount(server)
        .await;
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo test --test '*' 2>&1 | head -20`
Expected: Compiles (no test results yet since test files don't exist)

- [ ] **Step 3: Commit**

```bash
git add tests/common/mod.rs
git commit -m "test: add wiremock common helpers"
```

---

### Task 15: OpenAI chat tests

**Files:**
- Create: `tests/openai_chat.rs`

- [ ] **Step 1: Create tests/openai_chat.rs**

```rust
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

    let invocation = resp1.tool_invocations().next().unwrap();
    messages.push(resp1.into_message());
    messages.push(Message::Tool {
        invocation_id: invocation.id.clone(),
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
    assert!(matches!(err, LlmError::RateLimit { retry_after: Some(_) }));
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
```

- [ ] **Step 2: Run tests**

Run: `cargo test --test openai_chat`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/openai_chat.rs
git commit -m "test: add offline OpenAI chat tests"
```

---

### Task 16: OpenAI stream tests

**Files:**
- Create: `tests/openai_stream.rs`

- [ ] **Step 1: Create tests/openai_stream.rs**

```rust
mod common;

use llm_adapter::{
    openai::OpenAIProvider,
    types::*,
    Provider,
};
use futures::StreamExt;
use wiremock::MockServer;

fn provider(server: &MockServer) -> OpenAIProvider {
    OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .build()
}

#[tokio::test]
async fn stream_simple_collects_to_response() {
    let server = MockServer::start().await;
    common::mock_stream(&server, include_str!("fixtures/openai/stream_simple.txt")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert_eq!(resp.text(), "Hello!");
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert_eq!(resp.usage.input_tokens, 10);
    assert_eq!(resp.usage.output_tokens, 2);
}

#[tokio::test]
async fn stream_events_fire_in_order() {
    let server = MockServer::start().await;
    common::mock_stream(&server, include_str!("fixtures/openai/stream_simple.txt")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Hi".into())]))
        .build();

    let mut handle = provider(&server).chat_stream(&req).await.unwrap();
    let events = handle.events();

    let mut content_starts = 0usize;
    let mut text_deltas = 0usize;
    let mut content_stops = 0usize;
    let mut message_stops = 0usize;

    while let Some(event) = events.next().await {
        match event.unwrap() {
            StreamEvent::ContentStart { .. } => content_starts += 1,
            StreamEvent::TextDelta { .. } => text_deltas += 1,
            StreamEvent::ContentStop { .. } => content_stops += 1,
            StreamEvent::MessageStop { .. } => message_stops += 1,
            _ => {}
        }
    }

    assert_eq!(content_starts, 1);
    assert!(text_deltas >= 1);
    assert_eq!(content_stops, 1);
    assert_eq!(message_stops, 1);
}

#[tokio::test]
async fn stream_tool_call_collects_invocation() {
    let server = MockServer::start().await;
    common::mock_stream(&server, include_str!("fixtures/openai/stream_tool_call.txt")).await;

    let req = ChatRequest::builder("gpt-4o", 512)
        .message(Message::User(vec![RequestContent::Text("Weather in Paris?".into())]))
        .tool(ToolDef {
            name: "get_weather".into(),
            description: "Get weather".into(),
            input_schema: serde_json::json!({}),
        })
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

- [ ] **Step 2: Run tests**

Run: `cargo test --test openai_stream`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/openai_stream.rs
git commit -m "test: add offline OpenAI stream tests"
```

---

### Task 17: DeepSeek tests

**Files:**
- Create: `tests/deepseek_chat.rs`
- Create: `tests/deepseek_stream.rs`

- [ ] **Step 1: Create tests/deepseek_chat.rs**

```rust
mod common;

use llm_adapter::{
    deepseek,
    types::*,
    Provider,
};
use wiremock::MockServer;

fn provider(server: &MockServer) -> llm_adapter::openai::OpenAIProvider {
    // deepseek::client() returns OpenAIProvider; rebuild it with the mock base URL
    llm_adapter::openai::OpenAIProvider::builder("test-key")
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
    assert_eq!(resp.text(), "Hello from DeepSeek!");
    assert_eq!(resp.usage.cached_input_tokens, 0);
}

#[tokio::test]
async fn deepseek_reasoning_response() {
    let server = MockServer::start().await;
    common::mock_chat(&server, include_str!("fixtures/deepseek/chat_reasoning.json")).await;

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text("What is the answer?".into())]))
        .build();

    let resp = provider(&server).chat(&req).await.unwrap();

    // Reasoning should appear as ResponseContent::Reasoning before the text
    assert!(resp.reasoning().is_some());
    assert!(resp.reasoning().unwrap().contains("think"));
    assert_eq!(resp.text(), "The answer is 42.");
    // Reasoning block comes first (index 0), text second (index 1)
    assert!(matches!(&resp.content[0], ResponseContent::Reasoning(_)));
    assert!(matches!(&resp.content[1], ResponseContent::Text(_)));
}
```

- [ ] **Step 2: Create tests/deepseek_stream.rs**

```rust
mod common;

use llm_adapter::{
    openai::OpenAIProvider,
    types::*,
    Provider,
};
use wiremock::MockServer;

fn provider(server: &MockServer) -> OpenAIProvider {
    OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .parse_reasoning_content(true)
        .build()
}

#[tokio::test]
async fn deepseek_stream_reasoning_then_text() {
    let server = MockServer::start().await;
    common::mock_stream(&server, include_str!("fixtures/deepseek/stream_reasoning.txt")).await;

    let req = ChatRequest::builder("deepseek-v4-flash", 512)
        .message(Message::User(vec![RequestContent::Text("What is the answer?".into())]))
        .build();

    let handle = provider(&server).chat_stream(&req).await.unwrap();
    let resp = handle.collect().await.unwrap();

    assert!(resp.reasoning().is_some());
    let reasoning = resp.reasoning().unwrap();
    assert!(reasoning.contains("Let me"));
    assert_eq!(resp.text(), "Answer: 42");
    // Reasoning block has lower index than text block
    let reasoning_idx = resp.content.iter().position(|c| matches!(c, ResponseContent::Reasoning(_))).unwrap();
    let text_idx = resp.content.iter().position(|c| matches!(c, ResponseContent::Text(_))).unwrap();
    assert!(reasoning_idx < text_idx);
}
```

- [ ] **Step 3: Run all tests**

Run: `cargo test`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/deepseek_chat.rs tests/deepseek_stream.rs
git commit -m "test: add offline DeepSeek chat and stream tests"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task |
|---|---|
| Canonical types (RequestContent, ResponseContent, ImageSource, ToolInvocation) | Task 3 |
| Message ADT with role-encoded variants | Task 4 |
| ChatRequest + SamplingParams builders | Task 5 |
| ChatResponse helpers (text, tool_invocations, reasoning, into_message) | Task 6 |
| StreamEvent + ContentKind | Task 6 |
| Provider trait (chat, chat_stream) | Task 7 |
| StreamHandle (events, collect) | Task 7 |
| LlmError with all variants | Task 2 |
| OpenAI wire types | Task 8 |
| OpenAI conversion (ChatRequest ↔ OAI, OAI → canonical) | Task 9 |
| OpenAI SSE stream parser | Task 10 |
| OpenAIProvider (chat, chat_stream, chat_native, chat_stream_native) | Task 11 |
| OpenAIExt builder + OpenAIRequest | Task 9 |
| DeepSeek factory | Task 12 |
| parse_reasoning_content flag | Tasks 9, 10, 11 |
| Offline tests for all scenarios | Tasks 13–17 |
| Usage fields (cached_input, reasoning_tokens) | Tasks 3–9 |
| ToolChoice variants | Task 5 |
| Error normalization (401→Unauthorized, 429→RateLimit, 500→ServerError) | Tasks 11, 15 |

**Missing from Plan 1 (deferred to Plan 2):** Anthropic provider, AnthropicExt, AnthropicRequest/Response, cache_hints, thinking config.

### Placeholder scan

No TBD, TODO, or "similar to Task N" entries found.

### Type consistency check

- `ToolInvocation` defined in Task 3, used identically in Tasks 6, 7, 9
- `StopReason::ToolUse` used in tests matches the variant defined in Task 6
- `OaiUsage.prompt_cache_hit_tokens` deserialized in Task 8, consumed in Task 9 (`oai_usage_to_canonical`)
- `OaiStreamState` defined and exported in Task 10, imported in the unit test in `src/lib.rs`

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-03-foundation-openai-deepseek.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks

**2. Inline Execution** — execute tasks in this session with checkpoints

**Which approach?**
