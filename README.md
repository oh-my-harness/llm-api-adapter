# llm_adapter

*[中文版](README.zh-CN.md)*

A Rust adapter layer that normalises the wire formats of multiple LLM
providers (OpenAI Chat Completions, Anthropic Messages, DeepSeek, and any
OpenAI-compatible endpoint) into a single canonical type system, with
first-class support for provider-specific extensions, streaming, and tool
calling.

**Status:** pre-release (v0.1.0). Public API is unstable until v1.0.

[![codecov](https://codecov.io/gh/hhllhhyyds/llm-api-adapter/branch/main/graph/badge.svg)](https://codecov.io/gh/hhllhhyyds/llm-api-adapter)

## What it is

A small, focused library that gives you:

- One canonical request/response shape independent of any provider
- A `Provider` trait you can hold as `Box<dyn Provider>` and swap at runtime
- Streaming as a first-class type (`StreamHandle`), iterable as events or
  collected into a complete response
- An escape hatch (`OpenAIExt` / `extra_body`) for provider-specific fields
  without polluting the canonical layer
- Errors normalised across providers (`Unauthorized`, `RateLimit`,
  `Overloaded`, etc.) with `retry_after` exposed where available
- Offline-testable: every offline test runs against `wiremock` fixtures
  recorded from real APIs; `cargo test` needs no network and no API key

## What it isn't

To keep the surface area honest, this library deliberately does **not** do:

- Retry / backoff (the caller decides)
- Agent loops, prompt templating, conversation persistence
- Batch API, embeddings, TTS, image generation
- Azure OpenAI (different auth scheme + deployment-based routing)
- The stateful OpenAI Responses API

## Supported providers

| Provider | Universal path | Native extensions | Streaming | Tool calls | Reasoning | Structured output |
|---|---|---|---|---|---|---|
| OpenAI Chat Completions | ✅ | `OpenAIExt`: seed, penalties, `reasoning_effort`, `response_format`, `extra_body` | ✅ | ✅ | `reasoning_effort` (o-series / GPT-5) | ✅ (json_object, json_schema) |
| DeepSeek (`api.deepseek.com`) | ✅ (factory `deepseek::client`) | `OpenAIExt` (e.g. `extra_body: {"thinking": {"type": "disabled"}}`) | ✅ | ✅ | `reasoning_content` parsed into `ResponseContent::Reasoning` | via `extra_body` |
| Anthropic Messages | ✅ | `AnthropicExt`: top_k, thinking, metadata, cache_hints, `structured_output` | ✅ | ✅ | Extended Thinking (thinking block → `ResponseContent::Reasoning`) | ✅ (Strict Tool Use) |
| Any OpenAI-compatible gateway | ✅ via `base_url` + `chat_path` overrides | `extra_body` for non-standard fields | ✅ | ✅ | depends on endpoint | via `extra_body` |

Tested compatible gateways for the path-override path: GitHub Models
(`/chat/completions` no `/v1`), OpenRouter (`/api/v1/...`), Groq
(`/openai/v1/...`). The library makes no special accommodation for these —
they work because the schema is OpenAI Chat Completions.

## Quick start

`Cargo.toml`:

```toml
[dependencies]
llm_adapter = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

### Simple chat

```rust
use llm_adapter::{openai::OpenAIProvider, types::*, Provider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let p = OpenAIProvider::builder(std::env::var("OPENAI_API_KEY")?)
        .build();

    let req = ChatRequest::builder("gpt-4o-mini", 512)
        .message(Message::System("You are concise.".into()))
        .message(Message::User(vec![RequestContent::Text(
            "Say hi in five words.".into(),
        )]))
        .temperature(0.7)
        .build();

    let resp = p.chat(&req).await?;
    println!("{}", resp.text());
    println!("tokens: {} in / {} out", resp.usage.input_tokens, resp.usage.output_tokens);
    Ok(())
}
```

### DeepSeek with one factory call

```rust
use llm_adapter::{deepseek, types::*, Provider};

let p = deepseek::client(std::env::var("DEEPSEEK_API_KEY")?);
// Returns a pre-configured OpenAIProvider:
//   - base_url:                https://api.deepseek.com
//   - parse_reasoning_content: enabled
//   - tolerant_keepalive:      enabled

let req = ChatRequest::builder("deepseek-v4-flash", 2048)
    .message(Message::User(vec![RequestContent::Text(
        "What is 7 times 8?".into(),
    )]))
    .build();

let resp = p.chat(&req).await?;
println!("answer: {}", resp.text());
if let Some(reasoning) = resp.reasoning() {
    println!("(thought process: {})", reasoning);
}
```

### Runtime provider switching

```rust
use llm_adapter::{anthropic::AnthropicProvider, deepseek, openai::OpenAIProvider, types::*, Provider};

let provider: Box<dyn Provider> = if use_deepseek {
    Box::new(deepseek::client(deepseek_key))
} else if use_anthropic {
    Box::new(AnthropicProvider::builder(anthropic_key).build())
} else {
    Box::new(OpenAIProvider::builder(openai_key).build())
};

// Identical canonical call site:
let resp = provider.chat(&req).await?;
```

### Streaming

```rust
use futures::StreamExt;
use llm_adapter::{types::*, Provider};

let mut handle = provider.chat_stream(&req).await?;

while let Some(event) = handle.events().next().await {
    match event? {
        StreamEvent::TextDelta { text, .. } => print!("{text}"),
        StreamEvent::ReasoningDelta { text, .. } => eprint!("[think]{text}"),
        StreamEvent::ToolDelta { arguments, .. } => print!("[tool-args]{arguments}"),
        StreamEvent::MessageStop { stop_reason, usage } => {
            println!("\n-- stop={stop_reason:?} usage={usage:?}");
        }
        _ => {}
    }
}
```

Or fold the whole stream into a final response without consuming events
individually:

```rust
let handle = provider.chat_stream(&req).await?;
let resp = handle.collect().await?;
println!("{}", resp.text());
```

### Tool calling

```rust
use llm_adapter::types::*;
use serde_json::json;

let weather_tool = ToolDef::new(
    "get_weather",
    "Get current weather for a city.",
    json!({
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    }),
);

let req = ChatRequest::builder("gpt-4o-mini", 512)
    .message(Message::User(vec![RequestContent::Text(
        "What's the weather in Paris?".into(),
    )]))
    .tool(weather_tool.clone())
    .build();

let resp = provider.chat(&req).await?;
if matches!(resp.stop_reason, StopReason::ToolUse) {
    let mut messages = req.messages().to_vec();
    let invocation = resp.tool_invocations().next().unwrap().clone();
    messages.push(resp.into_message());
    messages.push(Message::Tool {
        invocation_id: invocation.id,
        content: vec![RequestContent::Text("Sunny, 22°C".into())],
        is_error: false,
    });

    let follow_up = ChatRequest::builder("gpt-4o-mini", 512)
        .messages(messages)
        .tool(weather_tool)
        .build();
    let final_resp = provider.chat(&follow_up).await?;
    println!("{}", final_resp.text());
}
```

### Structured output (OpenAI and Anthropic)

**OpenAI** — use the canonical `response_format` builder:

```rust
use llm_adapter::types::*;
use serde_json::json;

let schema = json!({
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"}
    },
    "required": ["title", "year"]
});

let req = ChatRequest::builder("gpt-4o-mini", 512)
    .message(Message::User(vec![RequestContent::Text(
        "Return info about 1984 by Orwell.".into(),
    )]))
    .response_format(ResponseFormat::JsonSchema {
        name: "book_info".into(),
        schema,
        strict: Some(true),
    })
    .build();

let resp = provider.chat(&req).await?;
let parsed: serde_json::Value = serde_json::from_str(&resp.text())?;
println!("{}", parsed["title"]);
```

**Anthropic** — use the native path with `structured_output`:

```rust
use llm_adapter::anthropic::{AnthropicExt, AnthropicRequest};
use serde_json::json;

let schema = json!({
    "type": "object",
    "properties": {"title": {"type": "string"}, "year": {"type": "integer"}},
    "required": ["title", "year"]
});

let ext = AnthropicExt::builder()
    .structured_output(schema)
    .build();

let anth_req = AnthropicRequest { base: chat_req, ext };
let resp = anthropic_provider.chat_native(&anth_req).await?;

// Extract the structured result from the tool invocation
let inv = resp.into_canonical()
    .tool_invocations()
    .next()
    .unwrap();
println!("{}", inv.input["title"]);
```

### Provider-specific extensions (native path)

The universal `Provider::chat` path is schema-neutral. When you need
provider-specific fields — OpenAI `seed`, `reasoning_effort`, or a
non-standard field accepted by an OpenAI-compatible endpoint — use the
native path:

```rust
use llm_adapter::openai::{OpenAIExt, OpenAIRequest};
use serde_json::json;

let ext = OpenAIExt::builder()
    .seed(42)
    .reasoning_effort("low")
    // DeepSeek V4 has thinking on by default and rejects tool_choice=Required
    // while thinking is enabled. Disable it through extra_body:
    .extra_body(json!({"thinking": {"type": "disabled"}}))
    .build();

let oai_req = OpenAIRequest { base: chat_req, ext };
let resp = openai_provider.chat_native(&oai_req).await?;
let stream = openai_provider.chat_stream_native(&oai_req).await?;
```

Anthropic native path follows the same pattern:

```rust
use llm_adapter::anthropic::{AnthropicExt, AnthropicMetadata, AnthropicRequest, ThinkingConfig};

let ext = AnthropicExt::builder()
    .top_k(40)
    .thinking(ThinkingConfig { budget_tokens: 2000 })
    .metadata(AnthropicMetadata { user_id: Some("my-user".into()) })
    .disable_parallel_tool_use()
    .build();

let anth_req = AnthropicRequest { base: chat_req, ext };
let resp = anthropic_provider.chat_native(&anth_req).await?;
let stream = anthropic_provider.chat_stream_native(&anth_req).await?;
```

## Provider configuration

### OpenAIProvider

| Method | Default | Description |
|---|---|---|
| `base_url(url)` | `https://api.openai.com` | API endpoint host |
| `chat_path(path)` | `/v1/chat/completions` | Chat completion path |
| `parse_reasoning_content(false)` | disabled | Parse `reasoning_content` field (DeepSeek) |
| `tolerant_keepalive(false)` | disabled | Tolerate keepalive bytes in SSE (DeepSeek) |

### AnthropicProvider

| Method | Default | Description |
|---|---|---|
| `base_url(url)` | `https://api.anthropic.com` | API endpoint host |
| `messages_path(path)` | `/v1/messages` | Messages API path |

### OpenAIExt

| Method | Default | Description |
|---|---|---|
| `seed(n)` | none | Deterministic sampling seed |
| `frequency_penalty(f)` | none | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty(f)` | none | Presence penalty (-2.0 to 2.0) |
| `user(id)` | none | End-user identifier for monitoring |
| `reasoning_effort(s)` | none | `"low"` \| `"medium"` \| `"high"` (o-series) |
| `response_format(fmt)` | none | `ResponseFormat::JsonObject` or `JsonSchema { name, schema, strict }` |
| `extra_body(val)` | none | Arbitrary JSON fields for gateway compat |
| `parse_reasoning_content()` | disabled | Parse DeepSeek `reasoning_content` |
| `tolerant_keepalive()` | disabled | Tolerate keepalive bytes in SSE |

### AnthropicExt

| Method | Default | Description |
|---|---|---|
| `top_k(n)` | none | Only sample from top K tokens |
| `thinking(config)` | none | Extended thinking with `budget_tokens` |
| `metadata(m)` | none | `user_id` for Anthropic dashboard |
| `disable_parallel_tool_use()` | disabled | Force single tool call per turn |
| `cache_hint(hint)` | none | Ephemeral prompt cache control for a block index |
| `structured_output(schema)` | none | JSON schema for structured output (via Strict Tool Use) |

### Same prompt, different providers

Because the library normalises all responses to the same canonical types,
you can compare how providers behave on identical input:

| Behaviour | OpenAI (gpt-4o) | Anthropic (claude-sonnet) | DeepSeek (deepseek-v4-flash) |
|---|---|---|---|
| Tools | stops with `ToolUse` | stops with `ToolUse` | stops with `ToolUse` |
| Reasoning | via `reasoning_effort` → `ResponseContent::Reasoning` | Extended Thinking → `ResponseContent::Reasoning { signature }` | Always-on (V4 default) → `ResponseContent::Reasoning` |
| Image input | supported | supported | not supported |
| Multi-turn thinking | N/A | requires `signature` round-trip via `AnthropicNativeResponse::into_message()` | N/A |

## Core concepts

### Canonical types

The semantics come from "having a conversation with an LLM", not from any
provider's wire format. Role is encoded in the `Message` enum so the type
system enforces correct content per role:

```rust
pub enum Message {
    System(String),
    User(Vec<RequestContent>),                            // text + images
    Assistant(Vec<ResponseContent>),                      // text + tool calls + reasoning
    Tool { invocation_id: String, content: Vec<RequestContent>, is_error: bool },
}

pub enum RequestContent  { Text(String), Image(ImageSource) }
pub enum ResponseContent { Text(String), ToolInvocation(ToolInvocation), Reasoning { text: String, signature: Option<String> } }
pub enum StopReason      { EndTurn, MaxTokens, StopSequence, ToolUse, ContentFilter, Other(String) }
```

Putting a `Reasoning` block inside `Message::User` is a compile error.

### Builder pattern everywhere

Every constructable type (`ChatRequest`, `SamplingParams`, `OpenAIExt`)
takes a builder, with private fields. Adding optional fields later is not a
breaking change.

### Universal vs native path

|  | `Provider::chat` (universal) | `chat_native` (native) |
|---|---|---|
| Request | `&ChatRequest` | `&OpenAIRequest` / `&AnthropicRequest = { base: ChatRequest, ext }` |
| Object-safe (`Box<dyn>`) | ✅ | ❌ (concrete provider type) |
| Provider-specific fields | ❌ | ✅ seed, penalties, top_k, thinking, cache_hints, extra_body |
| Use when | runtime-pluggable providers | you need provider-specific features |

### StreamHandle

```rust
impl StreamHandle {
    pub fn events(&mut self) -> &mut impl Stream<Item = Result<StreamEvent, LlmError>>;
    pub async fn collect(self) -> Result<ChatResponse, LlmError>;
}
```

Use `events()` for incremental UX (print tokens as they arrive). Use
`collect()` when you only need the final `ChatResponse` and don't want to
implement the fold yourself.

### Errors

`LlmError` is a normalised cross-provider error type:

```rust
pub enum LlmError {
    InvalidRequest(String),
    Unauthorized,
    Forbidden,
    RateLimit   { retry_after: Option<Duration> },
    Overloaded  { retry_after: Option<Duration> },
    ServerError(String),
    Timeout,
    Stream(String),
    Network(reqwest::Error),
    Decode(serde_json::Error),
    Provider    { code: String, message: String },   // 402 / unknown statuses
}
```

The library does no retry. `RateLimit` and `Overloaded` expose
`retry_after` from the `Retry-After` header for the caller to act on.

## Connecting to OpenAI-compatible gateways

The library supports any endpoint that speaks the OpenAI Chat Completions
schema, through three builder knobs:

```rust
OpenAIProvider::builder(key)
    .base_url("https://models.github.ai/inference")   // any host
    .chat_path("/chat/completions")                   // any path (default /v1/chat/completions)
    .build()
```

- **base_url** — scheme + host, optionally with a fixed path prefix;
  a trailing slash is trimmed before `chat_path` is appended.
- **chat_path** — set when the gateway omits or rewrites the `/v1` prefix.
  GitHub Models needs `/chat/completions`; OpenRouter needs `/api/v1/chat/completions`
  (so its `base_url` would be `https://openrouter.ai/api`).
- **extra_body** (on `OpenAIExt`) — top-level JSON fields not in the
  canonical schema. Mirrors the OpenAI Python SDK's `extra_body` parameter.

**Coverage estimate:** about 70% of "OpenAI-compatible" providers work
straight out of the box (DeepSeek, Together, Mistral, Cerebras, Moonshot,
Ollama, LM Studio, ...); the rest need `chat_path` or `extra_body` and
still work. **Azure OpenAI is not supported** — it uses an `api-key`
header instead of `Bearer`, and per-deployment paths.

## Testing

### Offline (default)

```
./test.sh              # or: cargo test
```

Runs ~75 unit and offline integration tests. No network, no API keys.
Fixtures live in `tests/fixtures/{openai,deepseek,anthropic}/` and were recorded
from a real API so that wire-level details (`system_fingerprint`,
`completion_tokens_details.reasoning_tokens`, SSE chunk boundaries, ...)
match production behaviour.

`test.sh` sets `NO_PROXY=127.0.0.1,localhost` so a macOS system HTTP proxy
doesn't intercept wiremock's loopback server.

### Integration (opt-in)

```
./test.sh --features integration
```

Hits real provider APIs. Configuration via environment variables (loaded
from `.env` if present):

```bash
# OpenAI / any OpenAI-compatible endpoint
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com         # default
export OPENAI_CHAT_PATH=/v1/chat/completions          # default
export OPENAI_MODEL=gpt-4o-mini                       # default
export OPENAI_REASONING_MODEL=o4-mini                 # optional; skip if unset

# DeepSeek
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=https://api.deepseek.com     # default
export DEEPSEEK_MODEL=deepseek-v4-flash               # default
export DEEPSEEK_REASONING_MODEL=deepseek-v4-flash     # optional

# Anthropic (can also target DeepSeek's /anthropic endpoint)
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=https://api.anthropic.com    # default
export ANTHROPIC_MESSAGES_PATH=/v1/messages            # default
export ANTHROPIC_MODEL=claude-sonnet-4-6-20250514      # default
```

Tests for a provider whose `*_API_KEY` is unset are silently skipped (no
failure). One integration test exercises the `extra_body + tool_choice:
Required` combination against real DeepSeek to validate the
provider-specific escape hatch.

### Re-recording fixtures

When upstream behaviour changes, regenerate fixtures:

```
cargo run --example record_fixtures
```

The tool reads `.env`, calls each configured provider with a stable
prompt, and writes responses verbatim (preserving server field order,
undocumented fields, and stream chunk boundaries) into
`tests/fixtures/`.

## Project layout

```
src/
├─ lib.rs                # public re-exports
├─ error.rs              # LlmError
├─ types/                # canonical types: Message, ChatRequest, ChatResponse, StreamEvent, ...
├─ provider.rs           # Provider trait
├─ stream_handle.rs      # StreamHandle + collect()
├─ openai/
│  ├─ mod.rs             # OpenAIProvider + builder + merge_extra_body
│  ├─ schema.rs          # OpenAI wire types (pub(crate))
│  ├─ convert.rs         # ChatRequest <-> OAI conversion, OpenAIExt builder
│  └─ stream.rs          # SSE parser, OaiStreamState
├─ anthropic/
│  ├─ mod.rs             # AnthropicProvider + builder
│  ├─ schema.rs          # Anthropic wire types (pub(crate))
│  ├─ convert.rs         # ChatRequest <-> Anthropic conversion, AnthropicExt builder
│  └─ stream.rs          # Anthropic SSE parser, AnthropicStreamState
└─ deepseek.rs           # factory returning a pre-configured OpenAIProvider

tests/
├─ fixtures/             # recorded API responses (JSON + SSE)
├─ common/mod.rs         # wiremock helpers + .env loader
├─ openai_*.rs           # offline tests
├─ deepseek_*.rs         # offline tests
├─ anthropic_*.rs        # offline tests
└─ integration_*.rs      # opt-in tests against real APIs

examples/
└─ record_fixtures.rs    # regenerate fixtures from live providers
```

## Known limitations

- **Azure OpenAI** not supported (different auth header + deployment paths).
- **OpenAI Responses API** not supported; only Chat Completions.
- **Prompt caching** is observable in `Usage.cached_input_tokens` but not
  proactively triggerable from the canonical layer. Use the native path
  (`AnthropicExt.cache_hints`) for cache-control markers.
- **Retry / backoff** is not built in. Look at `LlmError::RateLimit` /
  `LlmError::Overloaded`'s `retry_after` and decide in your application.

## License

MIT. See [LICENSE](LICENSE).
