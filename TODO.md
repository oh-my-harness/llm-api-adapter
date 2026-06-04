# TODO

## Current state (as of commit `e18319c`)

- Foundation, canonical types, `Provider` trait, `StreamHandle` — done
- OpenAI Chat Completions provider — done (`chat`, `chat_stream`, `chat_native`, `chat_stream_native`)
- DeepSeek factory (`deepseek::client`) — done; verified default V4 thinking-on behavior
- `OpenAIExt`: seed, frequency/presence penalty, user, `reasoning_effort`, `extra_body`, parse_reasoning_content, tolerant_keepalive — done
- OpenAI-compatible gateway support via `base_url` + `chat_path` — done
- 29 offline tests (wiremock + recorded fixtures), 15 integration tests (real APIs, gated by `--features integration`) — all green
- Fixture recording tool (`examples/record_fixtures.rs`) — done
- Bilingual README (en + zh-CN), MIT license — done
- `cargo fmt` + `cargo clippy --all-targets --features integration` — clean

---

## 1. Anthropic provider (Plan 2)

**Goal:** implement the Anthropic Messages API provider so users can hold
`Box<dyn Provider>` and switch to Anthropic at runtime, with a native path
for Anthropic-specific features.

**Spec references** (already designed in `docs/superpowers/specs/2026-06-03-llm-adapter-design.md`):
- §2.2 directory: `src/anthropic/{mod.rs, schema.rs, convert.rs, stream.rs}`
- §5.1 native types: `AnthropicContent`, `AnthropicMessage`, `AnthropicExt`,
  `AnthropicRequest`, `AnthropicResponse`, `AnthropicUsage`
- §3.1 `ResponseContent::Reasoning` semantics for Anthropic: read-only at
  canonical layer (signature dropped on `into_canonical`)
- §7 dependencies (`anthropic-version` header `2023-06-01`)

**Concrete deliverables:**

1. `src/anthropic/schema.rs` — wire types matching `POST /v1/messages`
   - request: `messages` array, top-level `system`, `tools` (`input_schema`
     field, no `function` wrapper), `tool_choice` (`{type: "auto"|"any"|"tool"|"none"}`),
     `thinking` config, `metadata`
   - response: structured `content` array (text / tool_use / thinking /
     redacted_thinking), `stop_reason`, `usage` with
     `cache_creation_input_tokens` and `cache_read_input_tokens`
   - SSE event taxonomy: `message_start`, `content_block_start`,
     `content_block_delta` (with `text_delta`/`input_json_delta`/
     `thinking_delta`/`signature_delta`), `content_block_stop`,
     `message_delta`, `message_stop`
2. `src/anthropic/convert.rs` — `ChatRequest <-> Anthropic` conversion:
   - `Message::System` → top-level `system` field (not part of `messages`)
   - `Message::Tool` → `tool_result` block inside the next `User` message
   - Strict user/assistant alternation enforcement
3. `src/anthropic/stream.rs` — Anthropic SSE parser. Unlike OpenAI, the
   message_start chunk doesn't have content; tool_use args arrive as
   `input_json_delta` chunks that must be concatenated then parsed.
4. `src/anthropic/mod.rs` — `AnthropicProvider` + builder; implements
   `Provider` trait (universal path) and exposes `chat_native`/`chat_stream_native`.
5. `AnthropicExt`: `top_k`, `disable_parallel_tool_use`, `cache_hints`,
   `thinking` config, `metadata`.
6. Errors: map 529 (overloaded) and 413 (request_too_large) correctly;
   `LlmError::Overloaded` already exists.

**Fixtures + tests:**

- Recording tool needs an Anthropic branch — most likely a free way to get
  Anthropic-compatible responses is **DeepSeek's `/anthropic` endpoint**
  (`https://api.deepseek.com/anthropic/v1/messages`, see
  `docs/research/05-deepseek.md` §4). Caveats: no image, no document, no
  built-in tools. Use it to record the basic flows; record the rest
  (thinking, prompt caching) against real Anthropic if a key becomes
  available.
- Offline tests: `tests/anthropic_chat.rs`, `tests/anthropic_stream.rs`
- Integration test: `tests/integration_anthropic.rs` reading
  `ANTHROPIC_API_KEY` + `ANTHROPIC_BASE_URL` + `ANTHROPIC_MODEL` from `.env`
- Update `.env` template, `README.md`, `README.zh-CN.md` to mark
  Anthropic from "planned" → "supported"

**Estimated work:** comparable to the original OpenAI plan — write a
plan file under `docs/superpowers/plans/` then execute it with
`subagent-driven-development`.

---

## 2. Cargo.toml metadata + crates.io publication readiness

Before `cargo publish` we need a complete `[package]` section. Currently
only `name`, `version`, `edition`, `license` are set.

**To add:**

```toml
description = "Rust adapter layer for LLM providers — OpenAI Chat Completions, DeepSeek, and any OpenAI-compatible endpoint, with first-class streaming and tool calling."
repository = "<github URL once published>"
homepage = "<same>"
documentation = "https://docs.rs/llm_adapter"
readme = "README.md"
keywords = ["llm", "openai", "deepseek", "anthropic", "ai"]  # max 5, alphanumeric + -
categories = ["api-bindings", "asynchronous"]
authors = ["<your name or org>"]
rust-version = "1.85"   # confirm — edition 2024 requires Rust 1.85+
```

**Other pre-publish checks:**

- Update `LICENSE` copyright holder if "hhllhhyyds" isn't the legal name
  to attribute work to
- Decide whether `pub fn _check() {}` removal commit (`09aae46`) makes the
  next minor version bump appropriate, or stay at 0.1.0 — for pre-1.0
  this is fine
- Add a `CHANGELOG.md` (optional but conventional)
- `cargo package --list` to inspect what gets uploaded; consider an
  `exclude = ["docs/", "examples/record_fixtures.rs", ".env", ...]` if
  `tests/fixtures/` shouldn't be packaged (probably should keep
  fixtures so downstream can run offline tests)
- Run `cargo publish --dry-run` and address any errors
- Tag the release commit (e.g. `v0.1.0`) and push the tag
- `cargo publish` — irreversible; the name is taken once accepted

**Suggested order:** finish (1) first since it changes the public API
surface — better to publish a complete tri-provider 0.1.0 than to ship
OpenAI-only then immediately break in 0.2.0 when Anthropic lands.
