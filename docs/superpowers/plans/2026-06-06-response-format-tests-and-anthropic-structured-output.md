# Response-Format Integration Tests + Anthropic Structured Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Issue #3 (integration tests for OpenAI `ResponseFormat`) and Issue #2 (`AnthropicExt::structured_output` via Strict Tool Use), executed in that order.

**Architecture:** Issue #3 adds two tests to an existing file with no logic changes. Issue #2 extends `AnthropicExt` with a `structured_output: Option<serde_json::Value>` field, wires it into `to_anthropic()` by injecting a synthetic `"structured_result"` tool and forcing `tool_choice`, then adds a unit test and an integration test.

**Tech Stack:** Rust 2024 edition, `serde_json`, `reqwest`, `tokio`, `async_trait`. All tests gate on `#[cfg(feature = "integration")]` and skip when the API key env-var is unset.

---

## File Map

| File | Change |
|------|--------|
| `tests/integration_openai.rs` | Add 2 new integration tests at the bottom |
| `src/anthropic/convert.rs` | Add `structured_output` field + builder method to `AnthropicExt`/`AnthropicExtBuilder`; wire into `to_anthropic()`; add 2 unit tests |
| `tests/integration_anthropic.rs` | Add 1 new integration test at the bottom |

---

## Part 1 — Issue #3: OpenAI ResponseFormat integration tests

### Task 1: Integration tests for `ResponseFormat` on OpenAI

**Files:**
- Modify: `tests/integration_openai.rs`

- [ ] **Step 1: Write the failing tests**

Append both tests to the bottom of `tests/integration_openai.rs`:

```rust
#[tokio::test]
async fn integration_response_format_json_object() {
    let Some((p, model)) = provider() else {
        return;
    };
    let req = ChatRequest::builder(model, 512)
        .message(Message::User(vec![RequestContent::Text(
            "Return a JSON object with a single key \"greeting\" \
             whose value is a short greeting string."
                .into(),
        )]))
        .response_format(ResponseFormat::JsonObject)
        .build();

    let resp = p.chat(&req).await.expect("chat failed");
    let parsed: serde_json::Value =
        serde_json::from_str(resp.text()).expect("response must be valid JSON");
    assert!(
        parsed.get("greeting").is_some(),
        "expected 'greeting' key in JSON response; got: {}",
        resp.text()
    );
}

#[tokio::test]
async fn integration_response_format_json_schema() {
    let Some((p, model)) = provider() else {
        return;
    };
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "title":  {"type": "string"},
            "year":   {"type": "integer"},
            "author": {"type": "string"}
        },
        "required": ["title", "year", "author"],
        "additionalProperties": false
    });
    let req = ChatRequest::builder(model, 512)
        .message(Message::User(vec![RequestContent::Text(
            "Return information about the novel '1984' by George Orwell.".into(),
        )]))
        .response_format(ResponseFormat::JsonSchema {
            name: "book_info".into(),
            schema,
            strict: Some(true),
        })
        .build();

    let resp = p.chat(&req).await.expect("chat failed");
    let parsed: serde_json::Value =
        serde_json::from_str(resp.text()).expect("response must be valid JSON");
    let title = parsed
        .get("title")
        .and_then(|v| v.as_str())
        .expect("missing 'title' field");
    assert!(!title.is_empty(), "title should be non-empty");
    let year = parsed
        .get("year")
        .and_then(|v| v.as_i64())
        .expect("missing 'year' field");
    assert_eq!(year, 1949, "1984 was published in 1949");
    let author = parsed
        .get("author")
        .and_then(|v| v.as_str())
        .expect("missing 'author' field");
    assert!(
        author.to_lowercase().contains("orwell"),
        "expected Orwell as author; got: {author}"
    );
}
```

- [ ] **Step 2: Run the new tests to verify they compile (skip OK, no API key needed)**

```bash
cargo test --features integration integration_response_format 2>&1 | head -40
```

Expected: tests either pass (if `OPENAI_API_KEY` is set) or print `[skip]` and exit 0.

- [ ] **Step 3: Run the full unit-test suite to confirm no regressions**

```bash
cargo test 2>&1 | tail -5
```

Expected: `test result: ok. N passed; 0 failed`.

- [ ] **Step 4: Commit**

```bash
git add tests/integration_openai.rs
git commit -m "test(openai): integration tests for ResponseFormat json_object and json_schema"
```

---

## Part 2 — Issue #2: Anthropic structured output via Strict Tool Use

### Task 2: Add `structured_output` to `AnthropicExt` and wire into `to_anthropic()`

**Files:**
- Modify: `src/anthropic/convert.rs`

The implementation has three parts that must land together (field → builder → wiring):

**Part A — Add the field to `AnthropicExt` and `AnthropicExtBuilder`**

In `src/anthropic/convert.rs`, add `structured_output` to the struct and the builder:

```rust
// In AnthropicExt (after cache_hints):
    /// JSON schema for structured output via Strict Tool Use.
    ///
    /// When set, `to_anthropic()` injects a synthetic `"structured_result"` tool
    /// carrying this schema and forces `tool_choice` to that tool.  The caller
    /// extracts the result from `resp.tool_invocations().next()`.
    pub structured_output: Option<serde_json::Value>,
```

```rust
// In AnthropicExtBuilder (after cache_hints):
    structured_output: Option<serde_json::Value>,
```

Add the builder method after `cache_hint`:

```rust
    /// Force the model to return a structured JSON result conforming to `schema`.
    ///
    /// Implemented via Strict Tool Use: a synthetic `"structured_result"` tool is
    /// injected into the request and `tool_choice` is forced to that tool.
    pub fn structured_output(mut self, schema: serde_json::Value) -> Self {
        self.structured_output = Some(schema);
        self
    }
```

Update the `build()` method to include the field:

```rust
    pub fn build(self) -> AnthropicExt {
        AnthropicExt {
            top_k: self.top_k,
            disable_parallel_tool_use: self.disable_parallel_tool_use,
            thinking: self.thinking,
            metadata: self.metadata,
            cache_hints: self.cache_hints,
            structured_output: self.structured_output,
        }
    }
```

Update the `#[derive(Debug, Clone, Default)]` default (the field is `Option`, so it derives `Default` automatically — no manual impl needed).

**Part B — Wire into `to_anthropic()`**

Replace the tools and tool_choice block (lines ~236–251 of convert.rs) with:

```rust
    // Convert tools.
    let mut tools: Vec<super::schema::AnthropicTool> = req
        .tools()
        .iter()
        .map(|t| super::schema::AnthropicTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect();

    // Inject synthetic structured-output tool if requested.
    if let Some(schema) = &ext.structured_output {
        tools.push(super::schema::AnthropicTool {
            name: "structured_result".into(),
            description: "Return the structured result.".into(),
            input_schema: schema.clone(),
        });
    }

    // Convert tool choice.
    let tool_choice = if ext.structured_output.is_some() {
        // Force the synthetic tool regardless of any canonical tool_choice.
        Some(serde_json::json!({"type": "tool", "name": "structured_result"}))
    } else if req.tools().is_empty() {
        None
    } else {
        Some(tool_choice_to_anthropic(req.tool_choice()))
    };
```

**Part C — Add two unit tests**

Add inside `#[cfg(test)] mod tests { ... }` at the bottom of `convert.rs`:

```rust
    #[test]
    fn ext_builder_with_structured_output() {
        let schema = serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}});
        let ext = AnthropicExt::builder()
            .structured_output(schema.clone())
            .build();
        assert_eq!(ext.structured_output, Some(schema));
    }

    #[test]
    fn structured_output_injects_tool_and_forces_choice() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        });
        let req = ChatRequest::builder("claude-sonnet-4-20250514", 1024)
            .message(Message::User(vec![RC::Text("Answer the question.".into())]))
            .build();
        let ext = AnthropicExt::builder()
            .structured_output(schema.clone())
            .build();

        let wire = to_anthropic(&req, Some(&ext), false).unwrap();

        // A single synthetic tool must be injected.
        assert_eq!(wire.tools.len(), 1, "expected exactly one injected tool");
        assert_eq!(wire.tools[0].name, "structured_result");
        assert_eq!(
            wire.tools[0].input_schema,
            schema,
            "injected schema must match the one provided"
        );

        // tool_choice must be forced to the synthetic tool.
        assert_eq!(
            wire.tool_choice,
            Some(serde_json::json!({"type": "tool", "name": "structured_result"})),
            "tool_choice must be forced to structured_result"
        );
    }
```

- [ ] **Step 1: Implement Part A (field + builder method)**

Add `structured_output: Option<serde_json::Value>` to `AnthropicExt` and `AnthropicExtBuilder`, the builder method, and update `build()` as shown above.

- [ ] **Step 2: Run tests to confirm Part A compiles**

```bash
cargo test 2>&1 | tail -5
```

Expected: all tests pass (the new field is not wired yet, but existing tests still pass).

- [ ] **Step 3: Implement Part B (wiring in `to_anthropic()`)**

Replace the tools + tool_choice block in `to_anthropic()` with the code from Part B above.

- [ ] **Step 4: Write the two unit tests (Part C)**

Append the two test functions to the `tests` module in `convert.rs`.

- [ ] **Step 5: Run unit tests and confirm both new tests pass**

```bash
cargo test -p llm_adapter 2>&1 | grep -E "test |FAILED|ok\."
```

Expected output includes:
```
test anthropic::convert::tests::ext_builder_with_structured_output ... ok
test anthropic::convert::tests::structured_output_injects_tool_and_forces_choice ... ok
```

- [ ] **Step 6: Run `cargo fmt` and fix any formatting issues**

```bash
cargo fmt && cargo fmt -- --check 2>&1
```

Expected: silent (no output means clean).

- [ ] **Step 7: Commit**

```bash
git add src/anthropic/convert.rs
git commit -m "feat(anthropic): structured_output via Strict Tool Use in AnthropicExt"
```

---

### Task 3: Integration test for Anthropic structured output

**Files:**
- Modify: `tests/integration_anthropic.rs`

The test uses `AnthropicProvider::chat_native()` which accepts `AnthropicRequest { base, ext }`.

- [ ] **Step 1: Write the failing integration test**

Add to the import line at the top of `tests/integration_anthropic.rs`:

```rust
use llm_adapter::{
    Provider,
    anthropic::{AnthropicExt, AnthropicProvider, AnthropicRequest},
    types::*,
};
```

Append the test to the bottom of the file:

```rust
#[tokio::test]
async fn integration_structured_output() {
    let Some((p, model)) = provider() else {
        return;
    };
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "title":  {"type": "string"},
            "year":   {"type": "integer"},
            "author": {"type": "string"}
        },
        "required": ["title", "year", "author"]
    });
    let base = ChatRequest::builder(model, 512)
        .message(Message::User(vec![RequestContent::Text(
            "Return information about the novel '1984' by George Orwell.".into(),
        )]))
        .build();
    let ext = AnthropicExt::builder()
        .structured_output(schema)
        .build();
    let anthropic_req = AnthropicRequest { base, ext };

    let resp = p
        .chat_native(&anthropic_req)
        .await
        .expect("chat_native failed");
    let canon = resp.into_canonical();

    assert!(
        matches!(canon.stop_reason, StopReason::ToolUse),
        "expected ToolUse stop reason; got: {:?}",
        canon.stop_reason
    );

    let inv = canon
        .tool_invocations()
        .next()
        .expect("expected at least one tool invocation");
    assert_eq!(
        inv.name, "structured_result",
        "tool name must match the synthetic tool"
    );

    let title = inv
        .input
        .get("title")
        .and_then(|v| v.as_str())
        .expect("missing 'title' field");
    assert!(!title.is_empty(), "title should be non-empty");

    let year = inv
        .input
        .get("year")
        .and_then(|v| v.as_i64())
        .expect("missing 'year' field");
    assert_eq!(year, 1949, "1984 was published in 1949");

    let author = inv
        .input
        .get("author")
        .and_then(|v| v.as_str())
        .expect("missing 'author' field");
    assert!(
        author.to_lowercase().contains("orwell"),
        "expected Orwell as author; got: {author}"
    );
}
```

- [ ] **Step 2: Run the new integration test (skip OK if no API key)**

```bash
cargo test --features integration integration_structured_output 2>&1 | head -40
```

Expected: test passes (if `ANTHROPIC_API_KEY` set) or prints `[skip]` and exits 0.

- [ ] **Step 3: Run full unit-test suite for regressions**

```bash
cargo test 2>&1 | tail -5
```

Expected: `test result: ok. N passed; 0 failed`.

- [ ] **Step 4: Run `cargo fmt --check`**

```bash
cargo fmt -- --check 2>&1
```

Expected: silent.

- [ ] **Step 5: Commit**

```bash
git add tests/integration_anthropic.rs
git commit -m "test(anthropic): integration test for structured_output via Strict Tool Use"
```

---

## Self-Review

### Spec coverage

- Issue #3 (ResponseFormat integration tests): ✅ `json_object` test + `json_schema` test with real assertions both added to `tests/integration_openai.rs`.
- Issue #2 (Anthropic structured output):
  - ✅ `structured_output` field on `AnthropicExt`
  - ✅ Builder method `AnthropicExtBuilder::structured_output(schema)`
  - ✅ Wired into `to_anthropic()` as synthetic tool + forced tool_choice
  - ✅ Unit test for builder field assignment
  - ✅ Unit test for wire-format correctness (tool injected, tool_choice forced)
  - ✅ Integration test via `chat_native()`
- Skip pattern (no API key → print `[skip]` → return): ✅ both integration files.

### Placeholder scan

None detected — every step contains actual code.

### Type consistency

- `AnthropicExt::structured_output` is `Option<serde_json::Value>` throughout (field, builder, wiring, tests).
- `super::schema::AnthropicTool` has fields `name: String`, `description: String`, `input_schema: serde_json::Value` — matches usage in wiring code.
- `AnthropicRequest { base: ChatRequest, ext: AnthropicExt }` — matches the struct definition in `convert.rs` line 118.
- `tool_choice: Option<serde_json::Value>` — matches `AnthropicRequest.tool_choice` in `schema.rs`.
