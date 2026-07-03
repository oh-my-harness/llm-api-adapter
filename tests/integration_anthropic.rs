//! Integration tests hitting a real Anthropic-compatible endpoint.
//!
//! Run with: `./test.sh --features integration`
//!
//! Skipped when `ANTHROPIC_API_KEY` is unset.
//! Configuration via environment (`.env` is loaded automatically):
//!   ANTHROPIC_API_KEY       required
//!   ANTHROPIC_BASE_URL      default https://api.anthropic.com
//!   ANTHROPIC_MESSAGES_PATH default /v1/messages
//!   ANTHROPIC_MODEL         default claude-sonnet-4-6-20250514

#![cfg(feature = "integration")]

mod common;

use futures::StreamExt;
use llm_adapter::{
    Provider,
    anthropic::{AnthropicExt, AnthropicProvider, AnthropicRequest, ThinkingConfig},
    types::*,
};

/// Returns `None` when the test should be skipped.
fn provider() -> Option<(AnthropicProvider, String)> {
    common::load_dotenv();
    let api_key = match common::opt_env("ANTHROPIC_API_KEY") {
        Some(k) => k,
        None => {
            println!("[skip] ANTHROPIC_API_KEY not set");
            return None;
        }
    };
    let base_url =
        common::opt_env("ANTHROPIC_BASE_URL").unwrap_or_else(|| "https://api.anthropic.com".into());
    let messages_path =
        common::opt_env("ANTHROPIC_MESSAGES_PATH").unwrap_or_else(|| "/v1/messages".into());
    let model =
        common::opt_env("ANTHROPIC_MODEL").unwrap_or_else(|| "claude-sonnet-4-6-20250514".into());

    let p = AnthropicProvider::builder(api_key)
        .base_url(base_url)
        .messages_path(messages_path)
        .build();
    Some((p, model))
}

#[tokio::test]
async fn integration_simple_chat() {
    let Some((p, model)) = provider() else {
        return;
    };
    let req = ChatRequest::builder(model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "Say hi in exactly five words.".into(),
        )]))
        .build();

    let resp = p.chat(&req).await.expect("chat failed");
    assert!(matches!(
        resp.stop_reason,
        StopReason::EndTurn | StopReason::MaxTokens
    ));
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
}

#[tokio::test]
async fn integration_multi_turn() {
    let Some((p, model)) = provider() else {
        return;
    };

    let req1 = ChatRequest::builder(&model, 1024)
        .message(Message::System("Respond in one short sentence.".into()))
        .message(Message::User(vec![RequestContent::Text(
            "My favourite colour is blue. Remember that.".into(),
        )]))
        .build();
    let resp1 = p.chat(&req1).await.expect("turn 1 failed");
    assert!(!resp1.text().is_empty());

    let mut messages = req1.messages().to_vec();
    messages.push(resp1.into_message());
    messages.push(Message::User(vec![RequestContent::Text(
        "What did I say my favourite colour was?".into(),
    )]));

    let req2 = ChatRequest::builder(model, 1024).messages(messages).build();
    let resp2 = p.chat(&req2).await.expect("turn 2 failed");
    let answer = resp2.text().to_lowercase();
    assert!(
        answer.contains("blue"),
        "model should remember the favourite colour; got: {}",
        resp2.text()
    );
}

fn weather_tool() -> ToolDef {
    ToolDef::new(
        "get_weather",
        "Get current weather for a city.",
        serde_json::json!({
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }),
    )
}

#[tokio::test]
async fn integration_tool_call() {
    let Some((p, model)) = provider() else {
        return;
    };
    let req = ChatRequest::builder(model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "You have a get_weather tool. Call it to find the weather in Paris.".into(),
        )]))
        .tool(weather_tool())
        .build();

    let resp = p.chat(&req).await.expect("tool call failed");
    assert!(
        matches!(resp.stop_reason, StopReason::ToolUse),
        "expected tool_use stop reason; got: {:?}, text: {:?}",
        resp.stop_reason,
        resp.text()
    );
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
    assert!(
        invocations[0].input.get("location").is_some(),
        "expected `location` in tool arguments, got: {}",
        invocations[0].input
    );
}

#[tokio::test]
async fn integration_tool_call_round_trip() {
    let Some((p, model)) = provider() else {
        return;
    };

    // Round 1: ask for a tool call.
    let req1 = ChatRequest::builder(&model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "You have a get_weather tool. Call it to find the weather in Paris.".into(),
        )]))
        .tool(weather_tool())
        .build();
    let resp1 = p.chat(&req1).await.expect("round 1 failed");
    let invocation = resp1
        .tool_invocations()
        .next()
        .expect("expected at least one tool invocation")
        .clone();

    // Round 2: feed back a fake tool result and let the model answer.
    let mut messages = req1.messages().to_vec();
    messages.push(resp1.into_message());
    messages.push(Message::Tool {
        invocation_id: invocation.id,
        content: vec![RequestContent::Text("Sunny, 22°C".into())],
        is_error: false,
    });

    let req2 = ChatRequest::builder(model, 1024).messages(messages).build();
    let resp2 = p.chat(&req2).await.expect("round 2 failed");
    assert!(matches!(resp2.stop_reason, StopReason::EndTurn));
    assert!(!resp2.text().is_empty());
}

#[tokio::test]
async fn integration_stream_simple() {
    let Some((p, model)) = provider() else {
        return;
    };
    let req = ChatRequest::builder(model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "Count from one to five.".into(),
        )]))
        .build();

    let mut handle = p.chat_stream(&req).await.expect("stream failed");

    let mut text_deltas = 0usize;
    let mut got_stop = false;
    let mut final_usage = Usage::default();
    while let Some(event) = handle.events().next().await {
        match event.expect("stream error") {
            StreamEvent::TextDelta { .. } => text_deltas += 1,
            StreamEvent::MessageStop { usage, .. } => {
                got_stop = true;
                final_usage = usage;
            }
            _ => {}
        }
    }
    assert!(text_deltas > 0, "expected at least one TextDelta");
    assert!(got_stop, "expected MessageStop");
    assert!(final_usage.input_tokens > 0);
    assert!(final_usage.output_tokens > 0);
}

#[tokio::test]
async fn integration_stream_tool_call() {
    let Some((p, model)) = provider() else {
        return;
    };
    let req = ChatRequest::builder(model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "You have a get_weather tool. Call it to find the weather in Paris.".into(),
        )]))
        .tool(weather_tool())
        .build();

    let handle = p.chat_stream(&req).await.expect("stream failed");
    let resp = handle.collect().await.expect("collect failed");
    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert!(!invocations.is_empty());
    assert_eq!(invocations[0].name, "get_weather");
    assert!(invocations[0].input.get("location").is_some());
}

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
    let ext = AnthropicExt::builder().structured_output(schema).build();
    let anthropic_req = AnthropicRequest { base, ext };

    let resp = match p.chat_native(&anthropic_req).await {
        Ok(resp) => resp,
        // DeepSeek V4 enables thinking by default; thinking mode rejects the
        // specific tool_choice that structured_output requires. Skip gracefully
        // when the backend is a DeepSeek-based Anthropic-compatible endpoint.
        Err(llm_adapter::LlmError::InvalidRequest(msg))
            if msg.contains("Thinking mode does not support") =>
        {
            println!("[skip] structured_output skipped: thinking mode conflict");
            return;
        }
        Err(e) => panic!("chat_native failed: {e}"),
    };
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
    assert_eq!(inv.name, "structured_result");

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

#[tokio::test]
async fn integration_thinking_via_canonical_request() {
    let Some((prov, model)) = provider() else {
        eprintln!("[skip] ANTHROPIC_API_KEY not set");
        return;
    };
    let req = ChatRequest::builder(&model, 4096)
        .message(Message::User(vec![RequestContent::Text(
            "Explain why the sky is blue in two sentences.".into(),
        )]))
        .thinking(ThinkingLevel::Budget(2000))
        .build();
    let resp = prov.chat(&req).await.unwrap();
    assert!(resp.reasoning().is_some());
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn integration_thinking_adaptive_via_canonical_request() {
    // Tests the 4.7+ adaptive thinking path (thinking.type="adaptive", effort=...).
    // Requires a 4.7+/mythos model via ANTHROPIC_ADAPTIVE_MODEL; skipped otherwise.
    common::load_dotenv();
    let Some(api_key) = common::opt_env("ANTHROPIC_API_KEY") else {
        eprintln!("[skip] ANTHROPIC_API_KEY not set");
        return;
    };
    let Some(model) = common::opt_env("ANTHROPIC_ADAPTIVE_MODEL") else {
        eprintln!("[skip] ANTHROPIC_ADAPTIVE_MODEL not set (need 4.7+/mythos model)");
        return;
    };
    let base_url =
        common::opt_env("ANTHROPIC_BASE_URL").unwrap_or_else(|| "https://api.anthropic.com".into());
    let messages_path =
        common::opt_env("ANTHROPIC_MESSAGES_PATH").unwrap_or_else(|| "/v1/messages".into());
    let prov = AnthropicProvider::builder(api_key)
        .base_url(base_url)
        .messages_path(messages_path)
        .build();
    let req = ChatRequest::builder(&model, 4096)
        .message(Message::User(vec![RequestContent::Text(
            "Explain why the sky is blue in two sentences.".into(),
        )]))
        .thinking(ThinkingLevel::High)
        .build();
    let resp = prov.chat(&req).await.unwrap();
    assert!(!resp.text().is_empty());
    // adaptive mode may or may not return thinking text (depends on `display`),
    // but the request must not 400 — that's the core assertion.
}

#[tokio::test]
async fn integration_thinking_tokens_in_usage() {
    // When thinking is enabled, output_tokens_details.thinking_tokens should
    // be populated and mapped to canonical usage.reasoning_tokens.
    let Some((prov, model)) = provider() else {
        eprintln!("[skip] ANTHROPIC_API_KEY not set");
        return;
    };
    let req = ChatRequest::builder(&model, 4096)
        .message(Message::User(vec![RequestContent::Text(
            "Explain why the sky is blue in two sentences.".into(),
        )]))
        .thinking(ThinkingLevel::Budget(2000))
        .build();
    let resp = prov.chat(&req).await.unwrap();
    assert!(resp.reasoning().is_some(), "expected reasoning content");
    assert!(
        resp.usage.reasoning_tokens > 0,
        "expected reasoning_tokens > 0 in usage, got: {}",
        resp.usage.reasoning_tokens
    );
}

#[tokio::test]
async fn integration_thinking_via_native_budget_config() {
    // Native path: explicitly choose ThinkingConfig::Budget (4.6 models).
    let Some((prov, model)) = provider() else {
        eprintln!("[skip] ANTHROPIC_API_KEY not set");
        return;
    };
    let ext = AnthropicExt::builder()
        .thinking(ThinkingConfig::Budget {
            budget_tokens: 2000,
        })
        .build();
    let req = AnthropicRequest {
        base: ChatRequest::builder(&model, 4096)
            .message(Message::User(vec![RequestContent::Text(
                "Explain why the sky is blue in two sentences.".into(),
            )]))
            .build(),
        ext,
    };
    let resp = prov.chat_native(&req).await.unwrap();
    assert!(
        resp.as_response().reasoning().is_some(),
        "expected reasoning content"
    );
    assert!(!resp.as_response().text().is_empty());
}

#[tokio::test]
async fn integration_thinking_via_native_adaptive_config() {
    // Native path: explicitly choose ThinkingConfig::Adaptive (4.7+ models).
    // Requires ANTHROPIC_ADAPTIVE_MODEL; skipped otherwise.
    common::load_dotenv();
    let Some(api_key) = common::opt_env("ANTHROPIC_API_KEY") else {
        eprintln!("[skip] ANTHROPIC_API_KEY not set");
        return;
    };
    let Some(model) = common::opt_env("ANTHROPIC_ADAPTIVE_MODEL") else {
        eprintln!("[skip] ANTHROPIC_ADAPTIVE_MODEL not set (need 4.7+/mythos model)");
        return;
    };
    let base_url =
        common::opt_env("ANTHROPIC_BASE_URL").unwrap_or_else(|| "https://api.anthropic.com".into());
    let messages_path =
        common::opt_env("ANTHROPIC_MESSAGES_PATH").unwrap_or_else(|| "/v1/messages".into());
    let prov = AnthropicProvider::builder(api_key)
        .base_url(base_url)
        .messages_path(messages_path)
        .build();
    let ext = AnthropicExt::builder()
        .thinking(ThinkingConfig::Adaptive {
            effort: "standard".into(),
        })
        .build();
    let req = AnthropicRequest {
        base: ChatRequest::builder(&model, 4096)
            .message(Message::User(vec![RequestContent::Text(
                "Explain why the sky is blue in two sentences.".into(),
            )]))
            .build(),
        ext,
    };
    let resp = prov.chat_native(&req).await.unwrap();
    // Must not 400 — that's the core assertion for adaptive wire format.
    assert!(!resp.as_response().text().is_empty());
}

#[tokio::test]
async fn integration_thinking_with_tool_call() {
    // Thinking + tool_choice=auto should coexist (thinking restricts to auto/none).
    let Some((prov, model)) = provider() else {
        eprintln!("[skip] ANTHROPIC_API_KEY not set");
        return;
    };
    let req = ChatRequest::builder(&model, 4096)
        .message(Message::User(vec![RequestContent::Text(
            "You have a get_weather tool. Call it to find the weather in Paris.".into(),
        )]))
        .tool(weather_tool())
        .thinking(ThinkingLevel::Budget(2000))
        .build();
    let resp = prov.chat(&req).await.unwrap();
    assert!(
        matches!(resp.stop_reason, StopReason::ToolUse),
        "expected tool_use stop reason; got: {:?}, text: {:?}",
        resp.stop_reason,
        resp.text()
    );
    // Thinking content should be present before the tool_use block.
    assert!(resp.reasoning().is_some(), "expected reasoning content");
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
}

#[tokio::test]
async fn integration_thinking_stream_reasoning_precedes_text() {
    // In thinking mode, reasoning block should arrive before text in the stream.
    let Some((prov, model)) = provider() else {
        eprintln!("[skip] ANTHROPIC_API_KEY not set");
        return;
    };
    let req = ChatRequest::builder(&model, 4096)
        .message(Message::User(vec![RequestContent::Text(
            "Explain why the sky is blue in two sentences.".into(),
        )]))
        .thinking(ThinkingLevel::Budget(2000))
        .build();
    let handle = prov.chat_stream(&req).await.expect("stream failed");
    let resp = handle.collect().await.expect("collect failed");
    assert!(
        resp.reasoning().is_some(),
        "expected reasoning content in stream"
    );
    assert!(!resp.text().is_empty());
    // Reasoning block must precede text block in canonical order.
    let reasoning_idx = resp
        .content
        .iter()
        .position(|c| matches!(c, ResponseContent::Reasoning { .. }))
        .expect("reasoning block missing");
    let text_idx = resp
        .content
        .iter()
        .position(|c| matches!(c, ResponseContent::Text(_)))
        .expect("text block missing");
    assert!(reasoning_idx < text_idx, "reasoning must precede text");
    assert!(
        resp.usage.reasoning_tokens > 0,
        "expected reasoning_tokens > 0 in streamed usage"
    );
}

#[tokio::test]
async fn integration_thinking_off_omits_field() {
    // ThinkingLevel::Off on the canonical path should omit the thinking field
    // entirely (not send thinking.type="disabled"). The model should still
    // respond normally without thinking content.
    let Some((prov, model)) = provider() else {
        eprintln!("[skip] ANTHROPIC_API_KEY not set");
        return;
    };
    let req = ChatRequest::builder(&model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "Say hello in one word.".into(),
        )]))
        .thinking(ThinkingLevel::Off)
        .build();
    let resp = prov.chat(&req).await.unwrap();
    assert!(!resp.text().is_empty());
    // No thinking → no reasoning content, no thinking tokens.
    assert!(
        resp.reasoning().is_none(),
        "Off should produce no reasoning"
    );
    assert_eq!(
        resp.usage.reasoning_tokens, 0,
        "Off should produce 0 reasoning_tokens"
    );
}
