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
use llm_adapter::{Provider, anthropic::AnthropicProvider, types::*};

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
