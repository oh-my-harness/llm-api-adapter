//! Integration tests hitting a real OpenAI-compatible endpoint.
//!
//! Run with: `./test.sh --features integration`
//!
//! Skipped when `OPENAI_API_KEY` is unset (printed but treated as success
//! per spec §8). Configuration via environment (`.env` is loaded
//! automatically):
//!   OPENAI_API_KEY      required
//!   OPENAI_BASE_URL     default https://api.openai.com
//!   OPENAI_CHAT_PATH    default /v1/chat/completions
//!   OPENAI_MODEL        default gpt-4o-mini
//!   OPENAI_REASONING_MODEL  optional; reasoning test skipped if absent

#![cfg(feature = "integration")]

mod common;

use futures::StreamExt;
use llm_adapter::{Provider, openai::OpenAIProvider, types::*};

/// Returns `None` when the test should be skipped.
fn provider() -> Option<(OpenAIProvider, String)> {
    common::load_dotenv();
    let api_key = match common::opt_env("OPENAI_API_KEY") {
        Some(k) => k,
        None => {
            println!("[skip] OPENAI_API_KEY not set");
            return None;
        }
    };
    let base_url =
        common::opt_env("OPENAI_BASE_URL").unwrap_or_else(|| "https://api.openai.com".into());
    let chat_path =
        common::opt_env("OPENAI_CHAT_PATH").unwrap_or_else(|| "/v1/chat/completions".into());
    let model = common::opt_env("OPENAI_MODEL").unwrap_or_else(|| "gpt-4o-mini".into());

    let p = OpenAIProvider::builder(api_key)
        .base_url(base_url)
        .chat_path(chat_path)
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
    assert!(!resp.text().is_empty(), "expected non-empty text");
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
async fn integration_tool_call_single_turn() {
    let Some((p, model)) = provider() else {
        return;
    };
    // Use Auto + an imperative prompt: tool_choice = Required is rejected
    // by some compatible providers when their model has thinking mode on
    // (e.g. DeepSeek V4).
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

    let req2 = ChatRequest::builder(model, 1024)
        .messages(messages)
        .tool(weather_tool())
        .build();
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
        serde_json::from_str(&resp.text()).expect("response must be valid JSON");
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
        serde_json::from_str(&resp.text()).expect("response must be valid JSON");
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
