//! Integration tests hitting the real DeepSeek API.
//!
//! Run with: `./test.sh --features integration`
//!
//! Skipped when `DEEPSEEK_API_KEY` is unset (printed but treated as success
//! per spec §8). Configuration via environment (`.env` is loaded
//! automatically):
//!   DEEPSEEK_API_KEY     required
//!   DEEPSEEK_BASE_URL    default https://api.deepseek.com
//!   DEEPSEEK_MODEL       default deepseek-v4-flash
//!   DEEPSEEK_REASONING_MODEL  optional; reasoning tests skipped if absent

#![cfg(feature = "integration")]

mod common;

use futures::StreamExt;
use llm_adapter::{Provider, deepseek, openai::OpenAIProvider, types::*};

/// Build a DeepSeek-configured provider honouring optional base_url override.
/// Returns `None` when the test should be skipped.
fn provider() -> Option<(OpenAIProvider, String)> {
    common::load_dotenv();
    let api_key = match common::opt_env("DEEPSEEK_API_KEY") {
        Some(k) => k,
        None => {
            println!("[skip] DEEPSEEK_API_KEY not set");
            return None;
        }
    };
    let model = common::opt_env("DEEPSEEK_MODEL").unwrap_or_else(|| "deepseek-v4-flash".into());

    let p = if let Some(base) = common::opt_env("DEEPSEEK_BASE_URL") {
        // Custom endpoint: rebuild via OpenAIProvider builder with same flags
        // the deepseek::client factory applies.
        OpenAIProvider::builder(api_key)
            .base_url(base)
            .parse_reasoning_content(true)
            .tolerant_keepalive(true)
            .build()
    } else {
        deepseek::client(api_key)
    };
    Some((p, model))
}

fn reasoning_model() -> Option<String> {
    common::opt_env("DEEPSEEK_REASONING_MODEL")
}

#[tokio::test]
async fn integration_simple_chat() {
    let Some((p, model)) = provider() else {
        return;
    };
    // Note: DeepSeek V4 enables thinking by default — we don't disable it here
    // because the canonical ChatRequest doesn't expose thinking control. The
    // response may still include reasoning_content, which parse_reasoning_content
    // turns into a Reasoning content block.
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
    assert!(
        resp2.text().to_lowercase().contains("blue"),
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
    // NOTE: DeepSeek V4 enables thinking by default, and thinking mode does
    // not accept tool_choice = Required/Specific. We rely on Auto + an
    // imperative prompt to elicit a tool call.
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
    assert!(invocations[0].input.get("location").is_some());
}

#[tokio::test]
async fn integration_tool_call_round_trip() {
    let Some((p, model)) = provider() else {
        return;
    };

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
    let mut got_text = false;
    let mut got_stop = false;
    while let Some(event) = handle.events().next().await {
        match event.expect("stream error") {
            StreamEvent::TextDelta { .. } => got_text = true,
            StreamEvent::MessageStop { usage, .. } => {
                got_stop = true;
                assert!(usage.input_tokens > 0);
                assert!(usage.output_tokens > 0);
            }
            _ => {}
        }
    }
    assert!(got_text, "expected at least one TextDelta");
    assert!(got_stop, "expected MessageStop");
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
}

#[tokio::test]
async fn integration_reasoning_chat() {
    let Some((p, _)) = provider() else {
        return;
    };
    let Some(rm) = reasoning_model() else {
        println!("[skip] DEEPSEEK_REASONING_MODEL not set");
        return;
    };

    let req = ChatRequest::builder(rm, 2048)
        .message(Message::User(vec![RequestContent::Text(
            "What is 7 times 8? Answer with just the number.".into(),
        )]))
        .build();

    let resp = p.chat(&req).await.expect("reasoning chat failed");
    assert!(!resp.text().is_empty());
    // Thinking mode emits reasoning_content; with the deepseek::client
    // factory (parse_reasoning_content = true), it surfaces as a Reasoning
    // block plus a non-zero reasoning_tokens count.
    assert!(
        resp.reasoning().is_some(),
        "expected reasoning content from thinking model"
    );
    assert!(
        resp.usage.reasoning_tokens > 0,
        "expected reasoning_tokens > 0 from thinking model"
    );
}

#[tokio::test]
async fn integration_reasoning_stream() {
    let Some((p, _)) = provider() else {
        return;
    };
    let Some(rm) = reasoning_model() else {
        println!("[skip] DEEPSEEK_REASONING_MODEL not set");
        return;
    };

    let req = ChatRequest::builder(rm, 2048)
        .message(Message::User(vec![RequestContent::Text(
            "What is 7 times 8? Answer with just the number.".into(),
        )]))
        .build();

    let handle = p.chat_stream(&req).await.expect("stream failed");
    let resp = handle.collect().await.expect("collect failed");
    assert!(
        resp.reasoning().is_some(),
        "expected reasoning content from streamed thinking model"
    );
    assert!(!resp.text().is_empty());
    // Reasoning block precedes text block in canonical order.
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
    assert!(reasoning_idx < text_idx);
}

#[tokio::test]
async fn integration_tool_call_required_via_extra_body_disable_thinking() {
    // DeepSeek V4 rejects `tool_choice: required` while thinking is enabled.
    // Use OpenAIExt.extra_body to send `thinking: {type: disabled}` and then
    // Required becomes legal.
    let Some((p, model)) = provider() else {
        return;
    };
    use llm_adapter::openai::{OpenAIExt, OpenAIRequest};

    let ext = OpenAIExt::builder()
        .extra_body(serde_json::json!({"thinking": {"type": "disabled"}}))
        .build();
    let oai_req = OpenAIRequest {
        base: ChatRequest::builder(model, 256)
            .message(Message::User(vec![RequestContent::Text(
                "What's the weather in Paris?".into(),
            )]))
            .tool(weather_tool())
            .tool_choice(ToolChoice::Required)
            .build(),
        ext,
    };

    let resp = p.chat_native(&oai_req).await.expect("native chat failed");
    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    let invocations: Vec<_> = resp.tool_invocations().collect();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].name, "get_weather");
}

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
