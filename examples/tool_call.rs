//! Two-turn tool calling: model emits a tool_use, the caller injects a
//! fake tool result, the model produces a final answer.
//!
//! Run with: `cargo run --example tool_call`
//!
//! Reads OPENAI_* from .env. If OPENAI_BASE_URL points at DeepSeek
//! (V4 has thinking on by default), prefer `cargo run --example
//! native_extra_body` instead — that one disables thinking so
//! `tool_choice: Required` works.

use llm_adapter::{Provider, openai::OpenAIProvider, types::*};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();

    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY is not set (try `source .env` first)")?;
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".into());
    let chat_path =
        std::env::var("OPENAI_CHAT_PATH").unwrap_or_else(|_| "/v1/chat/completions".into());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());

    let p = OpenAIProvider::builder(api_key)
        .base_url(base_url)
        .chat_path(chat_path)
        .build();

    let weather_tool = ToolDef {
        name: "get_weather".into(),
        description: "Get current weather for a city.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }),
    };

    // --- Turn 1: ask, expect a tool call ---
    let req1 = ChatRequest::builder(&model, 512)
        .message(Message::User(vec![RequestContent::Text(
            "You have a get_weather tool. Call it to find the weather in Paris.".into(),
        )]))
        .tool(weather_tool.clone())
        .build();

    let resp1 = p.chat(&req1).await?;
    if !matches!(resp1.stop_reason, StopReason::ToolUse) {
        return Err(format!(
            "expected ToolUse, got {:?}: text={:?}",
            resp1.stop_reason,
            resp1.text()
        )
        .into());
    }
    let invocation = resp1
        .tool_invocations()
        .next()
        .ok_or("no tool invocation in response")?
        .clone();
    println!(
        "[turn 1] model wants to call `{}` with args: {}",
        invocation.name, invocation.input,
    );

    // --- Turn 2: feed back a fake tool result ---
    let mut messages = req1.messages().to_vec();
    messages.push(resp1.into_message());
    messages.push(Message::Tool {
        invocation_id: invocation.id,
        content: vec![RequestContent::Text("Sunny, 22°C".into())],
        is_error: false,
    });

    let req2 = ChatRequest::builder(model, 512)
        .messages(messages)
        .tool(weather_tool)
        .build();

    let resp2 = p.chat(&req2).await?;
    println!("\n[turn 2] {}", resp2.text());
    Ok(())
}

fn load_env() {
    let Ok(s) = std::fs::read_to_string(".env") else {
        return;
    };
    for raw in s.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        if let Some((k, v)) = line.split_once('=') {
            let val = v.trim().trim_matches(|c| c == '"' || c == '\'');
            if !val.is_empty() && std::env::var_os(k.trim()).is_none() {
                unsafe {
                    std::env::set_var(k.trim(), val);
                }
            }
        }
    }
}
