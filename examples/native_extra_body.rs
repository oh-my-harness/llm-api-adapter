//! Native-path call with `extra_body` to inject a non-canonical request field.
//!
//! Demonstrates the DeepSeek interop fix: V4 enables thinking by default,
//! and rejects `tool_choice: Required` while thinking is on. Setting
//! `extra_body: {thinking: {type: disabled}}` works around it.
//!
//! Run with: `cargo run --example native_extra_body`
//!
//! Reads DEEPSEEK_* from .env. (Pointing OpenAI vars at DeepSeek works
//! too — base_url + a DeepSeek-compatible model.)

use llm_adapter::{
    openai::{OpenAIExt, OpenAIProvider, OpenAIRequest},
    types::*,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();

    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .map_err(|_| "DEEPSEEK_API_KEY is not set (try `source .env` first)")?;
    let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into());

    let p = OpenAIProvider::builder(api_key)
        .base_url("https://api.deepseek.com")
        .parse_reasoning_content(true)
        .build();

    let weather_tool = ToolDef::new(
        "get_weather",
        "Get current weather for a city.",
        json!({
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }),
    );

    let ext = OpenAIExt::builder()
        // The escape hatch: arbitrary top-level JSON fields merged into
        // the outgoing request body. Mirrors OpenAI Python SDK's
        // `extra_body=...`.
        .extra_body(json!({"thinking": {"type": "disabled"}}))
        .build();

    let oai_req = OpenAIRequest {
        base: ChatRequest::builder(model, 512)
            .message(Message::User(vec![RequestContent::Text(
                "What's the weather in Paris?".into(),
            )]))
            .tool(weather_tool)
            .tool_choice(ToolChoice::Required) // forbidden while thinking is enabled
            .build(),
        ext,
    };

    let resp = p.chat_native(&oai_req).await?;
    let invocations: Vec<_> = resp.tool_invocations().collect();
    println!(
        "stop_reason={:?}\ntool calls: {}",
        resp.stop_reason,
        invocations.len(),
    );
    for inv in invocations {
        println!("  - {}({})", inv.name, inv.input);
    }
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
