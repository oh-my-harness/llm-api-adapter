//! Native-path Anthropic call demonstrating provider-specific extensions.
//!
//! Shows top_k, thinking config, metadata, and disable_parallel_tool_use
//! — fields that only exist in the Anthropic Messages API and have no
//! canonical equivalent.
//!
//! Run with: `cargo run --example anthropic_native`
//!
//! Reads ANTHROPIC_* from .env.

use llm_adapter::{
    anthropic::{
        AnthropicExt, AnthropicMetadata, AnthropicProvider, AnthropicRequest, ThinkingConfig,
    },
    types::*,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY is not set (try `source .env` first)")?;
    let base_url =
        std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| "https://api.anthropic.com".into());
    let messages_path =
        std::env::var("ANTHROPIC_MESSAGES_PATH").unwrap_or_else(|_| "/v1/messages".into());
    let model =
        std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-6-20250514".into());

    let p = AnthropicProvider::builder(api_key)
        .base_url(base_url)
        .messages_path(messages_path)
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

    let ext = AnthropicExt::builder()
        .top_k(40)
        .thinking(ThinkingConfig {
            budget_tokens: 1000,
        })
        .metadata(AnthropicMetadata {
            user_id: Some("example-user".into()),
        })
        .disable_parallel_tool_use()
        .build();

    let anth_req = AnthropicRequest {
        base: ChatRequest::builder(model, 512)
            .message(Message::User(vec![RequestContent::Text(
                "What's the weather in Paris?".into(),
            )]))
            .tool(weather_tool)
            .build(),
        ext,
    };

    let resp = p.chat_native(&anth_req).await?;
    let invocations: Vec<_> = resp.tool_invocations().collect();
    println!(
        "stop_reason={:?}\ntool calls: {}",
        resp.stop_reason,
        invocations.len(),
    );
    for inv in invocations {
        println!("  - {}({})", inv.name, inv.input);
    }
    if let Some(reasoning) = resp.reasoning() {
        println!("reasoning: {reasoning}");
    }
    println!(
        "usage: {} in / {} out / {} cached",
        resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.cached_input_tokens,
    );
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
