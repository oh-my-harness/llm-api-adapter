//! Minimal one-turn chat with Anthropic that prints the model's text and
//! token usage.
//!
//! Run with: `cargo run --example anthropic_simple_chat`
//!
//! Reads from .env (or process env):
//!   ANTHROPIC_API_KEY        required
//!   ANTHROPIC_BASE_URL       default https://api.anthropic.com
//!   ANTHROPIC_MESSAGES_PATH  default /v1/messages
//!   ANTHROPIC_MODEL          default claude-sonnet-4-6-20250514

use llm_adapter::{Provider, anthropic::AnthropicProvider, types::*};

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

    let req = ChatRequest::builder(model, 512)
        .message(Message::System("Be concise.".into()))
        .message(Message::User(vec![RequestContent::Text(
            "Say hi in five words.".into(),
        )]))
        .temperature(0.7)
        .build();

    let resp = p.chat(&req).await?;
    println!("AI:\n{}", resp.text());
    if let Some(reasoning) = resp.reasoning() {
        println!("\n(thought process: {reasoning})");
    }
    println!(
        "\n[tokens: {} in / {} out / {} cached / {} cache-creation]",
        resp.usage.input_tokens,
        resp.usage.output_tokens,
        resp.usage.cached_input_tokens,
        resp.usage.cache_creation_input_tokens,
    );
    Ok(())
}

/// Minimal `.env` loader: parses `[export ] KEY=VALUE` lines, ignores
/// comments / blanks, and only sets keys that aren't already in env.
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
