//! Minimal one-turn chat that prints the model's text and token usage.
//!
//! Run with: `cargo run --example simple_chat`
//!
//! Reads from .env (or process env):
//!   OPENAI_API_KEY   required
//!   OPENAI_BASE_URL  default https://api.openai.com
//!   OPENAI_CHAT_PATH default /v1/chat/completions
//!   OPENAI_MODEL     default gpt-4o-mini

use llm_adapter::{Provider, openai::OpenAIProvider, types::*};

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

    let req = ChatRequest::builder(model, 512)
        .message(Message::System("Be concise.".into()))
        .message(Message::User(vec![RequestContent::Text(
            "Say hi in five words.".into(),
        )]))
        .temperature(0.7)
        .build();

    let resp = p.chat(&req).await?;
    println!("AI:\n{}", resp.text());
    println!(
        "\n[tokens: {} in / {} out / {} cached]",
        resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.cached_input_tokens,
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
