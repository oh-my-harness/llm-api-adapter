//! Hold `Box<dyn Provider>` and pick OpenAI, DeepSeek, or Anthropic at runtime
//! via a `PROVIDER` env var. The downstream call site doesn't know which.
//!
//! Demonstrates both the non-streaming and streaming paths with the same
//! request, highlighting reasoning blocks when present.
//!
//! Run with:
//!   PROVIDER=openai    cargo run --example runtime_switch
//!   PROVIDER=deepseek  cargo run --example runtime_switch
//!   PROVIDER=anthropic cargo run --example runtime_switch
//!
//! Reads from .env. Same OPENAI_* / DEEPSEEK_* / ANTHROPIC_* variables as the
//! other examples.

use futures::StreamExt;
use llm_adapter::{
    Provider, anthropic::AnthropicProvider, deepseek, openai::OpenAIProvider, types::*,
};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();

    let kind = std::env::var("PROVIDER").unwrap_or_else(|_| "openai".into());

    let (provider, model): (Box<dyn Provider>, String) = match kind.as_str() {
        "deepseek" => {
            let api_key =
                std::env::var("DEEPSEEK_API_KEY").map_err(|_| "DEEPSEEK_API_KEY is not set")?;
            let model =
                std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into());
            (Box::new(deepseek::client(api_key)), model)
        }
        "openai" => {
            let api_key =
                std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY is not set")?;
            let base_url = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com".into());
            let chat_path =
                std::env::var("OPENAI_CHAT_PATH").unwrap_or_else(|_| "/v1/chat/completions".into());
            let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
            let p = OpenAIProvider::builder(api_key)
                .base_url(base_url)
                .chat_path(chat_path)
                .build();
            (Box::new(p), model)
        }
        "anthropic" => {
            let api_key =
                std::env::var("ANTHROPIC_API_KEY").map_err(|_| "ANTHROPIC_API_KEY is not set")?;
            let base_url = std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".into());
            let messages_path =
                std::env::var("ANTHROPIC_MESSAGES_PATH").unwrap_or_else(|_| "/v1/messages".into());
            let model = std::env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-6-20250514".into());
            let p = AnthropicProvider::builder(api_key)
                .base_url(base_url)
                .messages_path(messages_path)
                .build();
            (Box::new(p), model)
        }
        other => {
            return Err(format!(
                "Unknown PROVIDER={other} (expected openai, deepseek, or anthropic)"
            )
            .into());
        }
    };

    println!("[using provider={kind} model={model}]");

    let req = ChatRequest::builder(model, 512)
        .message(Message::User(vec![RequestContent::Text(
            "Name three rivers. One word each.".into(),
        )]))
        .build();

    // --- Non-streaming path ---
    println!("--- chat() ---");
    let resp = provider.chat(&req).await?;
    println!("{}", resp.text());
    if let Some(r) = resp.reasoning() {
        println!("(reasoning: {r})");
    }

    // --- Streaming path (same request) ---
    println!("--- chat_stream() ---");
    let mut handle = provider.chat_stream(&req).await?;
    let mut reasoning_count = 0usize;

    while let Some(event) = handle.events().next().await {
        match event? {
            StreamEvent::TextDelta { text, .. } => {
                print!("{text}");
                std::io::stdout().flush().ok();
            }
            StreamEvent::ContentStart {
                kind: ContentKind::Reasoning,
                ..
            } => {
                eprint!("[reasoning] ");
                std::io::stderr().flush().ok();
            }
            StreamEvent::ReasoningDelta { text, .. } => {
                eprint!("{text}");
                std::io::stderr().flush().ok();
                reasoning_count += 1;
            }
            StreamEvent::ContentStop { .. } => {}
            _ => {}
        }
    }

    if reasoning_count > 0 {
        println!("(stream contained {reasoning_count} reasoning blocks)");
    }
    println!();
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
