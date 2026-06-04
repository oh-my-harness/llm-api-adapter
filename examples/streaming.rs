//! Streaming chat: prints text deltas as they arrive, then summary stats
//! from the MessageStop event.
//!
//! Run with: `cargo run --example streaming`
//!
//! Reads from .env. Defaults to OpenAI; set OPENAI_BASE_URL to point at any
//! OpenAI-compatible endpoint (DeepSeek, GitHub Models, OpenRouter, ...).

use futures::StreamExt;
use llm_adapter::{Provider, openai::OpenAIProvider, types::*};
use std::io::Write;

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

    let req = ChatRequest::builder(model, 1024)
        .message(Message::User(vec![RequestContent::Text(
            "Count slowly from one to ten. Put each number on a new line.".into(),
        )]))
        .build();

    let mut handle = p.chat_stream(&req).await?;
    let mut text_chars = 0usize;

    while let Some(event) = handle.events().next().await {
        match event? {
            StreamEvent::ContentStart { kind, .. } => {
                match kind {
                    ContentKind::Text => {}
                    ContentKind::Reasoning => print!("[reasoning] "),
                    ContentKind::ToolInvocation { name, .. } => print!("[tool:{name}] "),
                }
                std::io::stdout().flush().ok();
            }
            StreamEvent::TextDelta { text, .. } => {
                text_chars += text.chars().count();
                print!("{text}");
                std::io::stdout().flush().ok();
            }
            StreamEvent::ReasoningDelta { text, .. } => {
                eprint!("{text}");
                std::io::stderr().flush().ok();
            }
            StreamEvent::ToolDelta { arguments, .. } => {
                print!("{arguments}");
                std::io::stdout().flush().ok();
            }
            StreamEvent::ContentStop { .. } => {}
            StreamEvent::MessageStop { stop_reason, usage } => {
                println!(
                    "\n\n[done — {text_chars} chars streamed; stop={stop_reason:?}; \
                     tokens={} in / {} out]",
                    usage.input_tokens, usage.output_tokens,
                );
            }
        }
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
