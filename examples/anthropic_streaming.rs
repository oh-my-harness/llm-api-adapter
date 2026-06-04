//! Streaming chat with Anthropic: prints text/reasoning deltas as they
//! arrive, then summary stats from the MessageStop event.
//!
//! Run with: `cargo run --example anthropic_streaming`
//!
//! Reads from .env (or process env):
//!   ANTHROPIC_API_KEY        required
//!   ANTHROPIC_BASE_URL       default https://api.anthropic.com
//!   ANTHROPIC_MESSAGES_PATH  default /v1/messages
//!   ANTHROPIC_MODEL          default claude-sonnet-4-6-20250514

use futures::StreamExt;
use llm_adapter::{Provider, anthropic::AnthropicProvider, types::*};
use std::io::Write;

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
