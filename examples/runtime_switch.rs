//! Hold `Box<dyn Provider>` and pick OpenAI or DeepSeek at runtime via a
//! `PROVIDER` env var. The downstream call site doesn't know which.
//!
//! Run with:
//!   PROVIDER=openai   cargo run --example runtime_switch
//!   PROVIDER=deepseek cargo run --example runtime_switch
//!
//! Reads from .env. Same OPENAI_* / DEEPSEEK_* variables as the other examples.

use llm_adapter::{Provider, deepseek, openai::OpenAIProvider, types::*};

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
        other => {
            return Err(format!("Unknown PROVIDER={other} (expected openai or deepseek)").into());
        }
    };

    println!("[using provider={kind} model={model}]");

    let req = ChatRequest::builder(model, 512)
        .message(Message::User(vec![RequestContent::Text(
            "Name three rivers. One word each.".into(),
        )]))
        .build();

    // Identical call site regardless of which provider sits behind the box.
    let resp = provider.chat(&req).await?;
    println!("{}", resp.text());
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
