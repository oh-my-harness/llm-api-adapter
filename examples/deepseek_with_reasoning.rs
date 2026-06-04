//! DeepSeek via the convenience factory, printing both the final answer
//! and the model's chain-of-thought (`reasoning_content`).
//!
//! Run with: `cargo run --example deepseek_with_reasoning`
//!
//! Reads from .env:
//!   DEEPSEEK_API_KEY   required
//!   DEEPSEEK_MODEL     default deepseek-v4-flash
//!
//! DeepSeek V4 enables thinking mode by default, so a non-trivial prompt
//! will produce a `reasoning_content` block alongside the final answer.

use llm_adapter::{Provider, deepseek, types::*};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();

    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .map_err(|_| "DEEPSEEK_API_KEY is not set (try `source .env` first)")?;
    let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into());

    let p = deepseek::client(api_key);
    // `deepseek::client` is a pre-configured OpenAIProvider with
    //   base_url = https://api.deepseek.com
    //   parse_reasoning_content = true
    //   tolerant_keepalive       = true

    let req = ChatRequest::builder(model, 2048)
        .message(Message::User(vec![RequestContent::Text(
            "What is 7 times 8? Answer with just the number.".into(),
        )]))
        .build();

    let resp = p.chat(&req).await?;

    println!("Answer: {}", resp.text());

    if let Some(reasoning) = resp.reasoning() {
        println!("\n--- Reasoning ---\n{reasoning}");
    } else {
        println!("\n(No reasoning in response — thinking mode may be off.)");
    }

    println!(
        "\n[tokens: {} in / {} out / {} reasoning]",
        resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.reasoning_tokens,
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
