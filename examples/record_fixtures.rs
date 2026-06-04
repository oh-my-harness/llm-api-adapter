//! Record real API responses into tests/fixtures/.
//!
//! Loads credentials from .env, calls each provider with a stable prompt,
//! and writes responses verbatim — preserving server field order, undocumented
//! fields, and stream chunk boundaries. Replaces hand-crafted fixtures so the
//! offline test suite reflects real wire behaviour.
//!
//! Usage:
//!     cargo run --example record_fixtures
//!
//! Behaviour:
//! - Providers without an API key set: skipped
//! - Reasoning scenarios without a *_REASONING_MODEL set: skipped
//! - One scenario failing does not abort the rest

use std::path::{Path, PathBuf};

use futures::StreamExt;
use serde_json::json;

const OAI_DEFAULT_BASE: &str = "https://api.openai.com";
const DS_DEFAULT_BASE: &str = "https://api.deepseek.com";
const FIXTURES_ROOT: &str = "tests/fixtures";

#[derive(Clone, Copy, PartialEq)]
enum ProviderKind {
    /// True OpenAI Chat Completions semantics (reasoning_effort, max_completion_tokens).
    OpenAI,
    /// DeepSeek-style: thinking field controls reasoning, max_tokens is the cap.
    DeepSeek,
}

#[derive(Clone)]
struct ProviderConfig {
    name: &'static str,
    api_key: String,
    base_url: String,
    /// Path appended to base_url for chat completions.
    /// Default `/v1/chat/completions`; some compatible providers need a different
    /// path (e.g. GitHub Models uses `/chat/completions`).
    chat_path: String,
    /// Determines how reasoning is enabled and which token-limit field is used.
    kind: ProviderKind,
    model: Option<String>,
    reasoning_model: Option<String>,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    load_dotenv(".env");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .no_proxy()
        .build()
        .expect("build reqwest client");

    let openai_kind = match get_env("OPENAI_PROVIDER_KIND").as_str() {
        "deepseek" => ProviderKind::DeepSeek,
        _ => ProviderKind::OpenAI,
    };

    let openai = ProviderConfig {
        name: "openai",
        api_key: get_env("OPENAI_API_KEY"),
        base_url: get_env_or("OPENAI_BASE_URL", OAI_DEFAULT_BASE),
        chat_path: get_env_or("OPENAI_CHAT_PATH", "/v1/chat/completions"),
        kind: openai_kind,
        model: opt_env("OPENAI_MODEL"),
        reasoning_model: opt_env("OPENAI_REASONING_MODEL"),
    };
    let deepseek = ProviderConfig {
        name: "deepseek",
        api_key: get_env("DEEPSEEK_API_KEY"),
        base_url: get_env_or("DEEPSEEK_BASE_URL", DS_DEFAULT_BASE),
        chat_path: "/v1/chat/completions".into(),
        kind: ProviderKind::DeepSeek,
        model: opt_env("DEEPSEEK_MODEL"),
        reasoning_model: opt_env("DEEPSEEK_REASONING_MODEL"),
    };

    if openai.api_key.is_empty() {
        println!("[skip] openai: OPENAI_API_KEY not set");
    } else {
        record_openai(&client, &openai).await;
    }

    if deepseek.api_key.is_empty() {
        println!("[skip] deepseek: DEEPSEEK_API_KEY not set");
    } else {
        record_deepseek(&client, &deepseek).await;
    }

    println!("done.");
}

// ── OpenAI scenarios ─────────────────────────────────────────────────────────

async fn record_openai(client: &reqwest::Client, cfg: &ProviderConfig) {
    let Some(model) = cfg.model.as_deref() else {
        println!("[skip] openai: OPENAI_MODEL not set");
        return;
    };

    let dir = fixtures_dir("openai");
    let is_deepseek = cfg.kind == ProviderKind::DeepSeek;
    let tok_field = if is_deepseek { "max_tokens" } else { "max_completion_tokens" };

    // chat_simple: explicitly suppress reasoning on DeepSeek-style endpoints
    // (DeepSeek V4 enables thinking by default; real OpenAI ignores this field).
    let mut simple_body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say hi in exactly five words."}],
        "max_tokens": 256,
    });
    if is_deepseek {
        simple_body.as_object_mut().unwrap()
            .insert("thinking".into(), json!({"type": "disabled"}));
    }
    run_chat(client, cfg, &dir, "chat_simple.json", simple_body).await;

    // chat_tool_call
    let weather_tool = json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    });
    let mut tool_call_body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [weather_tool],
        "tool_choice": "required",
        "max_tokens": 256,
    });
    if is_deepseek {
        tool_call_body.as_object_mut().unwrap()
            .insert("thinking".into(), json!({"type": "disabled"}));
    }
    run_chat(client, cfg, &dir, "chat_tool_call.json", tool_call_body.clone()).await;

    // chat_after_tool: replay the recorded tool_call so the follow-up assumes
    // the same call_id / arguments. We craft this manually rather than reading
    // back because each scenario should be self-contained.
    let mut after_tool_body = json!({
        "model": model,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_recorded",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_recorded", "content": "Sunny, 22°C"},
        ],
        "tools": [weather_tool],
        "max_tokens": 256,
    });
    if is_deepseek {
        after_tool_body.as_object_mut().unwrap()
            .insert("thinking".into(), json!({"type": "disabled"}));
    }
    run_chat(client, cfg, &dir, "chat_after_tool.json", after_tool_body).await;

    // stream_simple
    let mut simple_stream = json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say hi in exactly five words."}],
        "max_tokens": 256,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if is_deepseek {
        simple_stream.as_object_mut().unwrap()
            .insert("thinking".into(), json!({"type": "disabled"}));
    }
    run_stream(client, cfg, &dir, "stream_simple.txt", simple_stream).await;

    // stream_tool_call
    run_stream(
        client, cfg, &dir, "stream_tool_call.txt",
        tool_call_body_with_stream(&tool_call_body),
    ).await;

    // Reasoning scenarios
    if let Some(rm) = cfg.reasoning_model.as_deref() {
        // DeepSeek V4: thinking is on by default — don't pass reasoning_effort
        //   (DeepSeek rejects unknown fields → 400).
        // True OpenAI o-series / GPT-5: use reasoning_effort + max_completion_tokens.
        let mut chat_body = json!({
            "model": rm,
            "messages": [{"role": "user", "content": "What is 7 times 8? Answer with just the number."}],
            tok_field: 2048,
        });
        if !is_deepseek {
            chat_body.as_object_mut().unwrap()
                .insert("reasoning_effort".into(), json!("low"));
        }
        run_chat(client, cfg, &dir, "chat_reasoning.json", chat_body.clone()).await;

        let mut stream_body = chat_body;
        let obj = stream_body.as_object_mut().unwrap();
        obj.insert("stream".into(), json!(true));
        obj.insert("stream_options".into(), json!({"include_usage": true}));
        run_stream(client, cfg, &dir, "stream_reasoning_usage.txt", stream_body).await;
    } else {
        println!("[skip] openai: OPENAI_REASONING_MODEL not set");
    }
}

fn tool_call_body_with_stream(base: &serde_json::Value) -> serde_json::Value {
    let mut body = base.clone();
    let obj = body.as_object_mut().unwrap();
    obj.insert("stream".into(), json!(true));
    obj.insert("stream_options".into(), json!({"include_usage": true}));
    body
}

// ── DeepSeek scenarios ───────────────────────────────────────────────────────

async fn record_deepseek(client: &reqwest::Client, cfg: &ProviderConfig) {
    let Some(model) = cfg.model.as_deref() else {
        println!("[skip] deepseek: DEEPSEEK_MODEL not set");
        return;
    };

    let dir = fixtures_dir("deepseek");

    // chat_simple: explicitly disable thinking mode (DeepSeek V4 enables it
    // by default; we want a baseline "no reasoning_content" fixture).
    run_chat(
        client, cfg, &dir, "chat_simple.json",
        json!({
            "model": model,
            "messages": [{"role": "user", "content": "Say hi in exactly five words."}],
            "max_tokens": 256,
            "thinking": {"type": "disabled"},
        }),
    ).await;

    if let Some(rm) = cfg.reasoning_model.as_deref() {
        // Thinking mode: DeepSeek V4 enables it via top-level "thinking" field
        // (extra_body in OpenAI SDK speak — at HTTP layer it's just a body field).
        // Legacy deepseek-reasoner triggers via model name.
        let is_legacy = rm.contains("reasoner");
        let mut chat_body = json!({
            "model": rm,
            "messages": [{"role": "user", "content": "What is 7 times 8? Answer with just the number."}],
            "max_tokens": 4096,
        });
        if !is_legacy {
            chat_body.as_object_mut().unwrap()
                .insert("thinking".into(), json!({"type": "enabled"}));
        }
        run_chat(client, cfg, &dir, "chat_reasoning.json", chat_body.clone()).await;

        let mut stream_body = chat_body;
        let obj = stream_body.as_object_mut().unwrap();
        obj.insert("stream".into(), json!(true));
        obj.insert("stream_options".into(), json!({"include_usage": true}));
        run_stream(client, cfg, &dir, "stream_reasoning.txt", stream_body).await;
    } else {
        println!("[skip] deepseek: DEEPSEEK_REASONING_MODEL not set");
    }
}

// ── HTTP helpers ─────────────────────────────────────────────────────────────

async fn run_chat(
    client: &reqwest::Client,
    cfg: &ProviderConfig,
    dir: &Path,
    filename: &str,
    body: serde_json::Value,
) {
    let path = dir.join(filename);
    print!("[{}] -> {} ... ", cfg.name, display_path(&path));
    let url = format!("{}{}", cfg.base_url.trim_end_matches('/'), &cfg.chat_path);

    let resp = match client
        .post(&url)
        .header("Authorization", format!("Bearer {}", cfg.api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => { println!("FAIL (send): {e}"); return; }
    };
    let status = resp.status();
    let text = match resp.text().await {
        Ok(t) => t,
        Err(e) => { println!("FAIL (read): {e}"); return; }
    };
    if !status.is_success() {
        println!("HTTP {} — body: {}", status, text);
        return;
    }
    if let Err(e) = std::fs::write(&path, &text) {
        println!("FAIL (write): {e}");
        return;
    }
    println!("ok ({} bytes)", text.len());
}

async fn run_stream(
    client: &reqwest::Client,
    cfg: &ProviderConfig,
    dir: &Path,
    filename: &str,
    body: serde_json::Value,
) {
    let path = dir.join(filename);
    print!("[{}] -> {} (stream) ... ", cfg.name, display_path(&path));
    let url = format!("{}{}", cfg.base_url.trim_end_matches('/'), &cfg.chat_path);

    let resp = match client
        .post(&url)
        .header("Authorization", format!("Bearer {}", cfg.api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => { println!("FAIL (send): {e}"); return; }
    };
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        println!("HTTP {} — body: {}", status, text);
        return;
    }

    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(b) => buf.extend_from_slice(&b),
            Err(e) => { println!("FAIL (stream): {e}"); return; }
        }
    }
    if let Err(e) = std::fs::write(&path, &buf) {
        println!("FAIL (write): {e}");
        return;
    }
    println!("ok ({} bytes)", buf.len());
}

// ── env / fs helpers ─────────────────────────────────────────────────────────

fn load_dotenv(path: &str) {
    let Ok(content) = std::fs::read_to_string(path) else { return; };
    for raw in content.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((k, v)) = line.split_once('=') else { continue; };
        let key = k.trim();
        let val = v.trim().trim_matches('"').trim_matches('\'');
        if !val.is_empty() && std::env::var_os(key).is_none() {
            unsafe { std::env::set_var(key, val); }
        }
    }
}

fn get_env(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

fn get_env_or(name: &str, fallback: &str) -> String {
    let v = get_env(name);
    if v.is_empty() { fallback.into() } else { v }
}

fn opt_env(name: &str) -> Option<String> {
    let v = get_env(name);
    if v.is_empty() { None } else { Some(v) }
}

fn fixtures_dir(provider: &str) -> PathBuf {
    let p = PathBuf::from(FIXTURES_ROOT).join(provider);
    std::fs::create_dir_all(&p).expect("create fixtures dir");
    p
}

fn display_path(p: &Path) -> String {
    p.strip_prefix(std::env::current_dir().unwrap_or_default())
        .unwrap_or(p)
        .display()
        .to_string()
}
