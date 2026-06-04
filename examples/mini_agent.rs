//! An interactive agent REPL that demonstrates multi-turn tool calling.
//!
//! Type a question, the agent calls tools as needed, answers, then waits for
//! your follow-up. Type `exit` or `Ctrl-C` to quit.
//!
//! Run with:
//! ```bash
//! cargo run --example mini_agent
//! cargo run --example mini_agent -- "What's 123 * 456?"
//! PROVIDER=anthropic cargo run --example mini_agent
//! PROVIDER=deepseek  cargo run --example mini_agent
//! ```
//!
//! Example session:
//! ```
//! You > What is 234 * 567?
//!   ── turn 1 tool: calculate({"expression":"234*567"}) ──
//!   → 132678
//!   ── turn 2 answer ──
//!   234 * 567 = 132,678
//!
//! You > Now add 1000 to that result.
//!   ── turn 1 tool: calculate({"expression":"132678+1000"}) ──
//!   → 133678
//!   ── turn 2 answer ──
//!   132,678 + 1,000 = 133,678
//! ```
//!
//! Environment variables (from .env or process env):
//!   PROVIDER            "openai" (default), "anthropic", or "deepseek"
//!   OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL / OPENAI_CHAT_PATH
//!   ANTHROPIC_API_KEY / ANTHROPIC_MODEL / ANTHROPIC_BASE_URL
//!   DEEPSEEK_API_KEY / DEEPSEEK_MODEL

use llm_adapter::{
    Provider, anthropic::AnthropicProvider, deepseek, openai::OpenAIProvider, types::*,
};
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Tools ──────────────────────────────────────────────────────────────────────

fn make_tools() -> Vec<ToolDef> {
    vec![
        ToolDef::new(
            "calculate",
            "Evaluate a mathematical expression. Supports + - * / ** ( ) and functions: sqrt, sin, cos, tan, log, ln, abs, ceil, floor, round, pi, e",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. \"234 * 567\""
                    }
                },
                "required": ["expression"],
            }),
        ),
        ToolDef::new(
            "get_date",
            "Get today's date in YYYY-MM-DD format",
            json!({
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["iso", "unix"],
                        "description": "'iso' for YYYY-MM-DD, 'unix' for epoch seconds"
                    }
                },
                "required": ["format"],
            }),
        ),
    ]
}

fn run_tool(name: &str, args: &serde_json::Value) -> String {
    match name {
        "calculate" => {
            let expr = args.get("expression").and_then(|v| v.as_str()).unwrap_or("");
            match eval_math(expr) {
                Ok(v) => format!("{v}"),
                Err(e) => format!("Error: {e}"),
            }
        }
        "get_date" => match args.get("format").and_then(|v| v.as_str()) {
            Some("unix") => format!(
                "{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ),
            _ => {
                let days = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    / 86400;
                let (y, m, d) = ymd_from_epoch(days as i64);
                format!("{y:04}-{m:02}-{d:02}")
            }
        },
        _ => format!("Unknown tool: {name}"),
    }
}

fn ymd_from_epoch(mut days: i64) -> (i64, u32, u32) {
    fn is_leap(y: i64) -> bool {
        y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
    }
    let mut y = 1970i64;
    loop {
        let yd = if is_leap(y) { 366 } else { 365 };
        if days < yd {
            break;
        }
        days -= yd;
        y += 1;
    }
    let dim = |m| match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        _ => 28,
    };
    let mut m = 1u32;
    loop {
        let dm = dim(m);
        if days < dm {
            break;
        }
        days -= dm;
        m += 1;
    }
    (y, m, (days + 1) as u32)
}

/// Minimal arithmetic expression evaluator (recursive descent, safe).
fn eval_math(expr: &str) -> Result<f64, String> {
    let s = expr.trim();
    if s.is_empty() {
        return Err("empty expression".into());
    }
    let cs: Vec<char> = s.chars().collect();
    let mut p = 0;
    let r = expr_inner(&cs, &mut p)?;
    skip_ws(&cs, &mut p);
    if p < cs.len() {
        return Err(format!("unexpected: {:?}", &s[p..]));
    }
    Ok(r)
}
fn skip_ws(cs: &[char], p: &mut usize) {
    while *p < cs.len() && cs[*p].is_ascii_whitespace() {
        *p += 1;
    }
}
fn expr_inner(cs: &[char], p: &mut usize) -> Result<f64, String> {
    let mut v = term(cs, p)?;
    loop {
        skip_ws(cs, p);
        if *p >= cs.len() {
            break;
        }
        match cs[*p] {
            '+' => {
                *p += 1;
                v += term(cs, p)?;
            }
            '-' => {
                *p += 1;
                v -= term(cs, p)?;
            }
            _ => break,
        }
    }
    Ok(v)
}
fn term(cs: &[char], p: &mut usize) -> Result<f64, String> {
    let mut v = power(cs, p)?;
    loop {
        skip_ws(cs, p);
        if *p >= cs.len() {
            break;
        }
        match cs[*p] {
            '*' => {
                *p += 1;
                v *= power(cs, p)?;
            }
            '/' => {
                *p += 1;
                let d = power(cs, p)?;
                if d == 0.0 {
                    return Err("division by zero".into());
                }
                v /= d;
            }
            _ => break,
        }
    }
    Ok(v)
}
fn power(cs: &[char], p: &mut usize) -> Result<f64, String> {
    let b = unary(cs, p)?;
    skip_ws(cs, p);
    if *p + 1 < cs.len() && cs[*p] == '*' && cs[*p + 1] == '*' {
        *p += 2;
        let e = power(cs, p)?;
        return Ok(b.powf(e));
    }
    Ok(b)
}
fn unary(cs: &[char], p: &mut usize) -> Result<f64, String> {
    skip_ws(cs, p);
    if *p >= cs.len() {
        return Err("unexpected end".into());
    }
    if cs[*p] == '-' {
        *p += 1;
        return Ok(-atom(cs, p)?);
    }
    if cs[*p] == '+' {
        *p += 1;
    }
    atom(cs, p)
}
fn atom(cs: &[char], p: &mut usize) -> Result<f64, String> {
    skip_ws(cs, p);
    if *p >= cs.len() {
        return Err("unexpected end".into());
    }
    if cs[*p] == '(' {
        *p += 1;
        let v = expr_inner(cs, p)?;
        skip_ws(cs, p);
        if *p >= cs.len() || cs[*p] != ')' {
            return Err("missing ')'".into());
        }
        *p += 1;
        return Ok(v);
    }
    if cs[*p].is_ascii_alphabetic() || cs[*p] == '_' {
        let s = *p;
        while *p < cs.len() && (cs[*p].is_ascii_alphanumeric() || cs[*p] == '_') {
            *p += 1;
        }
        let name: String = cs[s..*p].iter().collect();
        let name = name.to_lowercase();
        return match name.as_str() {
            "pi" => Ok(std::f64::consts::PI),
            "e" => Ok(std::f64::consts::E),
            _ => {
                skip_ws(cs, p);
                if *p < cs.len() && cs[*p] == '(' {
                    *p += 1;
                    let a = expr_inner(cs, p)?;
                    skip_ws(cs, p);
                    if *p >= cs.len() || cs[*p] != ')' {
                        return Err(format!("missing ) after {name}"));
                    }
                    *p += 1;
                    match name.as_str() {
                        "sqrt" => Ok(a.sqrt()),
                        "abs" => Ok(a.abs()),
                        "sin" => Ok(a.sin()),
                        "cos" => Ok(a.cos()),
                        "tan" => Ok(a.tan()),
                        "asin" => Ok(a.asin()),
                        "acos" => Ok(a.acos()),
                        "atan" => Ok(a.atan()),
                        "ln" | "log" => Ok(a.ln()),
                        "log10" => Ok(a.log10()),
                        "ceil" => Ok(a.ceil()),
                        "floor" => Ok(a.floor()),
                        "round" => Ok(a.round()),
                        _ => Err(format!("unknown: {name}")),
                    }
                } else {
                    Err(format!("unknown identifier: {name}"))
                }
            }
        };
    }
    let s = *p;
    let mut dot = false;
    while *p < cs.len() && (cs[*p].is_ascii_digit() || (!dot && cs[*p] == '.')) {
        if cs[*p] == '.' {
            dot = true;
        }
        *p += 1;
    }
    if *p == s {
        return Err(format!("expected number at {p}"));
    }
    cs[s..*p]
        .iter()
        .collect::<String>()
        .parse::<f64>()
        .map_err(|e| format!("bad number: {e}"))
}

// ── Agent loop ─────────────────────────────────────────────────────────────────

/// Process one user message through the full think→act→observe loop.
/// Returns accumulated messages including the new exchange.
async fn handle_turn(
    provider: &dyn Provider,
    messages: &mut Vec<Message>,
    user_text: &str,
    tools: &[ToolDef],
    model: &str,
    max_tool_turns: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    messages.push(Message::User(vec![RequestContent::Text(
        user_text.to_owned(),
    )]));

    for turn in 1..=max_tool_turns {
        let req = ChatRequest::builder(model, 1024)
            .messages(messages.clone())
            .tools(tools.to_vec())
            .build();

        let resp = provider.chat(&req).await?;

        // Show reasoning
        if let Some(r) = resp.reasoning() {
            println!("\n  ── reasoning ──\n{r}\n");
        }

        // Extract data before consuming resp
        let tool_invocations: Vec<_> = resp.tool_invocations().cloned().collect();
        let resp_text = resp.text();
        let resp_usage = (
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            resp.usage.cached_input_tokens,
            resp.usage.reasoning_tokens,
        );
        let assistant_msg = resp.into_message();
        messages.push(assistant_msg);

        if tool_invocations.is_empty() {
            // Final answer — done with this user message
            println!("  ── answer ──\n{resp_text}");
            println!(
                "  [input:{} output:{} cached:{} reasoning:{}]",
                resp_usage.0, resp_usage.1, resp_usage.2, resp_usage.3,
            );
            return Ok(());
        }

        // Execute tool calls
        for inv in &tool_invocations {
            println!(
                "  ── turn {turn} tool: {}({}) ──",
                inv.name, inv.input
            );
            let result = run_tool(&inv.name, &inv.input);
            println!("  → {result}");
            messages.push(Message::Tool {
                invocation_id: inv.id.clone(),
                content: vec![RequestContent::Text(result)],
                is_error: false,
            });
        }
    }

    Err("max tool turns reached".into())
}

// ── Entry point ────────────────────────────────────────────────────────────────

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

fn build_provider() -> Result<(Box<dyn Provider>, String), Box<dyn std::error::Error>> {
    let kind = std::env::var("PROVIDER").unwrap_or_else(|_| "openai".into());
    let (p, model) = match kind.as_str() {
        "deepseek" => {
            let key =
                std::env::var("DEEPSEEK_API_KEY").map_err(|_| "DEEPSEEK_API_KEY not set")?;
            let m = std::env::var("DEEPSEEK_MODEL")
                .unwrap_or_else(|_| "deepseek-v4-flash".into());
            (Box::new(deepseek::client(key)) as Box<dyn Provider>, m)
        }
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set")?;
            let base =
                std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".into());
            let path = std::env::var("OPENAI_CHAT_PATH")
                .unwrap_or_else(|_| "/v1/chat/completions".into());
            let m = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
            (
                Box::new(
                    OpenAIProvider::builder(key).base_url(base).chat_path(path).build(),
                ) as Box<dyn Provider>,
                m,
            )
        }
        "anthropic" => {
            let key =
                std::env::var("ANTHROPIC_API_KEY").map_err(|_| "ANTHROPIC_API_KEY not set")?;
            let base = std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".into());
            let path = std::env::var("ANTHROPIC_MESSAGES_PATH")
                .unwrap_or_else(|_| "/v1/messages".into());
            let m = std::env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-6-20250514".into());
            (
                Box::new(
                    AnthropicProvider::builder(key)
                        .base_url(base)
                        .messages_path(path)
                        .build(),
                ) as Box<dyn Provider>,
                m,
            )
        }
        other => return Err(format!("PROVIDER={other} (use openai|anthropic|deepseek)").into()),
    };
    Ok((p, model))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();
    let (provider, model) = build_provider()?;

    println!("╔══════════════════════════════════════════════╗");
    println!("║         llm_adapter — Mini Agent Demo        ║");
    println!("╚══════════════════════════════════════════════╝");
    println!(
        "  provider  {}",
        std::env::var("PROVIDER").unwrap_or_else(|_| "openai".into())
    );
    println!("  model     {model}");
    println!("  tools     calculate, get_date");
    println!();
    println!("Type a question, or `exit` to quit.");
    println!();

    let tools = make_tools();
    let mut messages = vec![
        Message::System(
            "You are a helpful assistant with access to tools: \
             calculate (evaluate math expressions) and get_date (current date). \
             Use tools when you need to compute something. \
             If the user refers to a previous result, compute it again—you \
             don't have memory of tool outputs across turns."
                .into(),
        ),
    ];

    // Check for one-shot question from CLI args
    let arg_text: String = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let one_shot = if arg_text.is_empty() { None } else { Some(arg_text) };

    if let Some(q) = &one_shot {
        println!("You > {q}");
        if let Err(e) = handle_turn(provider.as_ref(), &mut messages, q, &tools, &model, 5).await {
            eprintln!("  [{e}]");
        }
        println!();
        return Ok(());
    }

    // Interactive REPL
    let stdin = std::io::stdin();
    let mut input = String::new();
    loop {
        print!("You > ");
        use std::io::Write;
        std::io::stdout().flush()?;

        input.clear();
        if stdin.read_line(&mut input)? == 0 {
            break; // EOF
        }
        let line = input.trim();
        if line.is_empty() || line == "exit" || line == "quit" {
            break;
        }

        if let Err(e) = handle_turn(provider.as_ref(), &mut messages, line, &tools, &model, 5).await {
            eprintln!("  [{e}]");
        }
        println!();
    }

    println!("── bye ──");
    Ok(())
}
