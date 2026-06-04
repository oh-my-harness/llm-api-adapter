//! A mini agent that uses tool calling to answer questions.
//!
//! The agent has a `calculate` tool and runs the think→act→observe loop
//! automatically. Supports switching between OpenAI, Anthropic, and DeepSeek.
//!
//! Run with:
//! ```bash
//! cargo run --example mini_agent -- "What is 234 * 567?"
//! PROVIDER=anthropic cargo run --example mini_agent -- "If I have 123 apples and give away 45, how many remain?"
//! PROVIDER=deepseek  cargo run --example mini_agent -- "Calculate 2^10 + 3^5"
//! ```
//!
//! Environment variables (from .env or process env):
//!   PROVIDER            default "openai"
//!   OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL / OPENAI_CHAT_PATH
//!   ANTHROPIC_API_KEY / ANTHROPIC_MODEL / ANTHROPIC_BASE_URL
//!   DEEPSEEK_API_KEY / DEEPSEEK_MODEL

use llm_adapter::{
    Provider, anthropic::AnthropicProvider, deepseek, openai::OpenAIProvider, types::*,
};
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Tools ──────────────────────────────────────────────────────────────────────

/// Tool definitions shared across all providers.
fn tools() -> Vec<ToolDef> {
    vec![
        ToolDef::new(
            "calculate",
            "Evaluate a mathematical expression. Supports + - * / ** ( ) and built-in functions like sqrt, sin, cos, log, pi, e",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. 234 * 567 or sqrt(144)"
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
                        "description": "Date format: 'iso' for YYYY-MM-DD, 'unix' for epoch seconds"
                    }
                },
                "required": ["format"],
            }),
        ),
    ]
}

/// Execute a tool call and return the result.
fn run_tool(name: &str, args: &serde_json::Value) -> String {
    match name {
        "calculate" => {
            let expr = args.get("expression").and_then(|v| v.as_str()).unwrap_or("");
            match eval_math(expr) {
                Ok(result) => format!("{result}"),
                Err(e) => format!("Error: {e}"),
            }
        }
        "get_date" => match args.get("format").and_then(|v| v.as_str()) {
            Some("unix") => {
                let secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                format!("{secs}")
            }
            _ => {
                // ISO 8601 date via a simple approach — format literals.
                let secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                // Days since epoch, then compute year/month/day.
                let days = secs / 86400;
                let (year, month, day) = ymd_from_epoch(days as i64);
                format!("{year:04}-{month:02}-{day:02}")
            }
        },
        _ => format!("Unknown tool: {name}"),
    }
}

/// Compute year/month/day from days since Unix epoch.
fn ymd_from_epoch(mut days: i64) -> (i64, u32, u32) {
    // Days per month for a given year.
    fn days_in(y: i64, m: u32) -> i64 {
        match m {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                    29
                } else {
                    28
                }
            }
            _ => unreachable!(),
        }
    }
    // Find year (linear search from 1970; fine for our range).
    let mut y = 1970;
    loop {
        let yd = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366
        } else {
            365
        };
        if days < yd {
            break;
        }
        days -= yd;
        y += 1;
    }
    let mut m = 1u32;
    loop {
        let dm = days_in(y, m);
        if days < dm {
            break;
        }
        days -= dm;
        m += 1;
    }
    (y, m, (days + 1) as u32)
}

/// Minimal arithmetic expression evaluator (safe, no eval()).
fn eval_math(expr: &str) -> Result<f64, String> {
    let s = expr.trim();
    if s.is_empty() {
        return Err("empty expression".into());
    }
    // Simple recursive descent: parse term → factor → number / paren / function.
    let chars: Vec<char> = s.chars().collect();
    let mut pos = 0;
    let result = parse_expr(&chars, &mut pos)?;
    skip_ws(&chars, &mut pos);
    if pos < chars.len() {
        return Err(format!("unexpected trailing chars: {:?}", &s[pos..]));
    }
    Ok(result)
}

fn skip_ws(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

// expression := term ( '+' | '-' term )*
fn parse_expr(chars: &[char], pos: &mut usize) -> Result<f64, String> {
    let mut val = parse_term(chars, pos)?;
    loop {
        skip_ws(chars, pos);
        if *pos >= chars.len() {
            break;
        }
        match chars[*pos] {
            '+' => {
                *pos += 1;
                val += parse_term(chars, pos)?;
            }
            '-' => {
                *pos += 1;
                val -= parse_term(chars, pos)?;
            }
            _ => break,
        }
    }
    Ok(val)
}

// term := factor ( '*' | '/' factor )*
fn parse_term(chars: &[char], pos: &mut usize) -> Result<f64, String> {
    let mut val = parse_power(chars, pos)?;
    loop {
        skip_ws(chars, pos);
        if *pos >= chars.len() {
            break;
        }
        match chars[*pos] {
            '*' => {
                *pos += 1;
                val *= parse_power(chars, pos)?;
            }
            '/' => {
                *pos += 1;
                let denom = parse_power(chars, pos)?;
                if denom == 0.0 {
                    return Err("division by zero".into());
                }
                val /= denom;
            }
            _ => break,
        }
    }
    Ok(val)
}

// power := factor ( '**' factor )?
fn parse_power(chars: &[char], pos: &mut usize) -> Result<f64, String> {
    let base = parse_unary(chars, pos)?;
    skip_ws(chars, pos);
    if *pos < chars.len() && chars[*pos] == '*' && *pos + 1 < chars.len() && chars[*pos + 1] == '*' {
        *pos += 2;
        let exp = parse_power(chars, pos)?;
        return Ok(base.powf(exp));
    }
    Ok(base)
}

// unary := '-' factor | '+' factor | factor
fn parse_unary(chars: &[char], pos: &mut usize) -> Result<f64, String> {
    skip_ws(chars, pos);
    if *pos >= chars.len() {
        return Err("unexpected end of expression".into());
    }
    if chars[*pos] == '-' {
        *pos += 1;
        return Ok(-parse_atom(chars, pos)?);
    }
    if chars[*pos] == '+' {
        *pos += 1;
    }
    parse_atom(chars, pos)
}

// atom := number | '(' expr ')' | function_call | constant
fn parse_atom(chars: &[char], pos: &mut usize) -> Result<f64, String> {
    skip_ws(chars, pos);
    if *pos >= chars.len() {
        return Err("unexpected end of expression".into());
    }

    // '(' expr ')'
    if chars[*pos] == '(' {
        *pos += 1;
        let val = parse_expr(chars, pos)?;
        skip_ws(chars, pos);
        if *pos >= chars.len() || chars[*pos] != ')' {
            return Err("missing closing parenthesis".into());
        }
        *pos += 1;
        return Ok(val);
    }

    // Try to parse a function or named constant
    if chars[*pos].is_ascii_alphabetic() || chars[*pos] == '_' {
        let start = *pos;
        while *pos < chars.len() && (chars[*pos].is_ascii_alphanumeric() || chars[*pos] == '_') {
            *pos += 1;
        }
        let name: String = chars[start..*pos].iter().collect();
        let name = name.to_lowercase();

        // Named constants
        return match name.as_str() {
            "pi" => Ok(std::f64::consts::PI),
            "e" => Ok(std::f64::consts::E),
            // Functions with argument in parentheses
            _ => {
                skip_ws(chars, pos);
                if *pos < chars.len() && chars[*pos] == '(' {
                    *pos += 1; // '('
                    let arg = parse_expr(chars, pos)?;
                    skip_ws(chars, pos);
                    if *pos >= chars.len() || chars[*pos] != ')' {
                        return Err(format!("missing ) after {name}"));
                    }
                    *pos += 1; // ')'
                    match name.as_str() {
                        "sqrt" => Ok(arg.sqrt()),
                        "abs" => Ok(arg.abs()),
                        "sin" => Ok(arg.sin()),
                        "cos" => Ok(arg.cos()),
                        "tan" => Ok(arg.tan()),
                        "asin" => Ok(arg.asin()),
                        "acos" => Ok(arg.acos()),
                        "atan" => Ok(arg.atan()),
                        "ln" | "log" => Ok(arg.ln()),
                        "log10" => Ok(arg.log10()),
                        "ceil" => Ok(arg.ceil()),
                        "floor" => Ok(arg.floor()),
                        "round" => Ok(arg.round()),
                        _ => Err(format!("unknown function: {name}")),
                    }
                } else {
                    Err(format!("unknown identifier: {name}"))
                }
            }
        };
    }

    // Number
    parse_number(chars, pos)
}

fn parse_number(chars: &[char], pos: &mut usize) -> Result<f64, String> {
    skip_ws(chars, pos);
    let start = *pos;
    let mut has_dot = false;
    while *pos < chars.len() {
        match chars[*pos] {
            '0'..='9' => *pos += 1,
            '.' if !has_dot => {
                has_dot = true;
                *pos += 1;
            }
            _ => break,
        }
    }
    if *pos == start {
        return Err(format!(
            "expected number at position {pos}: {:?}",
            if start < chars.len() {
                chars[start..].iter().collect::<String>()
            } else {
                "<end>".into()
            }
        ));
    }
    let s: String = chars[start..*pos].iter().collect();
    s.parse::<f64>().map_err(|e| format!("invalid number '{s}': {e}"))
}

// ── Agent loop ─────────────────────────────────────────────────────────────────

/// Execute the full agent loop: call the LLM, handle tool calls, repeat until done.
async fn run_agent(
    provider: &dyn Provider,
    mut req: ChatRequest,
    max_turns: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    for turn in 1..=max_turns {
        // Call the LLM
        let resp = provider.chat(&req).await?;

        // Show reasoning if present
        if let Some(r) = resp.reasoning() {
            println!("\n── [turn {turn}] reasoning ──\n{r}\n");
        }

        // Collect tool invocations before moving resp
        let tool_invocations: Vec<_> = resp.tool_invocations().cloned().collect();
        let resp_text = resp.text();
        let resp_usage = (
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            resp.usage.cached_input_tokens,
            resp.usage.reasoning_tokens,
        );

        if tool_invocations.is_empty() {
            // No tool calls → final answer
            println!("── [turn {turn}] answer ──\n{resp_text}\n");
            println!(
                "[usage: {} in / {} out / {} cached / {} reasoning]",
                resp_usage.0, resp_usage.1, resp_usage.2, resp_usage.3,
            );
            return Ok(());
        }

        // Execute each tool call and build next request
        let mut messages = req.messages().to_vec();
        messages.push(resp.into_message());
        for inv in &tool_invocations {
            println!(
                "── [turn {turn}] tool: {}({}) ──",
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

        let mut builder = ChatRequest::builder(req.model().to_owned(), req.params().max_tokens())
            .messages(messages)
            .tools(req.tools().to_vec())
            .tool_choice(req.tool_choice().clone());
        if let Some(t) = req.params().temperature() {
            builder = builder.temperature(t);
        }
        if let Some(p) = req.params().top_p() {
            builder = builder.top_p(p);
        }
        req = builder.build();
    }

    eprintln!("[agent] max turns ({max_turns}) reached, stopping");
    Ok(())
}

// ── CLI ─────────────────────────────────────────────────────────────────────────

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();

    // ── Pick the provider ──────────────────────────────────────────────────────

    let kind = std::env::var("PROVIDER").unwrap_or_else(|_| "openai".into());

    let (provider, model): (Box<dyn Provider>, String) = match kind.as_str() {
        "deepseek" => {
            let api_key = std::env::var("DEEPSEEK_API_KEY")
                .map_err(|_| "DEEPSEEK_API_KEY is not set")?;
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
            let messages_path = std::env::var("ANTHROPIC_MESSAGES_PATH")
                .unwrap_or_else(|_| "/v1/messages".into());
            let model = std::env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-6-20250514".into());
            let p = AnthropicProvider::builder(api_key)
                .base_url(base_url)
                .messages_path(messages_path)
                .build();
            (Box::new(p), model)
        }
        other => {
            return Err(format!("Unknown PROVIDER={other} (expected openai, deepseek, or anthropic)").into());
        }
    };

    // ── Get the question ───────────────────────────────────────────────────────

    let question: String = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let question = if question.is_empty() {
        "What is 234 * 567? Then add 1000 to that result.".to_string()
    } else {
        question
    };

    println!("╔══════════════════════════════════════════════╗");
    println!("║         llm_adapter — Mini Agent Demo        ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("provider: {kind}");
    println!("model:    {model}");
    println!("question: {question}");
    println!();

    // ── First request ──────────────────────────────────────────────────────────

    let req = ChatRequest::builder(model, 1024)
        .message(Message::System(
            "You are a helpful assistant with access to tools. \
             When you need to calculate something, use the calculate tool. \
             Use get_date when you need the current date. \
             Think step by step — you may need multiple tool calls."
                .into(),
        ))
        .message(Message::User(vec![RequestContent::Text(question)]))
        .tools(tools())
        .build();

    // ── Agent loop ──────────────────────────────────────────────────────────────

    if let Err(e) = run_agent(provider.as_ref(), req, 5).await {
        eprintln!("agent stopped: {e}");
    }

    println!("── agent finished ──");
    Ok(())
}
