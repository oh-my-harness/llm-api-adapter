use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// Each test crate (openai_chat / openai_stream / deepseek_*) uses a subset of
// these helpers. dead_code warnings would fire from the crates that don't
// touch all three — suppress per-fn.

/// Mount a mock that replies to POST /v1/chat/completions with the given JSON body.
#[allow(dead_code)]
pub async fn mock_chat(server: &MockServer, body: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/json"))
        .mount(server)
        .await;
}

/// Mount a mock that replies to POST /v1/chat/completions with an SSE stream.
#[allow(dead_code)]
pub async fn mock_stream(server: &MockServer, body: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "text/event-stream")
                .append_header("cache-control", "no-cache"),
        )
        .mount(server)
        .await;
}

/// Mount a mock that replies with an HTTP error status and JSON error body.
#[allow(dead_code)]
pub async fn mock_error(server: &MockServer, status: u16, body: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(status).set_body_raw(body, "application/json"))
        .mount(server)
        .await;
}

// ── .env loader (used by integration tests) ──────────────────────────────────

/// Parse a `.env` file at `path` and set any unset keys in the process env.
///
/// Format: each line is `[export ] KEY=VALUE`. Values may be quoted with `"`
/// or `'`. Lines starting with `#`, blank lines, and lines with empty values
/// are ignored. Existing env vars are NOT overwritten so callers can override
/// via the shell.
#[allow(dead_code)]
pub fn load_dotenv() {
    let Ok(content) = std::fs::read_to_string(".env") else {
        return;
    };
    for raw in content.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let key = k.trim();
        let val = v.trim().trim_matches('"').trim_matches('\'');
        if !val.is_empty() && std::env::var_os(key).is_none() {
            unsafe {
                std::env::set_var(key, val);
            }
        }
    }
}

/// Read an env var, returning `None` if unset or empty.
#[allow(dead_code)]
pub fn opt_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}
