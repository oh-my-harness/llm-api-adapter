//! Structured diagnostics for abnormally terminated SSE streams.
//!
//! The runtime persists only the `Display` text of `LlmError`; the original
//! `tracing::warn!` payload never reaches the durable team journal. These
//! helpers therefore produce a short, redacted detail string that is carried
//! on `LlmError::StreamIncomplete` itself, so an operator can tell apart:
//!
//! * `done_without_finish_reason` — the provider ended the SSE protocol
//!   cleanly (`[DONE]`) but never sent a `finish_reason`.
//! * `transport_error` — reading the response body failed (connection reset,
//!   truncated chunk, TLS/body decode). The cause chain narrows down whether
//!   the failure was a timeout, connect, body or decode error.
//! * `eof_without_done` — the byte stream ended without `[DONE]` and without
//!   a client-visible transport error.
//! * `decode_error` — the bytes were not valid UTF-8 or valid SSE frames.
//!
//! No request headers, prompts, response bodies or API keys are copied into
//! the detail: only error-chain text, HTTP status, content type and a
//! whitelisted request-id header are retained, with URLs redacted.

use std::fmt;

use eventsource_stream::EventStreamError;

/// Maximum length of a diagnostic detail string.
const MAX_DETAIL_LEN: usize = 240;

/// Why an SSE stream ended without a terminal `MessageStop`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamEndKind {
    /// `[DONE]` arrived without a prior `finish_reason`.
    DoneWithoutFinishReason,
    /// The response body stream failed while being read.
    TransportError,
    /// The byte stream ended without `[DONE]` and without an error.
    EofWithoutDone,
    /// Bytes could not be decoded as UTF-8 or parsed as SSE.
    DecodeError,
}

impl StreamEndKind {
    /// Stable snake_case label used in persisted error strings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DoneWithoutFinishReason => "done_without_finish_reason",
            Self::TransportError => "transport_error",
            Self::EofWithoutDone => "eof_without_done",
            Self::DecodeError => "decode_error",
        }
    }
}

impl fmt::Display for StreamEndKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Whitelisted, non-sensitive response metadata captured before the body is
/// consumed.
#[derive(Debug, Clone)]
pub(crate) struct ResponseMeta {
    status: u16,
    content_type: Option<String>,
    request_id: Option<String>,
}

impl ResponseMeta {
    pub(crate) fn capture(response: &reqwest::Response) -> Self {
        let headers = response.headers();
        let content_type = headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|v| sanitize_header_value(v, 64));
        // Common gateway request ids. These are correlation handles, not
        // credentials, and are safe to persist for provider support.
        let request_id = ["x-request-id", "request-id", "x-amzn-requestid", "cf-ray"]
            .iter()
            .find_map(|name| headers.get(*name).and_then(|v| v.to_str().ok()))
            .map(|v| sanitize_header_value(v, 64));
        Self {
            status: response.status().as_u16(),
            content_type,
            request_id,
        }
    }

    /// Compact `k=v` prefix shared by every incomplete-stream detail.
    pub(crate) fn prefix(&self) -> String {
        let mut out = format!("http={}", self.status);
        if let Some(ct) = &self.content_type {
            out.push_str(" ct=");
            out.push_str(ct);
        }
        if let Some(rid) = &self.request_id {
            out.push_str(" rid=");
            out.push_str(rid);
        }
        out
    }
}

/// Classify an `eventsource-stream` error without leaking payload bytes.
pub(crate) fn classify_event_stream_error(
    error: &EventStreamError<reqwest::Error>,
) -> (StreamEndKind, String) {
    match error {
        EventStreamError::Transport(err) => (
            StreamEndKind::TransportError,
            format!("cause=transport {}", classify_reqwest_error(err)),
        ),
        EventStreamError::Utf8(err) => (
            StreamEndKind::DecodeError,
            format!("cause=utf8 {}", sanitize_detail(&err.to_string())),
        ),
        EventStreamError::Parser(err) => (
            StreamEndKind::DecodeError,
            format!("cause=sse_parser {}", sanitize_detail(&err.to_string())),
        ),
    }
}

/// Summarize a `reqwest::Error`: category flags plus the source chain.
fn classify_reqwest_error(err: &reqwest::Error) -> String {
    let mut flags = Vec::new();
    if err.is_timeout() {
        flags.push("timeout");
    }
    // reqwest gates `is_connect` to non-wasm targets.
    #[cfg(not(target_arch = "wasm32"))]
    if err.is_connect() {
        flags.push("connect");
    }
    if err.is_body() {
        flags.push("body");
    }
    if err.is_decode() {
        flags.push("decode");
    }
    if err.is_request() {
        flags.push("request");
    }
    if err.is_redirect() {
        flags.push("redirect");
    }

    let mut chain = vec![err.to_string()];
    let mut source = std::error::Error::source(err);
    while let Some(cause) = source {
        chain.push(cause.to_string());
        source = cause.source();
    }

    let mut out = format!("flags=[{}]", flags.join(","));
    if let Some(status) = err.status() {
        out.push_str(&format!(" status={}", status.as_u16()));
    }
    out.push_str(" chain=");
    out.push_str(&sanitize_detail(&chain.join(": ")));
    out
}

/// Trim, redact URLs/bearer tokens and cap the length.
pub(crate) fn sanitize_detail(input: &str) -> String {
    let mut out = String::new();
    let mut redact_next = false;
    for token in input.split_whitespace() {
        let lower = token.to_ascii_lowercase();
        let rendered = if lower.starts_with("http://")
            || lower.starts_with("https://")
            || lower.starts_with("bearer")
            || lower.starts_with("sk-")
        {
            "<redacted>"
        } else if redact_next {
            redact_next = false;
            "<redacted>"
        } else {
            if lower == "authorization:" || lower == "authorization" {
                redact_next = true;
            }
            token
        };
        if !out.is_empty() {
            out.push(' ');
        }
        out.extend(rendered.chars().filter(|c| !c.is_control()));
        if out.chars().count() >= MAX_DETAIL_LEN {
            break;
        }
    }
    truncate_chars(&mut out, MAX_DETAIL_LEN)
}

fn sanitize_header_value(value: &str, max: usize) -> String {
    let mut out: String = value
        .chars()
        .filter(|c| {
            c.is_ascii_alphanumeric() || matches!(c, '/' | '+' | '-' | '.' | '_' | ';' | '=')
        })
        .take(max)
        .collect();
    if out.is_empty() {
        out.push_str("unknown");
    }
    out
}

fn truncate_chars(value: &mut String, max: usize) -> String {
    if value.chars().count() <= max {
        return std::mem::take(value);
    }
    let truncated: String = value.chars().take(max.saturating_sub(3)).collect();
    *value = format!("{truncated}...");
    std::mem::take(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_labels_are_stable() {
        assert_eq!(StreamEndKind::TransportError.to_string(), "transport_error");
        assert_eq!(
            StreamEndKind::DoneWithoutFinishReason.as_str(),
            "done_without_finish_reason"
        );
    }

    #[test]
    fn sanitize_redacts_urls_and_tokens() {
        let raw = "error at https://api.example.com/v1/chat?key=secret sk-abc123 authorization: Bearer xyz";
        let clean = sanitize_detail(raw);
        assert!(!clean.contains("api.example.com"));
        assert!(!clean.contains("secret"));
        assert!(!clean.contains("sk-abc123"));
        assert!(!clean.contains("xyz"));
    }

    #[test]
    fn sanitize_caps_length_on_char_boundary() {
        let raw = "错误".repeat(300);
        let clean = sanitize_detail(&raw);
        assert!(clean.chars().count() <= MAX_DETAIL_LEN);
        assert!(clean.ends_with("..."));
    }
}
