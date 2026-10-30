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
//! the detail: only error-chain text, HTTP status, content type, a whitelisted
//! request-id header and numeric timing fields are retained, with URLs
//! redacted.
//!
//! Timing is measured at the byte level, where the body is read:
//! `elapsed_ms` counts from the start of the HTTP request (including waiting
//! for the response headers), `last_read_ms` is the offset of the last
//! successfully read body chunk and `idle_body_ms` is `elapsed_ms -
//! last_read_ms`. `idle_body_ms` is only rendered when at least one body byte
//! was actually read; a zero-byte body reports `last_read_ms=none
//! idle_body_ms=unavailable` instead of mis-attributing the header wait to
//! body idleness. All three fields degrade to `unavailable`/`none` once the
//! millisecond clock saturates, rather than printing a truncated (and thus
//! misleadingly growing) idle value. `read_timeout_ms` is the effective client
//! read timeout, so an operator can tell a timeout from an upstream stall
//! without consulting configuration; it is omitted entirely on platforms where
//! the client does not implement a read timeout (wasm).

use std::fmt;
use std::time::Duration;

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
    read_timeout_ms: Option<u64>,
}

impl ResponseMeta {
    pub(crate) fn capture(response: &reqwest::Response) -> Self {
        Self::capture_with_read_timeout(response, None)
    }

    /// Capture response metadata plus the effective client read timeout.
    ///
    /// The read timeout bounds *idle time between body reads*, not total
    /// generation time, so persisting it next to the measured idle gap lets an
    /// operator decide whether the failure matched the configured limit.
    ///
    /// `None` means the timeout is not implemented by the HTTP client on this
    /// platform (wasm); the field is then omitted from [`Self::prefix`] rather
    /// than reported as an effective value that never applies.
    pub(crate) fn capture_with_read_timeout(
        response: &reqwest::Response,
        read_timeout: Option<Duration>,
    ) -> Self {
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
            read_timeout_ms: read_timeout.map(|d| d.as_millis() as u64),
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
        if let Some(timeout_ms) = self.read_timeout_ms {
            out.push_str(" read_timeout_ms=");
            out.push_str(&timeout_ms.to_string());
        }
        out
    }
}

/// Render byte-level body-read timing as a compact `k=v` fragment.
///
/// * `elapsed_ms` — `None` means the millisecond clock saturated; the whole
///   timing triple is then reported as unavailable.
/// * `last_read_ms` — `None` means no body byte was ever read.
/// * `idle_body_ms` — only meaningful after a body read; when no byte was read
///   it is `None` and rendered as `unavailable`, never as the elapsed time.
pub(crate) fn timing_detail(
    elapsed_ms: Option<u64>,
    last_read_ms: Option<u64>,
    idle_body_ms: Option<u64>,
) -> String {
    let (elapsed, last, idle) = match (elapsed_ms, last_read_ms, idle_body_ms) {
        (Some(elapsed), Some(last), Some(idle)) => {
            (elapsed.to_string(), last.to_string(), idle.to_string())
        }
        (Some(elapsed), _, _) => (
            elapsed.to_string(),
            last_read_ms
                .map(|last| last.to_string())
                .unwrap_or_else(|| "none".into()),
            "unavailable".into(),
        ),
        // Saturated clock: nothing about the timing is trustworthy.
        (None, _, _) => ("unavailable".into(), "none".into(), "unavailable".into()),
    };
    format!("elapsed_ms={elapsed} last_read_ms={last} idle_body_ms={idle}")
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
        EventStreamError::Utf8(_) => (
            StreamEndKind::DecodeError,
            // The UTF-8 error's Display is a position/length summary, but the
            // body is never included. Keep only the structured category.
            "cause=utf8".into(),
        ),
        EventStreamError::Parser(_) => (
            StreamEndKind::DecodeError,
            // `nom::Error<String>` Display can embed the unparsed SSE input
            // (response bytes). Never render it: category only.
            "cause=sse_parser".into(),
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
    // reqwest wraps a body read timeout as a decode-class error
    // ("error decoding response body: ... operation timed out"). Labelling
    // that as `decode` makes every timeout look like a malformed payload, so
    // the redundant flag is dropped once the cause is already a timeout.
    // Genuine decode failures (timeout == false) keep the flag.
    if err.is_decode() && !err.is_timeout() {
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

/// Trim, redact URLs/credentials and cap the length.
///
/// Redaction is intentionally aggressive and format-tolerant:
/// * `http(s)://` is matched anywhere (including inside `(...)` or quotes) and
///   everything up to a structural delimiter is dropped.
/// * `Authorization` / `Bearer` / `sk-…` values are dropped, including when
///   wrapped in punctuation.
///
/// Only error-chain text reaches this function; prompts and response bodies
/// are never passed here (SSE parser input is excluded before sanitization).
pub(crate) fn sanitize_detail(input: &str) -> String {
    let redacted_urls = redact_urls(input);
    let mut out = String::new();
    let mut redact_next = false;
    for token in redacted_urls.split_whitespace() {
        let (lead, core, trail) = split_token_edges(token);
        let lower = core.to_ascii_lowercase();
        let rendered_core = if redact_next {
            if lower == "bearer" || lower.starts_with("bearer=") || lower.starts_with("bearer:") {
                // Keep the flag armed: `authorization: Bearer <secret>` must
                // redact the token after the scheme word too.
                "<redacted>"
            } else {
                redact_next = false;
                "<redacted>"
            }
        } else if lower == "bearer"
            || lower.starts_with("bearer=")
            || lower.starts_with("bearer:")
            || lower.starts_with("sk-")
        {
            "<redacted>"
        } else {
            if lower == "authorization:" || lower == "authorization" {
                redact_next = true;
            }
            core
        };
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(lead);
        out.extend(rendered_core.chars().filter(|c| !c.is_control()));
        out.push_str(trail);
        if out.chars().count() >= MAX_DETAIL_LEN {
            break;
        }
    }
    truncate_chars(&mut out, MAX_DETAIL_LEN)
}

/// Replace every `http(s)://…` run up to a structural delimiter.
fn redact_urls(input: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < input.len() {
        let rest = &input[i..];
        let bytes = rest.as_bytes();
        let scheme_len = if bytes.len() >= 8 && bytes[..8].eq_ignore_ascii_case(b"https://") {
            8
        } else if bytes.len() >= 7 && bytes[..7].eq_ignore_ascii_case(b"http://") {
            7
        } else {
            0
        };
        if scheme_len > 0 {
            out.push_str("<redacted-url>");
            i += scheme_len;
            while i < input.len() {
                let c = input[i..].chars().next().expect("char boundary");
                if c.is_whitespace() || matches!(c, ')' | ']' | '}' | '"' | '\'' | ',' | ';') {
                    break;
                }
                i += c.len_utf8();
            }
        } else {
            let c = rest.chars().next().expect("char boundary");
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

/// Split leading/trailing punctuation so `(sk-…),` is recognized as `sk-…`.
fn split_token_edges(token: &str) -> (&str, &str, &str) {
    let chars: Vec<(usize, char)> = token.char_indices().collect();
    let is_lead = |c: char| matches!(c, '(' | '[' | '{' | '"' | '\'' | '<');
    let is_trail = |c: char| matches!(c, ')' | ']' | '}' | '"' | '\'' | ',' | ';' | '>' | '.');
    let start_pos = chars
        .iter()
        .position(|(_, c)| !is_lead(*c))
        .unwrap_or(chars.len());
    let end_pos = chars
        .iter()
        .rposition(|(_, c)| !is_trail(*c))
        .map(|pos| pos + 1)
        .unwrap_or(start_pos);
    let start = chars.get(start_pos).map(|(i, _)| *i).unwrap_or(token.len());
    let end = chars.get(end_pos).map(|(i, _)| *i).unwrap_or(token.len());
    (&token[..start], &token[start..end], &token[end..])
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
    fn sanitize_redacts_urls_wrapped_in_punctuation() {
        let raw = "error sending request for url (https://api.example.com/v1/chat?api_key=SECRET_VALUE&x=1), retry";
        let clean = sanitize_detail(raw);
        assert!(!clean.contains("SECRET_VALUE"), "{clean}");
        assert!(!clean.contains("api.example.com"), "{clean}");
        assert!(clean.contains("<redacted-url>"), "{clean}");
        assert!(
            clean.contains("),"),
            "structural punctuation is kept: {clean}"
        );
    }

    #[test]
    fn sanitize_redacts_punctuated_credentials() {
        let raw = "authorization: (Bearer TOKEN_123), sk-SECRET_KEY; Bearer=OTHER";
        let clean = sanitize_detail(raw);
        for leaked in ["TOKEN_123", "SECRET_KEY", "OTHER"] {
            assert!(!clean.contains(leaked), "{leaked} leaked: {clean}");
        }
    }

    #[test]
    fn parser_and_utf8_details_never_include_payload() {
        use eventsource_stream::EventStreamError;
        use nom::error::{Error as NomError, ErrorKind};

        let parser: EventStreamError<reqwest::Error> = EventStreamError::Parser(NomError::new(
            "SECRET_RESPONSE_BODY".to_string(),
            ErrorKind::Tag,
        ));
        let (kind, detail) = classify_event_stream_error(&parser);
        assert_eq!(kind, StreamEndKind::DecodeError);
        assert_eq!(detail, "cause=sse_parser");
        assert!(!detail.contains("SECRET_RESPONSE_BODY"));

        let utf8 = String::from_utf8(vec![0xff]).expect_err("invalid utf8");
        let utf8: EventStreamError<reqwest::Error> = EventStreamError::Utf8(utf8);
        let (kind, detail) = classify_event_stream_error(&utf8);
        assert_eq!(kind, StreamEndKind::DecodeError);
        assert_eq!(detail, "cause=utf8");
    }

    #[test]
    fn sanitize_caps_length_on_char_boundary() {
        let raw = "错误".repeat(300);
        let clean = sanitize_detail(&raw);
        assert!(clean.chars().count() <= MAX_DETAIL_LEN);
        assert!(clean.ends_with("..."));
    }

    #[test]
    fn timing_detail_never_invents_body_idle_without_a_read() {
        assert_eq!(
            timing_detail(Some(150), None, None),
            "elapsed_ms=150 last_read_ms=none idle_body_ms=unavailable"
        );
        assert_eq!(
            timing_detail(Some(150), Some(120), Some(30)),
            "elapsed_ms=150 last_read_ms=120 idle_body_ms=30"
        );
    }

    #[test]
    fn timing_detail_marks_a_saturated_clock_as_unavailable() {
        let detail = timing_detail(None, None, None);
        assert_eq!(
            detail,
            "elapsed_ms=unavailable last_read_ms=none idle_body_ms=unavailable"
        );
        assert!(!detail.contains("idle_body_ms=0"), "{detail}");
    }

    #[test]
    fn prefix_omits_read_timeout_when_the_client_does_not_implement_one() {
        let meta = ResponseMeta {
            status: 200,
            content_type: Some("text/event-stream".into()),
            request_id: None,
            read_timeout_ms: None,
        };
        let prefix = meta.prefix();
        assert_eq!(prefix, "http=200 ct=text/event-stream");
        assert!(!prefix.contains("read_timeout_ms"), "{prefix}");

        let meta = ResponseMeta {
            read_timeout_ms: Some(60_000),
            ..meta
        };
        assert!(
            meta.prefix().contains("read_timeout_ms=60000"),
            "{}",
            meta.prefix()
        );
    }
}
