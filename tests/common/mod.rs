use wiremock::{Mock, MockServer, ResponseTemplate};
use wiremock::matchers::{method, path};

/// Mount a mock that replies to POST /v1/chat/completions with the given JSON body.
pub async fn mock_chat(server: &MockServer, body: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "application/json"),
        )
        .mount(server)
        .await;
}

/// Mount a mock that replies to POST /v1/chat/completions with an SSE stream.
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
pub async fn mock_error(server: &MockServer, status: u16, body: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(status)
                .set_body_raw(body, "application/json"),
        )
        .mount(server)
        .await;
}
