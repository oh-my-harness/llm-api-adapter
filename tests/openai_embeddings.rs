mod common;

use llm_adapter::{EmbeddingProvider, LlmError, openai::OpenAIProvider, types::*};
use wiremock::MockServer;

fn provider(server: &MockServer) -> OpenAIProvider {
    OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .build()
}

const EMBEDDINGS_RESPONSE: &str = r#"{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, 0.3]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.4, 0.5, 0.6]
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}"#;

#[tokio::test]
async fn embeddings_return_batched_vectors_and_usage() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/embeddings"))
        .and(wiremock::matchers::header(
            "authorization",
            "Bearer test-key",
        ))
        .and(wiremock::matchers::body_partial_json(serde_json::json!({
            "model": "text-embedding-3-small",
            "input": ["hello", "world"],
            "dimensions": 3,
            "encoding_format": "float"
        })))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_raw(EMBEDDINGS_RESPONSE, "application/json"),
        )
        .mount(&server)
        .await;

    let req = EmbeddingRequest::builder("text-embedding-3-small")
        .inputs(["hello", "world"])
        .dimensions(3)
        .encoding_format("float")
        .build();

    let resp = provider(&server).embed(&req).await.unwrap();

    assert_eq!(resp.model, "text-embedding-3-small");
    assert_eq!(resp.vectors.len(), 2);
    assert_eq!(resp.vectors[0].index, 0);
    assert_eq!(resp.vectors[0].embedding, vec![0.1, 0.2, 0.3]);
    assert_eq!(resp.vectors[1].index, 1);
    assert_eq!(resp.dimensions(), Some(3));
    let usage = resp.usage.unwrap();
    assert_eq!(usage.input_tokens, 8);
    assert_eq!(usage.total_tokens, 8);
}

#[tokio::test]
async fn custom_embeddings_path_for_compatible_gateways() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/embeddings"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_raw(EMBEDDINGS_RESPONSE, "application/json"),
        )
        .mount(&server)
        .await;

    let p = OpenAIProvider::builder("test-key")
        .base_url(server.uri())
        .embeddings_path("/embeddings")
        .build();
    let req = EmbeddingRequest::builder("text-embedding-3-small")
        .input("hello")
        .build();

    let resp = p.embed(&req).await.unwrap();

    assert_eq!(resp.vectors.len(), 2);
}

#[tokio::test]
async fn embedding_error_429_maps_to_rate_limit() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/embeddings"))
        .respond_with(
            wiremock::ResponseTemplate::new(429)
                .insert_header("retry-after", "30")
                .set_body_raw(
                    include_str!("fixtures/openai/error_rate_limit.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;

    let req = EmbeddingRequest::builder("text-embedding-3-small")
        .input("hello")
        .build();

    let err = provider(&server).embed(&req).await.unwrap_err();
    assert!(matches!(err, LlmError::RateLimit { retry_after: _ }));
}

#[tokio::test]
async fn http_error_classification_matches_chat() {
    // Embeddings share the chat status mapping: unknown-outcome 5xx must be
    // ServerError (never InvalidRequest), 400 stays a trusted rejection.
    for (status, expect_server_error) in [(504u16, true), (520, true), (599, true), (400, false)] {
        let server = MockServer::start().await;
        common::mock_embeddings_error(&server, status).await;

        let req = EmbeddingRequest::builder("text-embedding-3-small")
            .inputs(["hello", "world"])
            .build();
        let err = provider(&server).embed(&req).await.unwrap_err();
        if expect_server_error {
            assert!(
                matches!(err, LlmError::ServerError(_)),
                "HTTP {status}: {err:?}"
            );
        } else {
            assert!(
                matches!(err, LlmError::InvalidRequest(_)),
                "HTTP {status}: {err:?}"
            );
        }
    }
}
