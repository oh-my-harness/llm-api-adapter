use llm_adapter::{
    SearchError, SearchProvider, duckduckgo::DuckDuckGoSearchProvider, types::SearchRequest,
};
use wiremock::MockServer;

fn provider(server: &MockServer) -> DuckDuckGoSearchProvider {
    DuckDuckGoSearchProvider::builder()
        .base_url(server.uri())
        .build()
}

const SEARCH_RESPONSE: &str = r#"{
  "Heading": "Rust",
  "AbstractText": "Rust is a multi-paradigm programming language.",
  "AbstractURL": "https://duckduckgo.com/Rust_(programming_language)",
  "Results": [
    {
      "Text": "Rust official site - A language empowering everyone.",
      "FirstURL": "https://www.rust-lang.org/"
    }
  ],
  "RelatedTopics": [
    {
      "Text": "Cargo - Rust package manager.",
      "FirstURL": "https://doc.rust-lang.org/cargo/"
    },
    {
      "Name": "Grouped",
      "Topics": [
        {
          "Text": "Rustdoc - Documentation generator.",
          "FirstURL": "https://doc.rust-lang.org/rustdoc/"
        }
      ]
    }
  ]
}"#;

#[tokio::test]
async fn duckduckgo_search_returns_normalized_results() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::query_param("q", "rust language"))
        .and(wiremock::matchers::query_param("format", "json"))
        .and(wiremock::matchers::query_param("no_html", "1"))
        .and(wiremock::matchers::query_param("skip_disambig", "1"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_raw(SEARCH_RESPONSE, "application/json"),
        )
        .mount(&server)
        .await;

    let req = SearchRequest::builder("rust language")
        .max_results(3)
        .build();

    let resp = provider(&server).search(&req).await.unwrap();

    assert_eq!(resp.results.len(), 3);
    assert_eq!(resp.results[0].title, "Rust");
    assert_eq!(
        resp.results[0].url,
        "https://duckduckgo.com/Rust_(programming_language)"
    );
    assert_eq!(
        resp.results[0].snippet,
        "Rust is a multi-paradigm programming language."
    );
    assert_eq!(resp.results[0].source.as_deref(), Some("duckduckgo"));
    assert_eq!(resp.results[1].title, "Rust official site");
    assert_eq!(resp.results[2].title, "Cargo");
}

#[tokio::test]
async fn provider_max_results_caps_request_limit_and_sends_api_key() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::header(
            "authorization",
            "Bearer gateway-key",
        ))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_raw(SEARCH_RESPONSE, "application/json"),
        )
        .mount(&server)
        .await;

    let p = DuckDuckGoSearchProvider::builder()
        .base_url(server.uri())
        .api_key("gateway-key")
        .max_results(2)
        .build();
    let req = SearchRequest::builder("rust").max_results(10).build();

    let resp = p.search(&req).await.unwrap();

    assert_eq!(resp.results.len(), 2);
}

#[tokio::test]
async fn empty_query_is_rejected_before_http() {
    let server = MockServer::start().await;
    let req = SearchRequest::builder("   ").build();

    let err = provider(&server).search(&req).await.unwrap_err();

    assert!(matches!(err, SearchError::InvalidRequest(_)));
}

#[tokio::test]
async fn search_error_429_maps_to_rate_limit() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .respond_with(
            wiremock::ResponseTemplate::new(429)
                .insert_header("retry-after", "15")
                .set_body_string("rate limited"),
        )
        .mount(&server)
        .await;

    let req = SearchRequest::builder("rust").build();

    let err = provider(&server).search(&req).await.unwrap_err();

    assert!(matches!(err, SearchError::RateLimit { retry_after: _ }));
}
