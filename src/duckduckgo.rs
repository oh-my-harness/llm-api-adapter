use crate::{
    error::SearchError,
    provider::SearchProvider,
    types::{SearchRequest, SearchResponse, SearchResult},
};
use async_trait::async_trait;
use serde::Deserialize;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.duckduckgo.com";
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_RESULTS: usize = 5;

/// DuckDuckGo Instant Answer API search provider.
///
/// This provider uses DuckDuckGo's JSON endpoint and normalizes abstract,
/// direct result, and related-topic items into [`SearchResult`] values.
pub struct DuckDuckGoSearchProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    max_results: usize,
}

/// Builder for [`DuckDuckGoSearchProvider`].
pub struct DuckDuckGoSearchProviderBuilder {
    base_url: String,
    api_key: Option<String>,
    timeout: Duration,
    max_results: usize,
}

impl DuckDuckGoSearchProvider {
    /// Create a builder with default DuckDuckGo Instant Answer settings.
    pub fn builder() -> DuckDuckGoSearchProviderBuilder {
        DuckDuckGoSearchProviderBuilder {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_results: DEFAULT_MAX_RESULTS,
        }
    }

    async fn send(&self, req: &SearchRequest) -> Result<reqwest::Response, SearchError> {
        let query = req.query().trim();
        if query.is_empty() {
            return Err(SearchError::InvalidRequest(
                "search query must not be empty".into(),
            ));
        }

        let mut request = self
            .client
            .get(self.base_url.trim_end_matches('/'))
            .query(&[
                ("q", query),
                ("format", "json"),
                ("no_html", "1"),
                ("skip_disambig", "1"),
            ]);

        if let Some(api_key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().await?;
        let status = response.status();

        if !status.is_success() {
            let retry_after = extract_retry_after_header(response.headers());
            let error_text = response.text().await?;

            return match status.as_u16() {
                401 => Err(SearchError::Unauthorized),
                403 => Err(SearchError::Forbidden),
                429 => Err(SearchError::RateLimit { retry_after }),
                500 | 502 | 503 => Err(SearchError::ServerError(error_text)),
                _ => Err(SearchError::InvalidRequest(error_text)),
            };
        }

        Ok(response)
    }
}

impl DuckDuckGoSearchProviderBuilder {
    /// Set the base URL for the search API.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set an optional bearer API key for compatible gateways.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the HTTP request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the provider-level maximum number of returned results.
    pub fn max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }

    /// Build the provider.
    pub fn build(self) -> DuckDuckGoSearchProvider {
        DuckDuckGoSearchProvider {
            client: reqwest::Client::builder()
                .timeout(self.timeout)
                .build()
                .expect("failed to build HTTP client"),
            base_url: self.base_url,
            api_key: self.api_key,
            max_results: self.max_results,
        }
    }
}

#[async_trait]
impl SearchProvider for DuckDuckGoSearchProvider {
    async fn search(&self, req: &SearchRequest) -> Result<SearchResponse, SearchError> {
        let response = self.send(req).await?;
        let wire = response.json::<DuckDuckGoResponse>().await?;
        Ok(SearchResponse {
            results: normalize_response(wire, req.max_results().min(self.max_results)),
        })
    }
}

fn extract_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers.get(reqwest::header::RETRY_AFTER)?;
    let s = value.to_str().ok()?;
    s.trim().parse::<u64>().ok().map(Duration::from_secs)
}

#[derive(Debug, Deserialize)]
struct DuckDuckGoResponse {
    #[serde(default, rename = "Heading")]
    heading: String,
    #[serde(default, rename = "AbstractText")]
    abstract_text: String,
    #[serde(default, rename = "AbstractURL")]
    abstract_url: String,
    #[serde(default, rename = "Results")]
    results: Vec<DuckDuckGoTopic>,
    #[serde(default, rename = "RelatedTopics")]
    related_topics: Vec<DuckDuckGoTopic>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DuckDuckGoTopic {
    Result {
        #[serde(default, rename = "Text")]
        text: String,
        #[serde(default, rename = "FirstURL")]
        first_url: String,
    },
    Group {
        #[serde(default, rename = "Name")]
        name: String,
        #[serde(default, rename = "Topics")]
        topics: Vec<DuckDuckGoTopic>,
    },
    Other(serde_json::Value),
}

fn normalize_response(wire: DuckDuckGoResponse, limit: usize) -> Vec<SearchResult> {
    let mut out = Vec::new();

    if !wire.abstract_text.trim().is_empty() && !wire.abstract_url.trim().is_empty() {
        out.push(SearchResult {
            title: fallback_title(&wire.heading, &wire.abstract_text),
            url: wire.abstract_url,
            snippet: normalize_text(&wire.abstract_text),
            score: None,
            source: Some("duckduckgo".into()),
        });
    }

    collect_topics(wire.results, &mut out, limit);
    collect_topics(wire.related_topics, &mut out, limit);

    out.truncate(limit);
    out
}

fn collect_topics(topics: Vec<DuckDuckGoTopic>, out: &mut Vec<SearchResult>, limit: usize) {
    for topic in topics {
        if out.len() >= limit {
            return;
        }

        match topic {
            DuckDuckGoTopic::Result { text, first_url } => {
                let text = normalize_text(&text);
                let first_url = first_url.trim().to_string();
                if text.is_empty() || first_url.is_empty() {
                    continue;
                }

                out.push(SearchResult {
                    title: fallback_title("", &text),
                    url: first_url,
                    snippet: text,
                    score: None,
                    source: Some("duckduckgo".into()),
                });
            }
            DuckDuckGoTopic::Group { name, topics } => {
                let _ = name;
                collect_topics(topics, out, limit);
            }
            DuckDuckGoTopic::Other(value) => {
                let _ = value;
            }
        }
    }
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn fallback_title(title: &str, snippet: &str) -> String {
    let title = normalize_text(title);
    if !title.is_empty() {
        return title;
    }

    let snippet = normalize_text(snippet);
    snippet
        .split(" - ")
        .next()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&snippet)
        .chars()
        .take(120)
        .collect()
}
