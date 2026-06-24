/// Provider-neutral web search request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchRequest {
    pub(crate) query: String,
    pub(crate) max_results: usize,
}

impl SearchRequest {
    /// Create a builder with the query string.
    pub fn builder(query: impl Into<String>) -> SearchRequestBuilder {
        SearchRequestBuilder {
            query: query.into(),
            max_results: 5,
        }
    }

    /// Query text.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Maximum number of normalized results to return.
    pub fn max_results(&self) -> usize {
        self.max_results
    }
}

/// Builder for [`SearchRequest`].
pub struct SearchRequestBuilder {
    query: String,
    max_results: usize,
}

impl SearchRequestBuilder {
    /// Set the maximum number of results to return.
    pub fn max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }

    /// Consume the builder and produce a [`SearchRequest`].
    pub fn build(self) -> SearchRequest {
        SearchRequest {
            query: self.query,
            max_results: self.max_results,
        }
    }
}

/// Provider-neutral web search response.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

/// One normalized web search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub score: Option<f32>,
    pub source: Option<String>,
}

impl SearchResult {
    /// Create a new search result.
    pub fn new(
        title: impl Into<String>,
        url: impl Into<String>,
        snippet: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            url: url.into(),
            snippet: snippet.into(),
            score: None,
            source: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_request_builder_sets_query_and_limit() {
        let req = SearchRequest::builder("rust async trait")
            .max_results(3)
            .build();

        assert_eq!(req.query(), "rust async trait");
        assert_eq!(req.max_results(), 3);
    }
}
