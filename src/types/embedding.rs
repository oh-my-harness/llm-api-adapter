/// Provider-neutral embedding request.
///
/// `input` supports batching; providers should preserve the order of returned
/// vectors or include the original item index in each [`EmbeddingVector`].
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingRequest {
    pub(crate) model: String,
    pub(crate) input: Vec<String>,
    pub(crate) dimensions: Option<usize>,
    pub(crate) encoding_format: Option<String>,
}

impl EmbeddingRequest {
    /// Create a builder with the given model.
    pub fn builder(model: impl Into<String>) -> EmbeddingRequestBuilder {
        EmbeddingRequestBuilder {
            model: model.into(),
            input: Vec::new(),
            dimensions: None,
            encoding_format: None,
        }
    }

    /// The model identifier (e.g. `"text-embedding-3-small"`).
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Batched text input.
    pub fn input(&self) -> &[String] {
        &self.input
    }

    /// Requested vector dimensionality, if supported by the provider/model.
    pub fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }

    /// Provider encoding format hint, if any.
    pub fn encoding_format(&self) -> Option<&str> {
        self.encoding_format.as_deref()
    }
}

/// Builder for [`EmbeddingRequest`].
pub struct EmbeddingRequestBuilder {
    model: String,
    input: Vec<String>,
    dimensions: Option<usize>,
    encoding_format: Option<String>,
}

impl EmbeddingRequestBuilder {
    /// Append one input string.
    pub fn input(mut self, input: impl Into<String>) -> Self {
        self.input.push(input.into());
        self
    }

    /// Replace the full input batch.
    pub fn inputs<I, S>(mut self, inputs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.input = inputs.into_iter().map(Into::into).collect();
        self
    }

    /// Request a specific output dimensionality.
    pub fn dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set a provider encoding format hint, such as `"float"` or `"base64"`.
    pub fn encoding_format(mut self, encoding_format: impl Into<String>) -> Self {
        self.encoding_format = Some(encoding_format.into());
        self
    }

    /// Consume the builder and produce an [`EmbeddingRequest`].
    pub fn build(self) -> EmbeddingRequest {
        EmbeddingRequest {
            model: self.model,
            input: self.input,
            dimensions: self.dimensions,
            encoding_format: self.encoding_format,
        }
    }
}

/// One vector returned for a batched embedding request.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingVector {
    /// Position of the corresponding input item in the request batch.
    pub index: usize,
    /// Dense floating-point embedding.
    pub embedding: Vec<f32>,
}

/// Token usage for an embedding request.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmbeddingUsage {
    pub input_tokens: u32,
    pub total_tokens: u32,
}

/// Provider-neutral embedding response.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingResponse {
    pub model: String,
    pub vectors: Vec<EmbeddingVector>,
    pub usage: Option<EmbeddingUsage>,
}

impl EmbeddingResponse {
    /// Return the dimensionality of the first vector, if any.
    pub fn dimensions(&self) -> Option<usize> {
        self.vectors.first().map(|v| v.embedding.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_request_builder_sets_batch_options() {
        let req = EmbeddingRequest::builder("text-embedding-3-small")
            .inputs(["hello", "world"])
            .dimensions(256)
            .encoding_format("float")
            .build();

        assert_eq!(req.model(), "text-embedding-3-small");
        assert_eq!(req.input(), &["hello".to_string(), "world".to_string()]);
        assert_eq!(req.dimensions(), Some(256));
        assert_eq!(req.encoding_format(), Some("float"));
    }
}
