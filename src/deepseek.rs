use crate::{openai::OpenAIProvider, provider::ProviderCapabilities};

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";

/// Create an [`OpenAIProvider`] pre-configured for the DeepSeek API.
///
/// DeepSeek uses the OpenAI Chat Completions schema at a different base URL,
/// with `reasoning_content` parsing and tolerant keepalive enabled for
/// thinking-mode models.
///
/// Capabilities: reasoning ✓, json_schema ✓, images ✗
/// (DeepSeek V3/V4 do not accept image input; see integration test
/// `integration_image_input_returns_error` for verification.)
///
/// # Example
///
/// ```no_run
/// # use llm_adapter::deepseek;
/// let provider = deepseek::client("sk-your-api-key");
/// ```
pub fn client(api_key: impl Into<String>) -> OpenAIProvider {
    OpenAIProvider::builder(api_key)
        .base_url(DEEPSEEK_BASE_URL)
        .parse_reasoning_content(true)
        .tolerant_keepalive(true)
        .capabilities(ProviderCapabilities {
            images: false,
            reasoning: true,
            json_schema: true,
            embeddings: false,
        })
        .build()
}
