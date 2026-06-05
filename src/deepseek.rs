use crate::openai::OpenAIProvider;

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";

/// Create an [`OpenAIProvider`] pre-configured for the DeepSeek API.
///
/// DeepSeek uses the OpenAI Chat Completions schema at a different base URL,
/// with `reasoning_content` parsing and tolerant keepalive enabled for
/// thinking-mode models.
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
        .build()
}
