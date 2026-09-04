use crate::{openai::OpenAIProvider, provider::ProviderCapabilities};

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";

/// DeepSeek uses the OpenAI Chat Completions schema at a different base URL,
/// with `reasoning_content` parsing and tolerant keepalive enabled for
/// thinking-mode models.
///
/// Capabilities: reasoning ✓, json_schema ✗ (DeepSeek does not support
/// `response_format: json_schema` — use `json_object` or strict tool use),
/// images ✗ (DeepSeek V3/V4 have no vision: as of 2026-09 the API accepts
/// image_url content but silently ignores it — see integration test
/// `integration_image_input_is_accepted_but_ignored`), documents ✗
/// (no document input).
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
        .thinking_scheme(crate::openai::ThinkingScheme::ThinkingToggle)
        .capabilities(ProviderCapabilities {
            images: false,
            reasoning: true,
            json_schema: false,
            embeddings: false,
            documents: false,
            documents_inline: false,
        })
        .build()
}
