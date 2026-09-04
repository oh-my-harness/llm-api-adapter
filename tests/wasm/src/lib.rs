//! Real-LLM smoke test for `llm_adapter` compiled to wasm32, run under Node.
//!
//! Config is injected from JS (`std::env::var` does not work in wasm):
//!   `run_smoke(base_url, api_key, model)`
//!
//! Exercises the full universal path over reqwest's wasm fetch backend:
//! `Provider::chat`, `Provider::chat_stream`, `StreamHandle::collect`.

use llm_adapter::provider::Provider as _;
use llm_adapter::types::{ChatRequest, Message, RequestContent};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub async fn run_smoke(
    base_url: String,
    api_key: String,
    model: String,
) -> Result<String, JsValue> {
    let provider = llm_adapter::openai::OpenAIProvider::builder(api_key)
        .base_url(base_url)
        .build();

    let req = ChatRequest::builder(&model, 512)
        .message(Message::User(vec![RequestContent::Text(
            "Reply with exactly: WASM_OK".into(),
        )]))
        .build();

    // Non-streaming universal path.
    let resp = provider
        .chat(&req)
        .await
        .map_err(|e| JsValue::from_str(&format!("chat failed: {e}")))?;
    let mut report = format!("non-stream: {}\n", resp.text().trim());

    // Streaming universal path: chat_stream + StreamHandle::collect.
    let handle = provider
        .chat_stream(&req)
        .await
        .map_err(|e| JsValue::from_str(&format!("chat_stream failed: {e}")))?;
    let collected = handle
        .collect()
        .await
        .map_err(|e| JsValue::from_str(&format!("collect failed: {e}")))?;
    report.push_str(&format!("stream: {}\n", collected.text().trim()));

    web_sys::console::log_1(&JsValue::from_str(&report));
    Ok(report)
}
