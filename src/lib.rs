mod error;
pub use error::LlmError;

mod types;

pub fn _check() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_error_display() {
        let e = LlmError::Unauthorized;
        assert_eq!(e.to_string(), "unauthorized");
        let e2 = LlmError::InvalidRequest("bad param".into());
        assert!(e2.to_string().contains("bad param"));
    }

    #[test]
    fn content_construction() {
        use types::{RequestContent, ResponseContent, ImageSource, ToolInvocation};
        let _text = RequestContent::Text("hello".into());
        let _img = RequestContent::Image(ImageSource::Url("https://example.com/img.png".into()));
        let _resp = ResponseContent::Text("world".into());
        let _tool = ResponseContent::ToolInvocation(ToolInvocation {
            id: "call_1".into(),
            name: "fn".into(),
            input: serde_json::json!({"key": "val"}),
        });
        let _reasoning = ResponseContent::Reasoning("my reasoning".into());
    }
}
