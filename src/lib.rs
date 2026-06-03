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

    #[test]
    fn message_variants() {
        use types::{Message, RequestContent, ResponseContent};
        let _sys = Message::System("Be helpful".into());
        let _user = Message::User(vec![RequestContent::Text("Hi".into())]);
        let _asst = Message::Assistant(vec![ResponseContent::Text("Hello!".into())]);
        let _tool = Message::Tool {
            invocation_id: "call_1".into(),
            content: vec![RequestContent::Text("22°C".into())],
            is_error: false,
        };
    }

    #[test]
    fn chat_request_builder() {
        use types::{ChatRequest, Message, RequestContent, ToolChoice};
        let req = ChatRequest::builder("gpt-4o", 1024)
            .message(Message::System("You are helpful.".into()))
            .message(Message::User(vec![RequestContent::Text("Hello".into())]))
            .temperature(0.7)
            .build();

        assert_eq!(req.model(), "gpt-4o");
        assert_eq!(req.params().max_tokens, 1024);
        assert_eq!(req.params().temperature, Some(0.7));
        assert_eq!(req.messages().len(), 2);
        assert!(matches!(req.tool_choice(), ToolChoice::Auto));
    }

    #[test]
    fn chat_response_helpers() {
        use types::{ChatResponse, ResponseContent, StopReason, ToolInvocation, Usage};
        let resp = ChatResponse {
            id: "msg_1".into(),
            model: "gpt-4o".into(),
            content: vec![
                ResponseContent::Text("Hello ".into()),
                ResponseContent::Text("world".into()),
            ],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        };
        assert_eq!(resp.text(), "Hello world");
        assert!(resp.reasoning().is_none());
        assert_eq!(resp.tool_invocations().count(), 0);
    }

    #[test]
    fn stream_event_construction() {
        use types::{StreamEvent, ContentKind, StopReason, Usage};
        let _start = StreamEvent::ContentStart { index: 0, kind: ContentKind::Text };
        let _delta = StreamEvent::TextDelta { index: 0, text: "hi".into() };
        let _stop = StreamEvent::ContentStop { index: 0 };
        let _msg_stop = StreamEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        };
    }
}
