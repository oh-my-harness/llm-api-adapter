pub mod anthropic;
pub mod deepseek;
mod error;
pub mod openai;
pub mod provider;
mod stream_diag;
pub mod stream_handle;
pub mod types;

pub use error::LlmError;
pub use provider::{EmbeddingProvider, Provider, ProviderCapabilities};
pub use stream_diag::StreamEndKind;
pub use stream_handle::StreamHandle;
pub use types::*;

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
        use types::{ImageSource, RequestContent, ResponseContent, ToolInvocation};
        let _text = RequestContent::Text("hello".into());
        let _img = RequestContent::Image(ImageSource::Url {
            url: "https://example.com/img.png".into(),
        });
        let _resp = ResponseContent::Text("world".into());
        let _tool = ResponseContent::ToolInvocation(ToolInvocation {
            id: "call_1".into(),
            name: "fn".into(),
            input: serde_json::json!({"key": "val"}),
        });
        let _reasoning = ResponseContent::Reasoning {
            text: "my reasoning".into(),
            signature: None,
        };
    }

    #[test]
    fn reasoning_content_carries_signature() {
        use types::ResponseContent;
        let _with_sig = ResponseContent::Reasoning {
            text: "thinking".into(),
            signature: Some("sig123".into()),
        };
        let _without_sig = ResponseContent::Reasoning {
            text: "thinking".into(),
            signature: None,
        };
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
        use types::{ChatResponse, ResponseContent, StopReason, Usage};
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
        use types::{ContentKind, StopReason, StreamEvent, Usage};
        let _start = StreamEvent::ContentStart {
            index: 0,
            kind: ContentKind::Text,
        };
        let _delta = StreamEvent::TextDelta {
            index: 0,
            text: "hi".into(),
        };
        let _stop = StreamEvent::ContentStop {
            index: 0,
            signature: None,
        };
        let _msg_stop = StreamEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        };
    }

    #[test]
    fn provider_is_object_safe() {
        use provider::Provider;
        // Verify we can box the trait
        fn _accepts(_p: Box<dyn Provider>) {}
    }

    #[test]
    fn openai_provider_builds() {
        let _p = openai::OpenAIProvider::builder("test-key").build();
    }

    #[test]
    fn anthropic_provider_builds() {
        let _p = anthropic::AnthropicProvider::builder("test-key").build();
    }

    #[test]
    fn oai_response_deserializes() {
        use openai::schema::{OaiContent, OaiResponse};
        let json = r#"{
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8
            }
        }"#;
        let resp: OaiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "chatcmpl-1");
        match &resp.choices[0].message.content {
            Some(OaiContent::Text(text)) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn convert_simple_request_to_oai() {
        use openai::convert::to_oai;
        use types::{ChatRequest, Message, RequestContent};
        let req = ChatRequest::builder("gpt-4o", 512)
            .message(Message::System("You are helpful.".into()))
            .message(Message::User(vec![RequestContent::Text("Hi".into())]))
            .temperature(0.5)
            .build();
        let oai = to_oai(
            &req,
            None,
            openai::ThinkingScheme::None,
            openai::MaxTokensScheme::MaxTokens,
            false,
            true,
        )
        .unwrap();
        assert_eq!(oai.model, "gpt-4o");
        assert_eq!(oai.max_tokens, Some(512));
        assert_eq!(oai.temperature, Some(0.5));
        assert_eq!(oai.messages.len(), 2);
        assert_eq!(oai.messages[0].role, "system");
        assert_eq!(oai.messages[1].role, "user");
    }

    #[test]
    fn convert_oai_response_to_canonical() {
        use openai::{
            convert::from_oai,
            schema::{OaiChoice, OaiMessage, OaiResponse, OaiUsage},
        };
        use types::StopReason;
        let resp = OaiResponse {
            id: "chatcmpl-1".into(),
            model: "gpt-4o".into(),
            choices: vec![OaiChoice {
                message: OaiMessage {
                    role: "assistant".into(),
                    content: Some(openai::schema::OaiContent::Text("Hello!".into())),
                    tool_calls: vec![],
                    tool_call_id: None,
                    reasoning_content: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(OaiUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                ..Default::default()
            }),
        };
        let canon = from_oai(resp, false).unwrap();
        assert_eq!(canon.id, "chatcmpl-1");
        assert_eq!(canon.text(), "Hello!");
        assert!(matches!(canon.stop_reason, StopReason::EndTurn));
        assert_eq!(canon.usage.input_tokens, 10);
        assert_eq!(canon.usage.output_tokens, 5);
    }

    #[test]
    fn oai_stream_parser_emits_events() {
        use openai::schema::{OaiChunk, OaiDelta, OaiStreamChoice};
        use openai::stream::process_chunk;
        use types::{ContentKind, StreamEvent};

        let mut state = openai::stream::OaiStreamState::new();

        let chunk = OaiChunk {
            id: "c1".into(),
            model: "gpt-4o".into(),
            choices: vec![OaiStreamChoice {
                delta: OaiDelta {
                    content: Some("Hello".into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let events = process_chunk(&mut state, chunk, false);
        // First content delta should emit ContentStart then TextDelta
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            Ok(StreamEvent::ContentStart {
                index: 0,
                kind: ContentKind::Text
            })
        ));
        assert!(matches!(
            &events[1],
            Ok(StreamEvent::TextDelta { index: 0, .. })
        ));
    }

    #[test]
    fn deepseek_factory_returns_openai_provider() {
        use openai::OpenAIProvider;
        let _p: OpenAIProvider = deepseek::client("test-key");
    }

    #[test]
    fn provider_capabilities_openai() {
        let p = openai::OpenAIProvider::builder("key").build();
        let caps = p.capabilities();
        assert!(caps.supports_images());
        assert!(caps.supports_reasoning());
        assert!(caps.supports_json_schema());
        assert!(caps.supports_embeddings());
    }

    #[test]
    fn provider_capabilities_anthropic() {
        let p = anthropic::AnthropicProvider::builder("key").build();
        let caps = p.capabilities();
        assert!(caps.supports_images());
        assert!(caps.supports_reasoning());
        assert!(!caps.supports_json_schema());
        assert!(!caps.supports_embeddings());
    }

    #[test]
    fn provider_capabilities_deepseek() {
        use openai::OpenAIProvider;
        let p: OpenAIProvider = deepseek::client("key");
        let caps = p.capabilities();
        assert!(
            !caps.supports_images(),
            "DeepSeek V3/Flash does not support images"
        );
        assert!(caps.supports_reasoning());
        assert!(
            !caps.supports_json_schema(),
            "DeepSeek does not support json_schema response_format"
        );
        assert!(!caps.supports_embeddings());
    }

    #[test]
    fn provider_capabilities_box_dyn_provider() {
        let p: Box<dyn provider::Provider> =
            Box::new(openai::OpenAIProvider::builder("key").build());
        let caps = p.capabilities();
        assert!(caps.supports_images());
    }
}
