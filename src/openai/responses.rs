//! Direct OpenAI Responses transport with immutable count/execution binding.
//!
//! This is a provider-native API. It deliberately does not extend the
//! provider-neutral [`crate::provider::Provider`] trait: input-token admission,
//! request receipts, and Responses terminal events have semantics that the
//! Chat Completions abstraction cannot express.

use super::{OpenAIProvider, extract_retry_after_header};
use crate::{LlmError, Usage};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::fmt;

/// Wire contract for fields shared by input counting and execution.
pub const TOKEN_RELEVANT_REQUEST_PROJECTION_VERSION: &str =
    "openai-responses-token-relevant-request-projection/v2";
/// Wire contract for controls that do not affect input tokenization.
pub const RESPONSE_ONLY_CONTROLS_VERSION: &str = "openai-responses-controls/v2";
/// Sealed prepared-call contract joining projection, controls, and transport.
pub const PREPARED_RESPONSE_VERSION: &str = "openai-prepared-response/v2";
/// Receipt contract for count, execution, and failed-attempt evidence.
pub const PROVIDER_RECEIPT_VERSION: &str = "openai-responses-provider-receipt/v2";
/// Strict one-attempt transport policy contract.
pub const TRANSPORT_POLICY_VERSION: &str = "openai-responses-transport-policy/v1";
/// Canonical JSON algorithm used before SHA-256 for Responses digests.
pub const DIGEST_SERIALIZATION_PROFILE: &str = "rfc8785-jcs/v1";

#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct OpenAIResponsesBuildError {
    message: String,
}

impl OpenAIResponsesBuildError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesReasoningContext {
    Auto,
    CurrentTurn,
    AllTurns,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesReasoningMode {
    Standard,
    Pro,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAIResponsesReasoning {
    effort: OpenAIResponsesReasoningEffort,
    context: OpenAIResponsesReasoningContext,
    mode: OpenAIResponsesReasoningMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<OpenAIResponsesReasoningSummary>,
}

impl OpenAIResponsesReasoning {
    pub fn new(
        effort: OpenAIResponsesReasoningEffort,
        context: OpenAIResponsesReasoningContext,
        mode: OpenAIResponsesReasoningMode,
    ) -> Self {
        Self {
            effort,
            context,
            mode,
            summary: None,
        }
    }

    pub fn summary(mut self, summary: OpenAIResponsesReasoningSummary) -> Self {
        self.summary = Some(summary);
        self
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesTextVerbosity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIResponsesTextFormat {
    Text,
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAIResponsesText {
    format: OpenAIResponsesTextFormat,
    verbosity: OpenAIResponsesTextVerbosity,
}

impl OpenAIResponsesText {
    pub fn plain(verbosity: OpenAIResponsesTextVerbosity) -> Self {
        Self {
            format: OpenAIResponsesTextFormat::Text,
            verbosity,
        }
    }

    pub fn json_schema(
        name: impl Into<String>,
        schema: Value,
        strict: bool,
        verbosity: OpenAIResponsesTextVerbosity,
    ) -> Self {
        Self {
            format: OpenAIResponsesTextFormat::JsonSchema {
                name: name.into(),
                schema,
                strict,
                description: None,
            },
            verbosity,
        }
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        if let OpenAIResponsesTextFormat::JsonSchema {
            description: current,
            ..
        } = &mut self.format
        {
            *current = Some(description.into());
        }
        self
    }

    fn validate(&self) -> Result<(), OpenAIResponsesBuildError> {
        if let OpenAIResponsesTextFormat::JsonSchema {
            name,
            schema,
            description,
            ..
        } = &self.format
        {
            if name.is_empty()
                || name.len() > 64
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
            {
                return Err(OpenAIResponsesBuildError::new(
                    "Responses JSON schema name must be 1-64 ASCII letters, digits, '_' or '-'",
                ));
            }
            if !schema.is_object() {
                return Err(OpenAIResponsesBuildError::new(
                    "Responses JSON schema must be an object",
                ));
            }
            if description.as_ref().is_some_and(|value| value.is_empty()) {
                return Err(OpenAIResponsesBuildError::new(
                    "Responses JSON schema description cannot be empty",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesToolChoice {
    None,
    Auto,
    Required,
}

impl OpenAIResponsesToolChoice {
    fn as_wire(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Auto => "auto",
            Self::Required => "required",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesCachePolicy {
    Implicit,
    Explicit,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum OpenAIResponsesConversationState {
    Fresh,
    PreviousResponse(String),
    Conversation(String),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesTruncation {
    Disabled,
    Auto,
}

impl OpenAIResponsesTruncation {
    fn as_wire(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAIResponsesTokenProjection {
    schema_version: String,
    model: String,
    instructions: String,
    input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenAIResponsesReasoning>,
    text: OpenAIResponsesText,
    tools: Vec<Value>,
    tool_choice: OpenAIResponsesToolChoice,
    parallel_tool_calls: bool,
    cache_policy: OpenAIResponsesCachePolicy,
    conversation_state: OpenAIResponsesConversationState,
    truncation: OpenAIResponsesTruncation,
}

impl OpenAIResponsesTokenProjection {
    pub fn builder(
        model: impl Into<String>,
        instructions: impl Into<String>,
        input: Value,
    ) -> OpenAIResponsesTokenProjectionBuilder {
        OpenAIResponsesTokenProjectionBuilder {
            model: model.into(),
            instructions: instructions.into(),
            input,
            reasoning: None,
            text: OpenAIResponsesText::plain(OpenAIResponsesTextVerbosity::Medium),
            tools: Vec::new(),
            tool_choice: OpenAIResponsesToolChoice::None,
            parallel_tool_calls: false,
            cache_policy: OpenAIResponsesCachePolicy::Implicit,
            conversation_state: OpenAIResponsesConversationState::Fresh,
            truncation: OpenAIResponsesTruncation::Disabled,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn digest(&self) -> Result<String, OpenAIResponsesBuildError> {
        self.validate()?;
        let body = self.wire_body()?;
        digest_json(&json!({
            "digest_serialization_profile": DIGEST_SERIALIZATION_PROFILE,
            "schema_version": TOKEN_RELEVANT_REQUEST_PROJECTION_VERSION,
            "wire_body": body,
        }))
    }

    fn validate(&self) -> Result<(), OpenAIResponsesBuildError> {
        if self.schema_version != TOKEN_RELEVANT_REQUEST_PROJECTION_VERSION {
            return Err(OpenAIResponsesBuildError::new(
                "unsupported Responses token projection version",
            ));
        }
        if self.model.trim().is_empty() || self.model.contains('\0') {
            return Err(OpenAIResponsesBuildError::new(
                "Responses model identity cannot be empty or contain NUL",
            ));
        }
        if self.instructions.trim().is_empty() || self.instructions.contains('\0') {
            return Err(OpenAIResponsesBuildError::new(
                "Responses instructions cannot be empty or contain NUL",
            ));
        }
        if !self.input.is_string() && !self.input.is_array() && !self.input.is_object() {
            return Err(OpenAIResponsesBuildError::new(
                "Responses input must be a string, array, or object",
            ));
        }
        self.text.validate()?;
        if self.tools.iter().any(|tool| !tool.is_object()) {
            return Err(OpenAIResponsesBuildError::new(
                "every Responses tool definition must be an object",
            ));
        }
        if self.tools.is_empty() && self.tool_choice != OpenAIResponsesToolChoice::None {
            return Err(OpenAIResponsesBuildError::new(
                "an empty Responses tool set requires tool_choice=none",
            ));
        }
        match &self.conversation_state {
            OpenAIResponsesConversationState::Fresh => {}
            OpenAIResponsesConversationState::PreviousResponse(id)
            | OpenAIResponsesConversationState::Conversation(id)
                if id.trim().is_empty() || id.contains('\0') =>
            {
                return Err(OpenAIResponsesBuildError::new(
                    "Responses conversation identity cannot be empty or contain NUL",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn wire_body(&self) -> Result<Value, OpenAIResponsesBuildError> {
        self.validate()?;
        let mut body = Map::new();
        body.insert("model".into(), Value::String(self.model.clone()));
        body.insert(
            "instructions".into(),
            Value::String(self.instructions.clone()),
        );
        body.insert("input".into(), self.input.clone());
        if let Some(reasoning) = &self.reasoning {
            body.insert("reasoning".into(), serialize_value(reasoning)?);
        }
        body.insert("text".into(), serialize_value(&self.text)?);
        body.insert("tools".into(), Value::Array(self.tools.clone()));
        body.insert(
            "tool_choice".into(),
            Value::String(self.tool_choice.as_wire().into()),
        );
        body.insert(
            "parallel_tool_calls".into(),
            Value::Bool(self.parallel_tool_calls),
        );
        if self.cache_policy == OpenAIResponsesCachePolicy::Explicit {
            body.insert("prompt_cache_options".into(), json!({"mode": "explicit"}));
        }
        match &self.conversation_state {
            OpenAIResponsesConversationState::Fresh => {}
            OpenAIResponsesConversationState::PreviousResponse(id) => {
                body.insert("previous_response_id".into(), Value::String(id.clone()));
            }
            OpenAIResponsesConversationState::Conversation(id) => {
                body.insert("conversation".into(), Value::String(id.clone()));
            }
        }
        body.insert(
            "truncation".into(),
            Value::String(self.truncation.as_wire().into()),
        );
        Ok(Value::Object(body))
    }
}

pub struct OpenAIResponsesTokenProjectionBuilder {
    model: String,
    instructions: String,
    input: Value,
    reasoning: Option<OpenAIResponsesReasoning>,
    text: OpenAIResponsesText,
    tools: Vec<Value>,
    tool_choice: OpenAIResponsesToolChoice,
    parallel_tool_calls: bool,
    cache_policy: OpenAIResponsesCachePolicy,
    conversation_state: OpenAIResponsesConversationState,
    truncation: OpenAIResponsesTruncation,
}

impl OpenAIResponsesTokenProjectionBuilder {
    pub fn reasoning(mut self, reasoning: OpenAIResponsesReasoning) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub fn text(mut self, text: OpenAIResponsesText) -> Self {
        self.text = text;
        self
    }

    pub fn tools(mut self, tools: Vec<Value>) -> Self {
        self.tools = tools;
        self
    }

    pub fn tool_choice(mut self, tool_choice: OpenAIResponsesToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = enabled;
        self
    }

    pub fn cache_policy(mut self, cache_policy: OpenAIResponsesCachePolicy) -> Self {
        self.cache_policy = cache_policy;
        self
    }

    pub fn conversation_state(
        mut self,
        conversation_state: OpenAIResponsesConversationState,
    ) -> Self {
        self.conversation_state = conversation_state;
        self
    }

    pub fn truncation(mut self, truncation: OpenAIResponsesTruncation) -> Self {
        self.truncation = truncation;
        self
    }

    pub fn build(self) -> Result<OpenAIResponsesTokenProjection, OpenAIResponsesBuildError> {
        let projection = OpenAIResponsesTokenProjection {
            schema_version: TOKEN_RELEVANT_REQUEST_PROJECTION_VERSION.into(),
            model: self.model,
            instructions: self.instructions,
            input: self.input,
            reasoning: self.reasoning,
            text: self.text,
            tools: self.tools,
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            cache_policy: self.cache_policy,
            conversation_state: self.conversation_state,
            truncation: self.truncation,
        };
        projection.validate()?;
        Ok(projection)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesServiceTier {
    Default,
    Auto,
    Flex,
    Priority,
}

impl OpenAIResponsesServiceTier {
    fn as_wire(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Auto => "auto",
            Self::Flex => "flex",
            Self::Priority => "priority",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAIResponsesResponseControls {
    schema_version: String,
    max_output_tokens: u32,
    stream: bool,
    store: bool,
    background: bool,
    service_tier: OpenAIResponsesServiceTier,
    include: Vec<String>,
}

impl OpenAIResponsesResponseControls {
    pub fn builder(max_output_tokens: u32) -> OpenAIResponsesResponseControlsBuilder {
        OpenAIResponsesResponseControlsBuilder {
            max_output_tokens,
            stream: false,
            store: false,
            background: false,
            service_tier: OpenAIResponsesServiceTier::Default,
            include: Vec::new(),
        }
    }

    pub fn max_output_tokens(&self) -> u32 {
        self.max_output_tokens
    }

    pub fn stream(&self) -> bool {
        self.stream
    }

    pub fn store(&self) -> bool {
        self.store
    }

    pub fn background(&self) -> bool {
        self.background
    }

    pub fn service_tier(&self) -> OpenAIResponsesServiceTier {
        self.service_tier
    }

    pub fn include(&self) -> &[String] {
        &self.include
    }

    pub fn digest(&self) -> Result<String, OpenAIResponsesBuildError> {
        self.validate()?;
        digest_json(&json!({
            "digest_serialization_profile": DIGEST_SERIALIZATION_PROFILE,
            "controls": self,
        }))
    }

    fn validate(&self) -> Result<(), OpenAIResponsesBuildError> {
        if self.schema_version != RESPONSE_ONLY_CONTROLS_VERSION {
            return Err(OpenAIResponsesBuildError::new(
                "unsupported Responses control version",
            ));
        }
        if self.max_output_tokens == 0 {
            return Err(OpenAIResponsesBuildError::new(
                "Responses max_output_tokens must be positive",
            ));
        }
        if self.background {
            return Err(OpenAIResponsesBuildError::new(
                "background Responses are not supported by the synchronous receipt API",
            ));
        }
        if self.include.iter().any(|item| item.trim().is_empty()) {
            return Err(OpenAIResponsesBuildError::new(
                "Responses include entries cannot be empty",
            ));
        }
        Ok(())
    }
}

pub struct OpenAIResponsesResponseControlsBuilder {
    max_output_tokens: u32,
    stream: bool,
    store: bool,
    background: bool,
    service_tier: OpenAIResponsesServiceTier,
    include: Vec<String>,
}

impl OpenAIResponsesResponseControlsBuilder {
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn store(mut self, store: bool) -> Self {
        self.store = store;
        self
    }

    pub fn background(mut self, background: bool) -> Self {
        self.background = background;
        self
    }

    pub fn service_tier(mut self, service_tier: OpenAIResponsesServiceTier) -> Self {
        self.service_tier = service_tier;
        self
    }

    pub fn include(mut self, include: impl Into<String>) -> Self {
        self.include.push(include.into());
        self
    }

    pub fn build(self) -> Result<OpenAIResponsesResponseControls, OpenAIResponsesBuildError> {
        let controls = OpenAIResponsesResponseControls {
            schema_version: RESPONSE_ONLY_CONTROLS_VERSION.into(),
            max_output_tokens: self.max_output_tokens,
            stream: self.stream,
            store: self.store,
            background: self.background,
            service_tier: self.service_tier,
            include: self.include,
        };
        controls.validate()?;
        Ok(controls)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesRedirectPolicy {
    Reject,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesRetryPolicy {
    Never,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAIResponsesTransportPolicy {
    schema_version: String,
    redirect_policy: OpenAIResponsesRedirectPolicy,
    retry_policy: OpenAIResponsesRetryPolicy,
    max_http_attempts: u8,
}

impl OpenAIResponsesTransportPolicy {
    pub fn strict() -> Self {
        Self {
            schema_version: TRANSPORT_POLICY_VERSION.into(),
            redirect_policy: OpenAIResponsesRedirectPolicy::Reject,
            retry_policy: OpenAIResponsesRetryPolicy::Never,
            max_http_attempts: 1,
        }
    }

    pub fn redirect_policy(&self) -> OpenAIResponsesRedirectPolicy {
        self.redirect_policy
    }

    pub fn retry_policy(&self) -> OpenAIResponsesRetryPolicy {
        self.retry_policy
    }

    pub fn max_http_attempts(&self) -> u8 {
        self.max_http_attempts
    }

    pub fn digest(&self) -> Result<String, OpenAIResponsesBuildError> {
        self.validate()?;
        digest_json(&json!({
            "digest_serialization_profile": DIGEST_SERIALIZATION_PROFILE,
            "transport_policy": self,
        }))
    }

    fn validate(&self) -> Result<(), OpenAIResponsesBuildError> {
        if self != &Self::strict() {
            return Err(OpenAIResponsesBuildError::new(
                "Responses transport policy is not the canonical strict policy",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedOpenAIResponse {
    schema_version: String,
    projection: OpenAIResponsesTokenProjection,
    projection_digest: String,
    controls: OpenAIResponsesResponseControls,
    controls_digest: String,
    transport_policy: OpenAIResponsesTransportPolicy,
    transport_policy_digest: String,
}

impl PreparedOpenAIResponse {
    pub fn new(
        projection: OpenAIResponsesTokenProjection,
        controls: OpenAIResponsesResponseControls,
    ) -> Result<Self, OpenAIResponsesBuildError> {
        let projection_digest = projection.digest()?;
        let controls_digest = controls.digest()?;
        let transport_policy = OpenAIResponsesTransportPolicy::strict();
        let transport_policy_digest = transport_policy.digest()?;
        Ok(Self {
            schema_version: PREPARED_RESPONSE_VERSION.into(),
            projection,
            projection_digest,
            controls,
            controls_digest,
            transport_policy,
            transport_policy_digest,
        })
    }

    pub fn projection(&self) -> &OpenAIResponsesTokenProjection {
        &self.projection
    }

    pub fn projection_digest(&self) -> &str {
        &self.projection_digest
    }

    pub fn controls(&self) -> &OpenAIResponsesResponseControls {
        &self.controls
    }

    pub fn controls_digest(&self) -> &str {
        &self.controls_digest
    }

    pub fn transport_policy(&self) -> &OpenAIResponsesTransportPolicy {
        &self.transport_policy
    }

    pub fn transport_policy_digest(&self) -> &str {
        &self.transport_policy_digest
    }

    pub fn input_token_count_body(&self) -> Result<Value, OpenAIResponsesBuildError> {
        self.validate()?;
        self.projection.wire_body()
    }

    pub fn response_body(&self) -> Result<Value, OpenAIResponsesBuildError> {
        self.validate()?;
        let mut body = self
            .projection
            .wire_body()?
            .as_object()
            .expect("projection wire body is an object")
            .clone();
        body.insert(
            "max_output_tokens".into(),
            Value::from(self.controls.max_output_tokens),
        );
        body.insert("stream".into(), Value::Bool(self.controls.stream));
        body.insert("store".into(), Value::Bool(self.controls.store));
        body.insert("background".into(), Value::Bool(self.controls.background));
        body.insert(
            "service_tier".into(),
            Value::String(self.controls.service_tier.as_wire().into()),
        );
        if !self.controls.include.is_empty() {
            body.insert(
                "include".into(),
                Value::Array(
                    self.controls
                        .include
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
        Ok(Value::Object(body))
    }

    fn validate(&self) -> Result<(), OpenAIResponsesBuildError> {
        if self.schema_version != PREPARED_RESPONSE_VERSION {
            return Err(OpenAIResponsesBuildError::new(
                "unsupported prepared Responses call version",
            ));
        }
        if self.projection_digest != self.projection.digest()? {
            return Err(OpenAIResponsesBuildError::new(
                "prepared Responses projection digest mismatch",
            ));
        }
        if self.controls_digest != self.controls.digest()? {
            return Err(OpenAIResponsesBuildError::new(
                "prepared Responses control digest mismatch",
            ));
        }
        if self.transport_policy_digest != self.transport_policy.digest()? {
            return Err(OpenAIResponsesBuildError::new(
                "prepared Responses transport policy digest mismatch",
            ));
        }
        Ok(())
    }
}

fn serialize_value<T: Serialize>(value: &T) -> Result<Value, OpenAIResponsesBuildError> {
    serde_json::to_value(value).map_err(|error| {
        OpenAIResponsesBuildError::new(format!("serialize Responses request field: {error}"))
    })
}

fn digest_json<T: Serialize>(value: &T) -> Result<String, OpenAIResponsesBuildError> {
    let bytes = canonical_json_bytes(value)?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, OpenAIResponsesBuildError> {
    serde_jcs::to_vec(value).map_err(|error| {
        OpenAIResponsesBuildError::new(format!(
            "canonicalize Responses digest input with RFC 8785 JCS: {error}"
        ))
    })
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OpenAIResponsesProviderIdentity {
    provider: String,
    base_url: String,
}

impl OpenAIResponsesProviderIdentity {
    fn configured(base_url: &str) -> Self {
        Self {
            provider: "openai".into(),
            base_url: base_url.trim_end_matches('/').into(),
        }
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AttemptState {
    NotStarted,
    OutcomeUnknown,
    HttpRejected,
    ProviderErrorObserved,
    TerminalObserved,
}

impl AttemptState {
    fn has_provider_outcome(self) -> bool {
        matches!(
            self,
            Self::HttpRejected | Self::ProviderErrorObserved | Self::TerminalObserved
        )
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OpenAIResponsesProviderErrorEvidence {
    code: Option<String>,
    message: String,
    error_type: Option<String>,
    param: Option<String>,
}

impl OpenAIResponsesProviderErrorEvidence {
    pub fn code(&self) -> Option<&str> {
        self.code.as_deref()
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn error_type(&self) -> Option<&str> {
        self.error_type.as_deref()
    }

    pub fn param(&self) -> Option<&str> {
        self.param.as_deref()
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OpenAIResponsesReturnedReasoning {
    effort: Option<String>,
    context: Option<String>,
    mode: Option<String>,
    summary: Option<String>,
}

impl OpenAIResponsesReturnedReasoning {
    pub fn effort(&self) -> Option<&str> {
        self.effort.as_deref()
    }

    pub fn context(&self) -> Option<&str> {
        self.context.as_deref()
    }

    pub fn mode(&self) -> Option<&str> {
        self.mode.as_deref()
    }

    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OpenAIResponsesObservedRefusal {
    event_type: String,
    item_id: Option<String>,
    text: String,
    sequence_number: Option<u64>,
}

impl OpenAIResponsesObservedRefusal {
    pub fn event_type(&self) -> &str {
        &self.event_type
    }

    pub fn item_id(&self) -> Option<&str> {
        self.item_id.as_deref()
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn sequence_number(&self) -> Option<u64> {
        self.sequence_number
    }
}

#[derive(Debug, Clone)]
pub struct OpenAIResponsesAttemptReceipt {
    schema_version: &'static str,
    provider_identity: OpenAIResponsesProviderIdentity,
    endpoint: String,
    projection_digest: String,
    controls_digest: Option<String>,
    transport_policy_digest: String,
    state: AttemptState,
    status_code: Option<u16>,
    request_id: Option<String>,
    response_id: Option<String>,
    returned_model: Option<String>,
    terminal_status: Option<ResponsesTerminalStatus>,
    terminal_usage: Option<Usage>,
    returned_service_tier: Option<String>,
    provider_error: Option<OpenAIResponsesProviderErrorEvidence>,
    incomplete_reason: Option<String>,
    returned_reasoning: Option<OpenAIResponsesReturnedReasoning>,
    observed_tool_events: Vec<OpenAIResponsesObservedToolEvent>,
    observed_refusals: Vec<OpenAIResponsesObservedRefusal>,
}

impl OpenAIResponsesAttemptReceipt {
    pub fn schema_version(&self) -> &str {
        self.schema_version
    }

    pub fn state(&self) -> AttemptState {
        self.state
    }

    pub fn provider_identity(&self) -> &OpenAIResponsesProviderIdentity {
        &self.provider_identity
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn projection_digest(&self) -> &str {
        &self.projection_digest
    }

    pub fn controls_digest(&self) -> Option<&str> {
        self.controls_digest.as_deref()
    }

    pub fn transport_policy_digest(&self) -> &str {
        &self.transport_policy_digest
    }

    pub fn status_code(&self) -> Option<u16> {
        self.status_code
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn response_id(&self) -> Option<&str> {
        self.response_id.as_deref()
    }

    pub fn returned_model(&self) -> Option<&str> {
        self.returned_model.as_deref()
    }

    pub fn terminal_status(&self) -> Option<ResponsesTerminalStatus> {
        self.terminal_status
    }

    pub fn terminal_usage(&self) -> Option<&Usage> {
        self.terminal_usage.as_ref()
    }

    pub fn returned_service_tier(&self) -> Option<&str> {
        self.returned_service_tier.as_deref()
    }

    pub fn provider_error(&self) -> Option<&OpenAIResponsesProviderErrorEvidence> {
        self.provider_error.as_ref()
    }

    pub fn incomplete_reason(&self) -> Option<&str> {
        self.incomplete_reason.as_deref()
    }

    pub fn returned_reasoning(&self) -> Option<&OpenAIResponsesReturnedReasoning> {
        self.returned_reasoning.as_ref()
    }

    pub fn observed_tool_events(&self) -> &[OpenAIResponsesObservedToolEvent] {
        &self.observed_tool_events
    }

    pub fn observed_refusals(&self) -> &[OpenAIResponsesObservedRefusal] {
        &self.observed_refusals
    }
}

#[derive(Debug)]
pub struct OpenAIResponsesFailure {
    message: String,
    source: LlmError,
    attempt: OpenAIResponsesAttemptReceipt,
}

impl fmt::Display for OpenAIResponsesFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.message, self.source)
    }
}

impl std::error::Error for OpenAIResponsesFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl OpenAIResponsesFailure {
    pub fn attempt(&self) -> &OpenAIResponsesAttemptReceipt {
        &self.attempt
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAIResponsesError {
    #[error("invalid prepared Responses call: {0}")]
    InvalidPreparedCall(#[source] OpenAIResponsesFailure),
    #[error("Responses request failed: {0}")]
    Request(#[source] OpenAIResponsesFailure),
}

impl OpenAIResponsesError {
    pub fn attempt(&self) -> &OpenAIResponsesAttemptReceipt {
        match self {
            Self::InvalidPreparedCall(failure) | Self::Request(failure) => failure.attempt(),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ResponsesTerminalStatus {
    Completed,
    Incomplete,
    Failed,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OpenAIResponsesObservedToolEvent {
    item_type: String,
    item_id: Option<String>,
    call_id: Option<String>,
    name: Option<String>,
}

impl OpenAIResponsesObservedToolEvent {
    pub fn item_type(&self) -> &str {
        &self.item_type
    }

    pub fn item_id(&self) -> Option<&str> {
        self.item_id.as_deref()
    }

    pub fn call_id(&self) -> Option<&str> {
        self.call_id.as_deref()
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

#[derive(Debug, Clone)]
pub struct OpenAIResponseReceipt {
    schema_version: &'static str,
    provider_identity: OpenAIResponsesProviderIdentity,
    endpoint: String,
    request_id: String,
    status_code: u16,
    projection_digest: String,
    controls_digest: String,
    transport_policy_digest: String,
    requested_model: String,
    returned_model: String,
    response_id: String,
    terminal_status: ResponsesTerminalStatus,
    terminal_sequence_number: Option<u64>,
    usage: Usage,
    returned_service_tier: Option<String>,
    provider_error: Option<OpenAIResponsesProviderErrorEvidence>,
    incomplete_reason: Option<String>,
    returned_reasoning: Option<OpenAIResponsesReturnedReasoning>,
    observed_tool_events: Vec<OpenAIResponsesObservedToolEvent>,
    observed_refusals: Vec<OpenAIResponsesObservedRefusal>,
}

impl OpenAIResponseReceipt {
    pub fn schema_version(&self) -> &str {
        self.schema_version
    }

    pub fn provider_identity(&self) -> &OpenAIResponsesProviderIdentity {
        &self.provider_identity
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn status_code(&self) -> u16 {
        self.status_code
    }

    pub fn projection_digest(&self) -> &str {
        &self.projection_digest
    }

    pub fn controls_digest(&self) -> &str {
        &self.controls_digest
    }

    pub fn transport_policy_digest(&self) -> &str {
        &self.transport_policy_digest
    }

    pub fn requested_model(&self) -> &str {
        &self.requested_model
    }

    pub fn returned_model(&self) -> &str {
        &self.returned_model
    }

    pub fn response_id(&self) -> &str {
        &self.response_id
    }

    pub fn terminal_status(&self) -> ResponsesTerminalStatus {
        self.terminal_status
    }

    pub fn terminal_sequence_number(&self) -> Option<u64> {
        self.terminal_sequence_number
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    pub fn returned_service_tier(&self) -> Option<&str> {
        self.returned_service_tier.as_deref()
    }

    pub fn provider_error(&self) -> Option<&OpenAIResponsesProviderErrorEvidence> {
        self.provider_error.as_ref()
    }

    pub fn incomplete_reason(&self) -> Option<&str> {
        self.incomplete_reason.as_deref()
    }

    pub fn returned_reasoning(&self) -> Option<&OpenAIResponsesReturnedReasoning> {
        self.returned_reasoning.as_ref()
    }

    pub fn observed_tool_events(&self) -> &[OpenAIResponsesObservedToolEvent] {
        &self.observed_tool_events
    }

    pub fn observed_refusals(&self) -> &[OpenAIResponsesObservedRefusal] {
        &self.observed_refusals
    }
}

#[derive(Debug, Clone)]
pub struct OpenAIResponseResult {
    output_text: String,
    receipt: OpenAIResponseReceipt,
}

impl OpenAIResponseResult {
    pub fn output_text(&self) -> &str {
        &self.output_text
    }

    pub fn receipt(&self) -> &OpenAIResponseReceipt {
        &self.receipt
    }
}

#[derive(Debug, Clone)]
pub struct OpenAIResponseInputTokenCount {
    schema_version: &'static str,
    provider_identity: OpenAIResponsesProviderIdentity,
    endpoint: String,
    request_id: String,
    status_code: u16,
    requested_model: String,
    projection_digest: String,
    transport_policy_digest: String,
    input_tokens: u64,
}

impl OpenAIResponseInputTokenCount {
    pub fn schema_version(&self) -> &str {
        self.schema_version
    }

    pub fn provider_identity(&self) -> &OpenAIResponsesProviderIdentity {
        &self.provider_identity
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn status_code(&self) -> u16 {
        self.status_code
    }

    pub fn requested_model(&self) -> &str {
        &self.requested_model
    }

    pub fn projection_digest(&self) -> &str {
        &self.projection_digest
    }

    pub fn transport_policy_digest(&self) -> &str {
        &self.transport_policy_digest
    }

    pub fn input_tokens(&self) -> u64 {
        self.input_tokens
    }
}

#[derive(Debug)]
struct ParsedTerminal {
    response_id: String,
    returned_model: String,
    terminal_status: ResponsesTerminalStatus,
    sequence_number: Option<u64>,
    usage: Usage,
    returned_service_tier: Option<String>,
    provider_error: Option<OpenAIResponsesProviderErrorEvidence>,
    incomplete_reason: Option<String>,
    returned_reasoning: Option<OpenAIResponsesReturnedReasoning>,
    output_text: String,
    observed_tool_events: Vec<OpenAIResponsesObservedToolEvent>,
    observed_refusals: Vec<OpenAIResponsesObservedRefusal>,
}

impl OpenAIProvider {
    /// Count input tokens for a prepared Responses call.
    ///
    /// The count body is derived from the same immutable projection later used
    /// by [`create_response`](Self::create_response). This method performs one
    /// HTTP request and never retries.
    pub async fn count_response_input_tokens(
        &self,
        prepared: &PreparedOpenAIResponse,
    ) -> Result<OpenAIResponseInputTokenCount, OpenAIResponsesError> {
        let endpoint = self.responses_input_tokens_path.clone();
        let mut attempt = self.attempt_receipt(prepared, endpoint.clone(), false);
        let body = prepared
            .input_token_count_body()
            .map_err(|error| invalid_prepared_failure(error.to_string(), attempt.clone()))?;
        let url = endpoint_url(&self.base_url, &endpoint);
        let response = self
            .responses_client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|source| {
                attempt.state = AttemptState::OutcomeUnknown;
                request_failure(
                    "input-token request transport",
                    source.into(),
                    attempt.clone(),
                )
            })?;

        let status = response.status();
        attempt.status_code = Some(status.as_u16());
        attempt.request_id = response_request_id(response.headers());
        if !status.is_success() {
            attempt.state = AttemptState::HttpRejected;
            let retry_after = extract_retry_after_header(response.headers());
            let error_text = response.text().await.map_err(|source| {
                mark_outcome_unknown(&mut attempt);
                request_failure(
                    "read input-token error response",
                    source.into(),
                    attempt.clone(),
                )
            })?;
            attempt.provider_error = parse_http_error_evidence(&error_text);
            return Err(request_failure(
                "input-token endpoint rejected request",
                map_http_error(status.as_u16(), retry_after, error_text),
                attempt,
            ));
        }

        let bytes = response.bytes().await.map_err(|source| {
            mark_outcome_unknown(&mut attempt);
            request_failure("read input-token response", source.into(), attempt.clone())
        })?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|source| {
            mark_outcome_unknown(&mut attempt);
            request_failure(
                "decode input-token response",
                LlmError::Decode(source),
                attempt.clone(),
            )
        })?;
        let input_tokens = parse_input_token_count(&value).map_err(|message| {
            mark_outcome_unknown(&mut attempt);
            request_failure(
                "validate input-token response",
                LlmError::Stream(message),
                attempt.clone(),
            )
        })?;
        attempt.state = AttemptState::TerminalObserved;
        let request_id = attempt.request_id.clone().ok_or_else(|| {
            request_failure(
                "validate input-token receipt",
                LlmError::Stream("OpenAI response omitted x-request-id".into()),
                attempt.clone(),
            )
        })?;

        Ok(OpenAIResponseInputTokenCount {
            schema_version: PROVIDER_RECEIPT_VERSION,
            provider_identity: attempt.provider_identity,
            endpoint: attempt.endpoint,
            request_id,
            status_code: status.as_u16(),
            requested_model: prepared.projection.model.clone(),
            projection_digest: prepared.projection_digest.clone(),
            transport_policy_digest: prepared.transport_policy_digest.clone(),
            input_tokens,
        })
    }

    /// Execute one prepared Responses call and return terminal output plus an
    /// authoritative request/usage/tool receipt.
    ///
    /// Streaming calls complete at the first terminal Responses event and do
    /// not wait for SSE EOF. The transport performs no implicit retry.
    pub async fn create_response(
        &self,
        prepared: &PreparedOpenAIResponse,
    ) -> Result<OpenAIResponseResult, OpenAIResponsesError> {
        let endpoint = self.responses_path.clone();
        let mut attempt = self.attempt_receipt(prepared, endpoint.clone(), true);
        let body = prepared
            .response_body()
            .map_err(|error| invalid_prepared_failure(error.to_string(), attempt.clone()))?;
        let url = endpoint_url(&self.base_url, &endpoint);
        let response = self
            .responses_client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|source| {
                attempt.state = AttemptState::OutcomeUnknown;
                request_failure("Responses transport", source.into(), attempt.clone())
            })?;

        let status = response.status();
        attempt.status_code = Some(status.as_u16());
        attempt.request_id = response_request_id(response.headers());
        if !status.is_success() {
            attempt.state = AttemptState::HttpRejected;
            let retry_after = extract_retry_after_header(response.headers());
            let error_text = response.text().await.map_err(|source| {
                mark_outcome_unknown(&mut attempt);
                request_failure(
                    "read Responses error response",
                    source.into(),
                    attempt.clone(),
                )
            })?;
            attempt.provider_error = parse_http_error_evidence(&error_text);
            return Err(request_failure(
                "Responses endpoint rejected request",
                map_http_error(status.as_u16(), retry_after, error_text),
                attempt,
            ));
        }

        let terminal = if prepared.controls.stream {
            parse_stream_terminal(response, &mut attempt)
                .await
                .map_err(|source| {
                    mark_outcome_unknown(&mut attempt);
                    request_failure("parse Responses stream", source, attempt.clone())
                })?
        } else {
            let bytes = response.bytes().await.map_err(|source| {
                mark_outcome_unknown(&mut attempt);
                request_failure("read Responses response", source.into(), attempt.clone())
            })?;
            let value: Value = serde_json::from_slice(&bytes).map_err(|source| {
                mark_outcome_unknown(&mut attempt);
                request_failure(
                    "decode Responses response",
                    LlmError::Decode(source),
                    attempt.clone(),
                )
            })?;
            parse_terminal_response(&value, None, &mut attempt).map_err(|message| {
                mark_outcome_unknown(&mut attempt);
                request_failure(
                    "validate Responses terminal",
                    LlmError::Stream(message),
                    attempt.clone(),
                )
            })?
        };

        let request_id = attempt.request_id.clone().ok_or_else(|| {
            request_failure(
                "validate Responses receipt",
                LlmError::Stream("OpenAI response omitted x-request-id".into()),
                attempt.clone(),
            )
        })?;

        Ok(OpenAIResponseResult {
            output_text: terminal.output_text,
            receipt: OpenAIResponseReceipt {
                schema_version: PROVIDER_RECEIPT_VERSION,
                provider_identity: attempt.provider_identity,
                endpoint: attempt.endpoint,
                request_id,
                status_code: status.as_u16(),
                projection_digest: prepared.projection_digest.clone(),
                controls_digest: prepared.controls_digest.clone(),
                transport_policy_digest: prepared.transport_policy_digest.clone(),
                requested_model: prepared.projection.model.clone(),
                returned_model: terminal.returned_model,
                response_id: terminal.response_id,
                terminal_status: terminal.terminal_status,
                terminal_sequence_number: terminal.sequence_number,
                usage: terminal.usage,
                returned_service_tier: terminal.returned_service_tier,
                provider_error: terminal.provider_error,
                incomplete_reason: terminal.incomplete_reason,
                returned_reasoning: terminal.returned_reasoning,
                observed_tool_events: terminal.observed_tool_events,
                observed_refusals: terminal.observed_refusals,
            },
        })
    }

    fn attempt_receipt(
        &self,
        prepared: &PreparedOpenAIResponse,
        endpoint: String,
        include_controls: bool,
    ) -> OpenAIResponsesAttemptReceipt {
        OpenAIResponsesAttemptReceipt {
            schema_version: PROVIDER_RECEIPT_VERSION,
            provider_identity: OpenAIResponsesProviderIdentity::configured(&self.base_url),
            endpoint,
            projection_digest: prepared.projection_digest.clone(),
            controls_digest: include_controls.then(|| prepared.controls_digest.clone()),
            transport_policy_digest: prepared.transport_policy_digest.clone(),
            state: AttemptState::NotStarted,
            status_code: None,
            request_id: None,
            response_id: None,
            returned_model: None,
            terminal_status: None,
            terminal_usage: None,
            returned_service_tier: None,
            provider_error: None,
            incomplete_reason: None,
            returned_reasoning: None,
            observed_tool_events: Vec::new(),
            observed_refusals: Vec::new(),
        }
    }
}

fn invalid_prepared_failure(
    message: String,
    attempt: OpenAIResponsesAttemptReceipt,
) -> OpenAIResponsesError {
    OpenAIResponsesError::InvalidPreparedCall(OpenAIResponsesFailure {
        message,
        source: LlmError::InvalidRequest("prepared Responses call failed validation".into()),
        attempt,
    })
}

fn request_failure(
    message: impl Into<String>,
    source: LlmError,
    attempt: OpenAIResponsesAttemptReceipt,
) -> OpenAIResponsesError {
    OpenAIResponsesError::Request(OpenAIResponsesFailure {
        message: message.into(),
        source,
        attempt,
    })
}

fn endpoint_url(base_url: &str, path: &str) -> String {
    format!("{}{}", base_url.trim_end_matches('/'), path)
}

fn response_request_id(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
}

fn mark_outcome_unknown(attempt: &mut OpenAIResponsesAttemptReceipt) {
    if !attempt.state.has_provider_outcome() {
        attempt.state = AttemptState::OutcomeUnknown;
    }
}

fn parse_http_error_evidence(body: &str) -> Option<OpenAIResponsesProviderErrorEvidence> {
    let value: Value = serde_json::from_str(body).ok()?;
    let object = value.get("error")?.as_object()?;
    parse_provider_error_object(object, false).ok()
}

fn map_http_error(status: u16, retry_after: Option<std::time::Duration>, body: String) -> LlmError {
    match status {
        401 => LlmError::Unauthorized,
        403 => LlmError::Forbidden,
        402 => LlmError::Provider {
            code: "payment_required".into(),
            message: body,
        },
        429 => LlmError::RateLimit { retry_after },
        500 | 502 | 503 => LlmError::ServerError(body),
        529 => LlmError::Overloaded { retry_after },
        _ => LlmError::InvalidRequest(body),
    }
}

fn parse_input_token_count(value: &Value) -> Result<u64, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "input-token response must be an object".to_string())?;
    if object.get("object").and_then(Value::as_str) != Some("response.input_tokens") {
        return Err("input-token response object identity is invalid".into());
    }
    object
        .get("input_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| "input-token response omitted a valid input_tokens count".into())
}

async fn parse_stream_terminal(
    response: reqwest::Response,
    attempt: &mut OpenAIResponsesAttemptReceipt,
) -> Result<ParsedTerminal, LlmError> {
    let events = response
        .bytes_stream()
        .eventsource()
        .map(|event| event.map(|event| event.data));
    parse_stream_event_data(events, attempt).await
}

async fn parse_stream_event_data<S, E>(
    events: S,
    attempt: &mut OpenAIResponsesAttemptReceipt,
) -> Result<ParsedTerminal, LlmError>
where
    S: futures::Stream<Item = Result<String, E>>,
    E: fmt::Display,
{
    futures::pin_mut!(events);
    let mut last_sequence = None;

    while let Some(event) = events.next().await {
        let data = event.map_err(|error| LlmError::Stream(error.to_string()))?;
        if data == "[DONE]" {
            return Err(LlmError::Stream(
                "Responses stream ended before a terminal event".into(),
            ));
        }
        let value: Value = serde_json::from_str(&data)?;
        let sequence = value
            .get("sequence_number")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                LlmError::Stream("Responses stream event omitted sequence_number".into())
            })?;
        if last_sequence.is_some_and(|previous| sequence <= previous) {
            return Err(LlmError::Stream(
                "Responses stream sequence_number is not strictly increasing".into(),
            ));
        }
        last_sequence = Some(sequence);
        let event_type = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| LlmError::Stream("Responses stream event omitted type".into()))?;
        observe_stream_tool_event(&value, event_type, &mut attempt.observed_tool_events)?;
        observe_stream_refusal_event(&value, event_type, &mut attempt.observed_refusals)?;
        if matches!(
            event_type,
            "response.completed" | "response.incomplete" | "response.failed"
        ) {
            let response_value = value.get("response").ok_or_else(|| {
                LlmError::Stream("Responses terminal event omitted response".into())
            })?;
            let terminal = parse_terminal_response(response_value, Some(sequence), attempt)
                .map_err(LlmError::Stream)?;
            let expected = match event_type {
                "response.completed" => ResponsesTerminalStatus::Completed,
                "response.incomplete" => ResponsesTerminalStatus::Incomplete,
                "response.failed" => ResponsesTerminalStatus::Failed,
                _ => unreachable!(),
            };
            if terminal.terminal_status != expected {
                return Err(LlmError::Stream(
                    "Responses terminal event type does not match response status".into(),
                ));
            }
            return Ok(terminal);
        }
        if event_type == "error" {
            let error = parse_provider_error_object(
                value
                    .as_object()
                    .ok_or_else(|| LlmError::Stream("error event must be an object".into()))?,
                false,
            )
            .map_err(LlmError::Stream)?;
            attempt.provider_error = Some(error.clone());
            attempt.state = AttemptState::ProviderErrorObserved;
            return Err(LlmError::Stream(format!(
                "Responses stream returned error event: {}",
                error.message
            )));
        }
    }

    Err(LlmError::Stream(
        "Responses stream reached EOF before a terminal event".into(),
    ))
}

fn parse_terminal_response(
    value: &Value,
    sequence_number: Option<u64>,
    attempt: &mut OpenAIResponsesAttemptReceipt,
) -> Result<ParsedTerminal, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Responses terminal must be an object".to_string())?;
    if object.get("object").and_then(Value::as_str) != Some("response") {
        return Err("Responses terminal object identity is invalid".into());
    }
    let response_id = required_nonempty_string(object, "id")?;
    let returned_model = required_nonempty_string(object, "model")?;
    let terminal_status = match object.get("status").and_then(Value::as_str) {
        Some("completed") => ResponsesTerminalStatus::Completed,
        Some("incomplete") => ResponsesTerminalStatus::Incomplete,
        Some("failed") => ResponsesTerminalStatus::Failed,
        Some(other) => return Err(format!("Responses status is not terminal: {other}")),
        None => return Err("Responses terminal omitted status".into()),
    };

    attempt.state = AttemptState::TerminalObserved;
    attempt.response_id = Some(response_id.clone());
    attempt.returned_model = Some(returned_model.clone());
    attempt.terminal_status = Some(terminal_status);
    let usage = normalize_responses_usage(object.get("usage"));
    attempt.terminal_usage = Some(usage.clone());
    observe_terminal_payload(
        object,
        &mut attempt.observed_tool_events,
        &mut attempt.observed_refusals,
    );

    let provider_error = parse_optional_provider_error(object.get("error"), true)?;
    attempt.provider_error = provider_error.clone();
    let incomplete_reason = parse_incomplete_reason(object.get("incomplete_details"))?;
    attempt.incomplete_reason = incomplete_reason.clone();
    let returned_service_tier = Some(
        optional_nonempty_string_value(object.get("service_tier"), "Responses service_tier")?
            .ok_or_else(|| "Responses terminal omitted service_tier".to_string())?,
    );
    attempt.returned_service_tier = returned_service_tier.clone();
    let returned_reasoning = parse_returned_reasoning(object.get("reasoning"))?;
    attempt.returned_reasoning = returned_reasoning.clone();

    validate_terminal_consistency(terminal_status, &provider_error, &incomplete_reason)?;
    let output = object
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| "Responses terminal omitted output array".to_string())?;
    let (output_text, output_tools, output_refusals) = parse_response_output(output)?;
    for observed in output_tools {
        push_unique_tool(&mut attempt.observed_tool_events, observed);
    }
    for refusal in output_refusals {
        push_unique_refusal(&mut attempt.observed_refusals, refusal);
    }

    Ok(ParsedTerminal {
        response_id,
        returned_model,
        terminal_status,
        sequence_number,
        usage,
        returned_service_tier,
        provider_error,
        incomplete_reason,
        returned_reasoning,
        output_text,
        observed_tool_events: attempt.observed_tool_events.clone(),
        observed_refusals: attempt.observed_refusals.clone(),
    })
}

fn validate_terminal_consistency(
    status: ResponsesTerminalStatus,
    error: &Option<OpenAIResponsesProviderErrorEvidence>,
    incomplete_reason: &Option<String>,
) -> Result<(), String> {
    match status {
        ResponsesTerminalStatus::Completed if error.is_some() => {
            Err("completed Responses terminal cannot contain error".into())
        }
        ResponsesTerminalStatus::Completed if incomplete_reason.is_some() => {
            Err("completed Responses terminal cannot contain incomplete_details".into())
        }
        ResponsesTerminalStatus::Incomplete if error.is_some() => {
            Err("incomplete Responses terminal cannot contain error".into())
        }
        ResponsesTerminalStatus::Incomplete if incomplete_reason.is_none() => {
            Err("incomplete Responses terminal omitted incomplete_details.reason".into())
        }
        ResponsesTerminalStatus::Failed if error.is_none() => {
            Err("failed Responses terminal omitted error".into())
        }
        ResponsesTerminalStatus::Failed if incomplete_reason.is_some() => {
            Err("failed Responses terminal cannot contain incomplete_details".into())
        }
        _ => Ok(()),
    }
}

fn parse_optional_provider_error(
    value: Option<&Value>,
    require_code: bool,
) -> Result<Option<OpenAIResponsesProviderErrorEvidence>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(object)) => parse_provider_error_object(object, require_code).map(Some),
        Some(_) => Err("Responses error must be an object or null".into()),
    }
}

fn parse_provider_error_object(
    object: &Map<String, Value>,
    require_code: bool,
) -> Result<OpenAIResponsesProviderErrorEvidence, String> {
    let code = optional_nonempty_string_value(object.get("code"), "provider error code")?;
    if require_code && code.is_none() {
        return Err("Responses terminal error omitted code".into());
    }
    let message = object
        .get("message")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| "provider error omitted message".to_string())?;
    let error_type = optional_nonempty_string_value(object.get("type"), "provider error type")?;
    let param = optional_nonempty_string_value(object.get("param"), "provider error param")?;
    Ok(OpenAIResponsesProviderErrorEvidence {
        code,
        message,
        error_type,
        param,
    })
}

fn parse_incomplete_reason(value: Option<&Value>) -> Result<Option<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(object)) => object
            .get("reason")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_owned)
            .map(Some)
            .ok_or_else(|| "Responses incomplete_details omitted reason".into()),
        Some(_) => Err("Responses incomplete_details must be an object or null".into()),
    }
}

fn parse_returned_reasoning(
    value: Option<&Value>,
) -> Result<Option<OpenAIResponsesReturnedReasoning>, String> {
    let object = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(object)) => object,
        Some(_) => return Err("Responses reasoning must be an object or null".into()),
    };
    Ok(Some(OpenAIResponsesReturnedReasoning {
        effort: optional_nonempty_string_value(object.get("effort"), "reasoning effort")?,
        context: optional_nonempty_string_value(object.get("context"), "reasoning context")?,
        mode: optional_nonempty_string_value(object.get("mode"), "reasoning mode")?,
        summary: optional_nonempty_string_value(object.get("summary"), "reasoning summary")?,
    }))
}

fn optional_nonempty_string_value(
    value: Option<&Value>,
    field: &str,
) -> Result<Option<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(Value::String(_)) => Err(format!("{field} cannot be empty")),
        Some(_) => Err(format!("{field} must be a string or null")),
    }
}

fn required_nonempty_string(object: &Map<String, Value>, field: &str) -> Result<String, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("Responses terminal omitted valid {field}"))
}

fn parse_response_output(
    output: &[Value],
) -> Result<
    (
        String,
        Vec<OpenAIResponsesObservedToolEvent>,
        Vec<OpenAIResponsesObservedRefusal>,
    ),
    String,
> {
    let mut text = String::new();
    let mut tools = Vec::new();
    let mut refusals = Vec::new();
    for item in output {
        let object = item
            .as_object()
            .ok_or_else(|| "Responses output item must be an object".to_string())?;
        let item_type = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "Responses output item omitted type".to_string())?;
        match item_type {
            "message" => {
                let item_id =
                    optional_nonempty_string_value(object.get("id"), "Responses message id")?;
                let content = object
                    .get("content")
                    .and_then(Value::as_array)
                    .ok_or_else(|| "Responses message omitted content array".to_string())?;
                for block in content {
                    let block = block
                        .as_object()
                        .ok_or_else(|| "Responses message content must be an object".to_string())?;
                    match block.get("type").and_then(Value::as_str) {
                        Some("output_text") => {
                            let value =
                                block.get("text").and_then(Value::as_str).ok_or_else(|| {
                                    "Responses output_text block omitted text".to_string()
                                })?;
                            text.push_str(value);
                        }
                        Some("refusal") => {
                            let refusal = block
                                .get("refusal")
                                .and_then(Value::as_str)
                                .filter(|value| !value.trim().is_empty())
                                .ok_or_else(|| {
                                    "Responses refusal block omitted refusal".to_string()
                                })?;
                            push_unique_refusal(
                                &mut refusals,
                                OpenAIResponsesObservedRefusal {
                                    event_type: "message.refusal".into(),
                                    item_id: item_id.clone(),
                                    text: refusal.into(),
                                    sequence_number: None,
                                },
                            );
                        }
                        Some(other) => {
                            return Err(format!(
                                "unsupported Responses message content type: {other}"
                            ));
                        }
                        None => {
                            return Err("Responses message content omitted type".into());
                        }
                    }
                }
            }
            "reasoning" => {}
            _ => push_unique_tool(&mut tools, strict_tool_from_item(object, item_type)?),
        }
    }
    Ok((text, tools, refusals))
}

fn observe_stream_tool_event(
    event: &Value,
    event_type: &str,
    observed: &mut Vec<OpenAIResponsesObservedToolEvent>,
) -> Result<(), LlmError> {
    let object = event
        .as_object()
        .ok_or_else(|| LlmError::Stream("Responses stream event must be an object".into()))?;
    if let Some(item) = object.get("item") {
        let item = item
            .as_object()
            .ok_or_else(|| LlmError::Stream("Responses output item must be an object".into()))?;
        let item_type = item
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| LlmError::Stream("Responses output item omitted type".into()))?;
        if !matches!(item_type, "message" | "reasoning") {
            push_unique_tool(
                observed,
                strict_tool_from_item(item, item_type).map_err(LlmError::Stream)?,
            );
        }
    } else if is_tool_activity_event(event_type) {
        push_unique_tool(
            observed,
            OpenAIResponsesObservedToolEvent {
                item_type: event_type.into(),
                item_id: optional_nonempty_string_value(object.get("item_id"), "tool item_id")
                    .map_err(LlmError::Stream)?,
                call_id: optional_nonempty_string_value(object.get("call_id"), "tool call_id")
                    .map_err(LlmError::Stream)?,
                name: optional_nonempty_string_value(object.get("name"), "tool name")
                    .map_err(LlmError::Stream)?,
            },
        );
    }
    Ok(())
}

fn is_tool_activity_event(event_type: &str) -> bool {
    event_type.starts_with("response.")
        && (event_type.contains("_call") || event_type.starts_with("response.mcp_"))
}

fn strict_tool_from_item(
    object: &Map<String, Value>,
    item_type: &str,
) -> Result<OpenAIResponsesObservedToolEvent, String> {
    Ok(OpenAIResponsesObservedToolEvent {
        item_type: item_type.into(),
        item_id: optional_nonempty_string_value(object.get("id"), "tool item id")?,
        call_id: optional_nonempty_string_value(object.get("call_id"), "tool call id")?,
        name: optional_nonempty_string_value(object.get("name"), "tool name")?,
    })
}

fn lossy_tool_from_item(
    object: &Map<String, Value>,
    item_type: &str,
) -> OpenAIResponsesObservedToolEvent {
    OpenAIResponsesObservedToolEvent {
        item_type: item_type.into(),
        item_id: object.get("id").and_then(Value::as_str).map(str::to_owned),
        call_id: object
            .get("call_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
        name: object
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned),
    }
}

fn observe_stream_refusal_event(
    event: &Value,
    event_type: &str,
    observed: &mut Vec<OpenAIResponsesObservedRefusal>,
) -> Result<(), LlmError> {
    let text_field = match event_type {
        "response.refusal.delta" => "delta",
        "response.refusal.done" => "refusal",
        _ => return Ok(()),
    };
    let object = event
        .as_object()
        .ok_or_else(|| LlmError::Stream("Responses refusal event must be an object".into()))?;
    let item_id = object
        .get("item_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| LlmError::Stream("Responses refusal event omitted item_id".into()))?;
    let text = object
        .get(text_field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| LlmError::Stream(format!("Responses refusal event omitted {text_field}")))?;
    let sequence_number = object.get("sequence_number").and_then(Value::as_u64);
    push_unique_refusal(
        observed,
        OpenAIResponsesObservedRefusal {
            event_type: event_type.into(),
            item_id: Some(item_id.into()),
            text: text.into(),
            sequence_number,
        },
    );
    Ok(())
}

fn observe_terminal_payload(
    object: &Map<String, Value>,
    tools: &mut Vec<OpenAIResponsesObservedToolEvent>,
    refusals: &mut Vec<OpenAIResponsesObservedRefusal>,
) {
    let Some(output) = object.get("output").and_then(Value::as_array) else {
        return;
    };
    for item in output {
        let Some(item) = item.as_object() else {
            continue;
        };
        let Some(item_type) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        if item_type == "message" {
            let item_id = item.get("id").and_then(Value::as_str).map(str::to_owned);
            let Some(content) = item.get("content").and_then(Value::as_array) else {
                continue;
            };
            for block in content {
                let Some(block) = block.as_object() else {
                    continue;
                };
                if block.get("type").and_then(Value::as_str) == Some("refusal")
                    && let Some(text) = block.get("refusal").and_then(Value::as_str)
                {
                    push_unique_refusal(
                        refusals,
                        OpenAIResponsesObservedRefusal {
                            event_type: "message.refusal".into(),
                            item_id: item_id.clone(),
                            text: text.into(),
                            sequence_number: None,
                        },
                    );
                }
            }
        } else if item_type != "reasoning" {
            push_unique_tool(tools, lossy_tool_from_item(item, item_type));
        }
    }
}

fn push_unique_tool(
    tools: &mut Vec<OpenAIResponsesObservedToolEvent>,
    tool: OpenAIResponsesObservedToolEvent,
) {
    if !tools.contains(&tool) {
        tools.push(tool);
    }
}

fn push_unique_refusal(
    refusals: &mut Vec<OpenAIResponsesObservedRefusal>,
    refusal: OpenAIResponsesObservedRefusal,
) {
    if !refusals.contains(&refusal) {
        refusals.push(refusal);
    }
}

fn normalize_responses_usage(value: Option<&Value>) -> Usage {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Usage::missing();
    };
    let Some(object) = value.as_object() else {
        return Usage::default().reported_invalid();
    };
    let Some(input_tokens) = usage_u64(object, "input_tokens") else {
        return Usage::default().reported_invalid();
    };
    let Some(output_tokens) = usage_u64(object, "output_tokens") else {
        return Usage::default().reported_invalid();
    };
    let Some(total_tokens) = usage_u64(object, "total_tokens") else {
        return Usage::default().reported_invalid();
    };
    let Some(expected_total) = input_tokens.checked_add(output_tokens) else {
        return Usage::default().reported_invalid();
    };
    if total_tokens != expected_total {
        return Usage::default().reported_invalid();
    }

    let input_details = match optional_usage_details(object, "input_tokens_details") {
        Ok(details) => details,
        Err(()) => return Usage::default().reported_invalid(),
    };
    let output_details = match optional_usage_details(object, "output_tokens_details") {
        Ok(details) => details,
        Err(()) => return Usage::default().reported_invalid(),
    };
    let cached_input_tokens = match optional_usage_u64(input_details, "cached_tokens") {
        Ok(value) => value,
        Err(()) => return Usage::default().reported_invalid(),
    };
    let cache_creation_input_tokens = match optional_usage_u64(input_details, "cache_write_tokens")
    {
        Ok(value) => value,
        Err(()) => return Usage::default().reported_invalid(),
    };
    let reasoning_tokens = match optional_usage_u64(output_details, "reasoning_tokens") {
        Ok(value) => value,
        Err(()) => return Usage::default().reported_invalid(),
    };

    let Ok(input_tokens) = u32::try_from(input_tokens) else {
        return Usage::default().reported_invalid();
    };
    let Ok(output_tokens) = u32::try_from(output_tokens) else {
        return Usage::default().reported_invalid();
    };
    let Ok(cached_input_tokens) = u32::try_from(cached_input_tokens) else {
        return Usage::default().reported_invalid();
    };
    let Ok(cache_creation_input_tokens) = u32::try_from(cache_creation_input_tokens) else {
        return Usage::default().reported_invalid();
    };
    let Ok(reasoning_tokens) = u32::try_from(reasoning_tokens) else {
        return Usage::default().reported_invalid();
    };

    Usage {
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_creation_input_tokens,
        reasoning_tokens,
        ..Usage::default()
    }
    .reported()
}

fn usage_u64(object: &Map<String, Value>, field: &str) -> Option<u64> {
    object.get(field).and_then(Value::as_u64)
}

fn optional_usage_u64(object: Option<&Map<String, Value>>, field: &str) -> Result<u64, ()> {
    match object.and_then(|object| object.get(field)) {
        None | Some(Value::Null) => Ok(0),
        Some(value) => value.as_u64().ok_or(()),
    }
}

fn optional_usage_details<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<Option<&'a Map<String, Value>>, ()> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(details)) => Ok(Some(details)),
        Some(_) => Err(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{FutureExt, stream};

    #[test]
    fn digest_serialization_matches_rfc_8785_utf16_property_order() {
        let value = json!({
            "\u{20ac}": "Euro Sign",
            "\r": "Carriage Return",
            "\u{fb33}": "Hebrew Letter Dalet With Dagesh",
            "1": "One",
            "\u{1f600}": "Emoji: Grinning Face",
            "\u{80}": "Control",
            "\u{f6}": "Latin Small Letter O With Diaeresis"
        });

        let actual = String::from_utf8(canonical_json_bytes(&value).expect("RFC 8785 JCS"))
            .expect("canonical JSON is UTF-8");
        let expected = concat!(
            "{\"\\r\":\"Carriage Return\",",
            "\"1\":\"One\",",
            "\"\u{80}\":\"Control\",",
            "\"\u{f6}\":\"Latin Small Letter O With Diaeresis\",",
            "\"\u{20ac}\":\"Euro Sign\",",
            "\"\u{1f600}\":\"Emoji: Grinning Face\",",
            "\"\u{fb33}\":\"Hebrew Letter Dalet With Dagesh\"}"
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn digest_serialization_matches_rfc_8785_number_vector() {
        let value: Value = serde_json::from_str(
            r#"{"numbers":[333333333.33333329,1E30,4.50,2e-3,0.000000000000000000000000001]}"#,
        )
        .expect("RFC number fixture");

        let actual = String::from_utf8(canonical_json_bytes(&value).expect("RFC 8785 JCS"))
            .expect("canonical JSON is UTF-8");

        assert_eq!(
            actual,
            r#"{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27]}"#
        );
    }

    #[test]
    fn terminal_event_completes_before_a_permanently_pending_tail() {
        let terminal = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_first_terminal",
                "object": "response",
                "status": "completed",
                "service_tier": "default",
                "model": "gpt-test",
                "output": [],
                "usage": {
                    "input_tokens": 0,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 0
                    },
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 0
                }
            }
        })
        .to_string();
        let events = stream::once(async { Ok::<_, std::io::Error>(terminal) })
            .chain(stream::pending::<Result<String, std::io::Error>>());
        let mut attempt = OpenAIResponsesAttemptReceipt {
            schema_version: PROVIDER_RECEIPT_VERSION,
            provider_identity: OpenAIResponsesProviderIdentity::configured("https://example.test"),
            endpoint: "/v1/responses".into(),
            projection_digest: "sha256:projection".into(),
            controls_digest: Some("sha256:controls".into()),
            transport_policy_digest: "sha256:transport".into(),
            state: AttemptState::OutcomeUnknown,
            status_code: Some(200),
            request_id: Some("req_test".into()),
            response_id: None,
            returned_model: None,
            terminal_status: None,
            terminal_usage: None,
            returned_service_tier: None,
            provider_error: None,
            incomplete_reason: None,
            returned_reasoning: None,
            observed_tool_events: Vec::new(),
            observed_refusals: Vec::new(),
        };

        let result = parse_stream_event_data(events, &mut attempt)
            .now_or_never()
            .expect("the first terminal event must complete without polling the pending tail")
            .expect("valid terminal event");

        assert_eq!(result.response_id, "resp_first_terminal");
        assert_eq!(result.sequence_number, Some(1));
        assert_eq!(
            result.usage.provenance,
            crate::UsageProvenance::ReportedValid
        );
    }
}
