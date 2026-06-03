# llm_adapter 系统设计

> 日期：2026-06-03
> 状态：待实施

## 1. 背景与目标

设计一个 Rust 公开库（发布至 crates.io），提供统一的 LLM Provider 适配层。

**范围**：纯 wire 转换，不含 agent loop、retry 逻辑、conversation 持久化或 prompt 模板。

**首批 Provider**：OpenAI Chat Completions、Anthropic Messages、DeepSeek（复用 OpenAI 协议）。

**核心目标**：
- Canonical 类型系统自洽，语义来自"与 LLM 对话"这个领域，不锚定任何 provider
- 运行时动态切换 provider（`Box<dyn Provider>`）
- Provider 特有功能通过 native path 组合扩展，不污染 canonical 层
- 离线可测试，CI 无需真实 API key

---

## 2. 整体架构

### 2.1 分层

```
┌─────────────────────────────────────────┐
│           canonical 层（自洽）           │
│  types/：Message, Content, ChatRequest, │
│          ChatResponse, StreamEvent,     │
│          SamplingParams, StopReason,    │
│          Usage, LlmError               │
│  provider.rs：Provider trait            │
│  stream_handle.rs：StreamHandle         │
├──────────────┬──────────────────────────┤
│  openai/ 层  │      anthropic/ 层       │
│  wire types  │      wire types          │
│  OpenAIExt   │      AnthropicExt        │
│  OpenAIReq   │      AnthropicRequest    │
└──────────────┴──────────────────────────┘
```

### 2.2 目录结构

```
llm_adapter/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # 公开 API 总入口
│   ├── error.rs                # LlmError
│   ├── types/
│   │   ├── mod.rs
│   │   ├── content.rs          # RequestContent / ResponseContent
│   │   ├── message.rs          # Message ADT
│   │   ├── request.rs          # ChatRequest / SamplingParams / ToolDef / ToolChoice
│   │   ├── response.rs         # ChatResponse / StopReason / Usage
│   │   └── stream.rs           # StreamEvent / ContentKind
│   ├── provider.rs             # Provider trait
│   ├── stream_handle.rs        # StreamHandle
│   ├── openai/
│   │   ├── mod.rs              # OpenAIProvider + builder
│   │   ├── schema.rs           # OAI wire types（pub(crate)）
│   │   ├── convert.rs          # ChatRequest ↔ OAI schema
│   │   └── stream.rs           # OAI delta → StreamEvent
│   ├── anthropic/
│   │   ├── mod.rs              # AnthropicProvider + builder
│   │   ├── schema.rs           # Anthropic wire types（pub(crate)）
│   │   ├── convert.rs          # ChatRequest / AnthropicRequest ↔ Anthropic schema
│   │   └── stream.rs           # Anthropic events → StreamEvent
│   └── deepseek.rs             # 便利工厂，返回预配置 OpenAIProvider
└── tests/
    ├── fixtures/
    │   ├── openai/
    │   ├── anthropic/
    │   └── deepseek/
    ├── common/
    │   ├── mock_server.rs
    │   └── fixtures.rs
    ├── openai_chat.rs
    ├── anthropic_chat.rs
    ├── deepseek_chat.rs
    ├── openai_stream.rs
    ├── anthropic_stream.rs
    └── deepseek_stream.rs
```

---

## 3. Canonical 类型系统

### 3.1 内容类型

```rust
// 用户 / 工具产生的内容
pub enum RequestContent {
    Text(String),
    Image(ImageSource),
}

pub enum ImageSource {
    Url(String),
    Base64 { media_type: String, data: String },
}

pub struct ToolInvocation {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

// 模型产生的内容
pub enum ResponseContent {
    Text(String),
    ToolInvocation(ToolInvocation),
    Reasoning(String),
}
```

`ResponseContent::Reasoning` 在 canonical 层为**只读**：可从响应中读取推理文本，但不保证所有 provider 在多轮时能正确回传（Anthropic 需要 signature，走 native path；DeepSeek 无 signature 需求，universal path 可正常多轮）。

### 3.2 消息 ADT

```rust
pub enum Message {
    System(String),
    User(Vec<RequestContent>),
    Assistant(Vec<ResponseContent>),
    Tool {
        invocation_id: String,
        content: Vec<RequestContent>,
        is_error: bool,
    },
}
```

角色编码进 variant，编译器在各角色上强制正确的内容类型。`Reasoning` 只能出现在 `Assistant` variant，放入其他 variant 直接编译报错。

### 3.3 请求与采样参数

```rust
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub tool_choice: ToolChoice,
    pub params: SamplingParams,
}

pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub max_tokens: u32,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
}

pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

pub enum ToolChoice {
    Auto,
    None,
    Required,
    Specific(String),
}
```

`SamplingParams` 仅含三家真正共有的参数。`top_k`、`seed`、`frequency_penalty` 等 provider 特有参数归入各自扩展类型。

`ChatRequest`、`SamplingParams`、`AnthropicExt`、`OpenAIExt` 均通过 **builder pattern** 构造，不暴露 struct literal 构造方式。新增可选字段时只需加新 builder 方法，不产生 breaking change。

### 3.4 响应

```rust
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ResponseContent>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

impl ChatResponse {
    pub fn into_message(self) -> Message;       // → Message::Assistant(self.content)
    pub fn text(&self) -> String;               // 拼接所有 Text 块
    pub fn tool_invocations(&self) -> impl Iterator<Item = &ToolInvocation>;
    pub fn reasoning(&self) -> Option<&str>;
}

pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    ContentFilter,
    Other(String),
}

pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_input_tokens: u32,
    pub cache_creation_input_tokens: u32,  // Anthropic 独有，其他家为 0
    pub reasoning_tokens: u32,             // DeepSeek 独有，其他家为 0
}
```

### 3.5 流式类型

```rust
pub enum ContentKind {
    Text,
    Reasoning,
    ToolInvocation { id: String, name: String },  // 流式开始时 input 尚未完整，单独 delta
}

pub enum StreamEvent {
    ContentStart  { index: usize, kind: ContentKind },
    TextDelta     { index: usize, text: String },
    ReasoningDelta{ index: usize, text: String },
    ToolDelta     { index: usize, arguments: String },
    ContentStop   { index: usize },
    MessageStop   { stop_reason: StopReason, usage: Usage },
}
```

---

## 4. Provider Trait 与动态分发

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;
    async fn chat_stream(&self, req: &ChatRequest) -> Result<StreamHandle, LlmError>;
}
```

使用 `async-trait` crate 实现 dyn-compatible，支持 `Box<dyn Provider>`。每次 async 调用有一次 heap allocation，在网络 I/O 场景下可忽略。

### StreamHandle（两层流式）

```rust
pub struct StreamHandle {
    inner: BoxStream<'static, Result<StreamEvent, LlmError>>,
}

impl StreamHandle {
    // 底层：逐事件迭代
    pub fn events(&mut self) -> &mut impl Stream<Item = Result<StreamEvent, LlmError>>;
    // 上层：折叠成完整响应
    pub async fn collect(self) -> Result<ChatResponse, LlmError>;
}
```

---

## 5. Provider 特有类型（Native Path）

### 5.1 Anthropic

```rust
pub enum AnthropicImageSource {
    Url(String),
    Base64 { media_type: String, data: String },
}

pub enum AnthropicContent {
    Text(String),
    Image(AnthropicImageSource),
    ToolUse    { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: Vec<AnthropicContent>, is_error: bool },
    Thinking   { text: String, signature: String },
    RedactedThinking { data: String },
}

// 多轮 native 对话的消息单元（角色仅 User / Assistant，System → 顶层字段）
pub enum AnthropicMessage {
    User(Vec<AnthropicContent>),
    Assistant(Vec<AnthropicContent>),
}

pub struct CacheHint {
    pub target: CacheTarget,   // 标记哪条消息 / 哪个 content block
    pub ttl: CacheTtl,         // FiveMin | OneHour
}

pub struct ThinkingConfig {
    pub budget_tokens: u32,
}

pub struct AnthropicMetadata {
    pub user_id: Option<String>,
}

pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_input_tokens: u32,
    pub cache_creation_input_tokens: u32,
}

pub struct AnthropicExt {
    pub top_k: Option<u32>,
    pub disable_parallel_tool_use: bool,
    pub cache_hints: Vec<CacheHint>,
    pub thinking: Option<ThinkingConfig>,
    pub metadata: Option<AnthropicMetadata>,
}

pub struct AnthropicRequest {
    pub base: ChatRequest,
    pub ext: AnthropicExt,
}

pub struct AnthropicResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<AnthropicContent>,
    pub stop_reason: StopReason,
    pub usage: AnthropicUsage,
}

impl AnthropicResponse {
    pub fn into_message(self) -> AnthropicMessage;   // signature 完整保留，用于多轮
    pub fn into_canonical(self) -> ChatResponse;     // Reasoning 保留文本，signature 丢弃
}

impl AnthropicProvider {
    pub async fn chat_native(&self, req: &AnthropicRequest)
        -> Result<AnthropicResponse, LlmError>;
    pub async fn chat_stream_native(&self, req: &AnthropicRequest)
        -> Result<StreamHandle, LlmError>;
}
```

### 5.2 OpenAI

```rust
pub struct OpenAIExt {
    pub seed: Option<u64>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub user: Option<String>,
    pub parse_reasoning_content: bool,   // DeepSeek 启用
    pub tolerant_keepalive: bool,        // DeepSeek 思考模式排队时的 SSE keepalive
}

pub struct OpenAIRequest {
    pub base: ChatRequest,
    pub ext: OpenAIExt,
}

// OpenAI native response 与 canonical 结构相同，仅在字段映射上有差异
// 目前不单独定义 OpenAIResponse；native path 直接返回 ChatResponse
impl OpenAIProvider {
    pub async fn chat_native(&self, req: &OpenAIRequest)
        -> Result<ChatResponse, LlmError>;
    pub async fn chat_stream_native(&self, req: &OpenAIRequest)
        -> Result<StreamHandle, LlmError>;
}
```

### 5.3 DeepSeek（工厂，非新 Provider）

```rust
// src/deepseek.rs
pub fn client(api_key: impl Into<String>) -> OpenAIProvider {
    OpenAIProvider::builder()
        .api_key(api_key)
        .base_url("https://api.deepseek.com")
        .parse_reasoning_content(true)
        .tolerant_keepalive(true)
        .build()
}
```

### 5.4 两条路径对比

| | Universal path | Native path |
|---|---|---|
| 入口 | `Box<dyn Provider>` | 具体类型 |
| 请求 | `ChatRequest` | `AnthropicRequest` / `OpenAIRequest` |
| 响应 | `ChatResponse` | `AnthropicResponse` / `ChatResponse`（OpenAI） |
| Reasoning 多轮 | Anthropic: strip；DeepSeek: 正常 | ✅ 完整 |
| cache_hints / top_k | ❌ | ✅ |
| 运行时切换 provider | ✅ | ❌ |

---

## 6. 错误模型

```rust
pub enum LlmError {
    InvalidRequest(String),
    Unauthorized,
    Forbidden,
    RateLimit   { retry_after: Option<Duration> },
    Overloaded  { retry_after: Option<Duration> },
    ServerError(String),
    Timeout,
    Stream(String),
    Network(reqwest::Error),
    Decode(serde_json::Error),
    Provider    { code: String, message: String },
}
```

错误归一化后不内置 retry，由调用方自行处理。`RateLimit` 和 `Overloaded` 暴露 `retry_after` 供上层决策。

---

## 7. 技术选型

| 职责 | 选型 | 理由 |
|---|---|---|
| HTTP 客户端 | `reqwest` | 成熟稳定，tokio 原生 |
| async trait dyn | `async-trait` | 支持 `Box<dyn Provider>`，广泛验证 |
| SSE 解析 | `eventsource-stream` | 避免手写协议，处理 keepalive 边界 |
| 序列化 | `serde` + `serde_json` | 标准 |
| mock 测试 | `wiremock` | 离线 fixture 回放 |
| Rust edition | 2024 | 最新 stable（Rust 1.95） |

---

## 8. 测试策略

### 离线测试（默认）

- `cargo test` 无需任何环境变量
- 所有测试走 `wiremock` mock server，fixture 文件提交进 git
- 错误场景（429、529、500、流式断连）通过 mock 注入

### 集成测试（可选）

```bash
cargo test --features integration
```

每家 provider 支持三个环境变量：

| 变量 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | API key |
| `OPENAI_BASE_URL` | ❌ | `https://api.openai.com` | 可指向代理或本地服务 |
| `OPENAI_MODEL` | ❌ | `gpt-4o` | 测试用模型名 |
| `ANTHROPIC_API_KEY` | ✅ | — | API key |
| `ANTHROPIC_BASE_URL` | ❌ | `https://api.anthropic.com` | 可指向代理或本地服务 |
| `ANTHROPIC_MODEL` | ❌ | `claude-sonnet-4-6` | 测试用模型名 |
| `DEEPSEEK_API_KEY` | ✅ | — | API key |
| `DEEPSEEK_BASE_URL` | ❌ | `https://api.deepseek.com` | 可指向代理或本地服务 |
| `DEEPSEEK_MODEL` | ❌ | `deepseek-v4-flash` | 测试用模型名 |

未设置 `*_API_KEY` 的 provider，其集成测试自动跳过（不报错）。

### 每家 Provider 必须覆盖的场景

| 场景 | 离线 | 集成 |
|---|---|---|
| 普通 chat | ✅ | ✅ |
| 多轮对话 | ✅ | ✅ |
| Tool call 单轮 | ✅ | ✅ |
| Tool call 多轮 | ✅ | ✅ |
| 流式 chat | ✅ | ✅ |
| 流式 tool call | ✅ | ✅ |
| Reasoning 输出 | ✅ | ✅ |
| 错误归一化 | ✅ | — |
| 流式中途断连 | ✅ | — |

### 开发节奏

1. 录制真实 API 响应存为 fixture
2. 写测试，断言 canonical 类型转换结果
3. 写实现让测试通过
4. 集成测试最终验收

---

## 9. MVP 不覆盖

- Retry / backoff 逻辑（调用方自理）
- Prompt Caching 主动触发（P2，通过 AnthropicExt 的 `cache_hints` 支持）
- Structured Output / JSON Schema 强约束
- Batch API
- Embedding / TTS / Image generation
- Stateful API（OpenAI Responses API）
- DeepSeek Anthropic 兼容入口
