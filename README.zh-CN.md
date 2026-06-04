# llm_adapter

*[English](README.md)*

一个 Rust 编写的 LLM provider 适配层：把多家 provider（OpenAI Chat Completions、Anthropic Messages、DeepSeek、以及任何 OpenAI 兼容的 endpoint）的 wire 格式归一化到统一的 canonical 类型系统，原生支持流式、tool calling，并为 provider 特有功能保留独立的扩展通道。

**状态：** 预发布版本（v0.1.0）。在 v1.0 之前，公开 API 可能变动。

## 这个库是什么

一个目标聚焦的小型库，提供：

- 一套与具体 provider 无关的 canonical 请求/响应类型
- 一个 `Provider` trait，可作为 `Box<dyn Provider>` 持有，运行时切换 provider
- 流式作为一等类型（`StreamHandle`），可按事件迭代，也可折叠成完整响应
- 一个扩展通道（`OpenAIExt` / `extra_body`），让 provider 特有字段不污染 canonical 层
- 跨 provider 归一化的错误类型（`Unauthorized`、`RateLimit`、`Overloaded` 等），可获取 `retry_after`
- 离线可测试：所有离线测试基于从真实 API 录制的 `wiremock` fixture，`cargo test` 无需网络也无需 API key

## 这个库不做什么

为了让接口边界清晰，下列功能**不包含**在本库范围内：

- Retry / backoff（由调用方决定）
- Agent loop、prompt 模板、对话历史持久化
- Structured Output / JSON Schema 强约束
- Batch API、embedding、TTS、图像生成
- Azure OpenAI（认证方式不同，路径含 deployment ID）
- OpenAI 的有状态 Responses API

## 支持的 provider

| Provider | Universal 路径 | Native 扩展 | 流式 | Tool calling | Reasoning |
|---|---|---|---|---|---|
| OpenAI Chat Completions | ✅ | `OpenAIExt`：seed、penalties、`reasoning_effort`、`extra_body` | ✅ | ✅ | `reasoning_effort`（o 系列 / GPT-5） |
| DeepSeek（`api.deepseek.com`） | ✅（工厂函数 `deepseek::client`） | `OpenAIExt`（如 `extra_body: {"thinking": {"type": "disabled"}}`） | ✅ | ✅ | `reasoning_content` 解析为 `ResponseContent::Reasoning` |
| Anthropic Messages | ✅ | `AnthropicExt`：top_k、thinking、metadata、cache_hints | ✅ | ✅ | Extended Thinking（thinking 块 → `ResponseContent::Reasoning`） |
| 任意 OpenAI 兼容网关 | ✅，通过 `base_url` + `chat_path` 覆盖 | `extra_body` 注入非标字段 | ✅ | ✅ | 取决于 endpoint |

已经验证可通过路径覆盖支持的网关：GitHub Models（路径为 `/chat/completions`，无 `/v1`）、OpenRouter（路径为 `/api/v1/...`）、Groq（路径为 `/openai/v1/...`）。库本身不为这些网关做任何特殊处理，它们能工作纯粹是因为协议就是 OpenAI Chat Completions。

## 快速上手

`Cargo.toml`：

```toml
[dependencies]
llm_adapter = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

### 简单对话

```rust
use llm_adapter::{openai::OpenAIProvider, types::*, Provider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let p = OpenAIProvider::builder(std::env::var("OPENAI_API_KEY")?)
        .build();

    let req = ChatRequest::builder("gpt-4o-mini", 512)
        .message(Message::System("回答尽量简洁。".into()))
        .message(Message::User(vec![RequestContent::Text(
            "用五个字打个招呼。".into(),
        )]))
        .temperature(0.7)
        .build();

    let resp = p.chat(&req).await?;
    println!("{}", resp.text());
    println!("tokens: {} 入 / {} 出", resp.usage.input_tokens, resp.usage.output_tokens);
    Ok(())
}
```

### 一行工厂函数构造 DeepSeek 客户端

```rust
use llm_adapter::{deepseek, types::*, Provider};

let p = deepseek::client(std::env::var("DEEPSEEK_API_KEY")?);
// 返回一个预配置好的 OpenAIProvider：
//   - base_url:                https://api.deepseek.com
//   - parse_reasoning_content: 已启用
//   - tolerant_keepalive:      已启用

let req = ChatRequest::builder("deepseek-v4-flash", 2048)
    .message(Message::User(vec![RequestContent::Text(
        "7 乘 8 等于多少？".into(),
    )]))
    .build();

let resp = p.chat(&req).await?;
println!("答案：{}", resp.text());
if let Some(reasoning) = resp.reasoning() {
    println!("（思考过程：{}）", reasoning);
}
```

### 运行时切换 provider

```rust
use llm_adapter::{anthropic::AnthropicProvider, deepseek, openai::OpenAIProvider, types::*, Provider};

let provider: Box<dyn Provider> = if use_deepseek {
    Box::new(deepseek::client(deepseek_key))
} else if use_anthropic {
    Box::new(AnthropicProvider::builder(anthropic_key).build())
} else {
    Box::new(OpenAIProvider::builder(openai_key).build())
};

// 调用方代码完全一致：
let resp = provider.chat(&req).await?;
```

### 流式

```rust
use futures::StreamExt;
use llm_adapter::{types::*, Provider};

let mut handle = provider.chat_stream(&req).await?;

while let Some(event) = handle.events().next().await {
    match event? {
        StreamEvent::TextDelta { text, .. } => print!("{text}"),
        StreamEvent::ReasoningDelta { text, .. } => eprint!("[think]{text}"),
        StreamEvent::ToolDelta { arguments, .. } => print!("[tool-args]{arguments}"),
        StreamEvent::MessageStop { stop_reason, usage } => {
            println!("\n-- stop={stop_reason:?} usage={usage:?}");
        }
        _ => {}
    }
}
```

或者，跳过逐事件处理，直接折叠成完整响应：

```rust
let handle = provider.chat_stream(&req).await?;
let resp = handle.collect().await?;
println!("{}", resp.text());
```

### Tool calling

```rust
use llm_adapter::types::*;
use serde_json::json;

let weather_tool = ToolDef {
    name: "get_weather".into(),
    description: "获取某个城市的当前天气。".into(),
    input_schema: json!({
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    }),
};

let req = ChatRequest::builder("gpt-4o-mini", 512)
    .message(Message::User(vec![RequestContent::Text(
        "巴黎现在天气怎么样？".into(),
    )]))
    .tool(weather_tool.clone())
    .build();

let resp = provider.chat(&req).await?;
if matches!(resp.stop_reason, StopReason::ToolUse) {
    let mut messages = req.messages().to_vec();
    let invocation = resp.tool_invocations().next().unwrap().clone();
    messages.push(resp.into_message());
    messages.push(Message::Tool {
        invocation_id: invocation.id,
        content: vec![RequestContent::Text("晴，22°C".into())],
        is_error: false,
    });

    let follow_up = ChatRequest::builder("gpt-4o-mini", 512)
        .messages(messages)
        .tool(weather_tool)
        .build();
    let final_resp = provider.chat(&follow_up).await?;
    println!("{}", final_resp.text());
}
```

### Provider 特有扩展（native 路径）

通用的 `Provider::chat` 路径是 schema 中立的。当你需要传入 provider 特有字段——例如 OpenAI 的 `seed`、`reasoning_effort`，或某个 OpenAI 兼容 endpoint 接受的非标字段——走 native 路径：

```rust
use llm_adapter::openai::{OpenAIExt, OpenAIRequest};
use serde_json::json;

let ext = OpenAIExt::builder()
    .seed(42)
    .reasoning_effort("low")
    // DeepSeek V4 默认开启 thinking 模式，且在 thinking 启用时拒绝
    // tool_choice=Required。通过 extra_body 显式关闭：
    .extra_body(json!({"thinking": {"type": "disabled"}}))
    .build();

let oai_req = OpenAIRequest { base: chat_req, ext };
let resp = openai_provider.chat_native(&oai_req).await?;
let stream = openai_provider.chat_stream_native(&oai_req).await?;
```

Anthropic native 路径同理：

```rust
use llm_adapter::anthropic::{AnthropicExt, AnthropicMetadata, AnthropicRequest, ThinkingConfig};

let ext = AnthropicExt::builder()
    .top_k(40)
    .thinking(ThinkingConfig { budget_tokens: 2000 })
    .metadata(AnthropicMetadata { user_id: Some("my-user".into()) })
    .disable_parallel_tool_use()
    .build();

let anth_req = AnthropicRequest { base: chat_req, ext };
let resp = anthropic_provider.chat_native(&anth_req).await?;
let stream = anthropic_provider.chat_stream_native(&anth_req).await?;
```

## 核心概念

### Canonical 类型

类型语义来自"和 LLM 对话"这件事本身，不锚定任何 provider 的 wire 格式。角色被编码进 `Message` 的 enum variant，由类型系统确保每个角色只能携带正确的内容：

```rust
pub enum Message {
    System(String),
    User(Vec<RequestContent>),                            // 文本 + 图片
    Assistant(Vec<ResponseContent>),                      // 文本 + tool call + reasoning
    Tool { invocation_id: String, content: Vec<RequestContent>, is_error: bool },
}

pub enum RequestContent  { Text(String), Image(ImageSource) }
pub enum ResponseContent { Text(String), ToolInvocation(ToolInvocation), Reasoning(String) }
pub enum StopReason      { EndTurn, MaxTokens, StopSequence, ToolUse, ContentFilter, Other(String) }
```

把 `Reasoning` 放进 `Message::User` 是编译错误。

### Builder 模式

每个可构造的类型（`ChatRequest`、`SamplingParams`、`OpenAIExt`）都暴露 builder，字段保持私有。未来添加可选字段不构成 breaking change。

### Universal 路径 vs Native 路径

|  | `Provider::chat`（universal） | `chat_native`（native） |
|---|---|---|
| 请求类型 | `&ChatRequest` | `&OpenAIRequest` / `&AnthropicRequest = { base: ChatRequest, ext }` |
| Object-safe（`Box<dyn>`） | ✅ | ❌（具体 provider 类型） |
| Provider 特有字段 | ❌ | ✅ seed、penalties、top_k、thinking、cache_hints、extra_body |
| 适用场景 | 运行时可插拔 provider | 需要 provider 特有功能 |

### StreamHandle

```rust
impl StreamHandle {
    pub fn events(&mut self) -> &mut impl Stream<Item = Result<StreamEvent, LlmError>>;
    pub async fn collect(self) -> Result<ChatResponse, LlmError>;
}
```

需要"边到边渲染"的交互体验，用 `events()`。只要拿最终的 `ChatResponse`、不想自己写聚合逻辑，用 `collect()`。

### 错误

`LlmError` 是跨 provider 归一化的错误类型：

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
    Provider    { code: String, message: String },   // 402 / 未知状态码
}
```

库内不做 retry。`RateLimit` 和 `Overloaded` 会从 `Retry-After` header 提取 `retry_after`，由调用方决策何时重试。

## 连接 OpenAI 兼容网关

任何遵循 OpenAI Chat Completions schema 的 endpoint 都能接入，通过 builder 的三个旋钮配置：

```rust
OpenAIProvider::builder(key)
    .base_url("https://models.github.ai/inference")   // 任意 host
    .chat_path("/chat/completions")                   // 任意路径（默认 /v1/chat/completions）
    .build()
```

- **base_url** — scheme + host，可选附带固定路径前缀；尾部斜杠会被 trim 后再拼上 `chat_path`。
- **chat_path** — 当网关缺少或改写 `/v1` 前缀时使用。GitHub Models 需要 `/chat/completions`；OpenRouter 需要 `/api/v1/chat/completions`（所以 `base_url` 设为 `https://openrouter.ai/api`）。
- **extra_body**（位于 `OpenAIExt`） — 顶层注入 canonical schema 之外的 JSON 字段。对标 OpenAI Python SDK 的 `extra_body` 参数。

**兼容度估算：** 大约 70% 的"OpenAI 兼容"厂商开箱即用（DeepSeek、Together、Mistral、Cerebras、Moonshot、Ollama、LM Studio 等等）；其余通过 `chat_path` 或 `extra_body` 也能跑通。**Azure OpenAI 不支持** —— 它使用 `api-key` header 而非 `Bearer`，且路径里含每个 deployment 的 ID。

## 测试

### 离线测试（默认）

```
./test.sh              # 等价于 cargo test
```

运行约 75 个单元和离线集成测试，无需网络也无需 API key。fixture 文件位于 `tests/fixtures/{openai,deepseek,anthropic}/`，由真实 API 录制，因此 wire 层的细节（`system_fingerprint`、`completion_tokens_details.reasoning_tokens`、SSE chunk 边界等）都符合生产行为。

`test.sh` 会在调用 cargo test 前设 `NO_PROXY=127.0.0.1,localhost`，避免 macOS 系统 HTTP 代理拦截 wiremock 的本地服务。

### 集成测试（可选）

```
./test.sh --features integration
```

打真实 provider API。通过环境变量配置（如存在 `.env` 文件，会自动加载）：

```bash
# OpenAI / 任意 OpenAI 兼容 endpoint
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com         # 默认值
export OPENAI_CHAT_PATH=/v1/chat/completions          # 默认值
export OPENAI_MODEL=gpt-4o-mini                       # 默认值
export OPENAI_REASONING_MODEL=o4-mini                 # 可选；未设则跳过相关测试

# DeepSeek
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=https://api.deepseek.com     # 默认值
export DEEPSEEK_MODEL=deepseek-v4-flash               # 默认值
export DEEPSEEK_REASONING_MODEL=deepseek-v4-flash     # 可选

# Anthropic（也可以指向 DeepSeek 的 /anthropic 入口）
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=https://api.anthropic.com    # 默认值
export ANTHROPIC_MESSAGES_PATH=/v1/messages            # 默认值
export ANTHROPIC_MODEL=claude-sonnet-4-6-20250514      # 默认值
```

未设 `*_API_KEY` 的 provider，其集成测试会静默跳过（不报错）。其中一个测试专门验证 `extra_body + tool_choice: Required` 组合在真实 DeepSeek 上能跑通，证明 provider 特有扩展通道实际可用。

### 重新录制 fixture

当上游 API 行为变更，重新生成 fixture：

```
cargo run --example record_fixtures
```

工具会读取 `.env`，按固定 prompt 调每家 provider，把响应原样写入 `tests/fixtures/`（保留服务端字段顺序、未文档化字段、SSE chunk 边界）。

## 目录结构

```
src/
├─ lib.rs                # 公开 API 再导出
├─ error.rs              # LlmError
├─ types/                # canonical 类型：Message、ChatRequest、ChatResponse、StreamEvent 等
├─ provider.rs           # Provider trait
├─ stream_handle.rs      # StreamHandle + collect()
├─ openai/
│  ├─ mod.rs             # OpenAIProvider + builder + merge_extra_body
│  ├─ schema.rs          # OpenAI wire 类型（pub(crate)）
│  ├─ convert.rs         # ChatRequest <-> OAI 转换，OpenAIExt builder
│  └─ stream.rs          # SSE 解析器、OaiStreamState
├─ anthropic/
│  ├─ mod.rs             # AnthropicProvider + builder
│  ├─ schema.rs          # Anthropic wire 类型（pub(crate)）
│  ├─ convert.rs         # ChatRequest <-> Anthropic 转换，AnthropicExt builder
│  └─ stream.rs          # Anthropic SSE 解析器、AnthropicStreamState
└─ deepseek.rs           # 工厂函数，返回预配置好的 OpenAIProvider

tests/
├─ fixtures/             # 录制的 API 响应（JSON + SSE）
├─ common/mod.rs         # wiremock helper + .env 加载器
├─ openai_*.rs           # 离线测试
├─ deepseek_*.rs         # 离线测试
├─ anthropic_*.rs        # 离线测试
└─ integration_*.rs      # 可选的真实 API 测试

examples/
└─ record_fixtures.rs    # 从真实 provider 重新录制 fixture
```

## 已知限制

- **Azure OpenAI** 不支持（认证 header 不同 + 基于 deployment 的路径）。
- **OpenAI Responses API** 不支持；仅 Chat Completions。
- **Prompt caching**：可以从 `Usage.cached_input_tokens` 读到命中情况，但 canonical 层暂不提供主动触发。可以通过 native 路径（`AnthropicExt.cache_hints`）添加 cache-control 标记。
- **Retry / backoff** 未内建。请基于 `LlmError::RateLimit` / `LlmError::Overloaded` 中的 `retry_after` 在应用层处理。
- **Structured Output**（`response_format: json_schema`）暂未在 canonical 层暴露。当前可通过 `OpenAIExt.extra_body` 传递。

## License

MIT。详见 [LICENSE](LICENSE)。
