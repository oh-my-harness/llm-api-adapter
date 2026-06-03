# 适配层归一化设计要点

> 这份文档是基于 01/02/03/05/06 整理出的**设计输入**，不是设计稿。下一阶段（系统设计）会基于这些要点产出 spec.md。

## 1. 范围声明（再次确认）

第一版（MVP）覆盖：

- ✅ 非流式 chat（含 system / multi-turn / multimodal text+image）
- ✅ 流式 chat
- ✅ Tool use（定义 / 调用 / 多轮回喂）
- ✅ 基础参数（temperature / max_tokens / top_p / stop_sequences）
- ✅ Usage 统计（含 cache 命中信息）
- ✅ 错误归一化（含 retry-after 信号）
- ✅ Reasoning / Thinking 字段保留（DeepSeek 多轮工具调用强依赖；Anthropic signature 同样要保留）
- ✅ 三家：OpenAI Chat Completions + Anthropic Messages + DeepSeek（走 OpenAI 协议）

第一版**不覆盖**（保留扩展点）：

- ❌ Streaming text 之外的复杂 content（embedding / image-gen / TTS）
- ❌ Prompt Caching 的**主动**触发能力（Anthropic 的 `cache_control` 块标记 —— 可作 P2）
- ❌ Structured Output / JSON Schema 强约束
- ❌ Multiple completions（n>1）
- ❌ Logprobs
- ❌ Hosted tools / server-side tools
- ❌ Stateful APIs（Responses 的 previous_response_id、Conversations API）
- ❌ Batch API
- ❌ Audio / PDF / 复杂文档输入
- ❌ DeepSeek 的 Anthropic 兼容入口（`/anthropic`，价值不大，走 OpenAI 入口即可）

## 2. 核心抽象方向

### 2.1 Provider trait

```rust
// 伪代码示意，不是最终设计
#[async_trait]
trait Provider {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, Error>;

    async fn chat_stream(&self, req: ChatRequest)
        -> Result<BoxStream<Result<StreamEvent, Error>>, Error>;
}
```

- 单 trait，两个方法：流式 / 非流式
- `ChatRequest` 是 provider-agnostic 的统一请求模型
- `StreamEvent` 是归一化的流事件 enum（详见 `06-streaming-deep-dive.md` § 9.1）

### 2.2 内部 Content 模型（向 Anthropic 看齐）

```rust
enum Content {
    Text(String),
    Image(ImageSource),
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: Vec<Content>, is_error: bool },
    Thinking { text: String, signature: Option<String> },  // Anthropic thinking block / DeepSeek reasoning
}

enum Role { User, Assistant }

struct Message {
    role: Role,
    content: Vec<Content>,
}

struct ChatRequest {
    model: String,
    system: Option<String>,
    messages: Vec<Message>,
    tools: Vec<ToolDef>,
    tool_choice: ToolChoice,
    params: SamplingParams,
    thinking: Option<ThinkingConfig>,    // 控制是否启用思考及 budget
    cache_hints: Vec<CacheHint>,          // Anthropic 用，其他家忽略
}
```

**理由：**
- OpenAI / DeepSeek 可以从 block 数组**降级**到 string（取第一个 Text 块）
- Anthropic 直接对应
- `Thinking` block 统一承载 Anthropic 的 thinking 块（含 signature）和 DeepSeek 的 reasoning_content（signature=None）
- 内部模型有 `Thinking` block 是为了支持**多轮回传**：DeepSeek V4 思考模式 + 工具调用必须回传 reasoning_content，Anthropic 必须回传 thinking + signature

### 2.3 Tool 定义（向 Anthropic 看齐）

```rust
struct ToolDef {
    name: String,
    description: String,
    input_schema: serde_json::Value,  // JSON Schema
}

enum ToolChoice {
    Auto,
    None,
    Required,                   // 必须调一个，不限定哪个
    Specific(String),           // 指定 tool name
}

struct SamplingParams {
    temperature: Option<f32>,
    max_tokens: u32,
    top_p: Option<f32>,
    stop_sequences: Vec<String>,
    disable_parallel_tools: bool,
}

struct ThinkingConfig {
    enabled: bool,
    budget_tokens: Option<u32>,  // Anthropic 需要；DeepSeek 用模型名 / extra_body 控制
}

struct CacheHint {
    /// 标记 messages 数组中某条消息或其某个 content block 缓存
    target: CacheTarget,
    ttl: CacheTtl,        // FiveMin | OneHour
}
```

### 2.4 Stream Event 模型

详见 `06-streaming-deep-dive.md` § 9.1。要点：

- 向 Anthropic 嵌套结构看齐（`BlockStart` / `Delta` / `BlockStop`）
- 增加 `ReasoningDelta` 承载 DeepSeek `reasoning_content` 和 Anthropic `thinking_delta`
- 增加 `ThinkingSignature` 承载 Anthropic 多轮所需的 signature
- OpenAI / DeepSeek 适配器要"合成"block 边界事件（扁平 delta → 嵌套 block）

### 2.5 Stop Reason 归一化

```rust
enum StopReason {
    EndTurn,           // OAI:"stop"           | Anth:"end_turn"      | DS:"stop"
    MaxTokens,         // OAI:"length"         | Anth:"max_tokens"    | DS:"length"
    StopSequence,      // OAI:"stop"+stop arg  | Anth:"stop_sequence" | DS:"stop"
    ToolUse,           // OAI:"tool_calls"     | Anth:"tool_use"      | DS:"tool_calls"
    ContentFilter,     // OAI:"content_filter" | Anth:"refusal"       | DS:"content_filter"
    Other(String),
}
```

### 2.6 错误归一化

```rust
enum LlmError {
    InvalidRequest { message: String },             // 400 / 422
    Unauthorized,                                     // 401
    Forbidden,                                        // 403
    NotFound,                                         // 404
    RequestTooLarge,                                  // 413 (Anth)
    InsufficientBalance,                              // 402 (DeepSeek)
    RateLimit { retry_after: Option<Duration> },    // 429
    Overloaded { retry_after: Option<Duration> },   // 529 (Anth) / 503 (OAI/DS)
    ServerError { message: String },                  // 500
    Timeout,                                          // 504 (Anth) / 客户端超时
    Stream(StreamError),                              // 流式特有：中途断流、JSON 拼接失败等
    Network(std::io::Error),
    Decode(serde_json::Error),
    Provider {                                        // fallback：未识别的 provider 错误
        code: String,
        message: String,
        raw: serde_json::Value,
    },
}
```

错误里保留 `request_id`，方便用户排错时贴给 provider 支持。

### 2.7 Usage

```rust
struct Usage {
    input_tokens: u32,
    output_tokens: u32,

    /// 缓存命中的输入 token 数。三家映射：
    ///   OpenAI: usage.prompt_tokens_details.cached_tokens
    ///   Anthropic: usage.cache_read_input_tokens
    ///   DeepSeek: usage.prompt_cache_hit_tokens
    cached_input_tokens: u32,

    /// Anthropic 独有：写入缓存的 token 数（计费比正常 input 高）
    /// OpenAI / DeepSeek 此字段为 0
    cache_creation_input_tokens: u32,

    /// 推理 token 数。来源：
    ///   OpenAI o-series: 隐式扣减自 output_tokens（不单独暴露，置 0）
    ///   Anthropic thinking: 算入 output_tokens
    ///   DeepSeek: 单独计费，可单独累加
    reasoning_tokens: u32,
}
```

### 2.8 Provider 与 Endpoint 解耦（DeepSeek 引入的关键）

DeepSeek 不是新协议，而是 OpenAI 协议的另一个 endpoint。Provider 抽象必须考虑这点：

```rust
// ❌ 错误设计：把 base_url 写死在 Provider 里
struct OpenAIProvider { api_key: String }   // 隐式硬编码 api.openai.com

// ✅ 正确设计：base_url 作为配置项
struct OpenAIProvider {
    api_key: String,
    base_url: Url,            // 默认 api.openai.com，可指向 DeepSeek 或自建网关
    feature_flags: ProtocolFlags, // 是否处理 reasoning_content 等扩展字段
}

struct ProtocolFlags {
    /// 解析 delta.reasoning_content（DeepSeek、Doubao 等都用）
    parse_reasoning_content: bool,
    /// 容忍大量 `:` keep-alive 注释行（DeepSeek 思考模式排队时常见）
    tolerant_keepalive: bool,
}
```

调用方使用方式：

```rust
// 调原版 OpenAI
let openai = OpenAIProvider::builder()
    .api_key(env::var("OPENAI_API_KEY"))
    .build();

// 调 DeepSeek（OpenAI 协议入口）
let deepseek = OpenAIProvider::builder()
    .api_key(env::var("DEEPSEEK_API_KEY"))
    .base_url("https://api.deepseek.com")
    .parse_reasoning_content(true)
    .tolerant_keepalive(true)
    .build();

// 或者用便利工厂封装
let deepseek = DeepSeek::client(env::var("DEEPSEEK_API_KEY"));
// 内部其实返回 OpenAIProvider 实例，预设好 flags
```

**便利模块（不是新 Provider）：**

```rust
pub mod deepseek {
    pub fn client(api_key: impl Into<String>) -> OpenAIProvider {
        OpenAIProvider::builder()
            .api_key(api_key)
            .base_url("https://api.deepseek.com")
            .parse_reasoning_content(true)
            .tolerant_keepalive(true)
            .build()
    }
}
```

未来如果要加入 Doubao / 通义 / Moonshot 等其他 OpenAI 兼容的国内厂商，都按同样模式 —— **不写新 Provider，写新便利模块**。

## 3. 几个需要在 spec 阶段做决定的点

设计阶段需要回答：

1. **HTTP 客户端选 reqwest 还是 hyper？** —— reqwest 更省事，hyper 更轻、可控 streaming 细节。建议 reqwest。
2. **是否使用 async-trait？** —— 影响 trait object 设计。建议先用 `async-trait` crate（编译期 Box 化），后期看是否换成原生 AFIT。
3. **流式底层用什么？** —— `eventsource-stream` 或 `reqwest-eventsource`，避免自己写 SSE 解析。
4. **Provider key/config 如何注入？** —— builder pattern（支持 base_url、超时、代理、protocol flags）。
5. **Model 是字符串还是 enum？** —— 字符串（保持灵活，避免每次新模型都改库）。可以提供常量集作为便利（`models::CLAUDE_OPUS_4_8` / `models::DEEPSEEK_V4_FLASH`）。
6. **错误重试是否内置？** —— 第一版**不内置**，错误透传，调用方自己加 backoff（用 `backoff` crate）。简化设计。
7. **是否提供 `text()` 便利方法？** —— 是。`response.text()` 返回拼接好的纯文本（多个 text block join），方便简单场景。
8. **流式是否提供"事件 → 完整 Message"的折叠器？** —— 提供（参考 Anthropic SDK 的 `.stream().get_final_message()`）。
9. **思考内容（reasoning）的 API 表面**：是 `response.reasoning()` 单独方法，还是让调用方自己从 content blocks 里挑 `Thinking` 块？倾向后者（更一致）。
10. **DeepSeek 模块是 `crate::deepseek` 还是 `crate::openai::deepseek`？** —— 倾向前者（顶层，发现性更好），但实现复用 openai 模块。

## 4. 测试策略要点

- **录制 + 回放**：用 `wiremock-rs` 或 `mockito` 录制真实 provider 响应，离线回放写单元测试
- **真实集成测试**：用环境变量传 key，跑 `cargo test --features integration`，CI 跳过
- **流式专项**：用录制的 SSE 字节流，验证事件序列、累加结果与真实响应一致
- **错误注入**：用 mock server 注入 429 / 529 / 402 / 中途断连，验证错误归一化
- **三家覆盖**：核心场景（chat / tool 调用 / 流式 / 错误）都要在三家各跑一遍 fixture

## 5. 项目骨架草图（仅作参考，spec 阶段再敲定）

```
llm_adapter/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # 公开 API 总入口
│   ├── error.rs                # LlmError + Result alias
│   ├── types/                  # 公共类型
│   │   ├── mod.rs
│   │   ├── message.rs          # Content / Message / Role
│   │   ├── tool.rs             # ToolDef / ToolChoice
│   │   ├── usage.rs
│   │   ├── stream.rs           # StreamEvent / StopReason / BlockKind
│   │   ├── thinking.rs         # ThinkingConfig / CacheHint
│   │   └── request.rs          # ChatRequest / SamplingParams
│   ├── provider.rs             # Provider trait
│   ├── openai/
│   │   ├── mod.rs              # OpenAIProvider + builder（base_url 可配）
│   │   ├── schema.rs           # OAI 私有 schema（serde）
│   │   ├── convert.rs          # ChatRequest ↔ OAI schema
│   │   └── stream.rs           # OAI 扁平 delta → 合成 StreamEvent
│   ├── anthropic/
│   │   ├── mod.rs              # AnthropicProvider + builder
│   │   ├── schema.rs
│   │   ├── convert.rs
│   │   └── stream.rs           # Anthropic 嵌套 events → StreamEvent
│   └── deepseek.rs             # 便利工厂：返回预配置的 OpenAIProvider
└── tests/
    ├── fixtures/               # 录制的真实响应
    │   ├── openai/
    │   ├── anthropic/
    │   └── deepseek/
    ├── openai_chat.rs
    ├── anthropic_chat.rs
    ├── deepseek_chat.rs        # 主要测试 reasoning_content 处理
    ├── openai_stream.rs
    ├── anthropic_stream.rs
    └── deepseek_stream.rs      # 测试 keepalive 容忍 + reasoning 流
```

## 6. 下一步

进入**需求澄清 / 系统设计**阶段：

1. 用 `superpowers:brainstorming` 厘清第一版应支持的具体 use case（聊天、单轮调工具、多轮调工具、流式 + 工具、DeepSeek 思考模式 + 工具）
2. 用 `workflow-system-design` 把以上要点落成 `spec.md`，明确接口签名、错误语义、测试策略
3. spec 评审过后，用 `superpowers:writing-plans` 生成任务粒度的实施计划
4. `cargo init` 并按计划逐任务实施（TDD）
