# CLAUDE.md — llm_adapter 开发指引

## 项目定位

Rust 编写的 LLM Provider 适配层。把各家 provider 的 wire 格式归一化到 canonical 类型系统，保留 provider 特有功能的 native 扩展通道，不包含 agent/retry/prompt 模板。

## 构建与测试

### 构建

```bash
cargo build
cargo check --all-targets
cargo clippy --all-targets --features integration
cargo fmt -- --check
```

### 测试

```bash
# 离线（默认，无需 API key）
./test.sh

# 集成（打真实 API）
./test.sh --features integration
```

**注意**：macOS 系统 HTTP 代理（127.0.0.1:7897）会拦截 wiremock 和录制工具的请求。`test.sh` 已设 `NO_PROXY=127.0.0.1,localhost`。

### 运行示例

```bash
cargo run --example simple_chat
cargo run --example deepseek_with_reasoning
cargo run --example streaming
cargo run --example tool_call
PROVIDER=deepseek cargo run --example runtime_switch
cargo run --example native_extra_body
```

### 记录 fixture

```bash
cargo run --example record_fixtures
```

· .env 里的凭据决定录哪些 provider。未设 key 的 provider 跳过。
· 录制工具不走库代码（直接 reqwest），保留原始 server field order 和未文档化字段。

## 实现约定

### 类型设计

- 所有构造型类型用 **builder pattern**，字段私有。新增可选字段只加 builder 方法，不产生 breaking change。
- `Message` 是 **ADT**（enum variant 编码角色），不能给 `Message::User` 塞 `ResponseContent::Reasoning`——编译就错。
- `SamplingParams` 仅含三家共有的参数（temperature、max_tokens、top_p、stop）。`top_k`、`seed`、`frequency_penalty` 等 provider 特有参数归入各自的 `*Ext`。
- `ResponseContent::Reasoning` 在 canonical 层是只读的。拿响应文本需要调用 `resp.text()`，拿到 reasoning 用的是 `resp.reasoning()`。

### Provider 实现

- **Universal path**：`impl Provider for X`。输入 `&ChatRequest`，输出 `Result<ChatResponse, LlmError>`。object-safe（`Box<dyn Provider>` 可运行时切换）。
- **Native path**：作为 Provider 实现者的公开方法暴露（如 `chat_native`、`chat_stream_native`），直接访问 Provider 自己特有的 `*Request` 类型（如 `OpenAIRequest = { base: ChatRequest, ext: OpenAIExt }`）。
- **DeepSeek 复用 OpenAIProvider**：通过工厂函数返回预配置的 OpenAIProvider。不做单独类型。DeepSeek V4 默认开启 thinking 模式。

### 错误处理

- 不内建 retry。`LlmError::RateLimit` 和 `LlmError::Overloaded` 从 `Retry-After` header 提取 `retry_after`，调用方决定何时重试。
- HTTP 状态码映射规则见 `src/openai/mod.rs` 的 `send()` 方法。

### 代码风格

- uses `pub(crate)` 而非 `pub` 约束 visibility，wire types 和内部转换函数都不对外暴露。
- `#[allow(dead_code)]` 加在确知保留但当前未引用的 schema 字段上（如 `OaiChunk.id/model`、`OaiErrorBody.message/type`），而不是删除。
- `tests/common/mod.rs` 里的辅助函数跨多个 test crate 共享，用 `#[allow(dead_code)]` 避免每个 crate 各自的 dead_code 警告（每个 crate 只用其一部分）。

## 已知问题与边界

- **Azure OpenAI 不支持**：认证用 `api-key` header 而非 Bearer，路径含 deployment ID.
- **OpenAI Responses API 不支持**：只做 Chat Completions。
- **Prompt caching 只能读不能写**：`Usage.cached_input_tokens` 可查看缓存命中，但 canonical 层不支持 Anthropic 风格 cache_control 标记（随 Anthropic provider 加入）。
- **Structured Output**：OpenAI 通过 `ChatRequest::response_format()` 支持 json_object 和 json_schema；Anthropic 通过 `AnthropicExt::structured_output()` 用 Strict Tool Use 实现。
- **LlmError** 当前缺少 `PaymentRequired` 变体（HTTP 402 DeepSeek 独有），当前映射为 `Provider { code: "payment_required", ... }`。如果要 et 它的 display、Eq 等算子要看是否要加正式变体。

## 环境

- 平台：macOS（Darwin），但代码跨平台。
- 编辑器无关，不强制 IDE。
- `.env` 被 `.gitignore` 排除，不会意外提交。
