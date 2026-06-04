# Code Review 报告

## 审查范围
- **Spec**: `docs/superpowers/specs/2026-06-03-llm-adapter-design.md`
- **Tasks**: `docs/superpowers/plans/2026-06-04-anthropic-provider.md`
- **审查文件**: `src/lib.rs`, `src/error.rs`, `src/provider.rs`, `src/stream_handle.rs`, `src/types/*.rs`, `src/openai/*.rs`, `src/anthropic/*.rs`, `src/deepseek.rs`

## 总体结论: NEEDS_CHANGES

存在 2 个 P1 问题，建议合入前修复或确认计划。

## 裁决明细

### 已驳回 (DROP)

| Finding | Reviewer | 原始优先级 | Critic | 依据 |
|---------|----------|-----------|--------|------|
| F-1 | performance | P1 | ❌ 驳回 | 标准 Rust streaming 惯用法，2000 events 约 48KB 分配，对比 JSON 反序列化和 HTTP I/O 可忽略 |
| F-2 | performance | P1 | ❌ 驳回 | extra_body 为例外路径；to_value+to_string 是最简洁正确的 JSON merge 方案 |
| F-4 | performance | P2 | ❌ 驳回 | reqwest::Client::new() 懒初始化 (~0.1ms)，Provider 设计为单例 |
| F-5 | performance | P2 | ❌ 驳回 | 深拷贝微秒级，对比 LLM API 数百毫秒延迟可忽略 |
| F-6 | robustness | P1 | ❌ 驳回 | Anthropic 协议保证 block 严格顺序顺序（调研确认），LIFO 等价于 index 匹配 |
| F-9 | robustness | P2 | ❌ 驳回 | 防御性代码，注释明确标注 "shouldn't happen in normal flow" |
| F-11 | robustness | P2 | ❌ 驳回 | `unwrap_or_default()` 是标准 Rust 模式，空 usage → 零统计在语义上正确 |
| F-18 | contract | P0 | ❌ 驳回 | 同 F-6，Anthropic block 严格顺序，LIFO pop() 在此约束下安全 |

### 已确认 (KEEP)

| Finding | Reviewer | 原始优先级 | Critic | 终裁优先级 | 裁决依据 |
|---------|----------|-----------|--------|-----------|---------|
| F-8/F-14/F-17/F-23 | 多 review | P0/P1 | ⚠️ Not rebutted | **P1** | `tolerant_keepalive` 在 builder 中被接受但 `build()` 丢弃（`OpenAIProvider` 无此字段）；`parse_stream` 也不接收此参数；`deepseek::client().tolerant_keepalive(true)` 完全无效 |
| F-16/F-22 | contract/spec | P0/P1 | ⚠️ Not rebutted | **P1** | `AnthropicResponse` 是 `pub(crate)` wire 类型，非 spec §5.1 定义的公开类型；`chat_native()` 返回 `ChatResponse` 而非 `AnthropicResponse`；thinking signature 在 `from_anthropic()` (signature: _) 中丢弃；扩展思考多轮对话需要 signature 回传 |
| F-3 | performance | P1 | ⚠️ 降级 | **P2** | 第一次遍历确实无操作，但 N×M 规模极小（< 100 次空迭代），实际开销可忽略；应清理代码 |
| F-7/F-19/F-26 | robustness/spec | P1/P2 | ⚠️ Not proven | **P2** | `extract_retry_after` 接收 error body text 而非 HTTP headers（`response.text().await` 先消耗 response），且始终返回 `None`；TODO 明确标注，属已知技术债但结构上有缺陷 |
| F-10/F-21 | robustness/contract | P2 | ⚠️ Not proven | **P2** | `unwrap_or(Value::String(args))` 为设计取舍；建议增加 `tracing::warn!` 日志 |
| F-12 | robustness | P2 | ⚠️ Not proven | **P2** | `reqwest::Client::new()` 无默认超时；建议添加 `.timeout(Duration::from_secs(120))` 作为安全网 |
| F-13 | robustness | P2 | ⚠️ Not rebutted | **P2** | `anthropic_response_block_to_canonical` 缺少 `!text.is_empty()` 过滤，与计划文档不一致；实际影响有限 |
| F-15 | standards | P2 | ⚠️ Not rebutted | **P2** | `ToolDef` 是唯一不遵循 CLAUDE.md builder pattern 的公开构造类型；新增字段会有 breaking change |
| F-20/F-25 | contract/spec | P1/P2 | ⚠️ Not proven | **P2** | `AnthropicImageSource::media_type` 应为 `Option<String>` 而非 `String`；URL 类型图片发送 `"media_type": ""`；需真实 API 测试确认影响 |

### 转入 Follow-up Notes

| Finding | 原始问题 | 依据 |
|---------|---------|------|
| F-24 | CacheHint 结构不符 spec（`block_index/ephemeral` vs `target/ttl`） | 有意简化，功能等价；应更新 spec 匹配实际实现 |

## 正式问题

### P1（应该修复）

#### P1-1: `tolerant_keepalive` 完全失效
- **维度**: 健壮性 / 工程规范 / 契约 / 需求符合度
- **位置**:
  - `src/openai/mod.rs:24-30` （`OpenAIProviderBuilder` 声明 `tolerant_keepalive: bool` 字段）
  - `src/openai/mod.rs:188-196` （`build()` 不转移该字段，`OpenAIProvider` 无此字段）
  - `src/openai/stream.rs:234` （`parse_stream(response, parse_reasoning)` 不接收该参数）
  - `src/deepseek.rs:13` （`deepseek::client().tolerant_keepalive(true)` 调用但无效）
- **问题**: `tolerant_keepalive` 在 builder 和 `OpenAIExt` 中存在，但既不存入 `OpenAIProvider` 也不传入 stream parser。`deepseek::client()` 设置的 `.tolerant_keepalive(true)` 被静默丢弃。
- **证据**:
  - `OpenAIProvider` struct 无 `tolerant_keepalive` 字段
  - `build()` 不引用 `self.tolerant_keepalive`
  - `parse_stream(response, parse_reasoning)` 签名仅两个参数
  - `OpenAIExt.tolerant_keepalive` 在 `chat_native()`/`chat_stream_native()` 中从未读取
- **触发场景**: 使用 `deepseek::client()` 做流式推理，DeepSeek 在思考模式排队时发送 SSE keepalive（`: keep-alive` 或 `data: [KEEPALIVE]`），当前代码无法容忍处理
- **建议**:
  1. 为 `OpenAIProvider` 添加 `tolerant_keepalive: bool` 字段
  2. 在 `build()` 中转移 builder 的值
  3. 将 `tolerant_keepalive` 传入 `parse_stream()`，在解析器中跳过非 JSON 数据行
  4. 或者在确定不需要此功能时从 builder 和 `OpenAIExt` 中移除该字段（同时更新 spec 和 deepseek 工厂）

#### P1-2: Anthropic native path 丢弃 thinking signature
- **维度**: 契约 / 需求符合度
- **位置**:
  - `src/anthropic/schema.rs:156-176` （`AnthropicResponse` 是 `pub(crate)` wire 类型，非公开类型）
  - `src/anthropic/mod.rs:111-116` （`chat_native()` 返回 `Result<ChatResponse, LlmError>` 而非 `Result<AnthropicResponse, LlmError>`）
  - `src/anthropic/convert.rs:544-561` （`anthropic_response_block_to_canonical(Thinking{signature: _})` 丢弃 signature）
- **问题**: 规范 §5.1 定义公开的 `AnthropicResponse` 类型（含 `into_message()` → 保留 signature 和 `into_canonical()` → 丢弃 signature）。实际实现中 native path 直接返回 `ChatResponse`，thinking signature 在 `from_anthropic()` 中不可逆丢弃。启用扩展思考的多轮对话需要 signature 回传，当前实现会在第二轮失败。
- **证据**: 调用链 `chat_native()` → `from_anthropic()` → `anthropic_response_block_to_canonical(Thinking{thinking, signature: _})` → `ChatResponse` [signature 丢弃] → `into_message()` → `Message::Assistant(content)` [无 signature] → 重新提交给 Anthropic → API 400 错误
- **建议**:
  1. 实现公开的 `AnthropicResponse` 类型（从 `schema::AnthropicResponse` 映射）
  2. 实现 `into_canonical() -> ChatResponse`（丢弃 signature）和 `into_message() -> Message`（保留 signature → 通过 `ResponseContent::Reasoning` 携带 → 在 `assistant_content_to_wire` 中重建 `Thinking` block）
  3. `chat_native()` 返回 `AnthropicResponse`

### P2（建议改进）

#### P2-1: `apply_cache_hints` 第一次遍历为空操作
- **位置**: `src/anthropic/convert.rs:438-461`
- **问题**: 第一次完整遍历（lines 443-461）遍历全部消息和 content block 但内部只有注释，无实际代码
- **建议**: 删除第一个 pass

#### P2-2: `extract_retry_after` 始终返回 None（已知技术债）
- **位置**: `src/openai/mod.rs:102-107`, `src/anthropic/mod.rs:101-103`
- **问题**: `response.text().await?` 在提取 HTTP headers 之前消耗了 `response`；函数签名 `extract_retry_after(&self, _error_text: &str)` 接收 error body text，但 `Retry-After` 存在于 HTTP headers 中；两个实现均返回 `None`
- **建议**: 重构 `send()` 错误处理路径：在 `response.text()` 之前读取 `response.headers()`，提取 `Retry-After` 值（RFC 7231 支持 `seconds` 和 `HTTP-date` 两种格式），传递给错误构造函数

#### P2-3: `StreamHandle::collect()` ToolDelta JSON 解析静默回退
- **位置**: `src/stream_handle.rs:86`
- **问题**: `serde_json::from_str(&args).unwrap_or(Value::String(args))` 在 JSON 解析失败时静默回退，下游 `input.get("field")` 在 `Value::String` 上返回 `None`
- **建议**: 增加 `tracing::warn!` 日志记录回退事件，保持当前回退行为

#### P2-4: `reqwest::Client::new()` 无默认超时
- **位置**: `src/openai/mod.rs:189`, `src/anthropic/mod.rs:148`
- **问题**: 无 `.timeout()`、`.connect_timeout()` 或 `.read_timeout()`，provider 服务器挂起时请求无限等待
- **建议**: 使用 `reqwest::Client::builder().timeout(Duration::from_secs(120)).build()`，或将超时作为 builder 参数暴露

#### P2-5: Anthropic `from_anthropic` 不过滤空内容块
- **位置**: `src/anthropic/convert.rs:544-561`
- **问题**: `anthropic_response_block_to_canonical` 直接将空 `Text("")`/`Thinking("")` 推入输出，无空字符串检查
- **建议**: 添加 `if !text.is_empty()` 守卫

#### P2-6: `ToolDef` 无 builder，字段 `pub`
- **位置**: `src/types/request.rs:4-12`
- **问题**: 唯一不遵循 CLAUDE.md builder pattern 的公开构造类型；新增字段会有 breaking change
- **建议**: 考虑添加 builder 或使用 `#[non_exhaustive]` 保留向前兼容

#### P2-7: `AnthropicImageSource::media_type` 应为 `Option<String>`
- **位置**: `src/anthropic/schema.rs:100-113`
- **问题**: `media_type: String`（必需），URL 类型图片传入 `String::new()`，序列化为 `"media_type": ""`
- **建议**: 改为 `Option<String>`，URL 类型设 `None`，使用 `#[serde(skip_serializing_if = "Option::is_none")]`

## Follow-up Notes

- **CacheHint 简化**: 实际实现（`block_index: usize + ephemeral: bool`）与 spec §5.1（`target: CacheTarget + ttl: CacheTtl`）不一致，但这是有意简化（功能等效）。建议更新 spec 匹配实际实现。
- **LlmError::Timeout 应用**: 变体已定义但未被 `send()` 显式触发；`reqwest` 超时错误通过 `#[from]` 映射到 `Network` 而非 `Timeout`。如果调用方需要区分"网络错误"和"超时"，需在 `send()` 中显式检查。
- **PaymentRequired 缺失**: HTTP 402 (DeepSeek) 当前映射为 `LlmError::Provider { code: "payment_required", ... }`。如果需要正式变体需要单独 PR。
- **F-1/F-2/F-4/F-5** 已驳回：性能影响可忽略或为标准模式。
- **F-6/F-18** 已驳回：Anthropic 协议保证 block 严格顺序，LIFO pop() 安全。

---

*审查日期: 2026-06-04*
*审查范围: 完整 llm_adapter 库（OpenAI + Anthropic + DeepSeek provider）*
