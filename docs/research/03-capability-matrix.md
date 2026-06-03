# 能力对比矩阵（OpenAI vs Anthropic vs DeepSeek）

## 0. 三家关系澄清（重要前置）

DeepSeek **不是第三种 wire protocol**，而是在 OpenAI / Anthropic 协议上再开两个 endpoint：

```
                      ┌────────────────────┐
                      │ 内部统一接口        │
                      │ (Provider trait)   │
                      └─────────┬──────────┘
                                │
            ┌───────────────────┴───────────────────┐
            ↓                                       ↓
   ┌──────────────────┐                  ┌──────────────────┐
   │ OpenAI protocol  │                  │ Anthropic protocol│
   │ (chat/completions)│                  │ (messages)        │
   └────────┬──────────┘                  └────────┬──────────┘
            │                                      │
   ┌────────┴────────────────┐         ┌──────────┴──────────────┐
   ↓                         ↓         ↓                         ↓
api.openai.com    api.deepseek.com    api.anthropic.com    api.deepseek.com/anthropic
                  (+ /v1 别名)
```

**这意味着：**
- 适配 DeepSeek = **复用 OpenAI（或 Anthropic）适配器** + base_url 切换 + 处理 reasoning_content 等扩展字段
- 不需要写第三个 Provider 实现
- 设计阶段要把"协议"和"endpoint"两个概念分开（见 `04-design-implications.md` § 2.8）

下面的对比表大多分三列，但 OpenAI 列和 DeepSeek 列在大多数维度上**完全相同**（因为协议相同）；只有 DeepSeek 扩展或限制的部分才有差异。

---

## 1. 整体范式

| 维度 | OpenAI Chat Completions | Anthropic Messages | DeepSeek（OpenAI 入口） |
|---|---|---|---|
| 端点数 | 多个（chat / responses / embeddings / images...） | 一个（`/v1/messages`） | 一个（`/chat/completions`） |
| Base URL | `api.openai.com/v1` | `api.anthropic.com` | `api.deepseek.com` 或 `/v1` |
| System prompt 位置 | messages 数组第一条 `{role:"system"}` | 顶级字段 `system` | 同 OpenAI |
| Role 集 | `system, user, assistant, tool, developer` | `user, assistant` only | 同 OpenAI |
| Message 顺序约束 | 宽松 | 严格交替 user/assistant，首条必须 user | 同 OpenAI（宽松） |
| Response content 形态 | 单一 string + 可选 tool_calls 数组 | 结构化 content block 数组 | 同 OpenAI + `reasoning_content` 字段 |
| Stop reason 集 | stop, length, tool_calls, content_filter | end_turn, max_tokens, stop_sequence, tool_use, pause_turn, refusal | 同 OpenAI |
| Multiple completions | `n` 参数，返回 choices[] | 不支持 | 同 OpenAI |
| API version header | 不需要 | 需要 `anthropic-version` | 不需要 |
| 鉴权 header | `Authorization: Bearer <key>` | `x-api-key: <key>` | `Authorization: Bearer <key>` |
| 服务端状态 | Responses API 支持 `previous_response_id` | ❌ | ❌（明确说自己是 stateless） |

---

## 2. 工具调用对比

| 维度 | OpenAI | Anthropic | DeepSeek |
|---|---|---|---|
| Tool 定义嵌套 | `{type:"function", function:{name, description, parameters, strict}}` | `{name, description, input_schema}` 平铺 | 同 OpenAI |
| Schema 字段名 | `parameters` | `input_schema` | `parameters` |
| Strict 模式 | 显式 `strict: true` | 默认无（由模型可靠性保证） | ✅ 显式 `strict: true`（思考/非思考都支持） |
| Tool 调用结果格式 | `arguments` 是 **JSON 字符串** | `input` 是**结构化对象** | 同 OpenAI（字符串） |
| Tool 调用回喂 | 独立 `role: "tool"` 消息，关联 `tool_call_id` | user 消息里的 `tool_result` block，关联 `tool_use_id` | 同 OpenAI |
| 并行控制 | `parallel_tool_calls: false` | `tool_choice.disable_parallel_tool_use: true` | 同 OpenAI |
| tool_choice 值 | `"auto" \| "none" \| "required" \| {type, function}` | `{type:"auto" \| "any" \| "tool" \| "none", name?}` | 同 OpenAI |
| 强制单工具 | tool_choice 指定函数 | tool_choice `{type:"tool", name}` | 同 OpenAI |
| 强制任意工具 | tool_choice `"required"` | tool_choice `{type:"any"}` | 同 OpenAI |
| 工具响应可含图像 | ❌（必须字符串内容） | ✅（content 可以是 content block 数组） | ❌ |
| 思考模式 + 工具调用陷阱 | n/a | 启用 thinking 时 tool_choice 仅支持 auto/none | **必须**回传 `reasoning_content`，否则 400 |

---

## 3. 流式对比

详见 `06-streaming-deep-dive.md`。要点速览：

| 维度 | OpenAI | Anthropic | DeepSeek |
|---|---|---|---|
| Wire format | SSE，每行 `data: <json>`，以 `data: [DONE]` 终止 | SSE，每个事件 `event: <type>\ndata: <json>\n\n` | 同 OpenAI |
| 事件结构 | 扁平：所有内容在 `choices[].delta` | 嵌套：message → content_block → delta，有显式 start/stop | 同 OpenAI 扁平结构 |
| 文本增量字段 | `delta.content`（字符串） | `delta.text_delta.text` | `delta.content` |
| 推理增量字段 | ❌（不暴露） | `delta.thinking_delta.thinking` | **`delta.reasoning_content`**（独有） |
| Tool args 增量 | `delta.tool_calls[i].function.arguments`（字符串拼接） | `delta.input_json_delta.partial_json`（字符串拼接） | 同 OpenAI |
| 终止信号 | 最后 chunk `finish_reason != null`，再 `[DONE]` | `message_delta` 带 `stop_reason` + `message_stop` | 同 OpenAI |
| Usage 在流中 | 需 `stream_options.include_usage: true` | 默认就在 `message_delta` 累计返回 | 同 OpenAI |
| Keep-alive 行为 | 偶发 `:` 注释行 | `ping` 事件 | **大量** `:` 注释行（高负载排队时） |
| 推荐读超时 | 30s | 30s | **≥ 90s**（思考模式排队时长） |

---

## 4. Reasoning / Thinking

| 维度 | OpenAI | Anthropic | DeepSeek |
|---|---|---|---|
| 启用方式 | `reasoning_effort: "low/medium/high/..."`（仅 o 系列、GPT-5） | `thinking: {type:"enabled", budget_tokens: N}` | `extra_body={"thinking": {"type":"enabled"}}` 或选 `-reasoner` 模型 |
| 思考内容暴露 | **不暴露**（Chat Completions） | 显式 `thinking` content block，含明文 | **明文返回**于 `reasoning_content` 字段 |
| 加密版本 | Responses API 的 encrypted reasoning（ZDR） | `redacted_thinking` block | ❌ |
| 跨轮保留 | 用 `previous_response_id` 或回传加密 token | 回传 thinking block 时**必须保留 signature** | V4：无 tool 调用可省；**有 tool 调用必须回传 reasoning_content**，否则 400 |
| 与 tool use 共存 | GPT-5.4+ Chat Completions 下 `reasoning:none` 不支持 tool | 启用 thinking 时 tool_choice 仅支持 auto/none | 支持，但要正确回喂 reasoning_content |
| 思考时不支持的参数 | 各模型不同（o1 不支持 system） | temperature/top_p/top_k 失效 | temperature/top_p/penalty/logprobs **被忽略**（不报错） |
| 思考预算上限 | reasoning_effort 控制 | `budget_tokens` | reasoning_content 可达 32K，**不计入 64K 上下文** |
| 计费 | 算作 output token | 算作 output token | 算作 output token（独立计数） |

---

## 5. 多模态输入

| 类型 | OpenAI | Anthropic | DeepSeek |
|---|---|---|---|
| 图像 URL | `{type:"image_url", image_url:{url}}` | `{type:"image", source:{type:"url", url}}` | ❌（V4 当前无 vision） |
| 图像 base64 | `image_url.url = "data:image/jpeg;base64,..."` | `{type:"image", source:{type:"base64", media_type, data}}` | ❌ |
| PDF / 文档 | 通过 Files API 引用 | `{type:"document", source:{...}}` | ❌ |
| 音频 | 部分模型支持 audio input | ❌ | ❌ |

→ 第一版适配层多模态部分**先按 OpenAI / Anthropic 两套实现**，DeepSeek 走 OpenAI 协议但传图会被服务端拒绝，由调用方自己感知。

---

## 6. 速率限制与错误

| 维度 | OpenAI | Anthropic | DeepSeek |
|---|---|---|---|
| 限流维度 | RPM + TPM | RPM + ITPM + OTPM（input/output 分开） | **动态**（不发布固定表） |
| 限流 header | `x-ratelimit-*` 6 个 | `anthropic-ratelimit-*` | 不发布 |
| 排队 vs 拒绝 | 直接 429 | 直接 429 | **高负载先排队**（SSE 长 keep-alive），实在不行才 429 |
| 过载特殊状态码 | 503 | **529** | 503 |
| 余额耗尽 | 429 / 限额 | 鉴权失败相关 | **402 `insufficient_balance`**（独有） |
| Retry-After header | 有 | 有 | 偶有 |
| 缓存命中字段 | `usage.prompt_tokens_details.cached_tokens` | `usage.cache_read_input_tokens` + `cache_creation_input_tokens` | `usage.prompt_cache_hit_tokens` + `prompt_cache_miss_tokens` |
| Request ID header | `x-request-id` | `request-id` | `x-request-id` |

**完整错误码集合（需要适配层覆盖的）：**

| 状态码 | 三家含义 |
|---|---|
| 400 | 三家都是 `invalid_request_error` |
| 401 | 三家都是 `authentication_error` |
| 402 | **仅 DeepSeek**：余额不足 |
| 403 | OpenAI / Anthropic：权限不足；DeepSeek 不常见 |
| 404 | 三家都是端点/模型不存在 |
| 413 | Anthropic 独有：请求过大 |
| 422 | DeepSeek 独有：参数语义错（OpenAI 用 400） |
| 429 | 三家都是限流 |
| 500 | 三家都是内部错误 |
| 503 | OpenAI / DeepSeek：过载 |
| 504 | Anthropic：超时 |
| 529 | **仅 Anthropic**：全局过载（不计入配额） |

---

## 7. Prompt Caching（差异最大的一块）

| 维度 | OpenAI | Anthropic | DeepSeek |
|---|---|---|---|
| 触发方式 | **自动**（GPT-4o+ 及之后） | **显式** 用 `cache_control` 标记块 | **自动**（磁盘前缀缓存） |
| 缓存粒度 | 服务端决定 | 块级，至多 4 个 breakpoint | 64-token 单元，前缀字节匹配 |
| TTL 控制 | `prompt_cache_retention: "24h"` 可选 | `cache_control.ttl: "5m"\|"1h"` | 服务端定（约几小时） |
| 写入溢价 | 同价或略高 | **5m: 1.25×**，**1h: 2×** | 同价（写入不额外收费） |
| 读取折扣 | 约 50% off（取决于模型） | **0.1×（省 90%）** | **V4 Flash: 0.02×（省 98%）** |
| Usage 字段 | `cached_tokens` | `cache_read_input_tokens` + `cache_creation_input_tokens` | `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens` |
| 不变量 | n/a | `cache_read + cache_creation + 非缓存 == input_tokens` | `hit + miss == prompt_tokens` |
| 调用方控制 | 无（被动） | 主动（必须标记） | 无（被动，把稳定前缀靠前即可） |

→ 适配层需要：
1. 统一的 Usage 字段把三家命中数都映射到 `cached_input_tokens`
2. 提供一个**可选的** `CacheHint` API，让调用方在 Anthropic 上启用块标记（其他家忽略）

---

## 8. 独有能力（决定是否纳入第一版）

| 能力 | OpenAI | Anthropic | DeepSeek | 第一版建议 |
|---|---|---|---|---|
| 显式 Prompt caching | ❌（自动） | ✅ `cache_control` | ❌（自动） | Anthropic 暴露 `CacheHint`，其他家自动 |
| Structured Output | `response_format: json_schema` + strict | 无对应（靠 tool use 间接实现） | 同 OpenAI（兼容） | 暂不归一化，作 OpenAI/DeepSeek 专属 |
| 服务端状态 | Responses API `previous_response_id` | ❌ | ❌ | 不做 |
| 内置 hosted 工具 | web_search / file_search / computer_use / code_interpreter | computer_use / web_search / code_execution | ❌ | 不做（属于上层 agent 关注） |
| Logprobs | ✅ | ❌ | ✅（思考模式下被忽略） | 不做 |
| Multiple completions (n>1) | ✅ | ❌ | ✅ | 不做 |
| Batch API | ✅ | ✅（Message Batches） | ❌ | 不做 |
| 思考内容明文返回 | ❌ | ✅ thinking block | ✅ reasoning_content | **要做**（DeepSeek 多轮工具调用强依赖） |
| 思考签名（防伪） | ❌ | ✅ signature_delta | ❌ | 要做（Anthropic 多轮回传需保留） |
| Anthropic 风格 endpoint | ❌ | n/a（自己） | ✅ `/anthropic/v1/messages` | 不做（价值不大） |

---

## 9. 关键归一化挑战清单（v2，整合 DeepSeek）

按难度从低到高：

1. **角色与系统提示**：把 `system` 字段抽象成统一接口里的一个槽，OpenAI/DeepSeek 适配时塞 messages[0]，Anthropic 适配时塞顶级字段
2. **Content 块**：内部模型必须用**块数组**（向 Anthropic 看齐），OpenAI/DeepSeek 适配时把 string content 升级成单块 text，输出时按需展平
3. **Tool 定义 schema**：内部用 Anthropic 风格平铺（更简洁），输出 OpenAI/DeepSeek 时套 `{type:"function", function:{...}}` 外壳
4. **Tool 调用 arguments**：内部用 `serde_json::Value`（结构化），输出 OpenAI/DeepSeek 时序列化为 string，输入 Anthropic 时直接传
5. **Tool 结果回喂**：内部用统一 `ToolResult { id, content, is_error }`，OpenAI/DeepSeek 翻译成 `role:"tool"` 消息，Anthropic 翻译成 user content block
6. **流式事件**：必须设计统一的 stream event 枚举。向 Anthropic 看齐（有 block 概念），OpenAI/DeepSeek 适配器要"合成"block start/stop（详见 `06-streaming-deep-dive.md` § 9）
7. **Stop reason 归一化**：定义 enum：`Stop / Length / ToolUse / StopSequence / ContentFilter / Refusal / Other(String)`
8. **Usage 字段**：含三家不同的 cache 命中字段映射；含 reasoning token 计数
9. **错误归一化**：Provider-agnostic enum，需含 `InsufficientBalance(402)`（DeepSeek）和 `Overloaded(529)`（Anthropic）；保留原始错误信息和 retry-after 信号
10. **Tool choice 语义映射**：4 状态 + parallel disable 标记，做映射表
11. **Reasoning 字段一等公民**：DeepSeek V4 思考模式 + 工具调用要求 reasoning_content 必须保留回传；Anthropic thinking block 同样要保留 signature。内部 Message 必须有 `reasoning` 字段
12. **Prompt caching**：第一版暴露统一的 `CacheHint` 类型供 Anthropic 使用，其他家忽略；Usage 字段统一暴露 hit 数
13. **Provider/Endpoint 解耦**：DeepSeek 走 OpenAI 协议但 base_url 不同 —— Provider 抽象不能把 base_url 硬编码（详见 `04-design-implications.md` § 2.8）
