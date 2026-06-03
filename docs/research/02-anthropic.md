# Anthropic API 研究笔记

## 1. 整体形态

- **唯一主端点：** `POST /v1/messages`（Messages API），用一个 endpoint 覆盖文本、流式、工具、视觉、思考
- **配套：** `POST /v1/messages/count_tokens`（请求前预估 token）
- **鉴权：** `x-api-key` header；同时需要 `anthropic-version: 2023-06-01`（或更新）
- **可选 beta：** `anthropic-beta` header 启用预览功能（如 prompt caching extended TTL、context-management、tool search 等）

**关键差异（vs OpenAI）：**
- **没有 `system` role**。System prompt 是请求体的**顶级字段**，不是 messages 数组里的项
- messages 必须**严格交替** user / assistant，不允许连续两条同 role
- 首条消息必须是 user
- 响应是**结构化的 content block 数组**，不是单一 string

---

## 2. Messages API 核心 schema

### 2.1 请求体

```json
{
  "model": "claude-opus-4-8",
  "max_tokens": 4096,
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": [
      {"type": "text", "text": "What's the weather?"},
      {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
    ]}
  ],
  "system": "You are helpful." ,
  "tools": [...],
  "tool_choice": {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "..."} | {"type": "none"},
  "temperature": 1.0,
  "top_p": 0.999,
  "top_k": 250,
  "stop_sequences": ["..."],
  "stream": false,
  "metadata": {"user_id": "..."},
  "thinking": {"type": "enabled", "budget_tokens": 10000}
}
```

**Content block 类型（消息内 / 响应内通用）：**

| type | 出现位置 | 字段 |
|---|---|---|
| `text` | user / assistant | `text: string` |
| `image` | user | `source: {type: "base64", media_type, data}` 或 `{type: "url", url}` |
| `tool_use` | assistant 响应 | `id, name, input: object` |
| `tool_result` | user 消息 | `tool_use_id, content: string \| ContentBlock[], is_error?: bool` |
| `thinking` | assistant 响应（启用 extended thinking） | `thinking: string, signature: string` |
| `redacted_thinking` | assistant 响应 | `data: string`（加密） |
| `document` | user | `source: {...}`（PDF 等） |

`system` 也可以是数组形式（多块 + 各自 cache_control）：
```json
"system": [
  {"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}
]
```

### 2.2 响应体

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-4-8",
  "content": [
    {"type": "text", "text": "I'll check the weather."},
    {"type": "tool_use", "id": "toolu_...", "name": "get_weather", "input": {"location": "Paris"}}
  ],
  "stop_reason": "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" | "pause_turn" | "refusal",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

**和 OpenAI 的根本不同：**
1. **content 是块数组**，不是单一字符串 —— 一次响应可以同时含 text 和多个 tool_use
2. **tool_use.input 是结构化 object**，不是 JSON 字符串（OpenAI 是字符串）
3. `stop_reason: "tool_use"` 是判断"模型想调工具"的信号
4. 没有 `choices[]` 数组，一个请求一个响应

---

## 3. 流式（Streaming）

### 3.1 事件序列

启用方式：请求体加 `"stream": true`，SSE 响应。

**顶级序列：**
```
message_start
  content_block_start (index 0)
    content_block_delta × N
  content_block_stop
  content_block_start (index 1)
    content_block_delta × N
  content_block_stop
  ...
  message_delta
message_stop
```

途中可能穿插 `ping` 事件。

### 3.2 每种事件的 JSON 形态

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_xxx","type":"message","role":"assistant","content":[],"model":"...","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":42}}

event: message_stop
data: {"type":"message_stop"}
```

### 3.3 Delta 类型（按 block 类型分）

| Delta type | 字段 | 说明 |
|---|---|---|
| `text_delta` | `text` | 文本增量 |
| `input_json_delta` | `partial_json` | tool_use 输入的 JSON 字符串增量 |
| `thinking_delta` | `thinking` | 思考内容增量 |
| `signature_delta` | `signature` | thinking 签名（最后） |

### 3.4 tool_use 流式（重点）

```
content_block_start  → {content_block: {type:"tool_use", id, name, input: {}}}
content_block_delta  → {delta: {type:"input_json_delta", partial_json: "{\"loc"}}
content_block_delta  → {delta: {type:"input_json_delta", partial_json: "ation\":\"P"}}
content_block_delta  → {delta: {type:"input_json_delta", partial_json: "aris\"}"}}
content_block_stop
```

**注意：**
- `content_block_start` 的 `input` 是空 `{}` 占位，**真正的输入要从 partial_json 拼装**
- 默认行为：模型一次性吐一个完整 key-value，所以 deltas 之间可能有较长间隔
- 拼完所有 partial_json 后用 `JSON.parse` 得到完整 input
- 可选 beta `fine-grained-tool-streaming-2025-05-14` 取消缓冲（但可能输出非法 JSON，特别是 max_tokens 截断时）

### 3.5 usage 累计规则

- `message_start` 给初始 input_tokens 和 output_tokens=1
- 多个 `message_delta` 都会带 usage，但**累计值**，不是增量
- 客户端只需保留最后一个

---

## 4. 工具调用

### 4.1 Tool 定义

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location.",
  "input_schema": {
    "type": "object",
    "properties": {"location": {"type": "string"}},
    "required": ["location"]
  }
}
```

- 字段叫 `input_schema`（OpenAI 叫 `parameters`）
- 没有 `type: function` 这一层包裹（OpenAI Chat Completions 需要）
- name 全局唯一，不能和 Anthropic 内置 server tool 重名

### 4.2 tool_choice

| 取值 | 含义 |
|---|---|
| `{"type": "auto"}` | 默认（有 tools 时）。模型自由决定 |
| `{"type": "any"}` | 必须调用某个工具，但不指定哪个 |
| `{"type": "tool", "name": "..."}` | 强制调用指定工具 |
| `{"type": "none"}` | 禁止使用工具 |

**关键行为：**
- `any` 和 `tool` 模式下，模型**不会**在 tool_use 前输出文本（API 强制预填 assistant 消息）
- `extended thinking` 启用时，只支持 `auto` 和 `none`
- 配套 `disable_parallel_tool_use: true` 强制至多/恰好一个 tool 调用

### 4.3 多轮回喂格式

```json
// 第二轮 user 消息
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01ABC...",
      "content": "Sunny, 22°C"  // 可以是 string，也可以是 content block 数组（含 image 等）
    }
  ]
}
```

- `tool_result` 是 user 消息里的 content block，**不是独立 role**（OpenAI 用 `role: "tool"`）
- 多个 tool_use → 多个 tool_result，要在同一条 user 消息里
- `is_error: true` 可以表示工具执行失败

---

## 5. 错误码与重试

| 状态码 | 类型 | 可重试 | 说明 |
|---|---|---|---|
| 400 | `invalid_request_error` | ❌ | 请求格式或参数错 |
| 401 | `authentication_error` | ❌ | API key 无效 |
| 403 | `permission_error` | ❌ | 无权限 |
| 404 | `not_found_error` | ❌ | 端点或模型 ID 错 |
| 413 | `request_too_large` | ❌ | 请求超出大小限制 |
| 429 | `rate_limit_error` | ✅ | 限流（看 retry-after） |
| 500 | `api_error` | ✅ | 内部错误 |
| 504 | `timeout_error` | ✅ | 超时 |
| 529 | `overloaded_error` | ✅ | 全局过载（**不计入配额**） |

**错误响应体格式：**
```json
{"type": "error", "error": {"type": "invalid_request_error", "message": "..."}}
```

**重要 header：**
- `request-id`：每个响应都有，故障排查时给 Anthropic 必给
- `retry-after`：429 时给出秒数
- `anthropic-ratelimit-*`：限流余量信息

**429 vs 529 关键差别：**
- 429：你账户配额满了 → 退避 + 降低并发
- 529：服务端全局过载 → 退避 + 切换模型（Haiku 常更松）

**SDK 默认重试：** 重试 429 / 408 / 500 / 502 / 503 / 504 / 529，指数退避，默认 maxRetries=2。

---

## 6. 模型能力矩阵（2026-06 当前）

| 模型 | 用途 | Vision | Tool Use | Extended Thinking | Computer Use | Context |
|---|---|---|---|---|---|---|
| **Opus 4.8**（最新旗舰，2026-05） | 最复杂任务、agentic 编码 | ✅ | ✅ | ✅ | ✅ | 200K（1M beta on Sonnet） |
| **Opus 4.7 / 4.6 / 4.5** | 同上，老版本 | ✅ | ✅ | ✅ | ✅ | 200K |
| **Sonnet 4.6 / 4.5** | 默认主力 | ✅ | ✅ | ✅ | ✅ | 200K（1M beta） |
| **Haiku 4.5** | 速度/成本优先 | ✅ | ✅ | ✅ | ✅ | 200K |

**普遍能力：** 所有 4.x 模型支持文本 + 图像输入、工具使用、扩展思考。

**多模型策略（官方建议）：** Haiku 路由 / 简单任务 → Sonnet 中等复杂度 → Opus 最难的 10-15%，能省 60-70% 成本。

---

## 7. Prompt Caching（重要差异化能力）

### 7.1 用法

在要缓存的 content block 上加 `cache_control`：

```json
{
  "type": "text",
  "text": "<long system prompt or document>",
  "cache_control": {"type": "ephemeral", "ttl": "5m" | "1h"}
}
```

- 最多 **4 个 cache breakpoint** per 请求
- 缓存是**前缀缓存**，按 `tools → system → messages` 顺序匹配
- 任何前缀字节差异都导致 cache miss

### 7.2 定价

| 操作 | 倍率 |
|---|---|
| 5m 写入 | 1.25× input 价 |
| 1h 写入 | 2× input 价 |
| 缓存读取 | 0.1× input 价（**省 90%**） |

5m TTL：读 2 次以上就回本；1h TTL：读 3 次以上才回本。

### 7.3 响应 usage 字段

```json
"usage": {
  "input_tokens": 2048,
  "cache_read_input_tokens": 1800,
  "cache_creation_input_tokens": 248,
  "output_tokens": 503,
  "cache_creation": {
    "ephemeral_5m_input_tokens": 148,
    "ephemeral_1h_input_tokens": 100
  }
}
```

### 7.4 注意事项

- `tools → system → messages` 顺序：tool 定义变了会让 system 缓存也失效
- `tool_choice` 改变会让 message 块缓存失效（system/tools 不受影响）
- **best-effort**：写缓存可能失败，单次请求不保证一定命中
- 适配层若要支持，需要在统一接口里**显式标记**哪些 block 缓存（不能自动推断）

---

## 8. Extended Thinking（值得标注）

- 启用：`"thinking": {"type": "enabled", "budget_tokens": N}`
- 响应里多一个 `thinking` content block（明文）+ 可选 `redacted_thinking`（加密，给敏感场景）
- thinking 之后的 assistant 块要带 signature，多轮交互回传 thinking 时必须保留 signature
- 限制：`tool_choice` 只能 `auto` 或 `none`；不能用 `temperature`/`top_p`/`top_k`

适配层暂可不做 thinking，作为 P2 扩展。

---

## 9. 暂不展开（P2+）

- Citations / RAG 引用回填
- Computer Use（GUI 操作）
- Programmatic tool calling（lambda-like）
- Tool search tool（万级工具懒加载）
- Message Batches API（异步批量）
- Files API、PDF 输入
