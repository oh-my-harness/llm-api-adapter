# OpenAI API 研究笔记

## 1. 两套并存的 API

| 维度 | Chat Completions（老） | Responses（新，2025-03 推出） |
|---|---|---|
| 端点 | `POST /v1/chat/completions` | `POST /v1/responses` |
| 抽象单元 | `messages[]`（role-based 消息） | `input[]` + `output[]`（item 序列，type 联合） |
| Item 类型 | 只有 message（含 text/tool_calls） | message / function_call / reasoning / hosted_tool / ... |
| 状态 | 无状态，调用方自管历史 | 可选有状态：`store: true` + `previous_response_id` |
| 工具调用 schema | tool 定义需 `{type:"function", function:{...}}` 外层包裹 | tool 定义内联展开（internally-tagged） |
| Structured Output | `response_format: {type: "json_schema", ...}` | `text.format` |
| Reasoning 支持 | 仅参数 `reasoning_effort`，无独立 item | 有 reasoning item，可保留/加密返回 |
| System prompt | `{role:"system"}` 消息 | 顶层 `instructions` 字段 |
| 推荐度 | 仍支持，生态广 | 官方推荐新项目用 |

**适配层取舍：** 第一版先做 Chat Completions（兼容性、第三方网关、Azure OpenAI、众多模仿者都用这个 schema）。Responses 作为后续扩展。

---

## 2. Chat Completions 关键 schema

### 2.1 请求体核心字段

```json
{
  "model": "gpt-4.1",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
  ],
  "tools": [...],
  "tool_choice": "auto" | "none" | "required" | {"type":"function", "function":{"name":"..."}},
  "parallel_tool_calls": true,
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 4096,
  "max_completion_tokens": 4096,
  "stop": ["..."],
  "stream": false,
  "stream_options": {"include_usage": true},
  "response_format": {"type": "text"} | {"type": "json_object"} | {"type": "json_schema", "json_schema": {...}},
  "reasoning_effort": "none" | "minimal" | "low" | "medium" | "high" | "xhigh",
  "logprobs": false,
  "top_logprobs": 5,
  "prompt_cache_retention": "24h",
  "safety_identifier": "user-id-or-fingerprint",
  "n": 1
}
```

**关键约束：**
- `role` 枚举：`system`, `user`, `assistant`, `tool`，部分模型有 `developer`（取代 system）
- `messages[].content` 可以是 string 或 content-part 数组（含 text / image_url）
- assistant 消息可以同时有 `content` 和 `tool_calls`
- tool 消息必须紧跟 assistant 的 tool_calls，按 id 一对一回答

### 2.2 Tool 定义（Chat Completions schema）

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current temperature for a location.",
    "parameters": {
      "type": "object",
      "properties": {"location": {"type": "string"}},
      "required": ["location"],
      "additionalProperties": false
    },
    "strict": true
  }
}
```

- `strict: true` 强制 schema 严格匹配（仅 JSON Schema 子集），fine-tuned + 多 tool_call 时会失效
- 也支持新的 `custom` tool 类型（freeform 输入输出）

### 2.3 响应体

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "gpt-4.1-...",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "..." | null,
      "refusal": null,
      "tool_calls": [
        {
          "id": "call_xxx",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"Paris\"}"
          }
        }
      ]
    },
    "logprobs": null,
    "finish_reason": "stop" | "length" | "tool_calls" | "content_filter" | "function_call"
  }],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0}
  },
  "system_fingerprint": "fp_xxx"
}
```

**关键陷阱：**
- `tool_calls[].function.arguments` 是 **JSON 字符串**，不是结构化对象 —— 调用方需自己 `JSON.parse`
- 模型可能生成无效 JSON 或 hallucinate 字段，**必须验证**
- `finish_reason: "tool_calls"` 是判断"模型想调工具"的权威信号

---

## 3. 流式（Streaming）

### 3.1 启用与 wire 格式

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}
data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

- 设置 `stream: true`，响应是 SSE：每行 `data: <json>`，以 `data: [DONE]` 终止
- 设置 `stream_options.include_usage: true` 会在 `[DONE]` 前多一个 chunk 携带 `usage`

### 3.2 增量字段合并规则

- `delta.content`（字符串）：**拼接**
- `delta.refusal`（字符串）：**拼接**
- `delta.tool_calls`：**按 index 追加**，每个 tool_call 的 `function.arguments` 是字符串增量，需按 index 拼接

```python
# 伪代码：拼接 tool_calls
state = {}
for chunk in stream:
    for tc in chunk.delta.tool_calls or []:
        slot = state.setdefault(tc.index, {"id": None, "name": None, "args": ""})
        if tc.id: slot["id"] = tc.id
        if tc.function.name: slot["name"] = tc.function.name
        if tc.function.arguments: slot["args"] += tc.function.arguments
```

- 模型可能在同一个 stream 里**混合** text 内容和 tool_calls

---

## 4. 工具调用多轮流程

```
1. 调用方发 messages + tools
2. 模型返回 choices[0].message 含 tool_calls
3. 调用方将整个 assistant message append 到 messages
4. 对每个 tool_call 执行后，append:
   {role:"tool", tool_call_id: <id>, content: <result-as-string>}
5. 再次调用，模型基于 tool result 给出最终回答
```

**tool_choice 行为：**
| 值 | 含义 |
|---|---|
| `"auto"` | 默认（有 tools 时）。模型自己决定 |
| `"none"` | 默认（无 tools 时）。强制纯文本回答 |
| `"required"` | 强制调用某个 tool，允许并行 |
| `{type, function:{name}}` | 强制调用指定 tool |

**parallel_tool_calls：** 默认 true，可置 false 让模型每轮最多一个 tool_call。但**部分模型不支持**此参数（如 o3 / 4o 系列在某些组合下报错）。

---

## 5. 错误码与重试

| 状态码 | 类型 | 可重试 | 说明 |
|---|---|---|---|
| 400 | invalid_request | ❌ | 参数错误 |
| 401 | authentication | ❌ | API key 无效 |
| 403 | permission | ❌ | key 无权限 |
| 404 | not_found | ❌ | 端点/模型错误 |
| 408 | timeout | ✅ | 客户端超时 |
| 429 | rate_limit | ✅ | 限流（看 `Retry-After`） |
| 500 | server_error | ✅ | OpenAI 内部错误 |
| 502/503 | bad_gateway / unavailable | ✅ | 网关/过载 |

**重要 header（响应都会返回）：**
- `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, `x-ratelimit-reset-requests`
- `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-tokens`
- 429 时还有 `retry-after`

**重试策略：** 指数退避（1→2→4→8→16，cap 60s）+ jitter，最多 3-5 次。**绝不重试 4xx 非 429。**

---

## 6. 模型能力矩阵（核心 4 类）

| 系列 | Vision | Tools | Reasoning | Context |
|---|---|---|---|---|
| GPT-5.x（含 Fast / Pro） | ✅ | ✅ | ✅ 可调 effort（none/min/low/med/high/xhigh） | 大 |
| GPT-4.1（含 mini / nano） | ✅ | ✅ 强 | ❌ | 1M |
| GPT-4o（含 mini） | ✅ | ✅ | ❌ | 128K |
| o1 / o1-pro | ✅ | ✅ | ✅ | 128K |
| o3 / o3-pro / o4-mini | ✅ | ✅ | ✅ | 200K |

**特殊约束：**
- o1 系列**不支持 streaming**
- o1 系列**不支持 system message**（用 user 代替）
- **o 系列 / GPT-5.x 推理是模型原生架构，无法真正关闭**——省略 `reasoning_effort` 只是用默认 effort（gpt-5.5 默认 medium），要完全关闭推理需切换到非推理模型（如 gpt-4o）
- `reasoning_effort: "none"` 仅表示最小化推理（latency-critical 场景），不是关闭
- `reasoning_effort` 支持值因模型而异，部分模型仅支持子集（查 model page 确认）
- GPT-5.4+ 在 Chat Completions 下 `reasoning: none` 时不支持 tool calling
- 部分老模型不支持 `parallel_tool_calls`
---

## 7. Responses API 新增能力（暂不展开，留待第二版）

- **Items 时间线**：output[] 含 message / function_call / reasoning / hosted_tool 等异构 item
- **Encrypted reasoning**：ZDR 场景下可以拿到加密的 reasoning token 在下一轮回传
- **previous_response_id**：服务端保持状态，避免重发整段历史
- **内置工具**：`web_search`, `file_search`, `code_interpreter`, `computer_use`, MCP servers
- **Context compaction**：服务端自动压缩长对话
- **Background execution**：长任务后台跑，避免长连接

适配层若要做完整覆盖，Responses 是核心，但**第一版可以只做 Chat Completions**，先把多家归一化跑通。
