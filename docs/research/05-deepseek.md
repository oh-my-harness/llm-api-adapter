# DeepSeek API 研究笔记

## 1. 整体形态：双兼容协议

DeepSeek 不发明新协议，而是同时支持两种已有协议：

| 协议入口 | Base URL | 用途 |
|---|---|---|
| **OpenAI-compatible** | `https://api.deepseek.com`（也可加 `/v1` 别名） | 走 OpenAI SDK / Chat Completions schema |
| **Anthropic-compatible** | `https://api.deepseek.com/anthropic`（`/anthropic/v1/messages`） | 走 Anthropic SDK / Messages schema |

- **鉴权：** `Authorization: Bearer <key>`（OpenAI 风格）或 `x-api-key`（Anthropic 风格，看入口）
- 同一个 API key 两个入口都通用
- **关键意义**：从适配层角度，DeepSeek **不算第三种协议**，而是 OpenAI/Anthropic 协议的另一个 endpoint。复用 OpenAI/Anthropic 适配器，仅 base_url 切换 + 模型名切换即可基本跑通

但仍有**两类必须单独处理的差异**：DeepSeek-specific 字段（`reasoning_content`、缓存字段）+ DeepSeek 独有错误码（402）

---

## 2. 模型一览（2026-06 当前）

### 2.1 现役模型

| Model ID | 类型 | 上下文 | 备注 |
|---|---|---|---|
| `deepseek-v4-flash` | 快速/便宜 | 128K | 284B MoE（13B active），79% SWE-bench |
| `deepseek-v4-pro` | 旗舰推理 | 128K（1M beta `[1m]` 后缀） | 1.6T MoE（49B active），SWE-bench 80.6%，Codeforces #1 |
V4 模型的"思考模式"通过 `extra_body={"thinking": {"type": "enabled"}}` 开启，强度通过顶层 `reasoning_effort` 参数控制（**不是**嵌套在 thinking 对象内）。

### 2.2 即将废弃模型

⚠️ **2026-07-24 15:59 UTC 失效，不会回退**：

| 旧 ID | 当前实际指向 |
|---|---|
| `deepseek-chat` | `deepseek-v4-flash` 的非思考模式 |
| `deepseek-reasoner` | `deepseek-v4-flash` 的思考模式（内置推理，不使用 `thinking` 参数） |

### 2.3 Anthropic 入口的模型映射

走 `/anthropic` 入口时，请求 body 里的 model 字段会被 DeepSeek 自动映射：

| 请求中的名字 | 实际使用 |
|---|---|
| `claude-opus-*` | `deepseek-v4-pro` |
| `claude-sonnet-*`, `claude-haiku-*` | `deepseek-v4-flash` |
| 任何不识别的名字 | `deepseek-v4-flash`（兜底） |
| `deepseek-v4-pro` / `deepseek-v4-flash` | 原样使用 |

→ 适配层若走 Anthropic 入口，**建议直接传 DeepSeek 原生 model 名**，避免靠映射。

---

## 3. Chat Completions（OpenAI 入口）

### 3.1 Schema

完全兼容 OpenAI Chat Completions：

```json
{
  "model": "deepseek-v4-flash",
  "messages": [...],
  "tools": [...],
  "tool_choice": "auto",
  "temperature": 0.7,
  "max_tokens": 4096,
  "stream": false,
  "reasoning_effort": "high",
  "extra_body": {"thinking": {"type": "enabled"}}
}
```

**Wire format 要点：**
- `reasoning_effort` 是**顶层字段**（与 OpenAI 一致），不是嵌套在 `thinking` 对象内
- `thinking` 对象只有 `type: "enabled" | "disabled"`，不携带 effort
- thinking toggle 默认 enabled（不传 `thinking` 字段时 V4 默认开启思考）
- `reasoning_effort` 仅支持 `"high"`（默认）和 `"max"`；`low`/`medium` 降级为 `high`，`xhigh` 降级为 `max`

OpenAI SDK 直接换 `base_url` + `api_key` 即可。

### 3.2 DeepSeek 独有字段：`reasoning_content`

**响应消息结构**（思考模式开启时）：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The answer is 42.",            // 最终回答
      "reasoning_content": "Let me think...",     // CoT 推理过程
      "tool_calls": [...]                         // 可选
    }
  }]
}
```

- `reasoning_content` 与 `content` **同级**，是 DeepSeek 自定义扩展字段
- 思考模式下，`reasoning_content` 长度可达 32K（不计入 64K 总上下文配额）
- 思考模式下**不支持**的参数：`temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`（传了不报错，但被忽略）
- **billing：** reasoning tokens 单独计费（按 output token 价计）

### 3.3 多轮回喂 reasoning_content 的规则（重要陷阱）

| 模型 / 场景 | 是否要回传 reasoning_content |
|---|---|
| Legacy `deepseek-reasoner` | **绝对不能**，否则 400 错误 |
| V4 思考模式，无 tool call | 可省略，传了也会被忽略 |
| V4 思考模式，**有 tool call** | **必须回传**，否则 400 错误 |

```python
# V4 + tool call 多轮：
messages.append({
    'role': 'assistant',
    'content': response.choices[0].message.content,
    'reasoning_content': response.choices[0].message.reasoning_content,  # 必须！
    'tool_calls': response.choices[0].message.tool_calls,
})
```

**对适配层的影响：** 必须把 reasoning_content 作为 first-class 字段保留到内部 Message 模型，否则无法支持 V4 思考模式 + 工具调用。

### 3.4 流式

OpenAI 兼容的 SSE，但 delta 多了 `reasoning_content` 字段：

```python
for chunk in response:
    delta = chunk.choices[0].delta
    if delta.reasoning_content:
        reasoning_buf += delta.reasoning_content
    elif delta.content:
        content_buf += delta.content
```

**特殊：** reasoning_content 先到，content 后到。两者**不交错**：思考结束后才开始输出 content。

**Keep-alive 行为：** 高负载时，服务器可能返回 SSE 注释行（以 `:` 开头）作 keep-alive，**不是错误**，要跳过。10 分钟未开始推理则关连接。读超时建议 ≥ 60s。

### 3.5 Function Calling

完全兼容 OpenAI：tool 定义、`tool_calls` 响应、`role:"tool"` 回喂格式一模一样。strict 模式同样支持，**思考模式与非思考模式都支持**。

### 3.6 Structured Output（重要限制）

DeepSeek **不支持** `response_format: {type: "json_schema"}`——传入会返回 `invalid_request_error: "This response_format type is unavailable now"`。

`response_format.type` 仅支持两个值：
- `"text"`（默认）
- `"json_object"`（JSON 模式，保证输出合法 JSON）

**替代方案**（需要严格 Schema 约束时）：
1. **降级用 `json_object` + Prompt 约束**：在 system/user 消息里写清字段定义、类型、必填项、示例，配合 `response_format: {type: "json_object"}`，工程上加一层 JSON 校验兜底。
2. **用空工具调用迂回**：定义一个无实际执行逻辑的虚拟函数，把目标结构写在 `parameters` 里，开启 `strict: true`，强制模型输出符合 Schema 的函数参数。约束强度与 OpenAI `json_schema` 基本等价。

> **注意区分**：Function Calling 里的 `parameters` Schema 支持 `strict: true` 严格模式（完整 JSON Schema），这是**工具调用场景**的能力，不是 `response_format` 字段的能力。

---

## 4. Anthropic 入口（`/anthropic`）的兼容性

### 4.1 支持

- 文本 content blocks
- 工具调用（`tool_use` / `tool_result`）
- `system` 顶级字段
- 流式（同 Anthropic SSE 协议）

### 4.2 不支持

- `image` content block（**重要**：思考模式下也不支持，因为 V4 当前没有原生 vision）
- `document` content block（PDF 等）
- `search_result` block
- `server_tool` block（Anthropic 内置 web_search / code_execution 等）
- `web_search_tool` 内置工具
- MCP tool
- container upload

→ **第一版适配层若想统一走 OpenAI-compatible 入口**，DeepSeek 完全够用。Anthropic-compatible 入口主要价值是"无缝替换 Claude Code 的后端"，对适配层意义不大。

---

## 5. Context Caching（自动，无需显式标记）

### 5.1 工作机制

- 缓存粒度：**前缀缓存**，按字节匹配
- 缓存单元大小：**64 tokens**（小于不缓存）
- 缓存层级：磁盘缓存（不是内存）
- 服务端**自动检测**重复前缀并缓存，无需调用方加 `cache_control` 标记
- best-effort，无 100% 命中保证

### 5.2 响应字段

```json
"usage": {
  "prompt_tokens": 2048,
  "completion_tokens": 503,
  "total_tokens": 2551,
  "prompt_cache_hit_tokens": 1800,
  "prompt_cache_miss_tokens": 248
}
```

不变量：`prompt_tokens == prompt_cache_hit_tokens + prompt_cache_miss_tokens`

### 5.3 定价（2026-06）

- V4 Flash：cache miss $0.14/M，**cache hit $0.0028/M（省 98%）**，output $0.28/M
- V4 Pro（限时打折）：cache miss $0.435/M，cache hit $0.003625/M，output $0.87/M

### 5.4 对调用方的建议

把稳定的 system prompt / tool 定义 / few-shot 放前面，变化的用户内容放后面，命中率能到 70%+。

→ **对适配层：** 不需要"显式开启缓存"的 API（OpenAI 同样自动，Anthropic 显式）。适配层只需要把 hit/miss 字段映射到统一 Usage 结构。

---

## 6. 错误码

| 状态码 | 含义 | 可重试 | 说明 |
|---|---|---|---|
| 400 | 请求格式错 | ❌ | |
| 401 | 鉴权失败 | ❌ | 检查 API key |
| **402** | **余额不足** | ❌ | **DeepSeek 独有**，需充值 |
| 422 | 参数无效 | ❌ | 参数值非法（如 temperature 越界） |
| 429 | 限流 | ✅ | 退避重试 |
| 500 | 服务端错 | ✅ | |
| 503 | 服务过载 | ✅ | 退避 + circuit breaker |

**错误响应体格式**：与 OpenAI 一致（OpenAI SDK 的错误处理可直接用）。

**速率限制特点：** 不发布固定 RPM/TPM 表，**动态限流**。高负载时优先**排队**而非返回 429，可能见到很长的 SSE keep-alive。读超时要放宽。

---

## 7. 对适配层的特别影响

### 7.1 内部模型必须支持的字段（新增 / 调整）

**Message 增加 `reasoning_content` 字段：**

```rust
struct AssistantMessage {
    content: Vec<Content>,
    tool_calls: Vec<ToolUse>,
    reasoning: Option<String>,    // ← 新增。映射 DeepSeek reasoning_content；
                                   //   也可承载 Anthropic thinking block 的明文部分
}
```

**Usage 字段统一表示三家的缓存计数：**

```rust
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
    cached_input_tokens: u32,        // OAI: cached_tokens
                                      // Anth: cache_read_input_tokens
                                      // DS:   prompt_cache_hit_tokens
    cache_creation_input_tokens: u32, // Anth-only；OAI / DS 给 0
    reasoning_tokens: u32,            // DS / OAI o-series 的推理 token；Anth thinking
}
```

### 7.2 Provider 抽象的复用机会

DeepSeek 不是一个独立 provider 类型，而是 OpenAI/Anthropic 的"另一个 endpoint"。建议：

```rust
// 不要做 DeepSeekProvider，而是让 OpenAIProvider 支持 endpoint 切换
let ds = OpenAIProvider::builder()
    .api_key(key)
    .base_url("https://api.deepseek.com")
    .default_model("deepseek-v4-flash")
    .reasoning_field_mapping(true)  // 开启 reasoning_content 字段处理
    .build();
```

或者用工厂函数封装：

```rust
let ds = DeepSeek::openai_endpoint(key);   // 实际返回 OpenAIProvider
let ds = DeepSeek::anthropic_endpoint(key); // 实际返回 AnthropicProvider
```

### 7.3 新增的处理点

- ✅ OpenAI 适配器要识别 `reasoning_content` 字段并放入内部 `reasoning` 字段
- ✅ OpenAI 适配器流式解析要识别 `delta.reasoning_content` 增量
- ✅ Usage 字段要识别 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
- ✅ 错误码 enum 增加 `InsufficientBalance`（HTTP 402，DeepSeek 独有但可能未来 OpenAI 也会用）
- ✅ 思考模式多轮场景：当 `reasoning` 字段存在且历史中有 tool_call 时，回喂请求要保留 reasoning_content
- ⚠️ Anthropic 入口：第一版不优先支持（除非用户要把 DeepSeek 当 Anthropic 替身用）

### 7.4 模型字符串就让调用方自己传

不在库里硬编码模型常量。DeepSeek 模型名变化频繁（V3 → V4 Flash/Pro，legacy 别名 2026-07-24 失效），库要保持中立。

---

## 8. 来源

- 官方主入口：https://api-docs.deepseek.com/
- 中文版：https://api-docs.deepseek.com/zh-cn/
- Thinking Mode：https://api-docs.deepseek.com/guides/thinking_mode
- Function Calling：https://api-docs.deepseek.com/guides/function_calling
- Context Caching：https://api-docs.deepseek.com/guides/kv_cache
- Error Codes：https://api-docs.deepseek.com/quick_start/error_codes
- Anthropic-compatible API：https://api-docs.deepseek.com/guides/anthropic_api
- Models & Pricing：https://api-docs.deepseek.com/quick_start/pricing
