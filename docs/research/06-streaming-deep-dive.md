# 流式（Streaming）深度剖析

> 这份笔记是 `01-openai.md` / `02-anthropic.md` / `05-deepseek.md` 中流式部分的**统一深讲**，作为适配层流式模块设计的核心参考。

## 1. 底层：SSE 协议

LLM 流式输出基于 **Server-Sent Events（SSE）**，是 HTTP 之上一个非常简单的"服务器持续推送文本"协议。

### 1.1 SSE 的 HTTP 形态

```http
POST /v1/chat/completions HTTP/1.1
Content-Type: application/json
Accept: text/event-stream    ← 客户端意图

{"model": "...", "stream": true, ...}
```

服务器响应：

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream   ← 关键
Cache-Control: no-cache
Transfer-Encoding: chunked        ← 不预设 Content-Length

<然后开始持续往 socket 写数据，连接保持打开>
```

### 1.2 SSE 的消息格式

每条"事件"是一段以**空行结尾**的文本块，由若干个字段行组成：

```
field: value\n
field: value\n
\n                    ← 空行 = 一条事件结束
```

合法字段只有 5 个：`event`, `data`, `id`, `retry`，以及注释行 `:`。

**LLM 场景下基本只用 `event` 和 `data`：**

```
event: message_start
data: {"type":"message_start","message":{...}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{...}}
```

`data:` 行的值是字符串（通常是 JSON），客户端拿到后自己 `JSON.parse`。

### 1.3 注释行（很重要）

以 `:` 开头的行是 SSE 注释，客户端必须**忽略**它：

```
: keep-alive
```

LLM 服务端会在没有数据可推时定期发注释行，作用是**防止中间网关（CDN / 反向代理 / load balancer）把空闲连接断掉**。DeepSeek 高负载时甚至会发很长一串 `:` 来"占住"连接等待资源。

→ **写 SSE 解析器时第一个坑就是：忘记跳过 `:` 行，导致 JSON 解析报错。**

---

## 2. 为什么 LLM 必须用 streaming

四个理由，按重要性排序：

1. **用户体验**：模型生成几百 token 要 5-10 秒，逐字推送让用户立刻看到反馈。这是最常被提到但其实最不重要的。
2. **避免长连接超时**：当 `max_tokens` 很大（≥8K），单次响应可能要 30+ 秒。中间网关、CDN、移动网络都可能在 10-30s 上断空闲连接。流式让连接持续有数据流动，**这是技术上的硬需求**。Anthropic SDK 文档明确建议大 max_tokens 必须用流式或 Batch API。
3. **提前拿到结构化信号**：流到 `stop_reason: "tool_use"` 时就知道模型要调工具，可以**提前**开始准备工具执行环境（如启动沙箱、加载文件）。
4. **取消能力**：用户中途说"算了别答了"，流式可以直接关闭 socket 让服务端停止生成（计费也只算已生成的 token）。非流式只能等完。

---

## 3. OpenAI 流式：扁平 delta 模型

### 3.1 完整 wire 示例

一个普通文本响应的 wire 长这样：

```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700,"model":"gpt-4.1","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700,"model":"gpt-4.1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700,"model":"gpt-4.1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700,"model":"gpt-4.1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

### 3.2 设计观察

- **没有 `event:` 字段**，每条都只有 `data:`。所以每条事件**类型相同**，都是 `chat.completion.chunk`
- 流的终止用一个**特殊哨兵**：`data: [DONE]`（**不是 JSON**，是字面量字符串）
- 内容用 **delta** 字段表示"和上次相比的增量"：
  - `delta.role` 只在第一个 chunk 出现
  - `delta.content` 是文本增量
  - 中间什么都没生成的 chunk 可能是 `delta: {}`（空 delta）
  - 最后一个有数据的 chunk 带 `finish_reason`，再之后是 `[DONE]`

### 3.3 累加规则

OpenAI 的 delta 是**纯字符串拼接**模型：

| 字段 | 累加规则 |
|---|---|
| `delta.role` | 取第一次出现的值 |
| `delta.content` | 字符串拼接 |
| `delta.refusal` | 字符串拼接 |
| `delta.tool_calls` | 按 `index` 字段做数组合并，每项内字段拼接 |

伪代码：

```rust
struct State { role: Option<Role>, content: String, tool_calls: HashMap<usize, ToolCall> }

fn apply(s: &mut State, delta: Delta) {
    if let Some(r) = delta.role { s.role.get_or_insert(r); }
    if let Some(c) = delta.content { s.content.push_str(&c); }
    for tc in delta.tool_calls.unwrap_or_default() {
        let slot = s.tool_calls.entry(tc.index).or_default();
        if let Some(id) = tc.id { slot.id = id; }
        if let Some(name) = tc.function.name { slot.name = name; }
        if let Some(args) = tc.function.arguments { slot.args.push_str(&args); }
    }
}
```

### 3.4 stream_options.include_usage

默认流式响应**不带 usage 统计**（因为 usage 要全部生成完才能算出来）。要拿 usage：

```json
{"stream": true, "stream_options": {"include_usage": true}}
```

设置后，**`[DONE]` 之前会多一个 chunk**，`choices: []` 为空，`usage` 字段有完整统计：

```
data: {"id":"...","choices":[],"usage":{"prompt_tokens":50,"completion_tokens":12,"total_tokens":62,"prompt_tokens_details":{"cached_tokens":40}}}

data: [DONE]
```

→ 解析器要兼容 "choices 是空数组" 的最后一个 chunk。

---

## 4. Anthropic 流式：层级事件模型

### 4.1 完整 wire 示例

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-opus-4-8","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":42}}

event: message_stop
data: {"type":"message_stop"}

```

### 4.2 关键差异

| 维度 | OpenAI | Anthropic |
|---|---|---|
| 事件类型 | 隐式（全是 chunk） | 显式（`event:` 字段 + JSON 内 `type` 字段） |
| 终止信号 | `data: [DONE]` 哨兵 | `event: message_stop` 事件 |
| Usage 时机 | `[DONE]` 前一条 chunk（选）| `message_start` 给初值，`message_delta` 累计更新 |
| 内容结构 | 扁平 delta | 三级：message → content_block → delta |

### 4.3 事件类型一览

```
message_start          ─ 流开始，含 message 元数据（id, model, role, 初始 usage）
content_block_start    ─ 一个内容块开始，标 index 和 block 类型
content_block_delta    ─ 内容块的增量（delta 内还有 type 子字段）
content_block_stop     ─ 内容块结束（标 index）
message_delta          ─ 顶层消息更新（stop_reason、累计 usage）
message_stop           ─ 流结束
ping                   ─ 心跳，随时可能穿插，忽略即可
error                  ─ 错误（中途断流）
```

### 4.4 三级嵌套的含义

为什么 Anthropic 要搞这么复杂？因为一次响应可能有**多个内容块**并存：

```
content_block_start (index 0, type=text)        ← "我先思考一下..."
  content_block_delta × N
content_block_stop (index 0)

content_block_start (index 1, type=tool_use)    ← 然后决定调用工具
  content_block_delta × N (input_json_delta)
content_block_stop (index 1)
```

每个块用 `index` 编号，对应最终 `message.content[]` 数组的位置。**这种"块明确开始/结束"的设计在解析时比 OpenAI 的扁平 delta 更不容易出错**，因为状态变化点是显式的。

### 4.5 delta 内还有 type 子字段（容易忽略的细节）

`content_block_delta` 的 JSON 里有**两层 type**：

```json
{
  "type": "content_block_delta",           ← SSE 事件的 type
  "index": 0,
  "delta": {
    "type": "text_delta",                  ← delta 内容的 type
    "text": "Hello"
  }
}
```

delta 的 type 决定了字段名：

| `delta.type` | 字段 | 含义 |
|---|---|---|
| `text_delta` | `text` | 文本内容增量 |
| `input_json_delta` | `partial_json` | tool_use 输入的 JSON 字符串增量 |
| `thinking_delta` | `thinking` | extended thinking 的明文增量 |
| `signature_delta` | `signature` | thinking 块结束时的签名（一次性给完，不增量） |

---

## 5. DeepSeek 流式：OpenAI + reasoning_content 扩展

DeepSeek 走 OpenAI 协议，但 delta 里多一个字段：

```
data: {"id":"...","choices":[{"index":0,"delta":{"reasoning_content":"Let me think..."},"finish_reason":null}]}

data: {"id":"...","choices":[{"index":0,"delta":{"reasoning_content":" step by step."},"finish_reason":null}]}

data: {"id":"...","choices":[{"index":0,"delta":{"content":"The answer"},"finish_reason":null}]}

data: {"id":"...","choices":[{"index":0,"delta":{"content":" is 42."},"finish_reason":null}]}

data: {"id":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 关键观察

1. **reasoning_content 先到，content 后到**，两者不交错
2. delta 同时只会有一个：要么 `reasoning_content` 要么 `content`（不会同 chunk 给两个）
3. 累加规则同 OpenAI：字符串拼接
4. **高负载下大量出现 `:` keep-alive 注释行**，要跳过

### 累加伪代码

```rust
let mut reasoning = String::new();
let mut content = String::new();
for chunk in stream {
    let delta = &chunk.choices[0].delta;
    if let Some(r) = &delta.reasoning_content { reasoning.push_str(r); }
    if let Some(c) = &delta.content { content.push_str(c); }
}
```

---

## 6. Tool 调用流式：最复杂的部分

Tool 调用的 `arguments` 是 JSON。三家都用**字符串增量**传输（避免一次性大块塞），但累加细节不同。

### 6.1 OpenAI 的 tool_calls 流式

```
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"P"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"aris\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
```

**陷阱：**
- 第一个 chunk 才有 `id` 和 `function.name`，后续 chunk 只有 `function.arguments`
- 并行调用多个工具：每个 chunk 的 `delta.tool_calls` 数组里**只包含当前更新的那些**，要按 `index` 字段匹配
- arguments 是 raw 字符串拼接，**只能在 `finish_reason: "tool_calls"` 之后 `JSON.parse`**（中间是非法 JSON）

### 6.2 Anthropic 的 tool_use 流式

```
event: content_block_start
data: {"index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}

event: content_block_delta
data: {"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"loc"}}

event: content_block_delta
data: {"index":1,"delta":{"type":"input_json_delta","partial_json":"ation\":\"P"}}

event: content_block_delta
data: {"index":1,"delta":{"type":"input_json_delta","partial_json":"aris\"}"}}

event: content_block_stop
data: {"index":1}
```

**关键陷阱：**
- `content_block_start` 里的 `input` 是空 `{}` —— **占位**，不要直接用
- 真正的输入要从 `partial_json` 字符串增量拼接
- block 结束才能 `JSON.parse`（同 OpenAI）
- **类型不一致警告**：占位的 `input` 是 object，partial_json 是 string，这个不一致是设计如此（schema 上 `input` 是 object，但增量过程中是 string 累积）

### 6.3 一个加速选项：Anthropic 的 fine-grained tool streaming

默认行为：模型一次性输出一个完整 key-value pair，所以 `partial_json` 增量之间可能有较长间隔（模型在思考下一组）。

加 beta header `fine-grained-tool-streaming-2025-05-14`，模型立刻吐字符不再缓冲，**但**：

- 中间状态可能是非法 JSON
- 如果撞到 `max_tokens` 在半路截断，**最终的 partial_json 也可能是残缺的 JSON 字符串**
- 调用方要写额外的容错处理

第一版适配层**不要默认开**这个，作为可选 flag 暴露。

---

## 7. 推理流式（reasoning_content / thinking）

三家对"模型思考过程"的暴露方式完全不同。

### 7.1 OpenAI（Chat Completions 路径）

- **完全不暴露**。`reasoning_effort: "high"` 只控制思考强度，思考内容**不返回**（只是会被算进 output tokens 计费）
- Responses API 才有 `reasoning` item 类型，但仍然加密或只给 summary

### 7.2 DeepSeek

- 思考模式下，**明文返回**思考内容（`reasoning_content`）
- 流式时**先把 reasoning_content 全部推完，再开始推 content**
- 思考可达 32K token，**这段时间客户端只看到 reasoning_content 在长**，主答案要等很久才开始
- 调用方做 UI 时要分两栏（推理过程 / 最终回答），不能塞进同一个文本流

### 7.3 Anthropic（Extended Thinking）

- 启用 `"thinking": {"type": "enabled", "budget_tokens": N}` 才有
- 思考是**独立的 content block**（`type: "thinking"`），和 text block / tool_use block 并列在 content[] 数组里
- 流式时：

```
event: content_block_start
data: {"index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"index":0,"delta":{"type":"thinking_delta","thinking":"Let me consider..."}}

event: content_block_delta
data: {"index":0,"delta":{"type":"signature_delta","signature":"sig_xxx"}}

event: content_block_stop
data: {"index":0}

event: content_block_start
data: {"index":1,"content_block":{"type":"text","text":""}}
...
```

`signature_delta` 是 thinking 块结束时给出的**签名**。多轮对话回传 thinking 块时**必须保留 signature**，否则模型拒绝处理（防伪）。

---

## 8. 生产环境的坑

按踩坑频率排序：

### 8.1 忘记跳过 `:` 注释行

DeepSeek 高负载时会发大量 keep-alive `:` 行，OpenAI / Anthropic 在长 max_tokens 场景也会。**SSE 解析器必须忽略以 `:` 开头的行**。

### 8.2 把 `data: [DONE]` 当 JSON 解析

OpenAI 和 DeepSeek 的终止哨兵是字面量 `[DONE]`，不是 `"[DONE]"`。代码要在 `JSON.parse` 前先检查这个字符串。

### 8.3 读超时设置过短

OpenAI / Anthropic 在生成期间会持续推数据，超时短点没事（30s 够）。**DeepSeek 思考模式排队时可能 60s+ 没有数据**（只有 keep-alive），读超时建议 ≥ 90s。

### 8.4 中途断流（partial response）

任何 provider 都可能在生成中途因为网络/服务端问题断流。客户端可能看到：

- OpenAI：缺 `[DONE]`，最后一个有效 chunk 的 `finish_reason` 是 `null` 而非 `"stop"`
- Anthropic：缺 `message_stop`；或者中途收到 `event: error`

适配层要把"中途断流"作为一类**单独的错误**返回，调用方决定是否重试或拼凑已有结果。

### 8.5 max_tokens 截断时 tool_use 输入是非法 JSON

如果 `stop_reason` 是 `"max_tokens"` 而当时正在流 tool_use 的 partial_json，**最终拼出来的 input 可能是 `{"location": "Pari`** 这样残缺的 JSON。

适配层不要在 `JSON.parse` 失败时崩溃，应该把 partial_json 原文也保留下来，让调用方有挽救机会（比如截断到最近的合法位置）。

### 8.6 流式 + 工具的二次调用

模型一旦决定调工具，整个 stream 在 tool_use 块结束后就停了。**这不是"流式的结尾是最终答案"**，调用方拿到 stream 结束后，要：

1. 收集所有 tool_use 块
2. 执行工具
3. 用 tool_result 发**新的请求**（再次流式）
4. 这个 chain 可能循环多次

适配层不要假设"一次 chat_stream 调用对应一次完整对话"。

### 8.7 SDK reasoning_content 字段的多轮处理

DeepSeek V4 思考模式 + tool 调用场景：**reasoning_content 必须在历史里完整保留**（包括非流式接口），否则 400。

这不是流式独有的坑，但流式时容易遗漏：开发者在累加完 stream 后构造 history 时容易只塞 `content` 漏掉 `reasoning_content`。

---

## 9. 适配层的归一化设计建议

### 9.1 统一事件枚举（向 Anthropic 看齐 + 扩展）

```rust
enum StreamEvent {
    /// 流开始，给出消息元数据
    Start {
        message_id: String,
        model: String,
    },

    /// 一个内容块开始
    BlockStart {
        index: usize,
        kind: BlockKind,           // Text | ToolUse | Reasoning | Thinking
    },

    /// 文本内容增量
    TextDelta {
        index: usize,
        text: String,
    },

    /// 工具调用参数增量（JSON 片段）
    ToolUseDelta {
        index: usize,
        partial_json: String,
    },

    /// 推理/思考内容增量
    ReasoningDelta {
        index: usize,
        text: String,
    },

    /// thinking 块的签名（Anthropic 用）
    ThinkingSignature {
        index: usize,
        signature: String,
    },

    /// 内容块结束
    BlockStop {
        index: usize,
    },

    /// 累计 usage 更新（可能多次）
    UsageUpdate(Usage),

    /// 流结束
    End {
        stop_reason: StopReason,
    },
}
```

### 9.2 OpenAI / DeepSeek 适配器要做"合成"工作

OpenAI 的扁平 delta 模型里**没有显式的 block 开始/结束**，适配器要"重建"这些事件：

```rust
// 伪逻辑
struct OpenAiStreamAdapter {
    state: BlockState,    // 当前在哪个块里：None | InText | InToolUse(idx) | InReasoning
}

fn on_chunk(&mut self, chunk: OAIChunk) -> Vec<StreamEvent> {
    let delta = chunk.choices[0].delta;
    let mut out = vec![];

    // 文本块切换
    if delta.content.is_some() && !matches!(self.state, BlockState::InText) {
        self.end_current_block_if_any(&mut out);
        out.push(StreamEvent::BlockStart { index: 0, kind: BlockKind::Text });
        self.state = BlockState::InText;
    }
    if let Some(c) = delta.content {
        out.push(StreamEvent::TextDelta { index: 0, text: c });
    }

    // 推理内容（DeepSeek）
    if delta.reasoning_content.is_some() && !matches!(self.state, BlockState::InReasoning) {
        self.end_current_block_if_any(&mut out);
        out.push(StreamEvent::BlockStart { index: 0, kind: BlockKind::Reasoning });
        self.state = BlockState::InReasoning;
    }
    // ... 同理

    // 工具调用（每个 index 一个块）
    for tc in delta.tool_calls.unwrap_or_default() {
        // 第一次见到这个 index → BlockStart
        // 后续 → ToolUseDelta（push partial_json）
    }

    // 终止
    if let Some(reason) = chunk.choices[0].finish_reason {
        self.end_current_block_if_any(&mut out);
        out.push(StreamEvent::End { stop_reason: map_reason(reason) });
    }
    out
}
```

### 9.3 Anthropic 适配器几乎是 1:1 直传

```rust
fn map_event(ev: AnthropicEvent) -> Option<StreamEvent> {
    match ev {
        Ping => None,
        MessageStart { message } => Some(Start { message_id: message.id, model: message.model }),
        ContentBlockStart { index, content_block } => {
            let kind = match content_block.type_ {
                "text" => BlockKind::Text,
                "tool_use" => BlockKind::ToolUse,
                "thinking" => BlockKind::Thinking,
                _ => BlockKind::Other,
            };
            Some(BlockStart { index, kind })
        }
        ContentBlockDelta { index, delta } => match delta.type_ {
            "text_delta" => Some(TextDelta { index, text: delta.text }),
            "input_json_delta" => Some(ToolUseDelta { index, partial_json: delta.partial_json }),
            "thinking_delta" => Some(ReasoningDelta { index, text: delta.thinking }),
            "signature_delta" => Some(ThinkingSignature { index, signature: delta.signature }),
            _ => None,
        },
        ContentBlockStop { index } => Some(BlockStop { index }),
        MessageDelta { delta, usage } => Some(UsageUpdate(usage)),
        MessageStop => Some(End { stop_reason: ... }),
        Error(e) => Err(...),
    }
}
```

### 9.4 提供"事件流 → 完整消息"的折叠器

很多调用方只想要最终结果（不关心增量）。库提供便利方法：

```rust
impl Stream {
    async fn collect_message(self) -> Result<Message> {
        let mut blocks = Vec::new();
        let mut current_text = String::new();
        let mut current_tool_json = String::new();
        let mut stop_reason = None;
        let mut usage = Usage::default();

        while let Some(event) = self.next().await {
            match event? {
                BlockStart { kind, .. } => { /* 准备新块 */ }
                TextDelta { text, .. } => current_text.push_str(&text),
                ToolUseDelta { partial_json, .. } => current_tool_json.push_str(&partial_json),
                BlockStop { .. } => { /* 把当前块 push 到 blocks */ }
                UsageUpdate(u) => usage = u,
                End { stop_reason: r } => stop_reason = Some(r),
                _ => {}
            }
        }
        Ok(Message { content: blocks, stop_reason, usage })
    }
}
```

### 9.5 测试策略

流式测试最有效的方式是**录制真实响应字节流回放**：

1. 用真实 API key 跑一次，把所有 SSE 字节存到 `tests/fixtures/openai_text_stream.txt`
2. 测试时用 `mockito` / `wiremock-rs` 让请求命中本地 mock，mock 直接返回 fixture
3. 验证：
   - 事件序列完全匹配预期
   - `collect_message()` 拼出的最终消息和原始 SSE 单次响应一致
   - 注入"中途断流"、"非法 JSON"、`:` 心跳混入，验证容错

### 9.6 Rust 生态选型建议

- **HTTP / SSE：** `reqwest` + `reqwest-eventsource` 或 `eventsource-stream`。不要手写 SSE 解析（处理 chunked encoding、CR/LF、字段折行很容易出错）
- **流式接口：** `futures::Stream<Item = Result<StreamEvent>>`，让调用方用 `.next().await` 消费
- **取消：** 利用 `Stream` 被 drop 时自动关闭底层连接（reqwest 默认行为），无需特殊 API
- **超时：** 用 `tokio::time::timeout` 包裹 `next()`，区分"读超时"与"总超时"