# LLM Adapter 研究笔记 - 索引

> **更新日期：** 2026-07-03
> **范围：** OpenAI + Anthropic + DeepSeek 当前主力 API
> **目的：** 为 Rust 适配层设计提供归一化依据，不是完整 API 手册

## 文件结构

| 文件 | 内容 |
|---|---|
| `01-openai.md` | OpenAI Chat Completions API + Responses API + 流式 + 工具调用 + 错误 + 模型 |
| `02-anthropic.md` | Anthropic Messages API + 流式 + 工具调用 + 错误 + 模型 + Prompt Caching |
| `03-capability-matrix.md` | 三家能力侧边对比 + Rust 设计要点 |
| `04-design-implications.md` | 归一化策略建议（trait / enum / 保留 raw 字段） |
| `05-deepseek.md` | DeepSeek API（OpenAI 兼容 / Anthropic 兼容双入口）+ reasoning_content + Context Caching |
| `06-streaming-deep-dive.md` | 流式专题：SSE 协议 + 三家 wire format + tool/reasoning 流式 + 生产坑 + Rust 实现建议 |

## 信息来源（按权重）

> 笔记内容最初基于 2026-06-02 的网络搜索结果。2026-07-03 根据 OpenAI 官方 Reasoning models 文档、Anthropic 官方 Extended thinking 文档、DeepSeek 官方 Thinking Mode 文档进行了复核和修正。

**OpenAI 官方入口**
- API Reference：`platform.openai.com/docs/api-reference`（或 `developers.openai.com/api/reference`）
- Function calling：`developers.openai.com/api/docs/guides/function-calling`
- Streaming：`developers.openai.com/api/docs/guides/streaming-responses`
- Migrate to Responses：`developers.openai.com/api/docs/guides/migrate-to-responses`
- OpenAPI 规范：`github.com/openai/openai-openapi`（`openapi.yaml`）—— 单一可信源，强烈建议设计阶段用 spec 而非 doc

**Anthropic 官方入口**
- API Reference：`docs.anthropic.com/en/api/messages`（或新 `platform.claude.com/docs/en/api/messages`）
- Build with Claude：`docs.anthropic.com/en/docs/build-with-claude/{streaming,prompt-caching,extended-thinking,...}`
- Tool use：`docs.anthropic.com/en/docs/agents-and-tools/tool-use/{overview,implement-tool-use,fine-grained-tool-streaming}`
- Errors：`docs.anthropic.com/en/api/errors`
- 错误代码 cheat sheet：`github.com/anthropics/skills`（`skills/claude-api/shared/error-codes.md`）

**DeepSeek 官方入口**
- 主文档：`api-docs.deepseek.com/zh-cn/` 或 `api-docs.deepseek.com/`
- Thinking Mode：`api-docs.deepseek.com/guides/thinking_mode`
- Function Calling：`api-docs.deepseek.com/guides/function_calling`
- Context Caching（KV 缓存）：`api-docs.deepseek.com/guides/kv_cache`
- Error Codes：`api-docs.deepseek.com/quick_start/error_codes`
- Anthropic 兼容入口：`api-docs.deepseek.com/guides/anthropic_api`
- Models & Pricing：`api-docs.deepseek.com/quick_start/pricing`

## 重要时效说明

- 这些 API 更新很快。Opus 4.8（May 2026）、GPT-5.5、Responses API（2025-03 发布）、DeepSeek V4（2026 上半年）都是近期变动
- OpenAI 在推**新的 Responses API 替代 Chat Completions**，但 Chat Completions 仍长期支持
- 适配层设计应该考虑**同时支持两套 OpenAI API**，或先做 Chat Completions（更稳定、生态更广）后做 Responses
- DeepSeek 的 `deepseek-chat` / `deepseek-reasoner` **2026-07-24 失效**，库要保持模型名中立，不硬编码

## Provider 关系图（澄清 DeepSeek 的特殊定位）

```
DeepSeek 不是新协议，是在 OpenAI / Anthropic 协议上再开两个 endpoint：

  内部统一接口
       ↓
   ┌───┴────┐
   ↓        ↓
OpenAI    Anthropic    （两种 wire protocol）
protocol  protocol
   │         │
   ├──→ api.openai.com           ┌──→ api.anthropic.com
   ├──→ api.deepseek.com         └──→ api.deepseek.com/anthropic
   ├──→ api.deepseek.com/v1
   └──→ <future providers using OAI-compat>
```

→ **DeepSeek 适配 = 复用 OpenAI 适配器 + base_url 切换 + 处理 reasoning_content 字段**。设计阶段要把 endpoint 与 protocol 解耦。
