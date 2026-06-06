# TODO — remaining work after 0.1.0

## 1. 结构化输出 JSON Schema 的 canonical 层支持

**为什么要做：** 这是目前唯一多个 provider 都有但 canonical 层完全没有暴露的能力。OpenAI 有 `response_format`（`json_object` / `json_schema`），Anthropic 也有类似支持（虽然后端实现不同）。用户走 universal path 时无法使用结构化输出，只能绕道 native 路径用 `extra_body` 透传，失去了 canonical 层的统一性。Agent 框架依赖稳定的 JSON 输出来解析工具结果，如果 adapter 层不暴露，上游框架就得绕过 canonical 层直接操作 wire schema。

**要做的事：**
- 在 `ChatRequest` 支持 `response_format` 参数（`json_object` / `json_schema`）
- 确认 Anthropic 是否也需要暴露——若需，做归一化；若 Anthropic 比 OpenAI 差异大，各自走 native 路径
- 更新 README

## 2. Provider 能力声明（运行时可查询）

**为什么要做：** 当前没有任何运行时可查询的机制表明"这个 provider 支持什么"。信息散落在 README 表格、builder 注释和用户的踩坑经验中。比如用户想判断当前 provider 是否支持图片输入、是否支持 reasoning、是否支持 JSON schema，只能靠自己的 if/else 硬编码或查文档。缺少这种能力查询接口，上游框架（如 Agent）就没办法根据 provider 自适应——必须自己维护一张能力映射表，这在多 provider 场景下是不可持续的。另外 DeepSeek V4 Flash 是否支持图像输入目前没有验证，需要通过测试确认。

**要做的事：**
- 设计 `ProviderCapabilities` 类型，暴露接口如 `.supports_images()`、`.supports_reasoning()`、`.supports_json_schema()` 等
- 每个 Provider 实现返回自身能力
- 验证 DeepSeek V4 Flash 的图像支持（加集成测试，通过 OpenAI endpoint 发 `image_url` 请求看是否 4xx 还是正常返回）
- 更新 README
