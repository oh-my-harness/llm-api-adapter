# TODO — remaining work after 0.1.0

## 1. README + API 文档

所有 public 类型补全 doc comments（英文，统一风格），README 加 Quickstart 和功能展示。

- **Doc comments**：`ChatRequest`、`ChatRequestBuilder`、`Message`、`ResponseContent`、`ToolInvocation`、`Provider`、`StreamHandle`、`LlmError` 等核心类型的每个 public 字段、方法、enum variant 补上用途说明和适用场景
- **README Quickstart**：Cargo.toml 加依赖 → 创建 provider → 单次 chat → stream → 多轮 + tool call，3 分钟跑通
- **README 对比表**：同一 prompt 在 Claude 和 GPT 下的输出差异（text + reasoning + tool_call）
- **README extra_body**：展示通过 `extra_body` 透传 provider 特有参数
- **README 配置选项表**：各 provider builder 的可选项一览

## 2. 能跑的例子

- 改进 `examples/runtime_switch.rs`：演示 `Box<dyn Provider>` 动态切换，同一 `ChatRequest` 在 OpenAI 和 Anthropic 下的差异

## 3. CI + 测试覆盖率

参考 `~/Documents/projs/Lattice/.github/workflows/ci.yml`，建 GitHub Actions CI。

- `.github/workflows/ci.yml`：fmt + clippy + build + test + doc check + coverage
- Coverage 用 `cargo-llvm-cov` + Codecov，README 加 badge
- 暂不包括 integration tests
