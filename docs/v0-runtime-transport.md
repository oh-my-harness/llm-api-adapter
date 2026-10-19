# V0.1.1 模型请求传输接入说明

V0 Runtime 通过工作区的 `Cargo.toml` 和 `Cargo.lock` 将适配器固定为
`864050882254e12cbda33359debae40a03a4b35f`。不会自动使用同级适配器源码目录。
首次配置 V0 时，不要将此依赖换成软件包仓库版本或本地路径覆盖。

完整入口见 [V0 快速开始](https://github.com/oh-my-harness/senza-studio/blob/review/senza-studio-v011-20261019-658ae4bb/docs/v0-quickstart.md)。
本仓库当前文档 PR 基于更新后的 `main`，但本说明的行为及证据范围明确限定为上述 V0 固定提交；
不将本仓库后续新增的 Responses 接口行为混入 V0 验证结论。

V0 团队根据 Runtime 持久化的面板设置及成员覆盖配置组装 `OpenAIProvider`，
不从适配器 README 中的环境变量示例读取团队设置。
`OpenAIProvider::builder(key)` 显式接收密钥。自定义基础地址后会追加
`/v1/chat/completions`，因此应填写服务根地址，不要填写完整接口地址或已经带 `/v1` 的地址。
库提供 `chat_path` 构建参数，但当前 V0 成员／全局设置界面不提供该选项；
网关路径不兼容时应记录为部署限制。

本机原生目标在此版本使用默认的 reqwest `ClientBuilder`，未显式指定代理或调用 `.no_proxy()`。
Runtime 锁定的 reqwest 0.12.28 支持代理环境变量及 `NO_PROXY`/`no_proxy`。
应在启动 Runtime 的进程环境中、模型客户端构建前设置这些变量。
wasm 的浏览器传输属于另一条路径，不属于这里的 V0 源码启动方式。

适配器单独声明 `rustls-tls`；Runtime 在合并后的 reqwest 依赖上额外启用
`rustls-tls-native-roots`，因此 Runtime 本机原生构建可读取操作系统信任库，
或使用 rustls-native-certs 的 `SSL_CERT_FILE`/`SSL_CERT_DIR` 配置。
不能假定所有独立适配器使用方都继承了 Runtime 的 CA 特性。Windows 与 WSL 各自维护信任库。

2026-10-19 的本机对照中，同一合成请求经本地代理返回 HTTP 502、正文为空；直连返回 HTTP 200。
随后绕过验证观察到直连目标的网络连接及 HTTP 200。这些是本机、单次观察结果，
不是普遍绕过代理的建议。上游空错误可能缺少诊断细节，空 `server error` 不能证明
`tool_choice`、模型或任务本身导致故障。

本次文档审查不修改传输代码。按需网络配置、持久化限制和恢复记录见
[网络排错说明](https://github.com/oh-my-harness/senza-studio/blob/review/senza-studio-v011-20261019-658ae4bb/docs/v0-network.md)。
