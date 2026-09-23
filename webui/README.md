# YMZX AI 工作台

`index.html` 是部署在 Oracle 140 上的统一入口。它通过同源健康检查自动识别：

- `Qwen Image 2.1`：`/router-health/image` → 本地工作站的 Qwen 网关；
- `MiniMax H3`：`/router-health/h3` → 预留的 H3 网关；
- `LLM`：`/router-health/llm` → 当前实际运行的 vLLM 稳定入口。

`h3.html` 只在 H3 网关和 VAE 都在线时允许提交任务。当前工作站已验证 H3 TP4 DiT latent，但没有视频/音频 VAE，所以页面会保持未就绪，不会把 latent 伪装成成品视频。
