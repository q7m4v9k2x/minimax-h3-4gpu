# MiniMax H3 生成链路拆解与瓶颈报告

本报告把一次请求拆成可以独立计时和验收的阶段。文件中的早期 VAE 未测量结论已被 2026-09-24 的真实端到端实验更新；最新证据见 `docs/EXPERIMENTS.md` 和 `reports/remote-h3-e2e-20260924-api.json`。

## 流程图

<div class="mermaid">
flowchart LR
    A[API 请求与鉴权] --> B[参数校验<br/>尺寸/帧数/步数]
    B --> C[有界队列<br/>job_id]
    C --> D[文本/参考图 conditioning]
    D --> E[TP4 prepare<br/>packing/scheduler]
    E --> F[DiT 采样 N 步<br/>4×V100 同步]
    F --> G[释放或卸载 DiT]
    G --> H[VAE 解码<br/>空间 tile/时间 chunk]
    H --> I[媒体编码<br/>MP4/PNG]
    I --> J[临时文件写入<br/>fsync + 原子 rename]
    J --> K[历史状态与 API 响应]
    K --> L[刷新/断线恢复<br/>按 job_id 查询]
</div>

## 各阶段实测与问题

| 阶段 | 当前证据 | 瓶颈/问题 | 已做或应做的修复 |
|---|---|---|---|
| 请求、校验 | 现有仓库没有 API 计时 | 几何校验可被 target_shape 绕过；超限会把 OOM 推迟到 CUDA | 在入口同时限制像素数、帧数和 packed rows，超限返回 4xx |
| 队列、调度 | 没有服务端队列报告 | TP4 一次占用四卡，盲目并发会造成显存和 NCCL 竞争 | 单个常驻 TP4 worker + 有界队列；满载返回 429 queue_full |
| conditioning | 预计算 bundle 读取约 0.04 s | 这不是 Qwen3-VL 真正编码时间；冷启动和缓存命中没有测量 | 编码器独立进程或预计算；按 prompt/模型/编码器版本做有界 LRU |
| prepare | 480×864 约 0.055–0.068 s | 当前占比很小；高分辨率会增加 packing 和复制 | 固定 geometry bucket，避免请求级重复 dtype/device 转换 |
| DiT | 480×864 占 pipeline 约 99.6%，20 步约 121–124 s | 这是决定吞吐的主瓶颈；640×1152 接近 16GB 上限 | TP4 常驻、原生 SDPA、关闭每块 finite 检查；高分辨率实验使用 FF 分块 |
| VAE | 已完成首个端到端测量 | 864×480、124 帧的官方视频/音频 VAE 解码和 MP4 编码已通过；不同分辨率仍需独立计时 | 保持 DiT 退出后再解码；后续按 tile/time chunk 优化 |
| 编码、落盘 | API 端到端已通过 | API 返回可下载 MP4，仍应补充更多异常中断场景 | 临时文件、fsync、原子 rename；记录文件 SHA256 |
| 历史与响应 | 当前未测 | 浏览器状态不能作为任务状态；刷新/中断恢复缺少证据 | 以 job_id 持久化阶段、错误和最终文件路径，提供查询接口 |

## 已有 TP4 数据

阶段分析器：

    python scripts/analyze_h3_report.py reports/remote-h3-tp4-gate.json reports/remote-h3-tp4-gate-chunked-compare.json --json-out reports/stage-summary.json --markdown-out reports/stage-summary.md

未分块基线（480×864、124 帧、20 步）：

- conditioning：约 0.039 s；
- prepare：约 0.068 s；
- DiT：约 123.680 s，step P50/P95 约 6.162/6.181 s；
- pipeline：约 124.161 s，DiT 占约 99.6%；
- 峰值 allocated：12.48 GiB（13.40 GB），reserved：13.33 GiB（14.32 GB）。

关闭 block finite 检查并启用 h3_ff_chunk_rows=4096 后：

- conditioning：约 0.041 s；
- prepare：约 0.055 s；
- DiT：约 121.106 s，step P50/P95 约 6.035/6.052 s；
- 峰值 allocated：12.22 GiB（13.12 GB），reserved：12.81 GiB（13.76 GB）；
- latent 仍 finite，但分块改变 GEMM 舍入顺序，不能要求和基线逐位一致。

严格 A/B 结果显示，关闭 finite 检查本身才是 480×864 的主要速度收益：安全配置（无分块、无 finite 检查）DiT 119.218 s，较此前开启 finite 检查的 123.680 s 快约 3.6%；同样无 finite 检查下，4096 行分块为 121.106 s，约慢 1.6%，它的价值是降低峰值显存而不是提速。高分辨率时 8192 行分块在不增加峰值的情况下比 4096 行略快，因此只把它用于 640×1152 实验档。

640×1152 的 4096 行分块实验约 289 s，峰值约 13.99/14.70 GiB allocated/reserved；随后把分块大小调到 8192，实测 DiT 287.734 s、pipeline 288.300 s，峰值不变，step P50/P95 约 14.395/14.419 s。原始路径缺约 556 MiB OOM。CPU block offload 在 32 GiB 主机内存上也会被系统终止。当前高分辨率实验优先使用 8192，但仍需真实 conditioning 和最终 VAE 画质回归。

## 优化顺序

1. **先守住入口**：拒绝超过已验收 token bucket 的请求，避免把队列问题伪装成 CUDA OOM。
2. **让 TP4 常驻**：模型加载约 85.7 s，只允许在 worker 启动或崩溃恢复时发生。
3. **默认走 safe 配置**：config/lightx2v-v100-tp4-16gb-safe.json 关闭逐块 finite 检查；首次请求或固定采样再打开 debug 检查。
4. **高分辨率单独开关**：h3_ff_chunk_rows=4096 只用于 640×1152 实验，必须做真实 conditioning 和最终 VAE 画质回归。
5. **补齐 VAE 后再优化尾段**：先测 tile/time chunk，再决定是否让四卡分摊 VAE；不能用理论并行度代替实测。
6. **最后压测队列和持久化**：并发 1/2/4/8 只提交给一个 TP4 worker，记录 queue wait、P50/P95/P99、失败恢复和刷新可见性。

现阶段最明确的结论是：DiT 占据已测 pipeline 的绝大部分时间；conditioning/prepare 不是当前主瓶颈，VAE 和 API 尾段仍是测量空白。任何端到端吞吐承诺都要等这些阶段有实际报告后再下结论。
