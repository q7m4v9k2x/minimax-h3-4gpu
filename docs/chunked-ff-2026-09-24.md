# V100 分块前馈实验（2026-09-24）

为让 16GB V100 通过更大的空间尺寸，在 LightX2V 的 H3 推理器中加入了一个**显式选择**的 SwiGLU 行分块路径。配置项是 `h3_ff_chunk_rows`；默认值为 `0`，保持上游整段计算。`4096` 会按序列行分块计算 `in_proj → SiLU 门控 → out_proj`，每块写回 FP32 输出，避免一次性保留完整的 `[sequence_rows, 14336]` 中间张量。补丁见 [`patches/lightx2v-h3-chunked-ff.patch`](../patches/lightx2v-h3-chunked-ff.patch)，实验配置见 [`config/lightx2v-v100-tp4-16gb-chunked.experimental.json`](../config/lightx2v-v100-tp4-16gb-chunked.experimental.json)。

## 实机结果

目标工作站为物理 GPU 1–4（4×Tesla V100-SXM2-16GB，TP=4），固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和 `CUDA_VISIBLE_DEVICES=1,2,3,4`，原生 `torch_sdpa`，20 次 DiT 评估：

| 几何 | 结果 | 每卡峰值 allocated / reserved | DiT / pipeline | 输出 |
|---|---|---:|---:|---|
| 480×864、124 帧 | 分块对比通过 | 12.22 / 12.81 GiB（13.12 / 13.76 GB） | 121.11 / 121.59 s | finite |
| 640×1152、124 帧 | 分块通过 | 13.99 / 14.70 GiB（15.02 / 15.78 GB） | 288.52 / 289.10 s | finite |

640×1152 的未分块路径在同一环境下 OOM，缺少约 556 MiB；开启 CPU block offload 则耗尽约 32 GiB 主机内存并被系统终止。分块路径在 640×1152 的推理阶段四卡均达到约 100% 利用率，但该结果仍是 DiT-only，不能代表包含文本编码器和 VAE 的端到端生成时延。

## 数值边界

分块改变了 GEMM 的批次和舍入顺序，因此不能要求与未分块路径逐位一致。相同 prompt、seed、geometry 和 20 步的 480×864 对比中：

- 未分块 latent SHA256：`f569a163b0ea0526f8619d2e4d48d9624c93f5ac851b0a326c103e5957721088`；
- 分块 latent SHA256：`387d4ab6047025321741e1fbeef6764d276538b38d7527e24b2826b97aa09f46`；
- video latent 的最大绝对差约 `3.91`，均方根差约 `0.108`，两条路径均 `finite=true`。

因此 `h3_ff_chunk_rows=4096` 只应作为高分辨率实验开关；生产发布前必须用真实 conditioning、固定验收样例和最终 VAE 输出重新做画质回归。需要逐位复现时保持 `h3_ff_chunk_rows=0`。

## 使用建议

- 先使用 `480×864` 的 `tp4-safe` 配置；要尝试 `640×1152`，显式加载分块实验配置并设置有界队列。
- 不要把 `640×1152` 以上尺寸或更长时长视为已支持能力；dense attention 的序列和临时缓冲仍会快速增长。
- `h3_finite_check=true` 适合门禁和抽样，稳定后可在生产关闭，以减少全量 mask 和同步开销。
