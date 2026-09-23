# MiniMax H3 4×V100 静态优化审查（2026-09-23）

本文只基于代码和已有 TP4 gate 做静态审查，没有启动新的长时间 GPU 任务，也没有修改模型权重。

## 已验证基线

工作站的 TP4 gate 使用物理 GPU 1–4、FP16 权重、敏感层 FP32、原生 `torch_sdpa`，输入为 480×864、124 帧、20 次 DiT 评估：

- 权重分片加载约 85.7 秒；模型常驻后每卡 allocated 约 10.60 GiB。
- DiT 峰值约 13.4 GiB（每卡总显存约 15.77 GiB），20 步约 124.16 秒，即约 6.18 秒/步。
- 该 gate 使用随机 conditioning，只证明加载、TP 通信和 finite 输出，不能作为画质或生产吞吐承诺。

模型应保持单个 TP4 进程常驻，避免每个请求重新加载 10 个分片。

## 高分辨率的硬约束

H3 把文本、音频和视频拼成一条无 padding 序列。视频行数近似为：

```text
N_video = T_latent × (height / 32) × (width / 32)
N_total = text_rows + 2 × round(frames / 24 × 40) + N_video
```

其中 124 帧对应 37 个视频 latent 帧。基线的 `N_total` 约 15,431 行；同样时长下，768×1344 约 37,742 行；如果增加到 362 帧，约 109,094 行。注意力计算和临时缓冲不能按“总显存 64GB”线性相加，密集注意力的计算量近似随 `N²` 增长，激活和 TP gather 也随 `N` 增长。

代码层面存在一个入口风险：`packing.resolve_canvas_size()` 有 768×1344 的面积上限，但 `MiniMaxH3Runner._resolve_request_geometry()` 在收到 `target_shape` 时直接采用尺寸，`validate_t2av_geometry()` 只检查帧数和 32 的倍数，并没有检查面积或序列行数。因此 API/服务层可以绕过面积上限，直接提交会把进程推向 OOM 或极长计算的尺寸。

部署前应在请求入口同时执行：

1. `height × width <= MAX_PIXELS`（当前发布几何约为 768×1344 的 1MP 上限）；
2. 根据帧数计算 `N_total`，再与经过本机 gate 校准的 `max_sequence_rows` 比较；
3. 对超限请求返回 4xx 和可读的限制信息，不要等 CUDA OOM；
4. 将宽高、帧数、步数按有限的 bucket 归一化，便于复用编译/缓存和估算时延。

仅有面积上限仍不够：视频 token 还随帧数增长，必须同时限制时长。直接把 2K 图像或 15 秒视频接到当前 dense `torch_sdpa` 路径不应视为可用高分辨率方案。要支持更大尺寸，需要经过 SM70 验证的 block-sparse/局部注意力或空间/时间分块算法；目前工作站没有安装 SageAttention、FlashAttention 或 Sol-Attn，不能把现代 GPU 的稀疏配置直接套到 V100。

## 注意力与激活的具体瓶颈

- 当前 gate 的 `torch_sdpa` 是唯一已验证的注意力后端。V100 不应假设 FlashAttention/SageAttention 的现代架构内核可用。
- `MiniMaxH3TransformerInfer._gather_tp_last_dim()` 每个 block 都创建 `tp_size` 个 all-gather 缓冲并 `cat`。该临时内存按序列行数线性增长，和 FP32 residual/门控一起会压缩高分辨率余量。长期优化方向是分块调制或保持调制投影的本地分片，避免完整 modulation 的 gather/cat。
- `h3_v100_fp16` 为了数值安全把 residual、门控和部分中间值保留在 FP32；不能只把权重改成 FP16 就认为激活也减半。
- `h3_finite_check=true` 会在每个 block 创建 `isfinite` mask、执行全量检查并在 rank 0 读取 min/max。它适合首个 smoke test，不适合作为生产默认；高分辨率下这个 mask 本身就可能造成明显峰值和同步。建议首个请求或固定间隔抽检，稳定后关闭。
- `use_compile=false` 是当前合理默认。H3 序列长度、TP gather 和请求几何是动态的；若要试 `torch.compile`，应按固定 geometry bucket 单独编译并预热，不能让每个新尺寸触发重新编译。

## 并发模型

当前 H3 runner、scheduler、输入和 conditioning 都是有状态对象，TP4 一次生成会同步占用四张卡。不能在同一进程组内盲目启动多个请求，也不能把同一组卡当成四个独立 worker；这样会造成显存竞争、NCCL 集合通信交叉或 scheduler 数据互相覆盖。

建议的服务拓扑是：

- 一个常驻 TP4 worker（`CUDA_VISIBLE_DEVICES=1,2,3,4`），并发数固定为 1；
- API 层使用有界队列，返回 job id，队列满时快速返回 429/`queue_full`；
- 文本 conditioning 在 CPU 侧异步预计算，并按 prompt/模型版本/编码器版本做有界 LRU 缓存；不要每个请求重新加载 62GiB 文本编码器；
- worker 只在请求边界替换 conditioning，完成后释放请求级 latent 和输出引用；
- 单次生成失败后将该 TP4 进程标记为 unhealthy，重启整个 process group，不能在损坏的 NCCL group 上继续接请求。

在没有经过量化和显存复核前，不建议把四卡拆为 2×(TP2) 以追求并发：当前 AdaLN-pruned FP16 DiT 约 37.46 GiB，TP2 的权重分片理论上约 18.7 GiB/卡，已经超过 16GB，还没有算激活。`tensor_p_size=1, seq_p_size=4` 则会让每卡持有完整权重，同样不可行。TP2+SP2 只有在另一套量化/CPU offload 路径通过显存门禁后才可尝试。

## VAE 与高分辨率输出

MiniMax H3 VAE 已有时间分块（默认 17 帧）和空间 tile（默认 256 像素、64 像素重叠）。完整服务建议：

- DiT 完成后先释放/卸载 DiT，再进入 VAE；
- 让 VAE 采用 tile 解码，必要时打开 `vae_decode_parallel`，由四个 rank 分摊 tile，再只在 rank 0 拼接；
- `return_cpu`/输出编码尽量在 CPU 侧进行，避免把整段高分辨率视频长期留在 GPU；
- 记录 tile 数量和每个 tile 的解码时延，避免把 VAE 的峰值误算成 DiT 峰值。

VAE 分块只能解决解码峰值，不能消除 DiT 的 `N²` 注意力成本；它不是绕过高分辨率 DiT 限制的方法。

## 推荐的发布门禁

| 档位 | 几何与配置 | 用途 |
|---|---|---|
| `tp4-safe` | 480×864、124 帧、20–21 步、`torch_sdpa`、`h3_finite_check=false`、无 CPU offload | 当前唯一有真实 V100 gate 的生产候选 |
| `tp4-debug` | 与 safe 相同，打开 `h3_finite_check`，只跑首个/抽样请求 | 数值回归与升级验收 |
| `tp4-highres-experimental` | 逐级增加 token bucket；超出 gate 的尺寸默认拒绝，除非单独有报告 | 研发测试，不对外承诺 |
| `tp4-queue` | 单 TP4 常驻 worker、有界队列、conditioning LRU、失败重启 | 对外 API 的并发控制 |

每次放宽几何或改变注意力后端，都应记录：`N_total`、步数、每卡峰值 allocated/reserved、每步时延、NCCL 错误、输出 finite、VAE tile 峰值和队列等待时间。没有这些数据，不应宣称“高分辨率”“高并发”或“满载”。

## 2026-09-24 实测补充：SwiGLU 行分块

已在下游推理器加入可选的 `h3_ff_chunk_rows`。设为 `4096` 时，SwiGLU 的大中间张量按序列行分块，默认值 `0` 仍走原始路径。目标工作站的 TP4 对比结果如下：

- 480×864、124 帧、20 步：分块峰值约 12.22 / 12.81 GiB（13.12 / 13.76 GB）allocated/reserved，DiT 约 121.11 秒；未分块峰值约 12.48 / 13.33 GiB（13.40 / 14.32 GB），123.68 秒。两者均 finite。
- 640×1152、124 帧、20 步：分块通过，峰值约 13.99 / 14.70 GiB（15.02 / 15.78 GB），DiT 约 288.52 秒；未分块同条件 OOM，约差 556 MiB。
- CPU block offload 在这台约 32 GiB 主机上会耗尽主机内存并被系统终止，不能作为高分辨率兜底。

分块改变了 GEMM 批次和舍入顺序。相同 480×864 prompt/seed 的 video latent 最大绝对差约 3.91、RMSE 约 0.108，不能按 SHA256 逐位比较；真实 conditioning 和最终 VAE 输出必须单独做画质回归。因此该开关只进入 `tp4-highres-experimental`，不改变 `tp4-safe` 默认值。
