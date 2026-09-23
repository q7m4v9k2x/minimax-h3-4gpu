# 研究记录：4×V100-SXM2-16GB 运行 MiniMax H3

## 硬件事实

远程工作站实测（2026-09-23）：

- 物理 GPU 0：Quadro K620 2GiB，不能参加 H3 TP。
- 物理 GPU 1–4：Tesla V100-SXM2-16GB，SM70，每卡约 15.77GiB 可用。
- V100 之间拓扑为 NV2，P2P 可用；适合跨卡 all-to-all/TP。
- 驱动 580.178.04、PyTorch 2.10.0+cu128、CUDA runtime 12.8；V100 不应依赖 CUDA 13 专属内核。
- 主机内存约 32GiB，swap 8GiB。这个内存容量是低显存方案的主要瓶颈。

## 模型事实

官方 `MiniMaxAI/MiniMax-H3` 模型卡和仓库说明：

- H3-Omni-Transformer 是约 33B 的稠密单流 Transformer。
- 文本编码器使用 Qwen3-VL-32B 的前 50 层输出。
- FL2VA、Ref2VA 是两个任务分区；每个官方 BF16 分区约 134GiB，完整双分区需要约 270GiB 磁盘。
- 官方本地部署示例面向现代 GPU，并不承诺 V100 的 BF16/FP8 内核。
- H3 使用单独的 MiniMax H3 Community License；模型许可证存在地域排除和商业规模条款，部署前必须按当前许可证核对所在地和用途。代码仓库的 MIT 许可不覆盖模型权重。

因此，4×16GB 的关键不是“总显存 64GB”，而是是否把 DiT 和文本编码器真正分片，并把中间激活、VAE 和输入序列控制在每卡 16GB 内。

## 方案比较

| 路线 | 公开验证 | 对 4×16GB 的判断 |
|---|---|---|
| 官方 Diffusers/SGLang/vLLM-Omni | 官方支持现代 GPU；需要 BF16/新注意力后端 | 不能作为 V100 首选 |
| `dg1kjd` ComfyUI Ulysses | 8×V100-SXM2-32GB；每卡完整模型副本 | 4×16GB 需大量卸载，吞吐和显存风险高 |
| `rwashy/H3-V100` | 1/2×16GB，INT8 ConvRot/缩放 FP8 | 可作 ComfyUI 低显存基线，但不是真正 TP4；其预编译 CUDA 算子不能默认覆盖 SM70 |
| `Amduraznak` FP16 fix | V100 原生 FP16 数值安全 | 是精度修复，不解决模型容量 |
| `LightX2V-V100` TP4 | 4×V100 PCIe-32GB，TP4 + 独立 VAE | 最接近本机目标；4×16GB 需要量化/卸载实测 |
| Abiray pruned GGUF | Q3/Q4 单文件约 8.9/11.6GB，ComfyUI-GGUF 生态 | 可能适合低显存，但未找到 H3 扩散 TP4 实现；GGUF CUDA kernel 和质量必须实测 |

## 选定路线

本项目优先复用 LightX2V 的真实 TP4 设计，并用 V100 FP16 安全岛替代 BF16 Tensor Core 路径：

1. 只加载一个任务分区，优先 FL2VA。
2. DiT 首选 AdaLN-pruned FP16 + 原生 PyTorch SDPA；INT8/GGUF 只在确认 SM70 kernel 后作为压缩存储。不要把 FP8 当成 V100 原生计算格式。
3. 文本编码器使用预计算 conditioning、CPU 或可在 Volta 上运行的 FP16/INT8 分片方案；`nvfp4` 仅作为存储格式，不能假定 V100 能原生计算。
4. TP4 组内每张卡持有真实权重分片，避免四份完整 DiT 副本。
5. VAE 从 TP4 阶段拆出，使用单卡或时间块并行解码，释放 DiT 显存后再解码。
6. 先做单请求稳定性，再做排队并发；同一 TP4 组不做盲目多进程复制。

## 内存边界

社区测量的代表性文件大小（十进制 GB，实际实现会因版本和索引变化）：

- AdaLN-pruned FP16 DiT：约 37.46GiB（社区文件统计；TP4 理论约 9.4GiB/卡，未计激活）。
- pruned INT8 ConvRot DiT：约 19.53GiB（Comfy-Org 文件统计；SM70 CUDA kernel 尚未验证）。
- INT8 ConvRot 文本编码器：约 27.14GB。
- BF16 文本编码器：约 51.51GB。
- 官方视频/音频 VAE：约 11.0GB；部分社区量化/裁剪工作流约 5.8GB，不能默认套用。

这些数字是磁盘/主机权重上界，不等于运行时显存。对本机 32GiB RAM 来说，DiT、文本编码器和官方 VAE 若同时常驻会超限；需要预计算 conditioning、分阶段释放、磁盘映射或分片，并且 swap 不能作为性能方案。任何“4×16GB 已经稳定支持高分辨率”的结论都必须附带本机日志和峰值数据。

## 性能调优顺序

1. 确认 NVLink/P2P，避免误把 K620 纳入可见设备。
2. 使用 V100 可执行的 FP16 matmul，保留 residual、normalization、token refiner 和溢出保护的 FP32 安全岛。
3. 让 TP4 通信走 NVLink；固定 GPU 时钟前先记录功耗和 throttle reason，避免未经验证地改系统级时钟。
4. 复用文本 conditioning，避免每个采样步重复编码。
5. 使用 Turbo/少步数 LoRA 前先建立同 seed 的质量基线；EasyCache、SOL、CFG interval 都是质量/速度折中，必须单独记录。
6. 高分辨率按 token 数和时长逐级增加；优先降低帧数或步数，不要一开始就上 2K/15s。

## 尚未宣称的内容

截至本记录，尚无公开证据证明“4×V100-SXM2-16GB、32GiB RAM、官方 H3 权重、原生端到端高分辨率”已经稳定通过。因此仓库的脚本会给出容量风险和下一步，而不会伪造 benchmark 数字。

## 来源

- 官方代码和模型卡：<https://github.com/MiniMax-AI/MiniMax-H3>、<https://huggingface.co/MiniMaxAI/MiniMax-H3>
- TP4 下游实现：<https://github.com/Leonccaa/LightX2V-V100>
- ComfyUI 多卡 sequence parallel：<https://github.com/dg1kjd/comfyui-v100-sxm2-minimax-h3>
- V100 低显存优化：<https://github.com/rwashy/H3-V100>
- V100 FP16 数值修复：<https://github.com/Amduraznak/minimax-h3-fp16-fix>
- 16GB 内存估算和量化文件实测：<https://github.com/Tomiigo/minimax-h3-16gb>
- pruned GGUF 候选：<https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF>
