# MiniMax H3 on 4× Tesla V100-SXM2-16GB

这是一个面向 **4 张 Tesla V100-SXM2-16GB（SM70、NVLink）** 的 MiniMax H3 本地运行研究仓库。
仓库不包含模型权重，只保存硬件预检、显存/内存估算、依赖安装和压测入口；权重必须从官方或得到授权的 Hugging Face 镜像按模型许可证自行获取。

## 先说结论

- MiniMax H3 是约 **33.12B 参数的稠密 Omni Transformer**，官方权重以 BF16 发布，并同时包含 FL2VA、Ref2VA 两个任务分区。文本编码器使用 Qwen3-VL-32B 的保留层。
- V100（SM70）没有原生 BF16、FP8 Tensor Core。直接把完整 BF16 模型复制到每张 16GB 卡不可行；即使总显存相加也不会自动形成一个显存池。
- 已找到并核对的现成路线：
  - [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)：官方实现、权重和许可证。
  - [Leonccaa/LightX2V-V100](https://github.com/Leonccaa/LightX2V-V100)：真实 TP4/分片、FP16 敏感路径和单独 VAE 解码；公开基准在 4×32GB V100 上通过，适合作为 4 卡分片基线，但 **没有证明 4×16GB**。
  - [dg1kjd/comfyui-v100-sxm2-minimax-h3](https://github.com/dg1kjd/comfyui-v100-sxm2-minimax-h3)：ComfyUI Ulysses sequence parallel，模型副本驻留在每张卡；公开测试是 8×32GB，不能直接当作 4×16GB 的 TP 方案。
  - [rwashy/H3-V100](https://github.com/rwashy/H3-V100)：针对 V100 的 INT8 ConvRot/缩放 FP8 和显存管理，公开验证覆盖 1/2 张 16GB 卡；四卡并行仍需本机验证。
  - [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix)：V100 原生 FP16 数值安全修复，可作为 ComfyUI 单卡/分片路径的参考。
- 本仓库采用的目标架构是 **LightX2V 风格的真实 TP4**：4 卡共同持有分片，文本编码器也分片或卸载，VAE 单独解码；不把“4 卡分别复制完整模型”宣称为四卡共享显存。首选是 AdaLN-pruned 的 FP16/原生 PyTorch SDPA 路径；INT8/GGUF 只作为经过 SM70 内核检查后的存储压缩备选。
- 由于工作站只有约 32GiB 主机内存，官方 BF16 权重和完整双分区服务不适合直接部署。官方 VAE 文件本身约 11GiB；第一阶段应使用 AdaLN-pruned DiT + 预计算文本 conditioning 或 CPU/分片文本编码器，并按阶段释放组件，先通过 5 秒、低分辨率 T2VA smoke test，再逐步增加分辨率和时长。

## 快速开始

在工作站上（Ubuntu/Linux）：

```bash
git clone https://github.com/q7m4v9k2x/minimax-h3-4gpu
cd minimax-h3-4gpu
python3 scripts/preflight.py --json reports/preflight.json
python3 scripts/estimate_memory.py --preset v100-16gb
```

预检必须看到 4 张 V100、每张至少 14GiB、SM70，并且 GPU 之间 P2P/NVLink 可用。物理 Quadro K620 不得加入 `CUDA_VISIBLE_DEVICES`。

### 安装可复用的 ComfyUI 组件

```bash
COMFYUI_DIR=/home/ymzx/ComfyUI bash scripts/install_comfy_plugins.sh
```

脚本只克隆公开依赖，不修改 ComfyUI 核心文件；安装结果和 commit 会记录到 `reports/plugins.txt`。4×16GB 先不要启用需要每卡常驻完整模型副本的 sequence-parallel 节点，除非量化和显存预检已经通过。

### 生成启动环境

```bash
COMFYUI_DIR=/home/ymzx/ComfyUI \
GPU_IDS=1,2,3,4 \
bash scripts/launch_env.sh
```

脚本会打印可复制的环境变量和 ComfyUI 启动命令，不会替你停止现有服务。真正部署时可把输出保存到 systemd 或现有网关的受控服务单元中。

## 推荐的验证顺序

1. `preflight.py`：确认 4 张 V100 的 P2P、NVLink、驱动和 PyTorch CUDA。
2. `estimate_memory.py`：比较 BF16、INT8 DiT、INT8 文本编码器和 VAE 的主机内存上界。
3. 只下载一个任务分区（优先 FL2VA），避免同时保存 FL2VA 和 Ref2VA 两份权重。
4. 先用 LightX2V 的 TP4/DiT-only 入口跑 5 秒、低分辨率、少步数 smoke test，再用阶段遥测补丁测 VAE 和输出尾段。
5. 固定 seed，记录每卡峰值显存、NVLink/P2P 带宽、功耗、时延和输出是否 finite；每次只改一个变量。
6. 只有单请求稳定后，才测试并发。TP4 一次生成会占用 4 张卡；并发请求应排队，不能把同一 TP 组当成四个独立 worker。

## 已验证的 4×16GB gate

2026-09-23 在目标工作站完成了真实 TP4 DiT-only gate：FP16 权重分片加载约 85.7 秒，480×864、124 帧、20 次评估用时约 124.16 秒（约 6.18 秒/步），每卡峰值约 12.48 GiB allocated（13.40 GB），latent 输出为 finite。该结果证明 4×V100-SXM2-16GB 可以共同运行这条分片路径；conditioning 使用的是随机合成夹具，不能作为真实画质或生产吞吐承诺。复现命令和 `--transformer-path` 要求见 [实机运行手册](docs/runbook.md)，硬件细节见 [门禁记录](docs/hardware-2026-09-23.md)。

2026-09-24 又完成了一个显式实验开关 `h3_ff_chunk_rows`：它将 H3 的 SwiGLU 前馈按序列行分块，在相同 480×864 门禁上把峰值 allocated 降到约 12.22 GiB（13.12 GB）；640×1152、124 帧、20 步使用 `4096` 行约 289 秒，调到 `8192` 行后约 288.30 秒且峰值不变。未分块的 640×1152 路径会 OOM；CPU block offload 会耗尽约 32 GiB 主机内存。分块会改变 GEMM 舍入顺序，latent 不保证与未分块逐位一致，因此配置仍标为实验用途，完整数据和回归边界见 [分块前馈实验记录](docs/chunked-ff-2026-09-24.md)。

## 目录

| 路径 | 用途 |
|---|---|
| `docs/research.md` | 公开资料、方案比较和 4×16GB 可行性判断 |
| `config/v100-16gb.env.example` | 工作站环境变量模板 |
| `scripts/preflight.py` | GPU、P2P、NVLink、PyTorch 预检 |
| `scripts/estimate_memory.py` | 权重组合和上下文 token 的内存估算 |
| `scripts/install_comfy_plugins.sh` | 安装已核对的 V100 ComfyUI 组件 |
| `scripts/launch_env.sh` | 输出安全的 4 卡启动环境 |
| `scripts/benchmark_gpu.sh` | 记录生成期间 GPU 利用率、时钟、功耗和显存 |
| `scripts/analyze_h3_report.py` | 汇总 conditioning、prepare、DiT、VAE/输出缺口和瓶颈 |
| `scripts/check_weights.py` | 检查本地权重目录，避免漏下/混用分区 |
| `reports/` | 本机预检和压测结果（默认被 git 忽略） |

## 许可和权重

本仓库新增脚本采用 MIT。上游组件按各自许可证使用：`dg1kjd` 插件为 GPL-3.0，官方 MiniMax H3 使用 MiniMax H3 Community License，模型权重和派生量化文件均不随本仓库分发。该模型许可证当前列出的地域排除包括美国、欧盟、英国和韩国；商业部署前必须自行阅读当前版本协议、NOTICE 和安全限制。
