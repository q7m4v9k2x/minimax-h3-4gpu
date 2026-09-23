# 实机 TP4 运行手册（实验）

这个流程默认覆盖 **预计算 conditioning + DiT-only TP4**。它不把 Qwen3-VL 文本编码器塞进四卡采样进程，适合先验证 V100 FP16 数值路径和通信；文本编码可在另一台机器/CPU/量化环境生成后导出 conditioning bundle。要测 native VAE，先应用 `patches/lightx2v-stage-telemetry.patch`，再给 benchmark 加 `--with-vae`；没有 H3 VAE 权重时只能保留 DiT-only，不得把 pipeline 时间当作端到端时间。

## 1. 固定 GPU 映射并做通信门禁

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
python scripts/preflight.py --json reports/preflight.json
torchrun --standalone --nproc_per_node=4 scripts/nccl_smoke.py
```

如果日志出现 `Quadro K620` 或 `no kernel image is available`，先停止，不要继续加载模型。当前工作站实测必须显式设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。

## 2. 固定 LightX2V 下游版本

```bash
bash scripts/fetch_lightx2v.sh /home/ymzx/LightX2V-V100
cd /home/ymzx/LightX2V-V100
```

上游分支是 Apache-2.0 下游实验版本，commit 已固定在 `config/upstream-commits.txt`。它要求完整的 H3 模型/转换后 pruned transformer 和 conditioning bundle；本仓库不分发这些文件。

如果拿到的是 Comfy-Org 的单文件 pruned BF16 DiT，可用 `config/official-fl2va-transformer.json` 作为上游转换器的 `--base-config`；转换会产生新的分片文件，预计还需要约 40GiB 磁盘空间。转换前确认输入文件完整，且不要把 INT8/GGUF 文件误传给 BF16 curve 转换器。

## 3. 最小 TP4 gate

以 LightX2V 文档的 864×480、124 帧配置为起点，先把 `--trials` 设为 1、开启 finite check，并把输出写入 `reports/`。注意转换后的 pruned checkpoint 是平铺目录，必须显式传 `--transformer-path`，不能使用脚本默认的 `model_path/transformer`：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,4
export DTYPE=FP16 SENSITIVE_LAYER_DTYPE=FP32 LIGHTX2V_MINIMAL_IMPORT=1
export PYTHONPATH=/home/ymzx/h3-extras:/home/ymzx/LightX2V-V100
/home/ymzx/ComfyUI/venv/bin/torchrun --standalone --nproc_per_node=4 \
  /home/ymzx/LightX2V-V100/tools/minimax_h3/run_tp4_benchmark.py \
  --config /path/to/lightx2v-v100-tp4-16gb.experimental.json \
  --model-path /home/ymzx/models/minimax-h3/lightx2v-fl2v-pruned \
  --transformer-path /home/ymzx/models/minimax-h3/lightx2v-fl2v-pruned \
  --task t2av --condition-path /path/to/conditioning.safetensors \
  --prompt 'test prompt' --case-id h3-tp4 --frames 124 --nominal-seconds 5 \
  --height 480 --width 864 --trials 1 --block-finite-check \
  --evidence-dir /path/to/evidence --report /path/to/report.json \
  --effective-config /path/to/effective.json
```

2026-09-23 的实测摘要见 `docs/hardware-2026-09-23.md`。本地 conditioning 夹具只能验证加载、TP 通信和有限输出；接入真实文本编码器与 VAE 后还需单独验收画质、端到端时延和错误恢复。

本机 16GB 卡不要直接套用 32GB benchmark。第一次只验证 480×864、短时长和低步数；如果单卡峰值超过约 14GiB，先减少序列长度或改成阶段化卸载。

阶段分析：

    python scripts/analyze_h3_report.py reports/remote-h3-tp4-gate.json --markdown-out reports/stage-summary.md

它使用各 rank 的最大阶段耗时、step P50/P95 和峰值显存，避免把聚合 pipeline 字段误当成独立瓶颈；VAE、媒体编码、落盘和 API 响应缺少字段时会明确标记为未测量。

如果需要验证 640×1152，可改用仓库中的 `config/lightx2v-v100-tp4-16gb-chunked8192.experimental.json`。该配置将 `h3_ff_chunk_rows` 设为 `8192`，实测 4×V100 在 640×1152、124 帧、20 步通过，单卡峰值约 13.99 GiB allocated / 14.70 GiB reserved（15.02 / 15.78 GB），pipeline 约 288.30 秒。它会改变 SwiGLU 分块和舍入顺序，输出不保证与未分块路径逐位一致；只用于真实 conditioning 和 VAE 画质回归，不应直接替换 `tp4-safe` 默认配置。原始 640×1152 路径缺约 556 MiB OOM，CPU block offload 也不适合这台约 32 GiB 内存的主机。

## 4. 记录结果

生成期间另开终端运行：

```bash
bash /path/to/minimax-h3-4gpu/scripts/benchmark_gpu.sh reports/gpu-samples.csv
```

保留 `benchmark.json`、`effective-config.json`、GPU CSV、NCCL 门禁和输出媒体校验；不要提交模型、prompt 中的隐私内容、SSH 凭据或大文件。
