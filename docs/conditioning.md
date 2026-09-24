# MiniMax-H3 真实 Prompt Conditioning

H3 的 TP4 DiT 不直接接收字符串，而接收 Qwen3-VL-32B 在第 50 层输出的隐藏状态。`scripts/export_h3_conditioning.py` 使用 ComfyUI 已实现的 `MiniMaxH3Tokenizer` / `MiniMaxH3TEModel` 和官方 `int8+convrot` 权重导出兼容 LightX2V 的 safetensors bundle。

## 运行

在工作站上，等文本权重完整后执行：

```bash
source /home/ymzx/ComfyUI/venv/bin/activate
python /home/ymzx/minimax-h3-4gpu/scripts/export_h3_conditioning.py \
  --prompt 'A cinematic fox walking through a snowy forest, natural motion, high detail.' \
  --output /home/ymzx/models/minimax-h3/evidence/conditioning/fox.safetensors \
  --cpu
```

成功输出包含：

- `prompt_embeds`: `[tokens, 5120]` 的真实 Qwen3-VL 隐藏状态；
- `text_token_tags`: 与 token 一一对应的 H3 文本模态标签；
- `minimax_h3_bundle`: `task=t2av` 的 LightX2V bundle 元数据。

将输出路径传给 TP4 runner 的 `--condition-path` / `precomputed_condition_path`。导出脚本会检查形状和 finite 值，缺失权重、错误模型分支或 NaN 会直接失败，不会生成伪 conditioning。

文本编码和 DiT 采样应分阶段运行：文本编码完成后释放 encoder，再启动四卡 DiT，避免 32GiB 主机内存和 16GiB V100 显存被同时占满。
