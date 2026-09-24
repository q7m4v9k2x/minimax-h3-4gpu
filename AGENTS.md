# MiniMax H3 项目接手规则

本项目的本机源码和长期经验目录是 `D:\minimax-h3-4gpu`。默认中文沟通。接手时先读 [docs/HANDOFF.md](docs/HANDOFF.md)，再按任务读取具体实验记录；不要把别的项目历史或凭据自动加载进来。

## 机器和目录

- Windows 本机：`D:\minimax-h3-4gpu`，保存源码、文档和可公开的实验摘要。
- GPU 工作站：`192.168.1.140`，SSH 用户 `ymzx`，项目 `/home/ymzx/minimax-h3-4gpu`。
- Oracle 140：`140.245.81.67`，SSH 配置名 `oracle161`、用户 `ubuntu`，提供 HTTPS 工作台和代理。它不是 GPU 工作站。
- Oracle 150 是用户现有 VPN/出口；访问 Oracle 140 使用已有授权链路。不要把 VPN 链接、密码、私钥、token 写进代码、文档、日志或 Git。
- 使用 `ssh-server` 技能中当前授权的连接配置；不要把工作站登录方式套到 Oracle。

## 不可混淆的验收状态

1. 四卡 NCCL 通信成功，只证明通信。
2. 使用随机 conditioning 的 DiT-only latent 为 finite，只证明采样数值和容量门禁。
3. 真实 prompt conditioning 成功，只证明文本编码。
4. 只有真实 prompt → TP4 DiT → 官方视频/音频 VAE → 可播放媒体完成，才算端到端视频生成通过。
5. 网关文件检查成功、网页能打开或进度条跳动，都不能替代第 4 项。`--verified` / `H3_PIPELINE_VERIFIED=1` 必须有真实成品证据。
6. `reports/poster-dummy.jpg` 是测试图片；历史随机 latent、占位结果均不能当作用户的生成成品。

## 环境和四卡约束

- 物理 GPU 0 是 Quadro K620；计算卡为物理 GPU 1–4，4×V100-SXM2-16GB、SM70、NVLink。
- 启动前确认 `CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=1,2,3,4`。四卡 TP4 共同生成同一个视频，不是四个独立 worker。
- 当前目标采用分阶段进程：ComfyUI 环境编码文本，H3 环境运行 TP4，释放 DiT 后 VAE 解码。每次改动必须核对各阶段实际 Python、torchrun 和 PyTorch/CUDA 版本。
- `/home/ymzx/ComfyUI/venv` 与 `/home/ymzx/h3-venv` 不能当成同一个环境。不要直接混用不同 CUDA/PyTorch ABI 的 torchvision 或其他二进制扩展。
- V100 没有原生 BF16/FP8 Tensor Core；真实基线使用 FP16、敏感路径 FP32、`torch_sdpa`。现代 GPU 内核不能未经门禁直接替换。
- 主机约 32 GiB RAM。文本编码器、DiT、VAE 同时驻留和 CPU block offload 都存在内存上限，不能仅凭 64 GB 总显存判断可用。
- 生产候选先验证 864×480、124 帧、20 次模型评估。640×1152 的 FF 分块结果属于 DiT-only 实验；更大分辨率、更长视频、30 步和竖屏不能因参数允许就宣称已经实测。

## 修改与留档

- 配置和远端环境修改前备份。保留模型、输出、任务记录和用户文件。
- 同步前对比两端源码，先回收远端已验证的修复，避免旧本机代码覆盖修复。不要整目录盲目覆盖或清理。
- TP4 一次占用四卡；服务采用单生成 worker 和有界队列。并发 API 请求数不等于同时 GPU 采样数。
- 每次实测记录 prompt/seed、尺寸/帧数/步数、commit 或文件 hash、环境、完整命令、阶段耗时、每卡峰值、输出路径和校验结果。隐私 prompt 不公开。
- `docs/HANDOFF.md` 更新“已验证、正在运行、阻塞、下一步”；`docs/EXPERIMENTS.md` 追加经验与证据。记录完成后再结束长任务，不能只存在聊天摘要里。
- 老 README、runbook 或研究文档的“当前状态”可能过时，以接手文档的日期、原始日志和本轮复核为准；矛盾必须明确记录，不能自动选乐观结论。
- 用户要求真实成品下载到本机桌面。交付前检查 MP4 流、时长、分辨率、可解码性和内容；桌面路径通过 Windows 实际配置获取。
- 部署验收包含工作站健康检查，以及通过现有 VPN 从 Oracle HTTPS 路径验证就绪、真实任务、刷新恢复、进度事件和媒体下载。仅改网页提示不算部署完成。

## 公开仓库

用户已要求项目开源。发布前仅暂存本次明确属于源码或文档的文件；不要提交模型、视频大文件、凭据、私有任务日志或未经检查的 `reports/` 测试素材。MIT 代码许可证不覆盖上游组件或模型权重许可证。
