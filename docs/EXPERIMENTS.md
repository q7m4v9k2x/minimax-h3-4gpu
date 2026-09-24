# MiniMax H3 实测经验账本

只记录能指出证据的观察、修复和边界。计划不写为成功。更新时间：2026-09-24。

## 2026-09-24：真实中文 prompt 端到端与公网 API 验收

状态：成功。唯一主变量是把 TP4 采样和 VAE 解码固定到支持 SM70 的 `/home/ymzx/ComfyUI/venv`（PyTorch 2.10.0+cu128）；此前 `/home/ymzx/h3-venv` 的 PyTorch 2.14.0+cu130 只有 sm75 及以上内核，会在 V100 上报 `no kernel image is available`。纯 Python 依赖补齐到 h3-venv，但没有把其 CUDA 二进制用于推理。

- 请求：中文 prompt“金色夕阳下，红狐穿过薄雾竹林，电影感镜头，柔和体积光”，seed 20260924，864×480，124 帧，20 次模型评估，24 FPS。
- 文本编码：ComfyUI venv，生成真实 conditioning bundle；随后释放文本编码器再启动 TP4。
- TP4：`CUDA_VISIBLE_DEVICES=1,2,3,4`，FP16，敏感路径 FP32，torch SDPA，四卡共同生成同一 latent。手动 benchmark pipeline 约 119.92 秒；20 步平均约 5.96 秒/步；每卡峰值 allocated 13,396,084,224 bytes（约 12.49 GiB），四卡采样期间 100% 利用率，latent finite。
- VAE/媒体：官方视频和音频 VAE 解码，输出 864×480、124 帧、5.166667 秒 MP4，含 H.264 视频和 AAC 音频；手动成品 `h3-preview.mp4` 约 2.05 MiB。
- 公网 API：任务 `518dda3d36894eee7bdd4c0e` 从 `conditioning → loading → sampling → decode → encode → completed` 完成，返回 2,368,687 字节下载文件。报告：`reports/remote-h3-e2e-20260924-api.json`，媒体元数据：`reports/remote-h3-e2e-20260924-api-media.json`。
- 桌面成品：`C:\Users\Administrator\Desktop\minimax-h3-real-20260924-final2.mp4`（SHA256 `7EC23DD5940AD5329B9440D75418D82F70F7D3614017AF04F286063B736BD6D9`）；公网 API 成品 `C:\Users\Administrator\Desktop\minimax-h3-api-518dda3d36894eee7bdd4c0e.mp4`（SHA256 `63C6DAD8FB7381F06265081D83ED5FCA70C12F640889B73617D9DEC71C30B4E3`）。两者均已用 ffprobe 验证 864×480、124 帧、5.166667 秒。
- 部署：工作站 `/etc/systemd/system/minimax-h3.service` 已启用，监听 `127.0.0.1:8200`；既有反向隧道 `18100` 和 Oracle nginx 无需改 ACL。公网 `/router-health/h3`、`/h3-api/health` 返回 200。

边界：单 worker 仍会重复加载 TP4 权重，每次请求约 89 秒冷启动加 120 秒采样；并发请求按队列顺序执行，不会同时占用四卡组。高分辨率/30 步目前只有 DiT-only 历史门禁，尚未完成端到端媒体回归。

## 2026-09-24：15 秒三段公网端到端

状态：成功。任务 `22f6c4c0f1847484a1eb08ee` 使用 `duration=15,segments=3`，连续完成三次 864×480、124 帧、20 步的真实 TP4 采样和官方视频/音频 VAE 解码；总体进度事件从 `0/16` 到 `16/16`，随后用 FFmpeg 生成 360 帧成片。输出为 H.264 + AAC、864×480、24 FPS，视频、音频和容器时长均为 15.000000 秒，大小 58,877,109 bytes。本机对下载文件做 `ffprobe -count_frames` 和 FFmpeg `-xerror` 解码检查通过，成品位于 `C:\Users\Administrator\Desktop\minimax-h3-15s-22f6c4c0.mp4`；详细记录见 `docs/15s-segment-e2e-2026-09-24.md`。

边界：三个片段使用相邻 seed，能在 4×16GB 容量内交付 15 秒带声音视频，但不是单次 362 帧长序列，片段边界可能跳切；需要连续长镜头时仍需时间分块或更大显存方案。

## 2026-09-23～24：四卡 DiT-only 基线

硬件：4×V100-SXM2-16GB、NV2、约 32 GiB RAM。物理 K620 排除，`CUDA_DEVICE_ORDER=PCI_BUS_ID` 与 `CUDA_VISIBLE_DEVICES=1,2,3,4` 同时设置。历史运行环境为 PyTorch 2.10.0+cu128，不能自动代表后续 h3-venv。

| 宽×高 / 帧数 / 模型评估次数 | 设置 | DiT / 报告 pipeline 秒 | 每卡峰值 allocated / reserved GiB | 证据 |
|---|---|---:|---:|---|
| 864×480 / 124 / 20 | 无分块，block finite 检查开 | 123.68 / 124.16 | 12.48 / 13.33 | `remote-h3-tp4-gate.json` |
| 864×480 / 124 / 20 | 无分块，block finite 检查关 | 119.22 / 119.74 | 12.48 / 13.33 | `remote-h3-tp4-safe-nofinite.json` |
| 864×480 / 124 / 20 | FF 行分块 4096，检查关 | 121.11 / 121.59 | 12.22 / 12.81 | `remote-h3-tp4-gate-chunked-compare.json` |
| 1152×640 / 124 / 20 | FF 行分块 4096 | 288.52 / 289.10 | 13.99 / 14.70 | `remote-h3-tp4-640x1152-chunked.json` |
| 1152×640 / 124 / 20 | FF 行分块 8192 | 287.73 / 288.30 | 13.99 / 14.70 | `remote-h3-tp4-640x1152-chunked8192.json` |

上表文件位于本机 `reports/`，默认被 Git 忽略。输出均 finite；conditioning 为随机夹具，不足以判断语义或最终画质。pipeline 字段只包含该 benchmark 所测阶段，不包含真实文本编码、VAE 或媒体编码。

获得的结论：

- 关闭逐 block finite 检查减少检查及同步开销；稳定门禁之外可关闭，但改模型、精度或内核后仍需数值验收。
- 480p FF 分块主要节省显存；与同样关闭 finite 检查的基线相比略慢，不能宣传成通用提速。
- 1152×640 未分块约差 556 MiB 而 OOM；FF 分块让 DiT 容量门禁通过。8192 行相对 4096 行小幅更快、峰值相同，只是一次实验结果。
- CPU block offload 在约 32 GiB RAM 工作站耗尽内存，不能当成可靠兜底。
- 分块改变 GEMM 舍入顺序：历史固定输入对比 video latent 最大绝对差约 3.91、RMSE 约 0.108。必须用真实 conditioning 和 VAE 成品做质量回归，不能承诺画质无损或逐位一致。
- 历史权重加载约 85.7 秒。每请求重启 TP4 会重复这项成本；只有编排 worker 常驻不等于模型常驻。

## 2026-09-24：文本权重与运行环境修复

下表来自上一会话交接，仍需在本轮远端日志复核后补充完成状态。

| 问题 | 已报告的处理 | 验收/后续 |
|---|---|---|
| 文本编码器文件损坏 | 从 ModelScope 重下并做 SHA256 校验 | 核对 `HANDOFF.md` 中 hash；保留下载校验结果 |
| 固定文件大小假设错误 | 远端编排改为检查 safetensors 实际长度 | 本机初审仍是旧版，需回收修复 |
| 文本编码解释器混用 | 远端改为 ComfyUI venv 绝对 Python | 核验真实 conditioning 的形状、finite 和 metadata |
| 裸 `torchrun` 命中错误环境 | 远端改为 H3 当前环境绝对 torchrun | 本机初审仍是旧版，需回收修复 |
| 缺少 `gguf` | 交接报告已补 | 使用 H3 解释器验证 import/版本 |
| torchvision cu128 与 H3 cu130 ABI 冲突 | 交接报告已备份并使用纯 Python 兼容层 | 记录备份与修改文件，逐阶段验证，不复制不兼容二进制 |
| 缺少 `prometheus_client` | 尚待处理 | 补齐后单独重跑真实 conditioning 的 TP4，保存完整日志 |

不要用“不断复制 site-packages”代替最终可复现的环境配置。恢复运行后应记录可用的版本清单、安装来源和兼容补丁；不在研发中强行统一两个已承担不同职责的 PyTorch 环境。

## 每次新实测追加模板

```text
日期和实验 ID：
状态：进行中 / 成功 / 失败 / 中断
假设和唯一主要变量：
代码 commit、patch 或文件 SHA256：
Python / PyTorch / CUDA / NCCL / GPU 映射：
请求：prompt（公开样例）、seed、宽高、帧数、评估次数：
命令与日志路径：
文本编码 / 加载 / DiT / VAE / 编码 / 端到端秒数：
四卡峰值 allocated、reserved，主机 RAM 峰值：
finite 检查与异常：
MP4 路径、SHA256、ffprobe/抽帧结果：
桌面交付位置：
Web/API/Oracle 验收结果：
可支持的结论与未验证边界：
回滚点和下一步：
```
