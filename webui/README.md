# MiniMax H3 网关

`service.py` 是纯 Python 标准库 HTTP 服务。它只启动一个流水线 worker，任务和每次状态更新写入 `H3_DATA_DIR`（默认项目根目录的 `h3-jobs/`），因此进程重启后仍可查询历史任务；重启时正在运行的任务会标记为 `interrupted`，排队任务会恢复到队列。

启动（推荐从项目根目录执行，但不依赖当前工作目录）：

```bash
cd /home/ymzx/minimax-h3-4gpu
python3 webui/service.py --host 127.0.0.1 --port 8200 --verified
```

工作站的 V100 生产服务由 `minimax-h3.service` 管理。由于 H3 独立环境中的
PyTorch cu130 wheel 不包含 SM70，生产 unit 显式使用
`/home/ymzx/ComfyUI/venv/bin/python` 和同环境的 `torchrun`；该环境已验证
包含 `sm_70`。可参考 `config/minimax-h3.service.example`，不要改成裸
`torchrun` 或 `/home/ymzx/h3-venv/bin/torchrun`。

默认 worker 执行：

```text
/home/ymzx/minimax-h3-4gpu/scripts/generate_video.py \
  --request-json <job>/request.json --output-dir <job>
```

也可用 `H3_PIPELINE_PYTHON`、`H3_PIPELINE_SCRIPT`、`H3_DATA_DIR`、`H3_QUEUE_SIZE`、`H3_PUBLIC_PREFIX` 覆盖默认值。提交开关只有在流水线 `--check` 返回 `{"ready": true}` 且传入 `--verified` 后才打开。

接口基路径默认为 `/h3-api`：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` 或 `/healthz` | 服务和流水线资源检查；未就绪返回 503 |
| POST | `/h3-api/v1/videos/generations` | 创建任务，返回 202 和任务快照 |
| GET | `/h3-api/jobs` | 最近任务列表 |
| GET | `/h3-api/jobs/<id>` | 查询任务；页面刷新后用此接口恢复 |
| GET | `/h3-api/jobs/<id>/events` | SSE `job` 事件和心跳 |
| POST | `/h3-api/jobs/<id>/cancel` | 取消排队或正在运行的任务 |
| GET/HEAD | `/h3-api/files/<id>/<name>` | 下载结果，支持 `Range: bytes=...` |

任务请求示例：

```json
{"prompt":"海边日落，一只狗奔跑","width":480,"height":864,"frames":124,"steps":20,"seed":-1}
```

`h3.html` 将任务 id 保存在浏览器 `localStorage`，打开或刷新页面时先查询任务，再订阅 SSE；断线时自动回退到查询。结果文件仅从任务目录内校验过的流水线输出提供，避免路径穿越。
