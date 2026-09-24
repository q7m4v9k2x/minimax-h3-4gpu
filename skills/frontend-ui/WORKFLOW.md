# Workflow — 前端 UI 实施与验收

## 1. 审查现状

列出入口 HTML、样式和脚本；确认启动命令、静态资源路径、后端接口字段以及页面是否由反向代理提供。用浏览器或现有截图记录桌面和手机基线。不要读取或复制凭据。先记录会保持不变的 API 契约。

## 2. 画状态图

把用户动作、服务器事件和页面状态写成小型状态机：

```text
idle --submit--> queued --job accepted--> running
running --progress--> running
running --done--> succeeded
running --error--> failed --retry--> queued
```

每个状态定义：可见文案、可用按钮、`aria` 语义、数据来源和离开条件。取消、刷新、断线和重复提交都要有明确行为。刷新后优先用任务 ID 恢复快照；没有快照时显示可解释的空状态。

## 3. 先建令牌再写组件

把颜色、间距、圆角、阴影和断点放入 `:root` 变量，页面只组合 Shell、Card、Field、Button、Progress 和 Result。避免内联样式堆叠；同一状态在所有组件使用同一语义颜色。无构建项目可按 `index.html`、`styles.css`、`app.js` 拆分，也可沿用现有目录。

## 4. 接入真实进度

优先 SSE/WebSocket，其次退避轮询。事件处理要校验任务 ID、忽略过期事件、在断线时重连并在完成后停止订阅。进度条更新不应依赖定时器“假跳动”；如果后端只给阶段事件，显示阶段和不确定进度，并把最近事件时间告诉用户。

```js
const state = { phase: 'idle', percent: null, jobId: null, error: null };
function applyEvent(event) {
  if (event.job_id !== state.jobId) return;
  state.phase = event.status;
  state.percent = Number.isFinite(event.percent) ? event.percent : null;
  render(state);
}
```

## 5. 响应式和无障碍检查

在 390px 宽度先检查：无横向滚动、主按钮可见、输入和下载可操作、长文案可换行。使用键盘完成提交、取消、重试和下载；检查焦点可见、标签关联、错误关联和读屏状态。将动画缩减模式作为真实测试条件。

## 6. 验证和视觉回归

运行项目已有 lint/测试命令；无测试脚本时至少用浏览器做以下路径：首次加载、空输入、提交、排队、进行中、断线重连、成功预览/下载、失败重试、刷新恢复、手机窄屏。固定数据和时间后，在三种视口截图，与基线比较关键区域。任何差异都要说明是预期设计变化还是回归。

## 7. 交付记录

记录改动文件、状态事件映射、支持的断点、键盘和读屏检查、视觉截图路径及未解决限制。若页面由服务托管，最后用实际入口做一次健康检查；不要把本地调试地址或敏感配置写进文档。
