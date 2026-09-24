# Design System — 无构建依赖的前端 UI

## 设计目标

优先让用户知道「现在发生什么、下一步能做什么、出了问题如何恢复」。采用一套稳定的令牌和少量高复用组件，避免每个页面各写一套颜色、圆角和阴影。

## 令牌基线

在页面根元素定义变量；主题切换只改变量，不在组件里散落硬编码颜色。

```css
:root {
  color-scheme: dark;
  --bg: #0b1020;
  --surface: #121a2d;
  --surface-raised: #18233b;
  --border: #2a3857;
  --text: #edf3ff;
  --text-muted: #9aa9c4;
  --accent: #72a7ff;
  --accent-strong: #4f83ff;
  --success: #46d19b;
  --warning: #f4c86b;
  --danger: #ff7182;
  --focus: #b8d1ff;
  --radius-sm: 8px;
  --radius-md: 14px;
  --radius-lg: 22px;
  --shadow: 0 14px 40px rgb(0 0 0 / .25);
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --content-max: 1200px;
}
```

使用 4px 基准间距；正文行高至少 1.5；按钮和输入的可点击区域至少 44×44px。标题只表达层级，不靠颜色单独传达含义。图标旁配文字或 `aria-label`，不使用颜文字代替图标。

## 页面和组件

- **Shell**：`header`、主内容和可选侧栏；桌面采用双栏，窄屏变为单栏。
- **Card**：只承载一个任务或一组相关结果；标题、辅助说明、内容、操作顺序固定。
- **Button**：主操作一个视觉重点；危险操作用 `danger` 变体并提供确认或撤销。
- **Field**：`label` 与输入显式关联；提示、错误和帮助文本使用 `aria-describedby`。
- **Result**：缩略图、元数据、下载/复制操作和失败重试放在同一上下文。
- **Toast/Alert**：短消息用 toast，影响任务决策的错误放在相关控件附近并保留文本。

## 状态和进度

所有异步任务至少建模为 `idle → queued → running → succeeded | failed → retrying`。状态文案要具体（如「等待 GPU」或「解码第 12/30 步」），不要只显示「处理中」。

- 已知总量：用 `<progress max value>`，同时显示百分比和 `current / total`。
- 总量未知：用不确定进度条，并显示已耗时或最近事件；禁止伪造平滑百分比。
- 后端有 SSE/WebSocket：按事件更新进度，断线显示「连接中断，正在重连」，重连后先拉取当前任务快照。
- 轮询：使用退避（1s、2s、4s…上限 10s），页面隐藏时暂停；任务完成立即停止。
- 进度区域加 `role="status"` 或 `aria-live="polite"`，避免每个百分比变化都打断读屏。
- 失败必须说明原因、保留输入和提供重试；成功状态提供预览、下载和复制链接。

## 响应式和移动端

建议断点：`<640px` 手机、`640–1023px` 平板、`≥1024px` 桌面；优先内容宽度而非设备名称。使用 `min-width: 0`、可换行的操作组和 `clamp()` 字号，避免固定宽度导致横向滚动。移动端把次要操作收入菜单，但主操作保持可见；上传、预览和进度卡片按单列排列。

```css
@media (max-width: 639px) {
  .layout { grid-template-columns: 1fr; }
  .actions { display: grid; grid-template-columns: 1fr; }
  .page { padding: var(--space-4); }
}
```

## 无障碍

语义 HTML 优先；键盘顺序与视觉顺序一致；焦点样式不能移除；对比度目标正文 4.5:1、大字 3:1；错误不能只用颜色；动画尊重 `prefers-reduced-motion: reduce`；图片提供有意义的 `alt`，装饰图使用空 `alt`。

```css
:focus-visible { outline: 3px solid var(--focus); outline-offset: 3px; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; scroll-behavior: auto !important; }
}
```

## 视觉回归基线

固定浏览器、视口、主题、字体和数据；至少覆盖桌面 1440×900、平板 768×1024、手机 390×844。等待字体、图片和主要异步状态稳定后截图；比较布局、文字截断、焦点、错误和进度状态。动态时间、随机 ID 和动画应在测试夹具中冻结或关闭，阈值只容忍抗锯齿差异，不以放宽阈值掩盖布局变化。
