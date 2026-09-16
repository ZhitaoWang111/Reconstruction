# 可视化页面制作原则

本文逐条记录项目中可复用的可视化约定。当前先记录第 1 条，后续原则继续追加到本文，不重复维护多份版本。

## 1. 3D 可视化窗口：以离线点云检查页为例

参考页面：[scene0000_00 原始 mask 反投影检查页](../InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html)。

核心要求：**使用真正可交互的 3D 视窗，支持左键旋转、右键平移、滚轮缩放和一键重置；视图操作只改变显示，不修改实验数据。**

### 1.1 页面怎么组织

- 3D 视窗应有足够的独立显示空间，不能只放一张点云截图代替交互窗口。
- 参考页左侧展示当前帧的 2D 图像和实例列表，右侧展示对应的 3D 点云；选择同一个 mask 时，两个视图应同步。
- 视窗上方放常用控制：背景点云开关、RGB／实例色切换、重置视角。操作提示宜简短说明鼠标按键用途。
- 深色背景衬托点云；场景背景用低亮度灰色，当前观察对象使用更醒目的颜色和点大小。背景可以隐藏，方便检查被遮挡的点。
- 窗口缩放时同步更新画布分辨率、WebGL viewport 和透视宽高比，避免拉伸或模糊；高分屏考虑 `devicePixelRatio`。
- 窄屏改为上下排列，不产生整页横向溢出。参考页桌面画布最小高度为 620 px，窄屏为 480 px；这些数值是当前实现，不是所有页面的固定要求。

### 1.2 鼠标交互必须统一

| 操作 | 应有行为 | 注意事项 |
| --- | --- | --- |
| 左键拖动 | 旋转观察方向 | 不改变平移位置和缩放距离。 |
| 右键拖动 | 沿屏幕平面平移 | 点云跟随拖动方向移动，不触发旋转。 |
| 滚轮 | 拉近／拉远 | 设置距离上下限；在画布内阻止滚轮同时滚动页面。 |
| 松开鼠标 | 立即停止拖动 | 鼠标移出画布后松开，也不能留下“粘住”的拖动状态。 |
| 重置视角 | 恢复初始观察状态 | 同时恢复旋转、缩放和平移，不能只重置其中一部分。 |

重置视角不应重置当前帧、mask 选择、配色或背景开关。参考页也遵循这一约定。

实现约定：

- 使用 Pointer Events，在 `pointerdown` 区分左键 `button === 0` 与右键 `button === 2`，记录本次拖动模式和 `pointerId`。
- 拖动开始时使用 `setPointerCapture`；在 `pointerup`、`pointercancel`、`lostpointercapture` 时清理状态，并在需要时释放捕获。
- `pointermove` 只处理当前指针，并检查对应按键是否仍按下，避免取消或松键后继续移动。
- 只在 3D 画布上阻止 `contextmenu` 默认行为，不禁用整张网页的右键菜单。
- 右键平移按当前观察距离和画布高度调整速度，不能始终使用一个与缩放无关的固定步长。

参考实现采用 45° 垂直视野角，平移量为：

```text
units_per_pixel = 2 × distance × tan(fov_y / 2) / canvas_css_height
pan_x += mouse_dx × units_per_pixel
pan_y -= mouse_dy × units_per_pixel
```

这里 `distance`、`pan_x`、`pan_y` 使用归一化后的显示空间单位，不是点云的真实米制坐标；鼠标位移和画布高度均使用 CSS 像素。平移放在相机平面中，旋转之后右拖仍应沿屏幕方向移动。

### 1.3 数据与显示必须分开

- 根据整个场景的包围盒确定显示中心和尺度。切换帧、实例或配色时保留用户视角，不反复自动回到初始位置；需要恢复时由“重置视角”完成。
- 旋转、平移、缩放通过渲染矩阵完成，不改写原始点坐标，不把“显示缩放”当作尺度校正。
- 使用深度测试，让前后遮挡关系正确。参考页通过 WebGL 的 `gl.POINTS` 绘制点云，不依赖在线 3D 服务。
- RGB 配色和实例配色是两种显示方式，切换颜色不改变点的位置、数量或成员关系。
- 若一个点属于多个 mask，保留全部成员关系，并明确标记歧义。参考页在全部 mask 的实例色模式下用紫色显示重叠点，不静默替它挑选一个实例。
- 明确说明结果的含义：当前实例 ID 只在帧内有效，不能把每帧不同颜色误解为已经完成跨帧物体关联。
- 如未来需要为显示做抽样或过滤，必须标明显示点数／规则，并保留原始结果；不能在用户不知情时改变实验结论。

### 1.4 离线打开与加载方式

- 对于已有实验结果的查看页，应能通过 `file://` 直接打开，不要求启动推理模型或网站服务，页面依赖也应保存在本地。
- 参考页通过本地 `manifest.js` 提供索引，再按需加载帧对应的 JS 数据文件；点坐标、颜色等从 Base64 解码为 TypedArray 后上传到 WebGL 缓冲区。
- 浏览器展示的是预先导出的点云资产，并不是打开网页时重新推理或重新解析整份原始 AMB3R 结果。NPZ／PLY 另提供下载入口。
- 不一次性把全部帧装入内存。参考页的解码帧缓存上限为 3，场景背景单独缓存，并清理淘汰帧的注册数据和脚本节点。
- 快速切帧时，用请求序号防止较早的加载结果覆盖最新选择；加载失败应显示错误，不能一直停留在无解释的黑屏。

### 1.5 代码应该改在哪里

| 位置 | 用途 |
| --- | --- |
| [web/mask_pointcloud/index.html](../InstanceRecognition/web/mask_pointcloud/index.html) | 页面结构、画布和控制按钮。 |
| [web/mask_pointcloud/styles.css](../InstanceRecognition/web/mask_pointcloud/styles.css) | 布局、画布尺寸和窄屏适配。 |
| [web/mask_pointcloud/viewer.js](../InstanceRecognition/web/mask_pointcloud/viewer.js) | 数据加载、2D／3D 联动、WebGL 绘制及鼠标交互。 |
| [src/backproject_masks.py](../InstanceRecognition/src/backproject_masks.py) | 导出数据，并将页面模板复制到实验输出目录。 |
| [当前输出目录](../InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/) | 用户实际打开的页面及该次实验的可视化资产。 |

修复页面时，以 `web/mask_pointcloud/` 下的源码为准，同时同步用户当前打开的输出副本。**不能只改输出副本，否则下一次导出会丢失修复；也不能只改模板，让当前页面继续使用旧代码。** 只改交互时无需重跑模型或反投影。

关键函数：`setupMouse` 管鼠标事件；`computeMatrix` 管视图变换；`renderCloud` 管绘制；`updateCloudBuffers` 管当前显示点集；`setupControls` 包含重置逻辑。

### 1.6 完成前怎么验收

至少验证：

- 用真实浏览器通过 `file://` 打开，场景背景、当前帧和选中 mask 均正确显示。
- 真实右键拖动后，平移值、渲染矩阵及画布像素都发生变化，旋转角不变；不能只检查事件处理函数存在。
- 左键仍能旋转，滚轮仍能缩放；画布上的右键菜单被阻止，松键或取消后不再继续拖动。
- 重置同时恢复旋转、距离和平移；切换帧／实例不会意外清空当前视角。
- 快速切帧不会显示过期结果；桌面和窄屏没有整页横向溢出，控制台没有运行时异常。

现有真实浏览器回归脚本：[check_mask_pointcloud_viewer.py](../InstanceRecognition/tests/check_mask_pointcloud_viewer.py)。它包含鼠标交互检查以及当前 scene0000_00 的帧加载、显示点数、缓存和布局检查；其中场景帧数与背景点数断言是样例专用，复用于其他实验时需相应调整。取消拖动分支通过合成 `pointercancel` 事件覆盖，并非真实触屏手势验证。

复查当前参考页面：

```bash
python3 /home/wzt/wzt/CVPR_2027/InstanceRecognition/tests/check_mask_pointcloud_viewer.py \
  --page /home/wzt/wzt/CVPR_2027/InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/index.html \
  --output /home/wzt/wzt/CVPR_2027/InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/qa/principles_check
```

最近一次右键修复的验证记录：[browser_checks.json](../InstanceRecognition/runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw/qa/right_button_pan/browser_checks.json)。

边界：当前基准是桌面鼠标交互；中键平移、双指缩放、三维点选、自动聚焦单个物体等不能默认视为已实现。需要这些能力时另行增加实现和验收。
