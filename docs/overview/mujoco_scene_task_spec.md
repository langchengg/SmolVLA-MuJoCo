# MuJoCo Scene Task Spec

## 1. Scene Summary

场景名称：`tabletop_two_trays_v1`

用途：

- 作为 GitHub 展示用的轻量机械臂桌面抓取与排序场景
- 作为语言条件操作任务的可视化载体
- 作为 episode logging 和结果分析的统一输入来源

## 2. Workspace

桌面布局固定为三块容器区域：

- `left_tray`
- `right_tray`
- `back_bin`

桌面中央为待抓取区域，物体在该区域内随机初始化。

## 3. Robot

机器人类型：`lightweight_cartesian_arm`

设计原则：

- 视觉上明确是机械臂操作，不要求精确复刻 Panda
- 不依赖外部 menagerie 资产
- 适合在本仓库内自包含运行

动作原语：

1. hover above object
2. descend to pick
3. lift
4. move to target receptacle
5. descend to place
6. release
7. return home

## 4. Cameras

默认相机：

- `front_camera`
- `wrist_camera`
- `top_camera`

README hero 视频默认使用：

- `front_camera`

## 5. Objects

默认物体集合：

- `red_cube`
- `blue_cube`
- `green_cylinder`
- `yellow_block`
- `orange_cylinder`

默认数量规则：

- train-style layouts: `4` objects
- eval-style layouts: `5` objects

## 6. Task Families

固定支持 5 类任务：

### `single_object_pick_place`

示例：

`Pick up the red cube and place it in the left tray.`

### `color_grouping`

示例：

`Sort all blue objects into the right tray.`

### `shape_grouping`

示例：

`Place cylinders in the back bin and cubes in the front tray.`

### `compositional_sorting`

示例：

`Place warm-colored objects in the left tray and cool-colored objects in the right tray.`

### `paraphrase_generalization`

示例：

`Grasp the crimson block and drop it into the left tray.`

## 7. Visual Randomization

默认扰动集合：

- `nominal`
- `low_light`
- `clutter_background`
- `camera_yaw_20deg`

这些扰动不追求物理写实的工业级 domain randomization，而是为了支持 GitHub 上可讲清楚的 robustness story。

## 8. Episode Contract

每个 episode 至少要记录：

- 使用的指令
- 使用的模型变体
- 场景 seed
- 任务成功与否
- 目标选择是否正确
- 物体是否落入正确容器
- 平均策略延迟
- 碰撞次数
- regrasp 次数
- 视频文件路径

## 9. Output Contract

每次 showcase 运行必须产出：

- episode JSONL
- aggregate CSV
- summary Markdown
- showcase PNG figures
- per-episode MP4
- hero MP4 / GIF / thumbnail

默认输出目录：

- `artifacts/showcase/episodes`
- `artifacts/showcase/summary`
- `artifacts/showcase/figures`
- `artifacts/showcase/videos`

## 10. Non-goals

这个仓库当前版本不把下面这些作为默认目标：

- 精确动力学控制 benchmark
- 真实 Panda URDF/MJCF 资产复刻
- 抓取接触建模的高保真研究
- 真机控制闭环部署

这些内容可以在 advanced 路线中逐步接入，但不阻塞 GitHub demo 主线。
