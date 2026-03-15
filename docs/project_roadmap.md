# Project Roadmap

## 适合你的最终项目定位

最推荐的版本不是“我做了一个万能机械臂大模型”，而是一个双主线作品集:

- `Benchmark Track`: SmolVLA + LIBERO + MuJoCo
- `Showcase Track`: 文本指令驱动桌面抓取与多物体排序

整体项目定位可以写成:

`面向简历和 GitHub 的 simulation-first VLA benchmark + showcase 项目`

你的目标应该是把下面 4 件事讲清楚:

1. 我复现了什么
2. 我加了什么实验维度
3. 我从结果里得到了什么独特判断
4. 我如何把复杂能力可视化成视频 demo

## 机械臂和环境选择

### 机械臂

首选 `Franka Panda`

原因:

- 与 `LIBERO` 对齐
- 与 MoveIt 官方生态对齐
- 仿真资源丰富
- 写简历时辨识度高

### 环境

- 学习策略 benchmark: `MuJoCo + LIBERO`
- 规划系统 baseline: `MoveIt + Panda`
- 仿真桥接: `mujoco_ros2_control`

## 建议你怎么做成一个完整 GitHub 项目

### 1. 复现 baseline

先跑通:

- `SmolVLA zero-shot`
- `SmolVLA fine-tuned on LIBERO`

结果页至少给出:

- success rate
- 任务截图或短视频
- 训练/评估命令

### 2. 把你的创新点变成“实验套件”

不要只写“我做了语言泛化”。要写成 benchmark 套件:

- `Language Generalization Suite`
- `Spatial Perturbation Suite`
- `Visual Robustness Suite`
- `Chunking and Latency Suite`

这会让项目更像研究/工程作品，而不是课程作业。

### 3. 每个实验都要有一个核心问题

例如:

- 语言改写是否显著降低成功率?
- 微调是否能缓解视觉扰动下的性能下降?
- chunk size 增大后是更稳定还是更慢?
- 量化以后还能否维持 `>10Hz`?

### 4. 加一个桌面多物体排序 showcase

这里非常值得做。

原因:

- 更容易出视频
- 更容易让非机器人方向的人看懂
- 更像一个完整的应用场景
- 可以自然展示语言指令理解

但它最好的角色是:

`showcase layer, not benchmark replacement`

你可以直接做这些命令模板:

- `Pick up the red cube and put it into the left tray.`
- `Sort the blue objects into the right bin.`
- `Place cylinders at the back and cubes at the front.`

### 5. 最后加一个 MoveIt baseline

MoveIt 在这里的最好角色不是“替代 VLA”，而是:

`作为 planning-based reference system`

你可以比较:

- 任务完成率
- 动作是否更平滑
- 对视觉扰动是否敏感
- 对初始位姿变化的适应性

## 简历包装的正确方式

### 你应该强调

- benchmark 设计能力
- 结果分析能力
- 工程复现能力
- 对模型局限性的判断

### 你应该主动承认

- 这是 simulation-first 项目
- 没有做真实机器人部署
- 目前主要验证的是 benchmark 内泛化和效率

这种表达不会减分，反而会让人觉得你边界感很清楚。

## 如果你要写“独特思考”，写什么最有价值

### 1. 不要只看 success rate

加上:

- latency
- rollout frequency
- trajectory jerk
- perturbation drop

### 2. 不要只比模型大小

你更应该写:

`小模型在受限算力下是否还能保持可接受的控制频率和任务成功率`

### 3. 不要把 RL 当口号

如果你要加 RL，最好是下面两类:

- residual RL
- imitation initialized RL fine-tuning

这样更像“在已有策略上做改进”，不会显得散。

## 一个月版本的推荐节奏

### 第 1 周

- 跑通 LeRobot + LIBERO
- 跑通 baseline eval
- 整理日志格式

### 第 2 周

- 做语言/空间泛化实验
- 画 generalization matrix

### 第 3 周

- 做视觉扰动和 chunking 实验
- 补 latency 分析
- 搭建桌面抓取与多物体排序场景

### 第 4 周

- 接一个 MoveIt baseline
- 录制 showcase 视频
- 补 README、图、视频、简历 bullet

## 一个很重要的判断

如果你没有真机，千万不要因此否定自己。

你现在最该做的是:

`把“能复现开源 VLA + 能做严谨评测 + 能讲清结果”做到扎实`

这件事本身就已经比很多空泛项目强了。
