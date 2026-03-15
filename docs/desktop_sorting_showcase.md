# Desktop Sorting Showcase

## 相关文件

- `configs/desktop_sorting_showcase.yaml`
- `docs/mujoco_scene_task_spec.md`
- `templates/desktop_sorting_prompts.yaml`
- `docs/eval_log_format.md`
- `examples/desktop_sorting_eval_log.jsonl`
- `examples/desktop_sorting_eval_summary.csv`

## 这个方向值不值得加

值得，而且很适合你。

如果你把它作为 `benchmark` 之外的 `showcase task`，它会明显增强项目的可展示性:

- 视频更直观
- 指令驱动感更强
- 容易做对比演示
- 更适合放在 README 顶部

最好的项目结构是:

`LIBERO benchmark for rigor + desktop sorting showcase for storytelling`

## 推荐任务定义

任务名建议写成:

`Language-Driven Desktop Grasping and Multi-Object Sorting in MuJoCo`

或者:

`Text-Conditioned Desktop Sorting with SmolVLA Policies`

## 场景设计

- 机器人: `Franka Panda`
- 环境: `MuJoCo tabletop`
- 工作区: 桌面 + 2 到 3 个 bin 或 tray
- 物体数量: `3` 到 `6`
- 物体属性:
  - color
  - shape
  - size
  - position

## 推荐指令模板

### 单目标

- `Pick up the red cube and place it in the left tray.`
- `Move the blue cylinder to the back bin.`

### 多物体排序

- `Sort all blue objects into the right bin.`
- `Place cylinders in the back tray and cubes in the front tray.`
- `Move warm-colored objects to the top bin and cool-colored objects to the bottom bin.`

### 语言泛化

- `grasp the crimson block`
- `move the scarlet cube`
- `put the reddish object into the left tray`

## 为什么它特别适合 VLA

这个任务天然支持:

- 文本指令理解
- 视觉目标识别
- 多物体上下文
- 顺序决策
- 失败案例可视化

你的视频里可以非常容易做出这几种对比:

- `zero-shot vs fine-tuned`
- `exact wording vs paraphrase`
- `nominal lighting vs low light`
- `single-step vs chunked action prediction`

## 建议评估指标

- task success rate
- object-level sorting accuracy
- instruction grounding accuracy
- completion time
- average re-grasp count
- collision count
- trajectory jerk

## 推荐的视频组织方式

README 顶部的视频最好控制在 `20` 到 `35` 秒。

### Shot 1

显示文本指令和场景初始化:

- `Sort all blue objects into the right bin.`

### Shot 2

播放 fine-tuned policy 完成任务

### Shot 3

并排展示 zero-shot 和 fine-tuned

### Shot 4

展示一个失败案例:

- paraphrase 指令失败
- clutter background 失败
- low-light 失败

### Shot 5

展示结果统计图

## 数据集怎么来

### 最省力的方式

先把 desktop sorting 当作:

`small custom task family built on top of the SmolVLA fine-tuning pipeline`

也就是:

- benchmark 主体仍然使用 `LIBERO`
- desktop sorting 作为你自己的小型定制数据集或小规模模拟数据

### 如果你第一版不想采数据

你也可以先只做:

- few task templates
- qualitative video results
- 少量手工构造评测

这在作品集阶段是完全合理的。

## 你在简历里该怎么写

可以写:

- Built a MuJoCo desktop sorting showcase for text-conditioned robotic manipulation, turning VLA behavior into short, interpretable demo videos.
- Designed multi-object language grounding tasks with color, shape, and tray-based sorting instructions to complement LIBERO benchmark evaluation.

不要写:

- real-world autonomous desktop manipulation
- production-ready instruction-following robot

## 最重要的一句判断

是的，加这个方向会更好。

但请把它放在:

`benchmark-backed showcase`

而不是:

`only demo, no benchmark`
