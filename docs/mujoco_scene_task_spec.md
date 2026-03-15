# MuJoCo Scene Task Specification

## 1. 目标

这个文档定义桌面抓取与多物体排序场景的任务规格，用来统一:

- 场景搭建
- 数据记录
- 指令设计
- 评估标准
- 视频输出

目标任务名:

`Language-Driven Desktop Grasping and Multi-Object Sorting`

## 2. 场景目标

我们希望这个 MuJoCo 场景同时满足:

1. 能体现文本指令驱动
2. 能体现多物体上下文推理
3. 能方便做可视化视频
4. 能定义清晰 success criteria
5. 能支持泛化与鲁棒性实验

## 3. 机器人与控制

- robot: `Franka Panda`
- gripper: `parallel jaw gripper`
- control mode:
  - end-effector pose delta
  - gripper open/close command
- recommendation:
  - 先做相对控制
  - 后续再对比绝对控制

## 4. 工作空间

### 4.1 桌面

- shape: rectangle tabletop
- size recommendation:
  - width: `0.7m`
  - depth: `0.5m`
  - height: `0.75m`

### 4.2 容器

建议最小版本放:

- `left_tray`
- `right_tray`
- optional `back_bin`

tray/bin 的目标不是复杂，而是:

- 容易定义目标区域
- 容易判定成功
- 容易在视频里看清楚

## 5. 物体集合

### 5.1 数量

建议:

- train/easy scene: `3-4`
- eval/hard scene: `5-6`

### 5.2 属性

每个物体至少包含:

- `object_id`
- `shape`
- `color`
- `size`
- `spawn_pose`
- `target_container`

### 5.3 推荐对象类型

- cube
- cylinder
- block

### 5.4 推荐颜色

- red
- blue
- green
- yellow
- orange

## 6. 相机设置

建议至少保留 3 个视角:

1. `front_camera`
2. `wrist_camera`
3. `top_camera`

### 6.1 front_camera

- 用于主视频输出
- 最适合 GitHub 首页展示

### 6.2 wrist_camera

- 用于更细粒度观察抓取
- 对分析失败很有帮助

### 6.3 top_camera

- 适合检测排序结果
- 适合自动评估 object placement

## 7. 随机化设计

为了让任务不只是“背动作”，建议至少随机化:

- object initial pose
- object ordering
- lighting intensity
- camera yaw
- background texture

推荐范围:

- x/y object shift: `+-5cm`
- camera yaw: `0 / 10 / 20 deg`
- light level: `nominal / dim / bright`

## 8. 任务家族

### 8.1 Single-object pick-and-place

例子:

- `Pick up the red cube and place it in the left tray.`

### 8.2 Color grouping

例子:

- `Sort all blue objects into the right tray.`

### 8.3 Shape grouping

例子:

- `Place cylinders in the back bin and cubes in the front tray.`

### 8.4 Compositional instruction

例子:

- `Move warm-colored objects to the left tray and cool-colored objects to the right tray.`

### 8.5 Paraphrase generalization

例子:

- `Grasp the crimson block and drop it into the left tray.`

## 9. 成功判定

### 9.1 Episode-level success

满足以下条件则记为 success:

1. 所有目标物体进入正确容器
2. 非目标物体未被错误放置
3. 没有严重碰撞导致任务无效
4. 在时间上限内完成

### 9.2 Object-level success

每个物体单独记录:

- correct container
- incorrect container
- not moved
- dropped outside

## 10. 评估指标

建议统一记录:

- `task_success`
- `object_sort_accuracy`
- `instruction_grounding_accuracy`
- `completion_time_s`
- `mean_policy_latency_ms`
- `rollout_hz`
- `collision_count`
- `regrasp_count`
- `trajectory_jerk`

## 11. 失败类型标签

建议给失败 episode 打标签，后面做视频和分析特别有用:

- `wrong_object_selected`
- `wrong_container_selected`
- `grasp_failure`
- `drop_failure`
- `timeout`
- `collision_abort`
- `paraphrase_misunderstanding`
- `visual_perturbation_failure`

## 12. 视频导出建议

### 12.1 GitHub 首页视频

- duration: `20-35s`
- format: `mp4` or converted gif
- content:
  - prompt overlay
  - scene start
  - policy execution
  - final sorted state

### 12.2 分析视频

建议保留:

- nominal success
- paraphrase failure
- visual perturbation failure
- zero-shot vs fine-tuned side-by-side

## 13. 场景版本建议

### v0

- 2 trays
- 3 objects
- exact prompt only

### v1

- 2 trays + 1 back bin
- 4-5 objects
- paraphrase prompts

### v2

- multi-condition perturbations
- lighting/background/camera changes
- video comparison suite

## 14. 与 benchmark 的关系

这个 MuJoCo 桌面场景的角色是:

`showcase and targeted analysis environment`

不是为了替代 `LIBERO`，而是为了:

- 更容易出视频
- 更容易解释模型行为
- 更容易做可控扰动实验
