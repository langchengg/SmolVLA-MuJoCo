# Evaluation Log Format

## 1. 目的

评测日志要同时服务 3 件事:

1. 后续统计分析
2. GitHub 图表和表格
3. 失败案例回放与视频筛选

因此建议分成两层:

- `episode-level JSONL`
- `aggregate CSV`

## 2. Episode-level JSONL

每一行代表一个 episode。

推荐文件名:

`desktop_sorting_eval_log.jsonl`

### 必需字段

- `run_id`
- `timestamp`
- `model_variant`
- `checkpoint`
- `scene_id`
- `task_family`
- `instruction_id`
- `instruction_text`
- `language_variant`
- `visual_variant`
- `object_layout_seed`
- `camera_variant`
- `n_objects`
- `task_success`
- `object_sort_accuracy`
- `instruction_grounding_accuracy`
- `completion_time_s`
- `mean_policy_latency_ms`
- `rollout_hz`
- `collision_count`
- `regrasp_count`
- `trajectory_jerk`

### 推荐字段

- `failure_tag`
- `video_path`
- `notes`

### 字段说明

- `model_variant`: `zero_shot` / `finetuned`
- `language_variant`: `exact` / `paraphrase`
- `visual_variant`: `nominal` / `low_light` / `clutter_background` / `camera_yaw_20deg`
- `task_success`: `true` / `false`
- `object_sort_accuracy`: `0.0-1.0`
- `instruction_grounding_accuracy`: `0.0-1.0`

## 3. Aggregate CSV

每一行代表一个实验条件汇总。

推荐文件名:

`desktop_sorting_eval_summary.csv`

### 推荐字段

- `experiment_family`
- `model_variant`
- `task_family`
- `language_variant`
- `visual_variant`
- `n_episodes`
- `success_rate`
- `object_sort_accuracy`
- `instruction_grounding_accuracy`
- `completion_time_s`
- `mean_policy_latency_ms`
- `rollout_hz`
- `collision_count`
- `trajectory_jerk`

## 4. 记录原则

### 原则 1

每次跑实验都记录 `run_id`

### 原则 2

每条日志都保留原始 `instruction_text`

### 原则 3

失败一定打 `failure_tag`

### 原则 4

如果有视频文件，日志里保存 `video_path`

这样后面找视频和写 README 会快很多。

## 5. 最推荐的工作流

1. 跑 episode
2. 写入 JSONL
3. 聚合成 CSV
4. 用 CSV 画图
5. 用 JSONL 找失败案例视频

## 6. 为什么不用只存 CSV

因为只存 CSV 会丢失很多细粒度信息:

- 哪一句 prompt 失败了
- 哪个 seed 失败了
- 对应哪段视频
- 是 grasp 失败还是排序目标理解失败

所以:

- `CSV` 用来汇总
- `JSONL` 用来追 episode 细节
