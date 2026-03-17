# Evaluation Log Format

这个仓库统一采用：

- `episode-level`: JSONL
- `aggregate-level`: CSV

这样既适合程序化分析，也适合放到 GitHub 里直接查看。

## 1. Showcase Episode JSONL

路径示例：

- `examples/desktop_sorting_eval_log.jsonl`
- `artifacts/showcase/episodes/desktop_sorting_eval_log.jsonl`

一行对应一个 episode。

### Required fields

| field | meaning |
| --- | --- |
| `run_id` | 本次运行标识 |
| `timestamp` | 生成时间 |
| `model_variant` | `zero_shot` / `finetuned` |
| `checkpoint` | mock checkpoint 名称 |
| `scene_id` | MuJoCo scene 标识 |
| `task_family` | 任务族 |
| `instruction_id` | 指令模板 id |
| `instruction_text` | 文本指令 |
| `language_variant` | `exact` / `paraphrase` |
| `visual_variant` | `nominal` / `low_light` / `clutter_background` / `camera_yaw_20deg` |
| `object_layout_seed` | 物体布局 seed |
| `camera_variant` | 相机设置 |
| `n_objects` | 物体数量 |
| `task_success` | 是否成功 |
| `object_sort_accuracy` | 目标物体放置正确率 |
| `instruction_grounding_accuracy` | 指令 grounding 准确率 |
| `completion_time_s` | 完成时长 |
| `mean_policy_latency_ms` | 平均策略延迟 |
| `rollout_hz` | 推理频率 |
| `collision_count` | 碰撞次数 |
| `regrasp_count` | 重抓次数 |
| `trajectory_jerk` | 轨迹平滑性 proxy |
| `failure_tag` | 失败标签 |
| `video_path` | 视频路径 |
| `notes` | 备注 |

### Example

```json
{
  "run_id": "showcase_zero_shot_s7",
  "model_variant": "zero_shot",
  "task_family": "paraphrase_generalization",
  "instruction_id": "para_crimson_block_left_tray",
  "task_success": false,
  "failure_tag": "paraphrase_misunderstanding",
  "mean_policy_latency_ms": 70.0,
  "video_path": "artifacts/showcase/videos/zero_shot_para_crimson_block_left_tray.mp4"
}
```

## 2. Showcase Aggregate CSV

路径示例：

- `examples/desktop_sorting_eval_summary.csv`
- `artifacts/showcase/summary/desktop_sorting_eval_summary.csv`

聚合粒度：

- `model_variant`
- `task_family`
- `language_variant`
- `visual_variant`

### Required fields

| field | meaning |
| --- | --- |
| `experiment_family` | 固定为 `desktop_sorting` |
| `model_variant` | 模型变体 |
| `task_family` | 任务族 |
| `language_variant` | 语言条件 |
| `visual_variant` | 视觉条件 |
| `n_episodes` | episode 数 |
| `success_rate` | 成功率 |
| `object_sort_accuracy` | 平均放置准确率 |
| `instruction_grounding_accuracy` | 平均 grounding 准确率 |
| `completion_time_s` | 平均完成时长 |
| `mean_policy_latency_ms` | 平均延迟 |
| `rollout_hz` | 平均推理频率 |
| `collision_count` | 平均碰撞次数 |
| `regrasp_count` | 平均重抓次数 |
| `trajectory_jerk` | 平均 jerk proxy |

## 3. Benchmark Mock CSV

路径示例：

- `examples/sample_results.csv`
- `artifacts/benchmark/summary/mock_benchmark_results.csv`

聚合粒度：

- `experiment_family`
- `model_variant`
- `task_suite`
- `language_variant`
- `spatial_variant`
- `visual_variant`
- `chunk_size`
- `quantization`

### Key fields

| field | meaning |
| --- | --- |
| `experiment_family` | `generalization` / `visual_robustness` / `chunking` / `latency` |
| `model_variant` | `zero_shot` / `finetuned` |
| `task_suite` | benchmark suite 名称 |
| `success_rate` | 模拟成功率 |
| `latency_ms` | 平均推理延迟 |
| `rollout_hz` | 推理频率 |
| `trajectory_jerk` | 轨迹平滑度 proxy |
| `episode_time_s` | episode 用时 |

## 4. Why This Format

这套格式对 GitHub 项目很友好，因为它同时满足：

- 可复现
- 可脚本分析
- 可出图
- 可写简历 bullet
- 可链接到具体视频片段
