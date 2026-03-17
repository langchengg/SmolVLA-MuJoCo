# Evaluation Artifact Schema

This repository uses two distinct result layers:

- raw LeRobot eval outputs under `results/raw`
- promoted benchmark registry outputs under `results/real`

It also keeps a separate simulation demo schema for the desktop-sorting showcase.

## 1. Raw LeRobot output locations

Training:

- `results/raw/train/<run_name>/`

Evaluation:

- `results/raw/eval/<run_name>/`

Expected raw eval artifact:

- `results/raw/eval/<run_name>/eval_info.json`

The promotion pipeline expects `eval_info.json` to expose, at minimum:

- `aggregated.pc_success`
- `aggregated.eval_ep_s`
- `per_episode`

The current implementation also tolerates partial files and falls back to `per_episode.success` for success rate if needed.

## 2. Promoted benchmark registry schema

Canonical file:

- `results/real/benchmark_registry.csv`

Registry columns:

| column | meaning |
| --- | --- |
| `source` | `real` or `mock` |
| `run_name` | unique promoted run label |
| `policy_label` | human-readable policy identifier |
| `checkpoint_step` | checkpoint step if available |
| `train_dir` | raw training directory |
| `eval_dir` | raw eval directory |
| `policy_path` | HF repo id or local policy path |
| `experiment_family` | `generalization`, `visual_robustness`, `chunking`, or `latency` |
| `model_variant` | `zero_shot` or `finetuned` |
| `task_suite` | benchmark suite name |
| `language_variant` | instruction condition |
| `spatial_variant` | spatial condition |
| `visual_variant` | visual condition |
| `chunk_size` | action chunk size |
| `quantization` | quantization setting |
| `n_episodes` | number of eval episodes |
| `success_rate` | mean success rate |
| `latency_ms` | mean policy latency if provided |
| `rollout_hz` | rollout frequency if provided |
| `trajectory_jerk` | motion smoothness proxy if provided |
| `episode_time_s` | mean episode time |

Upsert key:

- `source`
- `run_name`
- `experiment_family`
- `model_variant`
- `task_suite`
- `language_variant`
- `spatial_variant`
- `visual_variant`
- `chunk_size`
- `quantization`

## 3. Published real benchmark outputs

Promotion regenerates:

- `results/real/summary/summary.md`
- `results/real/figures/real_overview.png`
- `results/real/figures/generalization_matrix.png`
- `results/real/figures/visual_robustness.png`
- `results/real/figures/chunking_tradeoff.png`
- `results/real/figures/latency_tradeoff.png`
- `examples/real_benchmark_results.csv`
- `reports/real/*.png`
- `docs/results/latest_real_results.md`

## 4. Mock desktop-sorting schema

Episode-level log:

- `examples/mock_desktop_sorting_eval_log.jsonl`

Aggregate-level summary:

- `examples/mock_desktop_sorting_eval_summary.csv`

Episode JSONL fields include:

- instruction id/text
- model variant
- visual condition
- success/failure tag
- latency
- collision and regrasp counts
- video path

Aggregate CSV groups by:

- model variant
- task family
- language variant
- visual variant

## 5. Mock benchmark schema

Mock benchmark sample:

- `examples/mock_benchmark_results.csv`

This uses the same normalized benchmark registry columns as the real registry, but with:

- `source=mock`
- synthetic run names and policy labels

This makes the analysis and plotting code reusable across both paths.
