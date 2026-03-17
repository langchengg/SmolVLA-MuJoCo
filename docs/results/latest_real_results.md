# Latest Real Results

This document is generated from `results/real/benchmark_registry.csv`.
Committed rows in this repository are sample promoted benchmark rows.

## Overview

- Promoted runs: 31
- Sources: real
- Model variants: zero_shot, finetuned
- Best nominal success: finetuned 89%
- Best paraphrase success: finetuned 78%
- Best realtime setting: fp16 at 10.4 Hz

## Findings

- Nominal generalization improves from 72% to 89%.
- Visual robustness coverage: 4 visual conditions across 2 model variants.
- Best chunking row in the current registry: chunk 8 with 89% success.
- Best quantized realtime row: fp16 at 10.4 Hz and 89% success.

## Registry Rows

| run_name | experiment_family | model_variant | task_suite | language_variant | spatial_variant | visual_variant | chunk_size | quantization | n_episodes | success_rate | latency_ms | rollout_hz | checkpoint_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_real_000 | generalization | zero_shot | libero_10 | exact | nominal | nominal | 8 | fp16 | 20 | 0.72 | 96 | 10.4 |  |
| sample_real_001 | generalization | zero_shot | libero_10 | exact | shifted_left_5cm | nominal | 8 | fp16 | 20 | 0.57 | 96 | 10.4 |  |
| sample_real_002 | generalization | zero_shot | libero_10 | exact | shifted_right_5cm | nominal | 8 | fp16 | 20 | 0.57 | 96 | 10.4 |  |
| sample_real_003 | generalization | zero_shot | libero_10 | paraphrase | nominal | nominal | 8 | fp16 | 20 | 0.49 | 96 | 10.4 |  |
| sample_real_004 | generalization | zero_shot | libero_10 | paraphrase | shifted_left_5cm | nominal | 8 | fp16 | 20 | 0.33 | 96 | 10.4 |  |
| sample_real_005 | generalization | zero_shot | libero_10 | paraphrase | shifted_right_5cm | nominal | 8 | fp16 | 20 | 0.33 | 96 | 10.4 |  |
| sample_real_006 | generalization | finetuned | libero_10 | exact | nominal | nominal | 8 | fp16 | 20 | 0.89 | 96 | 10.4 | 20000 |
| sample_real_007 | generalization | finetuned | libero_10 | exact | shifted_left_5cm | nominal | 8 | fp16 | 20 | 0.81 | 96 | 10.4 | 20000 |
| sample_real_008 | generalization | finetuned | libero_10 | exact | shifted_right_5cm | nominal | 8 | fp16 | 20 | 0.81 | 96 | 10.4 | 20000 |
| sample_real_009 | generalization | finetuned | libero_10 | paraphrase | nominal | nominal | 8 | fp16 | 20 | 0.78 | 96 | 10.4 | 20000 |
| sample_real_010 | generalization | finetuned | libero_10 | paraphrase | shifted_left_5cm | nominal | 8 | fp16 | 20 | 0.69 | 96 | 10.4 | 20000 |
| sample_real_011 | generalization | finetuned | libero_10 | paraphrase | shifted_right_5cm | nominal | 8 | fp16 | 20 | 0.69 | 96 | 10.4 | 20000 |
| sample_real_012 | visual_robustness | zero_shot | libero_10 | exact | nominal | nominal | 8 | fp16 | 20 | 0.72 | 96 | 10.4 |  |
| sample_real_013 | visual_robustness | zero_shot | libero_10 | exact | nominal | low_light | 8 | fp16 | 20 | 0.47 | 96 | 10.4 |  |
| sample_real_014 | visual_robustness | zero_shot | libero_10 | exact | nominal | clutter_background | 8 | fp16 | 20 | 0.51 | 96 | 10.4 |  |
| sample_real_015 | visual_robustness | zero_shot | libero_10 | exact | nominal | camera_yaw_15deg | 8 | fp16 | 20 | 0.55 | 96 | 10.4 |  |
| sample_real_016 | visual_robustness | finetuned | libero_10 | exact | nominal | nominal | 8 | fp16 | 20 | 0.89 | 96 | 10.4 | 20000 |
| sample_real_017 | visual_robustness | finetuned | libero_10 | exact | nominal | low_light | 8 | fp16 | 20 | 0.74 | 96 | 10.4 | 20000 |
| sample_real_018 | visual_robustness | finetuned | libero_10 | exact | nominal | clutter_background | 8 | fp16 | 20 | 0.78 | 96 | 10.4 | 20000 |
| sample_real_019 | visual_robustness | finetuned | libero_10 | exact | nominal | camera_yaw_15deg | 8 | fp16 | 20 | 0.81 | 96 | 10.4 | 20000 |
| sample_real_020 | chunking | zero_shot | libero_10 | exact | nominal | nominal | 1 | fp16 | 20 | 0.61 | 58 | 17.2 |  |
| sample_real_021 | chunking | zero_shot | libero_10 | exact | nominal | nominal | 4 | fp16 | 20 | 0.67 | 74 | 13.5 |  |
| sample_real_022 | chunking | zero_shot | libero_10 | exact | nominal | nominal | 8 | fp16 | 20 | 0.72 | 96 | 10.4 |  |
| sample_real_023 | chunking | zero_shot | libero_10 | exact | nominal | nominal | 16 | fp16 | 20 | 0.7 | 142 | 7.0 |  |
| sample_real_024 | chunking | finetuned | libero_10 | exact | nominal | nominal | 1 | fp16 | 20 | 0.76 | 58 | 17.2 | 20000 |
| sample_real_025 | chunking | finetuned | libero_10 | exact | nominal | nominal | 4 | fp16 | 20 | 0.84 | 74 | 13.5 | 20000 |
| sample_real_026 | chunking | finetuned | libero_10 | exact | nominal | nominal | 8 | fp16 | 20 | 0.89 | 96 | 10.4 | 20000 |
| sample_real_027 | chunking | finetuned | libero_10 | exact | nominal | nominal | 16 | fp16 | 20 | 0.87 | 142 | 7.0 | 20000 |
| sample_real_028 | latency | finetuned | libero_10 | exact | nominal | nominal | 8 | fp16 | 20 | 0.89 | 96 | 10.4 | 20000 |
| sample_real_029 | latency | finetuned | libero_10 | exact | nominal | nominal | 8 | int8 | 20 | 0.87 | 68 | 14.7 | 20000 |
| sample_real_030 | latency | finetuned | libero_10 | exact | nominal | nominal | 8 | int4 | 20 | 0.79 | 44 | 22.7 | 20000 |
