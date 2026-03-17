from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


BENCHMARK_COLUMNS = [
    "experiment_family",
    "model_variant",
    "task_suite",
    "language_variant",
    "spatial_variant",
    "visual_variant",
    "chunk_size",
    "quantization",
    "n_episodes",
    "success_rate",
    "latency_ms",
    "rollout_hz",
    "trajectory_jerk",
    "episode_time_s",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    task_suite: str


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return BenchmarkConfig(task_suite=payload["dataset"]["task_suite"])


def generate_mock_benchmark_frame(config: BenchmarkConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    task_suite = config.task_suite

    base_success = {"zero_shot": 0.72, "finetuned": 0.89}
    paraphrase_penalty = {"zero_shot": 0.23, "finetuned": 0.11}
    spatial_penalty = {"zero_shot": 0.15, "finetuned": 0.08}
    visual_penalty = {
        "nominal": 0.00,
        "low_light": 0.25,
        "clutter_background": 0.21,
        "camera_yaw_15deg": 0.17,
    }
    visual_penalty_ft = {
        "nominal": 0.00,
        "low_light": 0.15,
        "clutter_background": 0.11,
        "camera_yaw_15deg": 0.08,
    }

    for model_variant in ["zero_shot", "finetuned"]:
        for language_variant in ["exact", "paraphrase"]:
            for spatial_variant in ["nominal", "shifted_left_5cm", "shifted_right_5cm"]:
                success = base_success[model_variant]
                if language_variant == "paraphrase":
                    success -= paraphrase_penalty[model_variant]
                if spatial_variant != "nominal":
                    success -= spatial_penalty[model_variant]
                if language_variant == "paraphrase" and spatial_variant != "nominal":
                    success -= 0.01
                jerk = 0.31 if model_variant == "zero_shot" else 0.22
                jerk += 0.07 if language_variant == "paraphrase" else 0.0
                jerk += 0.04 if spatial_variant != "nominal" else 0.0
                episode_time = 12.8 if model_variant == "zero_shot" else 10.2
                episode_time += 2.9 if language_variant == "paraphrase" else 0.0
                episode_time += 1.2 if spatial_variant != "nominal" else 0.0
                rows.append(
                    {
                        "experiment_family": "generalization",
                        "model_variant": model_variant,
                        "task_suite": task_suite,
                        "language_variant": language_variant,
                        "spatial_variant": spatial_variant,
                        "visual_variant": "nominal",
                        "chunk_size": 8,
                        "quantization": "fp16",
                        "n_episodes": 20,
                        "success_rate": round(success, 2),
                        "latency_ms": 96,
                        "rollout_hz": 10.4,
                        "trajectory_jerk": round(jerk, 2),
                        "episode_time_s": round(episode_time, 1),
                    }
                )

    for model_variant in ["zero_shot", "finetuned"]:
        penalties = visual_penalty if model_variant == "zero_shot" else visual_penalty_ft
        nominal_success = base_success[model_variant]
        nominal_time = 12.8 if model_variant == "zero_shot" else 10.2
        nominal_jerk = 0.31 if model_variant == "zero_shot" else 0.22
        for visual_variant, penalty in penalties.items():
            rows.append(
                {
                    "experiment_family": "visual_robustness",
                    "model_variant": model_variant,
                    "task_suite": task_suite,
                    "language_variant": "exact",
                    "spatial_variant": "nominal",
                    "visual_variant": visual_variant,
                    "chunk_size": 8,
                    "quantization": "fp16",
                    "n_episodes": 20,
                    "success_rate": round(nominal_success - penalty, 2),
                    "latency_ms": 96,
                    "rollout_hz": 10.4,
                    "trajectory_jerk": round(nominal_jerk + penalty * 0.35, 2),
                    "episode_time_s": round(nominal_time + penalty * 14, 1),
                }
            )

    chunking_map = {
        "zero_shot": {
            1: (0.61, 58, 17.2, 0.48, 12.3),
            4: (0.67, 74, 13.5, 0.39, 12.0),
            8: (0.72, 96, 10.4, 0.31, 12.8),
            16: (0.70, 142, 7.0, 0.28, 14.1),
        },
        "finetuned": {
            1: (0.76, 58, 17.2, 0.51, 10.6),
            4: (0.84, 74, 13.5, 0.35, 9.9),
            8: (0.89, 96, 10.4, 0.22, 10.2),
            16: (0.87, 142, 7.0, 0.18, 11.6),
        },
    }
    for model_variant, values in chunking_map.items():
        for chunk_size, (success, latency, hz, jerk, episode_time) in values.items():
            rows.append(
                {
                    "experiment_family": "chunking",
                    "model_variant": model_variant,
                    "task_suite": task_suite,
                    "language_variant": "exact",
                    "spatial_variant": "nominal",
                    "visual_variant": "nominal",
                    "chunk_size": chunk_size,
                    "quantization": "fp16",
                    "n_episodes": 20,
                    "success_rate": success,
                    "latency_ms": latency,
                    "rollout_hz": hz,
                    "trajectory_jerk": jerk,
                    "episode_time_s": episode_time,
                }
            )

    latency_rows = {
        "fp16": (0.89, 96, 10.4, 0.22, 10.2),
        "int8": (0.87, 68, 14.7, 0.24, 10.5),
        "int4": (0.79, 44, 22.7, 0.31, 11.8),
    }
    for quantization, (success, latency, hz, jerk, episode_time) in latency_rows.items():
        rows.append(
            {
                "experiment_family": "latency",
                "model_variant": "finetuned",
                "task_suite": task_suite,
                "language_variant": "exact",
                "spatial_variant": "nominal",
                "visual_variant": "nominal",
                "chunk_size": 8,
                "quantization": quantization,
                "n_episodes": 20,
                "success_rate": success,
                "latency_ms": latency,
                "rollout_hz": hz,
                "trajectory_jerk": jerk,
                "episode_time_s": episode_time,
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)
