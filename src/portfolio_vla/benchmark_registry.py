from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROVENANCE_COLUMNS = [
    "source",
    "run_name",
    "policy_label",
    "checkpoint_step",
    "train_dir",
    "eval_dir",
    "policy_path",
]

BENCHMARK_METRIC_COLUMNS = [
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

BENCHMARK_COLUMNS = PROVENANCE_COLUMNS + BENCHMARK_METRIC_COLUMNS

UPSERT_KEY_COLUMNS = [
    "source",
    "run_name",
    "experiment_family",
    "model_variant",
    "task_suite",
    "language_variant",
    "spatial_variant",
    "visual_variant",
    "chunk_size",
    "quantization",
]


@dataclass(frozen=True)
class PromotedEvalRecord:
    row: dict[str, Any]
    eval_info: dict[str, Any]


def empty_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_COLUMNS)


def normalize_registry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    defaults: dict[str, Any] = {
        "source": "unknown",
        "run_name": "unspecified",
        "policy_label": "",
        "checkpoint_step": pd.NA,
        "train_dir": "",
        "eval_dir": "",
        "policy_path": "",
        "experiment_family": "",
        "model_variant": "",
        "task_suite": "",
        "language_variant": "exact",
        "spatial_variant": "nominal",
        "visual_variant": "nominal",
        "chunk_size": 8,
        "quantization": "fp16",
        "n_episodes": 0,
        "success_rate": pd.NA,
        "latency_ms": pd.NA,
        "rollout_hz": pd.NA,
        "trajectory_jerk": pd.NA,
        "episode_time_s": pd.NA,
    }

    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    for column in ["checkpoint_step", "chunk_size", "n_episodes"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ["success_rate", "latency_ms", "rollout_hz", "trajectory_jerk", "episode_time_s"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame[BENCHMARK_COLUMNS]


def load_benchmark_registry(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return empty_registry_frame()
    return normalize_registry_frame(pd.read_csv(path))


def save_benchmark_registry(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_registry_frame(frame).to_csv(path, index=False)


def upsert_registry_rows(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    existing = normalize_registry_frame(existing)
    incoming = normalize_registry_frame(incoming)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=UPSERT_KEY_COLUMNS, keep="last")
    combined = combined.sort_values(
        [
            "source",
            "experiment_family",
            "model_variant",
            "task_suite",
            "language_variant",
            "spatial_variant",
            "visual_variant",
            "chunk_size",
            "quantization",
            "run_name",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return normalize_registry_frame(combined)


def load_eval_info(eval_dir: str | Path) -> dict[str, Any]:
    eval_dir = Path(eval_dir)
    eval_info_path = eval_dir / "eval_info.json"
    with eval_info_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _list_length(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def _extract_n_episodes(payload: dict[str, Any]) -> int:
    per_episode = payload.get("per_episode", {})
    if isinstance(per_episode, dict):
        lengths = [_list_length(value) for value in per_episode.values()]
        lengths = [length for length in lengths if length > 0]
        if lengths:
            return max(lengths)
    aggregated = payload.get("aggregated", {})
    for key in ["n_episodes", "episodes", "num_episodes"]:
        value = aggregated.get(key, payload.get(key))
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _extract_success_rate(payload: dict[str, Any]) -> float:
    aggregated = payload.get("aggregated", {})
    if aggregated.get("pc_success") is not None:
        value = float(aggregated["pc_success"])
        return value / 100.0 if value > 1.0 else value
    per_episode = payload.get("per_episode", {})
    successes = per_episode.get("success")
    if isinstance(successes, list) and successes:
        return float(pd.Series(successes, dtype=float).mean())
    raise ValueError("Could not determine success rate from eval_info.json")


def _extract_episode_time(payload: dict[str, Any]) -> float | None:
    aggregated = payload.get("aggregated", {})
    for key in ["eval_ep_s", "episode_time_s"]:
        if aggregated.get(key) is not None:
            return float(aggregated[key])
    return None


def promote_eval_record(
    *,
    eval_dir: str | Path,
    run_name: str,
    experiment_family: str,
    model_variant: str,
    task_suite: str,
    language_variant: str,
    spatial_variant: str,
    visual_variant: str,
    chunk_size: int,
    quantization: str,
    policy_label: str | None = None,
    checkpoint_step: int | None = None,
    train_dir: str | Path | None = None,
    policy_path: str | Path | None = None,
    latency_ms: float | None = None,
    rollout_hz: float | None = None,
    trajectory_jerk: float | None = None,
) -> PromotedEvalRecord:
    eval_dir = Path(eval_dir).resolve()
    payload = load_eval_info(eval_dir)
    row = {
        "source": "real",
        "run_name": run_name,
        "policy_label": policy_label or model_variant,
        "checkpoint_step": checkpoint_step if checkpoint_step is not None else pd.NA,
        "train_dir": str(train_dir) if train_dir else "",
        "eval_dir": str(eval_dir),
        "policy_path": str(policy_path) if policy_path else "",
        "experiment_family": experiment_family,
        "model_variant": model_variant,
        "task_suite": task_suite,
        "language_variant": language_variant,
        "spatial_variant": spatial_variant,
        "visual_variant": visual_variant,
        "chunk_size": int(chunk_size),
        "quantization": quantization,
        "n_episodes": _extract_n_episodes(payload),
        "success_rate": _extract_success_rate(payload),
        "latency_ms": latency_ms,
        "rollout_hz": rollout_hz,
        "trajectory_jerk": trajectory_jerk,
        "episode_time_s": _extract_episode_time(payload),
    }
    return PromotedEvalRecord(row=normalize_registry_frame(pd.DataFrame([row])).iloc[0].to_dict(), eval_info=payload)
