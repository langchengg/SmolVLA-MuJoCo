from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SHOWCASE_REQUIRED_COLUMNS = {
    "run_id",
    "timestamp",
    "model_variant",
    "checkpoint",
    "scene_id",
    "task_family",
    "instruction_id",
    "instruction_text",
    "language_variant",
    "visual_variant",
    "object_layout_seed",
    "camera_variant",
    "n_objects",
    "task_success",
    "object_sort_accuracy",
    "instruction_grounding_accuracy",
    "completion_time_s",
    "mean_policy_latency_ms",
    "rollout_hz",
    "collision_count",
    "regrasp_count",
    "trajectory_jerk",
    "failure_tag",
    "video_path",
    "notes",
}


@dataclass(frozen=True)
class ShowcaseSummaryArtifacts:
    summary_frame: pd.DataFrame
    markdown: str


def load_episode_log(path: str | Path) -> pd.DataFrame:
    frame = pd.read_json(Path(path), lines=True)
    missing = SHOWCASE_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")
    return frame


def summarize_episode_log(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(
            ["model_variant", "task_family", "language_variant", "visual_variant"],
            as_index=False,
        )
        .agg(
            n_episodes=("task_success", "count"),
            success_rate=("task_success", "mean"),
            object_sort_accuracy=("object_sort_accuracy", "mean"),
            instruction_grounding_accuracy=("instruction_grounding_accuracy", "mean"),
            completion_time_s=("completion_time_s", "mean"),
            mean_policy_latency_ms=("mean_policy_latency_ms", "mean"),
            rollout_hz=("rollout_hz", "mean"),
            collision_count=("collision_count", "mean"),
            regrasp_count=("regrasp_count", "mean"),
            trajectory_jerk=("trajectory_jerk", "mean"),
        )
    )
    summary.insert(0, "experiment_family", "desktop_sorting")
    return summary


def build_showcase_markdown(summary_frame: pd.DataFrame) -> str:
    success = summary_frame.groupby("model_variant", as_index=False)["success_rate"].mean()
    perturbed = summary_frame[summary_frame["visual_variant"] != "nominal"]
    paraphrase = summary_frame[summary_frame["language_variant"] == "paraphrase"]
    success_map = dict(zip(success["model_variant"], success["success_rate"], strict=False))
    perturbed_map = dict(zip(perturbed["model_variant"], perturbed["success_rate"], strict=False))
    paraphrase_map = dict(zip(paraphrase["model_variant"], paraphrase["success_rate"], strict=False))
    return "\n".join(
        [
            "# Desktop Sorting Summary",
            "",
            "## Key findings",
            "",
            (
                f"- Finetuned mock policy averages {success_map.get('finetuned', 0.0):.0%} success, "
                f"versus {success_map.get('zero_shot', 0.0):.0%} for zero-shot."
            ),
            (
                f"- Across perturbed scenes, finetuned retains {perturbed_map.get('finetuned', 0.0):.0%} success, "
                f"versus {perturbed_map.get('zero_shot', 0.0):.0%} for zero-shot."
            ),
            (
                f"- On paraphrased instructions, finetuned reaches {paraphrase_map.get('finetuned', 0.0):.0%} success, "
                f"versus {paraphrase_map.get('zero_shot', 0.0):.0%} for zero-shot."
            ),
            "",
            "## Suggested GitHub framing",
            "",
            "- Built a self-contained MuJoCo desktop sorting showcase that renders short videos and episode-level logs.",
            "- Used mock zero-shot and fine-tuned policies to visualize language grounding, perturbation robustness, and failure modes.",
        ]
    )


def summarize_showcase(path: str | Path) -> ShowcaseSummaryArtifacts:
    frame = load_episode_log(path)
    summary = summarize_episode_log(frame)
    return ShowcaseSummaryArtifacts(summary_frame=summary, markdown=build_showcase_markdown(summary))
