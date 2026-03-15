from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
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
}


@dataclass(frozen=True)
class SummaryArtifacts:
    generalization_matrix: pd.DataFrame
    visual_robustness: pd.DataFrame
    chunking: pd.DataFrame
    latency: pd.DataFrame
    markdown: str


def load_results(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")
    return frame


def generalization_matrix(frame: pd.DataFrame, model_variant: str = "finetuned") -> pd.DataFrame:
    subset = frame[
        (frame["experiment_family"] == "generalization")
        & (frame["model_variant"] == model_variant)
    ]
    matrix = subset.pivot_table(
        index="language_variant",
        columns="spatial_variant",
        values="success_rate",
        aggfunc="mean",
    )
    return matrix.sort_index(axis=0).sort_index(axis=1)


def visual_robustness_table(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["experiment_family"] == "visual_robustness"]
    table = (
        subset.groupby(["model_variant", "visual_variant"], as_index=False)
        .agg(
            success_rate=("success_rate", "mean"),
            episode_time_s=("episode_time_s", "mean"),
        )
        .sort_values(["visual_variant", "model_variant"])
    )
    return table


def chunking_table(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["experiment_family"] == "chunking"]
    table = (
        subset.groupby(["model_variant", "chunk_size"], as_index=False)
        .agg(
            success_rate=("success_rate", "mean"),
            rollout_hz=("rollout_hz", "mean"),
            latency_ms=("latency_ms", "mean"),
            trajectory_jerk=("trajectory_jerk", "mean"),
        )
        .sort_values(["model_variant", "chunk_size"])
    )
    return table


def latency_table(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["experiment_family"] == "latency"]
    table = (
        subset.groupby(["model_variant", "quantization"], as_index=False)
        .agg(
            success_rate=("success_rate", "mean"),
            rollout_hz=("rollout_hz", "mean"),
            latency_ms=("latency_ms", "mean"),
        )
        .sort_values(["model_variant", "latency_ms"])
    )
    return table


def build_markdown_summary(frame: pd.DataFrame, realtime_threshold_hz: float = 10.0) -> str:
    gen_ft = generalization_matrix(frame, model_variant="finetuned")
    gen_zs = generalization_matrix(frame, model_variant="zero_shot")
    visual = visual_robustness_table(frame)
    chunking = chunking_table(frame)
    latency = latency_table(frame)

    ft_nominal = float(gen_ft.loc["exact", "nominal"])
    zs_nominal = float(gen_zs.loc["exact", "nominal"])
    ft_hard = float(
        gen_ft.loc["paraphrase", ["shifted_left_5cm", "shifted_right_5cm"]].mean()
    )
    zs_hard = float(
        gen_zs.loc["paraphrase", ["shifted_left_5cm", "shifted_right_5cm"]].mean()
    )

    visual_nominal = visual[visual["visual_variant"] == "nominal"].set_index("model_variant")
    visual_shifted = (
        visual[visual["visual_variant"] != "nominal"]
        .groupby("model_variant", as_index=True)["success_rate"]
        .mean()
    )
    ft_visual_drop = float(visual_nominal.loc["finetuned", "success_rate"] - visual_shifted.loc["finetuned"])
    zs_visual_drop = float(visual_nominal.loc["zero_shot", "success_rate"] - visual_shifted.loc["zero_shot"])

    chunk_candidates = chunking[
        (chunking["model_variant"] == "finetuned")
        & (chunking["rollout_hz"] >= realtime_threshold_hz)
    ].sort_values(["success_rate", "trajectory_jerk"], ascending=[False, True])
    best_chunk = chunk_candidates.iloc[0]

    latency_candidates = latency[
        (latency["model_variant"] == "finetuned")
        & (latency["rollout_hz"] >= realtime_threshold_hz)
    ].copy()
    best_success = float(latency_candidates["success_rate"].max())
    near_best = latency_candidates[latency_candidates["success_rate"] >= best_success - 0.03]
    best_latency = near_best.sort_values(["latency_ms", "success_rate"], ascending=[True, False]).iloc[0]

    return "\n".join(
        [
            "# Experiment Summary",
            "",
            "## Key findings",
            "",
            (
                f"- Fine-tuning improves nominal LIBERO success from {zs_nominal:.0%} "
                f"to {ft_nominal:.0%}."
            ),
            (
                f"- Under the hardest language+spatial conditions, zero-shot averages "
                f"{zs_hard:.0%} while fine-tuned averages {ft_hard:.0%}."
            ),
            (
                f"- Visual perturbation drop shrinks from {zs_visual_drop:.0%} "
                f"to {ft_visual_drop:.0%} after fine-tuning."
            ),
            (
                f"- Best realtime chunk size is {int(best_chunk['chunk_size'])}, with "
                f"{best_chunk['success_rate']:.0%} success at {best_chunk['rollout_hz']:.1f} Hz."
            ),
            (
                f"- Best quantization setting above {realtime_threshold_hz:.0f} Hz is "
                f"{best_latency['quantization']}, with {best_latency['success_rate']:.0%} success "
                f"at {best_latency['rollout_hz']:.1f} Hz."
            ),
            "",
            "## Suggested resume framing",
            "",
            (
                "- Built a simulation-first benchmark for SmolVLA on LIBERO Panda tasks, "
                "covering language, spatial, and visual generalization."
            ),
            (
                "- Quantified the accuracy-latency trade-off of action chunking and model "
                "quantization for realtime robotic control."
            ),
        ]
    )


def summarize_results(
    csv_path: str | Path,
    realtime_threshold_hz: float = 10.0,
) -> SummaryArtifacts:
    frame = load_results(csv_path)
    return SummaryArtifacts(
        generalization_matrix=generalization_matrix(frame, model_variant="finetuned"),
        visual_robustness=visual_robustness_table(frame),
        chunking=chunking_table(frame),
        latency=latency_table(frame),
        markdown=build_markdown_summary(frame, realtime_threshold_hz=realtime_threshold_hz),
    )
