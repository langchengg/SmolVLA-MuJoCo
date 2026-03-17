from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from portfolio_vla.benchmark_registry import BENCHMARK_COLUMNS, normalize_registry_frame


REQUIRED_COLUMNS = set(BENCHMARK_COLUMNS)


@dataclass(frozen=True)
class SummaryArtifacts:
    frame: pd.DataFrame
    generalization_matrix: pd.DataFrame
    visual_robustness: pd.DataFrame
    chunking: pd.DataFrame
    latency: pd.DataFrame
    overview_rows: list[tuple[str, str]]
    markdown: str


def load_results(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    frame = normalize_registry_frame(pd.read_csv(csv_path))
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")
    return frame


def _available_model_variants(frame: pd.DataFrame) -> list[str]:
    return [str(value) for value in frame["model_variant"].dropna().drop_duplicates().tolist()]


def generalization_matrix(frame: pd.DataFrame, model_variant: str = "finetuned") -> pd.DataFrame:
    subset = frame[
        (frame["experiment_family"] == "generalization")
        & (frame["model_variant"] == model_variant)
    ]
    if subset.empty:
        return pd.DataFrame()
    matrix = subset.pivot_table(
        index="language_variant",
        columns="spatial_variant",
        values="success_rate",
        aggfunc="mean",
    )
    return matrix.sort_index(axis=0).sort_index(axis=1)


def preferred_generalization_matrix(frame: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    for preferred in ["finetuned", "zero_shot"]:
        matrix = generalization_matrix(frame, model_variant=preferred)
        if not matrix.empty:
            return preferred, matrix
    return None, pd.DataFrame()


def visual_robustness_table(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["experiment_family"] == "visual_robustness"].dropna(subset=["success_rate"])
    if subset.empty:
        return pd.DataFrame(columns=["model_variant", "visual_variant", "success_rate", "episode_time_s"])
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
    subset = frame[frame["experiment_family"] == "chunking"].dropna(subset=["success_rate"])
    if subset.empty:
        return pd.DataFrame(columns=["model_variant", "chunk_size", "success_rate", "rollout_hz", "latency_ms", "trajectory_jerk"])
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
    subset = frame[frame["experiment_family"] == "latency"].dropna(subset=["success_rate"])
    if subset.empty:
        return pd.DataFrame(columns=["model_variant", "quantization", "success_rate", "rollout_hz", "latency_ms"])
    table = (
        subset.groupby(["model_variant", "quantization"], as_index=False)
        .agg(
            success_rate=("success_rate", "mean"),
            rollout_hz=("rollout_hz", "mean"),
            latency_ms=("latency_ms", "mean"),
        )
        .sort_values(["model_variant", "latency_ms"], na_position="last")
    )
    return table


def build_overview_rows(frame: pd.DataFrame, realtime_threshold_hz: float = 10.0) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    run_count = len(frame)
    sources = ", ".join(sorted({str(value) for value in frame["source"].dropna().unique()})) or "n/a"
    model_variants = ", ".join(_available_model_variants(frame)) or "n/a"
    rows.append(("Promoted runs", str(run_count)))
    rows.append(("Sources", sources))
    rows.append(("Model variants", model_variants))

    nominal = frame[
        (frame["experiment_family"] == "generalization")
        & (frame["language_variant"] == "exact")
        & (frame["spatial_variant"] == "nominal")
    ].dropna(subset=["success_rate"])
    if not nominal.empty:
        best_nominal = nominal.sort_values("success_rate", ascending=False).iloc[0]
        rows.append(
            (
                "Best nominal success",
                f"{best_nominal['model_variant']} {best_nominal['success_rate']:.0%}",
            )
        )

    paraphrase = frame[
        (frame["experiment_family"] == "generalization")
        & (frame["language_variant"] == "paraphrase")
    ].dropna(subset=["success_rate"])
    if not paraphrase.empty:
        best_para = paraphrase.sort_values("success_rate", ascending=False).iloc[0]
        rows.append(
            (
                "Best paraphrase success",
                f"{best_para['model_variant']} {best_para['success_rate']:.0%}",
            )
        )

    latency = latency_table(frame)
    realtime = latency[latency["rollout_hz"].fillna(-1) >= realtime_threshold_hz]
    if not realtime.empty:
        best_realtime = realtime.sort_values(["success_rate", "latency_ms"], ascending=[False, True]).iloc[0]
        rows.append(
            (
                "Best realtime setting",
                f"{best_realtime['quantization']} at {best_realtime['rollout_hz']:.1f} Hz",
            )
        )

    return rows


def build_markdown_summary(frame: pd.DataFrame, realtime_threshold_hz: float = 10.0, title: str = "Benchmark Results") -> str:
    overview_rows = build_overview_rows(frame, realtime_threshold_hz=realtime_threshold_hz)
    visual = visual_robustness_table(frame)
    chunking = chunking_table(frame)
    latency = latency_table(frame)

    lines = [f"# {title}", "", "## Overview", ""]
    for label, value in overview_rows:
        lines.append(f"- {label}: {value}")

    lines.extend(["", "## Findings", ""])

    nominal = frame[
        (frame["experiment_family"] == "generalization")
        & (frame["language_variant"] == "exact")
        & (frame["spatial_variant"] == "nominal")
    ].dropna(subset=["success_rate"])
    if len(nominal["model_variant"].drop_duplicates()) >= 2:
        nominal_map = nominal.groupby("model_variant", as_index=True)["success_rate"].mean()
        if "zero_shot" in nominal_map.index and "finetuned" in nominal_map.index:
            lines.append(
                f"- Nominal generalization improves from {nominal_map['zero_shot']:.0%} to {nominal_map['finetuned']:.0%}."
            )

    if not visual.empty and len(visual["visual_variant"].drop_duplicates()) > 1:
        lines.append(
            f"- Visual robustness coverage: {visual['visual_variant'].nunique()} visual conditions across {visual['model_variant'].nunique()} model variants."
        )

    if not chunking.empty:
        best_chunk = chunking.dropna(subset=["success_rate"]).sort_values(
            ["success_rate", "trajectory_jerk"], ascending=[False, True]
        ).iloc[0]
        lines.append(
            f"- Best chunking row in the current registry: chunk {int(best_chunk['chunk_size'])} with {best_chunk['success_rate']:.0%} success."
        )

    if not latency.empty:
        realtime = latency[latency["rollout_hz"].fillna(-1) >= realtime_threshold_hz]
        if not realtime.empty:
            best_latency = realtime.sort_values(["success_rate", "latency_ms"], ascending=[False, True]).iloc[0]
            lines.append(
                f"- Best quantized realtime row: {best_latency['quantization']} at {best_latency['rollout_hz']:.1f} Hz and {best_latency['success_rate']:.0%} success."
            )

    if lines[-1] == "":
        lines.append("- No promoted findings available yet.")

    return "\n".join(lines).rstrip() + "\n"


def summarize_results(
    csv_path: str | Path,
    realtime_threshold_hz: float = 10.0,
    title: str = "Benchmark Results",
) -> SummaryArtifacts:
    frame = load_results(csv_path)
    _, matrix = preferred_generalization_matrix(frame)
    return SummaryArtifacts(
        frame=frame,
        generalization_matrix=matrix,
        visual_robustness=visual_robustness_table(frame),
        chunking=chunking_table(frame),
        latency=latency_table(frame),
        overview_rows=build_overview_rows(frame, realtime_threshold_hz=realtime_threshold_hz),
        markdown=build_markdown_summary(frame, realtime_threshold_hz=realtime_threshold_hz, title=title),
    )
