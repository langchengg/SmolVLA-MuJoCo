from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from portfolio_vla.analysis import summarize_results
from portfolio_vla.benchmark_registry import (
    BENCHMARK_COLUMNS,
    load_benchmark_registry,
    normalize_registry_frame,
    promote_eval_record,
    save_benchmark_registry,
    upsert_registry_rows,
)
from portfolio_vla.plotting import (
    save_chunking_chart,
    save_generalization_heatmap,
    save_latency_chart,
    save_overview_card,
    save_visual_robustness_chart,
)


@dataclass(frozen=True)
class PromoteRealArgs:
    eval_dir: str | Path
    run_name: str
    experiment_family: str
    model_variant: str
    task_suite: str
    language_variant: str
    spatial_variant: str
    visual_variant: str
    chunk_size: int
    quantization: str
    registry_path: str | Path
    results_dir: str | Path
    docs_path: str | Path
    examples_path: str | Path
    reports_dir: str | Path
    policy_label: str | None = None
    checkpoint_step: int | None = None
    train_dir: str | Path | None = None
    policy_path: str | Path | None = None
    latency_ms: float | None = None
    rollout_hz: float | None = None
    trajectory_jerk: float | None = None


def render_real_results_bundle(
    registry_csv: str | Path,
    results_dir: str | Path,
    docs_path: str | Path,
    examples_path: str | Path,
    reports_dir: str | Path,
) -> dict[str, Path]:
    registry_csv = Path(registry_csv)
    results_dir = Path(results_dir)
    docs_path = Path(docs_path)
    examples_path = Path(examples_path)
    reports_dir = Path(reports_dir)

    summary_dir = results_dir / "summary"
    figures_dir = results_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_results(registry_csv, title="Latest Real Benchmark Results")

    summary_md_path = summary_dir / "summary.md"
    summary_md_path.write_text(summary.markdown, encoding="utf-8")

    save_overview_card(summary.overview_rows, figures_dir / "real_overview.png", title="Real Benchmark Overview")
    save_generalization_heatmap(summary.generalization_matrix, figures_dir / "generalization_matrix.png")
    save_visual_robustness_chart(summary.visual_robustness, figures_dir / "visual_robustness.png")
    save_chunking_chart(summary.chunking, figures_dir / "chunking_tradeoff.png")
    save_latency_chart(summary.latency, figures_dir / "latency_tradeoff.png")

    docs_path.write_text(build_latest_real_results_doc(summary.frame, summary.markdown), encoding="utf-8")
    shutil.copy2(registry_csv, examples_path)
    for source_name, target_name in [
        ("generalization_matrix.png", "real_generalization_matrix.png"),
        ("visual_robustness.png", "real_visual_robustness.png"),
        ("chunking_tradeoff.png", "real_chunking_tradeoff.png"),
        ("latency_tradeoff.png", "real_latency_tradeoff.png"),
        ("real_overview.png", "real_overview.png"),
    ]:
        shutil.copy2(figures_dir / source_name, reports_dir / target_name)

    return {
        "summary_md": summary_md_path,
        "real_overview": figures_dir / "real_overview.png",
        "generalization": figures_dir / "generalization_matrix.png",
        "visual": figures_dir / "visual_robustness.png",
        "chunking": figures_dir / "chunking_tradeoff.png",
        "latency": figures_dir / "latency_tradeoff.png",
        "docs_path": docs_path,
        "examples_path": examples_path,
    }


def build_latest_real_results_doc(frame: pd.DataFrame, summary_markdown: str) -> str:
    frame = normalize_registry_frame(frame)
    compact = frame[
        [
            "run_name",
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
            "checkpoint_step",
        ]
    ]

    header = "| " + " | ".join(compact.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in compact.itertuples(index=False, name=None)
    ]
    table = "\n".join([header, separator, *rows])
    summary_lines = summary_markdown.strip().splitlines()
    if summary_lines and summary_lines[0].startswith("# "):
        summary_lines = summary_lines[1:]
    summary_block = "\n".join(summary_lines).strip()
    return "\n".join(
        [
            "# Latest Real Results",
            "",
            "This document is generated from `results/real/benchmark_registry.csv`.",
            "Committed rows in this repository are sample promoted benchmark rows.",
            "",
            summary_block,
            "",
            "## Registry Rows",
            "",
            table,
            "",
        ]
    )


def promote_real_results(args: PromoteRealArgs) -> dict[str, Path]:
    registry_path = Path(args.registry_path)
    existing = load_benchmark_registry(registry_path)
    promoted = promote_eval_record(
        eval_dir=args.eval_dir,
        run_name=args.run_name,
        experiment_family=args.experiment_family,
        model_variant=args.model_variant,
        task_suite=args.task_suite,
        language_variant=args.language_variant,
        spatial_variant=args.spatial_variant,
        visual_variant=args.visual_variant,
        chunk_size=args.chunk_size,
        quantization=args.quantization,
        policy_label=args.policy_label,
        checkpoint_step=args.checkpoint_step,
        train_dir=args.train_dir,
        policy_path=args.policy_path,
        latency_ms=args.latency_ms,
        rollout_hz=args.rollout_hz,
        trajectory_jerk=args.trajectory_jerk,
    )
    incoming = pd.DataFrame([promoted.row], columns=BENCHMARK_COLUMNS)
    registry = upsert_registry_rows(existing, incoming)
    save_benchmark_registry(registry, registry_path)
    return render_real_results_bundle(
        registry_csv=registry_path,
        results_dir=args.results_dir,
        docs_path=args.docs_path,
        examples_path=args.examples_path,
        reports_dir=args.reports_dir,
    )
