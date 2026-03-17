from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_success_by_task_family(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    aggregated = (
        summary.groupby(["task_family", "model_variant"], as_index=False)["success_rate"]
        .mean()
    )
    pivot = aggregated.pivot(index="task_family", columns="model_variant", values="success_rate")
    ax = pivot.plot(kind="bar", figsize=(8.5, 4.8), color=["#6b7280", "#0f766e"])
    ax.set_title("Desktop Sorting Success by Task Family")
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Task family")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_success_by_language_variant(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    aggregated = (
        summary.groupby(["language_variant", "model_variant"], as_index=False)["success_rate"]
        .mean()
    )
    pivot = aggregated.pivot(index="language_variant", columns="model_variant", values="success_rate")
    ax = pivot.plot(kind="bar", figsize=(7.2, 4.6), color=["#94a3b8", "#2563eb"])
    ax.set_title("Success by Language Variant")
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Language variant")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_latency_vs_success(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"zero_shot": "#64748b", "finetuned": "#dc2626"}
    for _, row in summary.iterrows():
        ax.scatter(
            row["mean_policy_latency_ms"],
            row["success_rate"],
            color=colors[row["model_variant"]],
            s=90,
        )
        ax.annotate(
            f"{row['model_variant']}:{row['task_family']}",
            (row["mean_policy_latency_ms"], row["success_rate"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    ax.set_title("Latency vs Success")
    ax.set_xlabel("Mean policy latency (ms)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_failure_breakdown(frame: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    failures = frame[frame["failure_tag"] != ""].copy()
    if failures.empty:
        failures = pd.DataFrame(
            {
                "model_variant": ["zero_shot", "finetuned"],
                "failure_tag": ["none", "none"],
                "count": [0, 0],
            }
        )
    else:
        failures = (
            failures.groupby(["model_variant", "failure_tag"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

    pivot = failures.pivot(index="failure_tag", columns="model_variant", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(8.4, 4.8), color=["#64748b", "#b91c1c"])
    ax.set_title("Failure Tag Breakdown")
    ax.set_ylabel("Episode count")
    ax.set_xlabel("Failure tag")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
