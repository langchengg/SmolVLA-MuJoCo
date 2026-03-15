from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_generalization_heatmap(matrix: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(matrix.values, cmap="YlGn", vmin=0.0, vmax=1.0)
    ax.set_title("Fine-tuned Generalization Matrix")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for row_index, row_name in enumerate(matrix.index):
        for col_index, col_name in enumerate(matrix.columns):
            value = matrix.loc[row_name, col_name]
            ax.text(col_index, row_index, f"{value:.0%}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Success rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_visual_robustness_chart(table: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    pivot = table.pivot(index="visual_variant", columns="model_variant", values="success_rate")
    ax = pivot.plot(kind="bar", figsize=(8, 4.5), color=["#9ca3af", "#2563eb"])
    ax.set_title("Visual Robustness")
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Visual condition")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_chunking_chart(table: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for model_variant, color in [("zero_shot", "#6b7280"), ("finetuned", "#16a34a")]:
        subset = table[table["model_variant"] == model_variant]
        axes[0].plot(subset["chunk_size"], subset["success_rate"], marker="o", label=model_variant, color=color)
        axes[1].plot(
            subset["chunk_size"],
            subset["trajectory_jerk"],
            marker="o",
            label=model_variant,
            color=color,
        )

    axes[0].set_title("Chunk Size vs Success")
    axes[0].set_xlabel("Chunk size")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_title("Chunk Size vs Jerk")
    axes[1].set_xlabel("Chunk size")
    axes[1].set_ylabel("Trajectory jerk")
    axes[0].legend(title="Model")
    axes[1].legend(title="Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_latency_chart(table: pd.DataFrame, output_path: str | Path, realtime_threshold_hz: float = 10.0) -> None:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    subset = table[table["model_variant"] == "finetuned"]
    ax.scatter(subset["latency_ms"], subset["success_rate"], s=90, color="#dc2626")

    for _, row in subset.iterrows():
        ax.annotate(
            f"{row['quantization']} ({row['rollout_hz']:.1f}Hz)",
            (row["latency_ms"], row["success_rate"]),
            textcoords="offset points",
            xytext=(5, 6),
        )

    ax.set_title("Latency vs Success")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.text(
        0.98,
        0.05,
        f"Realtime threshold: {realtime_threshold_hz:.0f}Hz",
        transform=ax.transAxes,
        ha="right",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
