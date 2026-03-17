from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _save_empty_figure(title: str, output_path: str | Path, message: str) -> None:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.60, title, ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def save_overview_card(overview_rows: list[tuple[str, str]], output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    image = Image.new("RGB", (920, 360), color=(245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, 896, 336), radius=22, fill=(255, 255, 255), outline=(220, 226, 233))
    draw.text((52, 48), title, fill=(22, 29, 37), font=_font(28))

    y = 108
    for label, value in overview_rows[:6]:
        draw.text((56, y), label, fill=(92, 103, 115), font=_font(18))
        draw.text((340, y), value, fill=(20, 27, 34), font=_font(20))
        y += 38

    image.save(output_path)


def save_generalization_heatmap(matrix: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    if matrix.empty:
        _save_empty_figure("Generalization Matrix", output_path, "No promoted generalization rows available.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(matrix.values, cmap="YlGn", vmin=0.0, vmax=1.0)
    ax.set_title("Generalization Matrix")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for row_index, row_name in enumerate(matrix.index):
        for col_index, col_name in enumerate(matrix.columns):
            value = matrix.loc[row_name, col_name]
            if pd.isna(value):
                continue
            ax.text(col_index, row_index, f"{value:.0%}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Success rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_visual_robustness_chart(table: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    if table.empty:
        _save_empty_figure("Visual Robustness", output_path, "No promoted visual robustness rows available.")
        return

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
    if table.empty:
        _save_empty_figure("Chunking Tradeoff", output_path, "No promoted chunking rows available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = [("zero_shot", "#6b7280"), ("finetuned", "#16a34a")]
    seen = False
    for model_variant, color in colors:
        subset = table[table["model_variant"] == model_variant]
        if subset.empty:
            continue
        seen = True
        axes[0].plot(subset["chunk_size"], subset["success_rate"], marker="o", label=model_variant, color=color)
        axes[1].plot(
            subset["chunk_size"],
            subset["trajectory_jerk"],
            marker="o",
            label=model_variant,
            color=color,
        )

    if not seen:
        plt.close(fig)
        _save_empty_figure("Chunking Tradeoff", output_path, "No promoted chunking rows available.")
        return

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
    subset = table[table["model_variant"] == "finetuned"] if not table.empty else table
    if subset.empty:
        _save_empty_figure("Latency vs Success", output_path, "No promoted latency rows available.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(subset["latency_ms"], subset["success_rate"], s=90, color="#dc2626")

    for _, row in subset.iterrows():
        hz_label = "n/a" if pd.isna(row["rollout_hz"]) else f"{row['rollout_hz']:.1f}Hz"
        ax.annotate(
            f"{row['quantization']} ({hz_label})",
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
