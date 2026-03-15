#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
CACHE_DIR = REPO_ROOT / ".cache" / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from portfolio_vla.analysis import summarize_results
from portfolio_vla.plotting import (
    save_chunking_chart,
    save_generalization_heatmap,
    save_latency_chart,
    save_visual_robustness_chart,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze VLA benchmark results and export figures.")
    parser.add_argument("--input", required=True, help="Path to a CSV file of aggregated experiment results.")
    parser.add_argument("--output", required=True, help="Directory for plots and markdown summary.")
    parser.add_argument(
        "--realtime-threshold-hz",
        type=float,
        default=10.0,
        help="Rollout frequency threshold used for realtime recommendations.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_results(args.input, realtime_threshold_hz=args.realtime_threshold_hz)

    (output_dir / "summary.md").write_text(summary.markdown + "\n", encoding="utf-8")
    save_generalization_heatmap(summary.generalization_matrix, output_dir / "generalization_matrix.png")
    save_visual_robustness_chart(summary.visual_robustness, output_dir / "visual_robustness.png")
    save_chunking_chart(summary.chunking, output_dir / "chunking_tradeoff.png")
    save_latency_chart(
        summary.latency,
        output_dir / "latency_tradeoff.png",
        realtime_threshold_hz=args.realtime_threshold_hz,
    )

    print(f"Wrote analysis artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
