from __future__ import annotations

import argparse
import sys
from pathlib import Path

from portfolio_vla.analysis import summarize_results
from portfolio_vla.benchmark_mock import generate_mock_benchmark_frame, load_benchmark_config
from portfolio_vla.plotting import (
    save_chunking_chart,
    save_generalization_heatmap,
    save_latency_chart,
    save_visual_robustness_chart,
)
from portfolio_vla.readme_assets import export_readme_assets
from portfolio_vla.runner import generate_showcase_artifacts


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portfolio VLA demo utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    showcase = subparsers.add_parser("showcase", help="Generate MuJoCo desktop sorting demo artifacts.")
    showcase.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "desktop_sorting_showcase.yaml"),
        help="Path to the showcase config YAML.",
    )
    showcase.add_argument(
        "--prompts",
        default=str(REPO_ROOT / "templates" / "desktop_sorting_prompts.yaml"),
        help="Path to the prompt template YAML.",
    )
    showcase.add_argument(
        "--output",
        default=str(REPO_ROOT / "artifacts" / "showcase"),
        help="Directory for showcase artifacts.",
    )
    showcase.add_argument("--seed", type=int, default=7, help="Random seed for deterministic episode layouts.")

    benchmark = subparsers.add_parser("benchmark", help="Generate mock benchmark CSV and figures.")
    benchmark.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "benchmark.yaml"),
        help="Path to benchmark config YAML.",
    )
    benchmark.add_argument(
        "--output",
        default=str(REPO_ROOT / "artifacts" / "benchmark"),
        help="Directory for benchmark artifacts.",
    )

    readme = subparsers.add_parser("export-readme", help="Export lightweight README assets.")
    readme.add_argument(
        "--showcase-dir",
        default=str(REPO_ROOT / "artifacts" / "showcase"),
        help="Showcase artifacts directory.",
    )
    readme.add_argument(
        "--benchmark-dir",
        default=str(REPO_ROOT / "artifacts" / "benchmark"),
        help="Benchmark artifacts directory.",
    )
    readme.add_argument(
        "--output",
        default=str(REPO_ROOT / "assets" / "readme"),
        help="Directory for GitHub README assets.",
    )
    return parser


def run_showcase(args: argparse.Namespace) -> None:
    result = generate_showcase_artifacts(args.config, args.prompts, args.output, seed=args.seed)
    print(f"Showcase artifacts written to {args.output}")
    print(f"Hero video: {result['hero_video']}")


def run_benchmark(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    summary_dir = output_dir / "summary"
    figures_dir = output_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = load_benchmark_config(args.config)
    frame = generate_mock_benchmark_frame(config)
    csv_path = summary_dir / "mock_benchmark_results.csv"
    frame.to_csv(csv_path, index=False)

    summary = summarize_results(csv_path)
    (summary_dir / "summary.md").write_text(summary.markdown + "\n", encoding="utf-8")
    save_generalization_heatmap(summary.generalization_matrix, figures_dir / "generalization_matrix.png")
    save_visual_robustness_chart(summary.visual_robustness, figures_dir / "visual_robustness.png")
    save_chunking_chart(summary.chunking, figures_dir / "chunking_tradeoff.png")
    save_latency_chart(summary.latency, figures_dir / "latency_tradeoff.png")
    print(f"Benchmark artifacts written to {output_dir}")


def run_readme_export(args: argparse.Namespace) -> None:
    copied = export_readme_assets(args.showcase_dir, args.benchmark_dir, args.output)
    print(f"Copied {len(copied)} README assets to {args.output}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "showcase":
        run_showcase(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "export-readme":
        run_readme_export(args)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


def showcase_entrypoint() -> None:
    raise SystemExit(main(["showcase", *sys.argv[1:]]))


def benchmark_entrypoint() -> None:
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))


def readme_assets_entrypoint() -> None:
    raise SystemExit(main(["export-readme", *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
