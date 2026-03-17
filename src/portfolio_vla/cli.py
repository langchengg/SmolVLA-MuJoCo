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
from portfolio_vla.real_results import PromoteRealArgs, promote_real_results
from portfolio_vla.runner import generate_showcase_artifacts


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portfolio VLA utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    showcase = subparsers.add_parser("showcase", help="Generate MuJoCo desktop sorting demo artifacts.")
    showcase.add_argument("--config", default=str(REPO_ROOT / "configs" / "desktop_sorting_showcase.yaml"))
    showcase.add_argument("--prompts", default=str(REPO_ROOT / "templates" / "desktop_sorting_prompts.yaml"))
    showcase.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "showcase"))
    showcase.add_argument("--seed", type=int, default=7)

    benchmark = subparsers.add_parser("benchmark", help="Generate mock benchmark CSV and figures.")
    benchmark.add_argument("--config", default=str(REPO_ROOT / "configs" / "benchmark.yaml"))
    benchmark.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "mock_benchmark"))

    promote = subparsers.add_parser("promote-real", help="Promote one LeRobot eval run into the real benchmark registry.")
    promote.add_argument("--eval-dir", required=True)
    promote.add_argument("--run-name", required=True)
    promote.add_argument("--experiment-family", required=True, choices=["generalization", "visual_robustness", "chunking", "latency"])
    promote.add_argument("--model-variant", required=True, choices=["zero_shot", "finetuned"])
    promote.add_argument("--task-suite", default="libero_10")
    promote.add_argument("--language-variant", default="exact")
    promote.add_argument("--spatial-variant", default="nominal")
    promote.add_argument("--visual-variant", default="nominal")
    promote.add_argument("--chunk-size", type=int, default=8)
    promote.add_argument("--quantization", default="fp16")
    promote.add_argument("--policy-label")
    promote.add_argument("--checkpoint-step", type=int)
    promote.add_argument("--train-dir")
    promote.add_argument("--policy-path")
    promote.add_argument("--latency-ms", type=float)
    promote.add_argument("--rollout-hz", type=float)
    promote.add_argument("--trajectory-jerk", type=float)
    promote.add_argument("--registry-path", default=str(REPO_ROOT / "results" / "real" / "benchmark_registry.csv"))
    promote.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "real"))
    promote.add_argument("--docs-path", default=str(REPO_ROOT / "docs" / "results" / "latest_real_results.md"))
    promote.add_argument("--examples-path", default=str(REPO_ROOT / "examples" / "real_benchmark_results.csv"))
    promote.add_argument("--reports-dir", default=str(REPO_ROOT / "reports" / "real"))

    readme = subparsers.add_parser("export-readme", help="Export lightweight README assets.")
    readme.add_argument("--showcase-dir", default=str(REPO_ROOT / "artifacts" / "showcase"))
    readme.add_argument("--real-dir", default=str(REPO_ROOT / "results" / "real"))
    readme.add_argument("--output", default=str(REPO_ROOT / "assets" / "readme"))
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

    summary = summarize_results(csv_path, title="Mock Benchmark Results")
    (summary_dir / "summary.md").write_text(summary.markdown, encoding="utf-8")
    save_generalization_heatmap(summary.generalization_matrix, figures_dir / "generalization_matrix.png")
    save_visual_robustness_chart(summary.visual_robustness, figures_dir / "visual_robustness.png")
    save_chunking_chart(summary.chunking, figures_dir / "chunking_tradeoff.png")
    save_latency_chart(summary.latency, figures_dir / "latency_tradeoff.png")
    print(f"Mock benchmark artifacts written to {output_dir}")


def run_promote_real(args: argparse.Namespace) -> None:
    result = promote_real_results(
        PromoteRealArgs(
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
            registry_path=args.registry_path,
            results_dir=args.results_dir,
            docs_path=args.docs_path,
            examples_path=args.examples_path,
            reports_dir=args.reports_dir,
        )
    )
    print(f"Updated real benchmark registry: {args.registry_path}")
    print(f"Updated results summary: {result['summary_md']}")


def run_readme_export(args: argparse.Namespace) -> None:
    copied = export_readme_assets(args.showcase_dir, args.real_dir, args.output)
    print(f"Copied {len(copied)} README assets to {args.output}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "showcase":
        run_showcase(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "promote-real":
        run_promote_real(args)
    elif args.command == "export-readme":
        run_readme_export(args)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


def showcase_entrypoint() -> None:
    raise SystemExit(main(["showcase", *sys.argv[1:]]))


def benchmark_entrypoint() -> None:
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))


def promote_real_entrypoint() -> None:
    raise SystemExit(main(["promote-real", *sys.argv[1:]]))


def readme_assets_entrypoint() -> None:
    raise SystemExit(main(["export-readme", *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
