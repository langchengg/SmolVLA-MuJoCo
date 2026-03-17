from __future__ import annotations

from pathlib import Path
import shutil


def export_readme_assets(showcase_dir: str | Path, benchmark_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    showcase_dir = Path(showcase_dir)
    benchmark_dir = Path(benchmark_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        showcase_dir / "videos" / "hero_showcase.mp4": output_dir / "hero_showcase.mp4",
        showcase_dir / "videos" / "hero_showcase.gif": output_dir / "hero_showcase.gif",
        showcase_dir / "videos" / "hero_thumbnail.png": output_dir / "hero_thumbnail.png",
        showcase_dir / "figures" / "success_by_task_family.png": output_dir / "showcase_success_by_task_family.png",
        showcase_dir / "figures" / "success_by_language_variant.png": output_dir / "showcase_success_by_language_variant.png",
        showcase_dir / "figures" / "latency_vs_success.png": output_dir / "showcase_latency_vs_success.png",
        showcase_dir / "figures" / "failure_breakdown.png": output_dir / "showcase_failure_breakdown.png",
        benchmark_dir / "figures" / "generalization_matrix.png": output_dir / "benchmark_generalization_matrix.png",
        benchmark_dir / "figures" / "visual_robustness.png": output_dir / "benchmark_visual_robustness.png",
        benchmark_dir / "figures" / "chunking_tradeoff.png": output_dir / "benchmark_chunking_tradeoff.png",
        benchmark_dir / "figures" / "latency_tradeoff.png": output_dir / "benchmark_latency_tradeoff.png",
    }

    copied: dict[str, Path] = {}
    for source, target in mapping.items():
        shutil.copy2(source, target)
        copied[str(source)] = target
    return copied
