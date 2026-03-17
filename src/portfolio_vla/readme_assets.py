from __future__ import annotations

from pathlib import Path
import shutil


def export_readme_assets(showcase_dir: str | Path, real_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    showcase_dir = Path(showcase_dir)
    real_dir = Path(real_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        real_dir / "figures" / "real_overview.png": output_dir / "real_overview.png",
        real_dir / "figures" / "generalization_matrix.png": output_dir / "real_generalization.png",
        real_dir / "figures" / "chunking_tradeoff.png": output_dir / "real_chunking.png",
        real_dir / "figures" / "latency_tradeoff.png": output_dir / "real_latency.png",
        showcase_dir / "videos" / "hero_showcase.mp4": output_dir / "demo_showcase.mp4",
        showcase_dir / "videos" / "hero_showcase.gif": output_dir / "demo_showcase.gif",
        showcase_dir / "videos" / "hero_thumbnail.png": output_dir / "demo_showcase_thumbnail.png",
        showcase_dir / "figures" / "success_by_task_family.png": output_dir / "demo_success_by_task_family.png",
        showcase_dir / "figures" / "success_by_language_variant.png": output_dir / "demo_success_by_language_variant.png",
        showcase_dir / "figures" / "latency_vs_success.png": output_dir / "demo_latency_vs_success.png",
        showcase_dir / "figures" / "failure_breakdown.png": output_dir / "demo_failure_breakdown.png",
    }

    copied: dict[str, Path] = {}
    for source, target in mapping.items():
        if not source.exists():
            continue
        shutil.copy2(source, target)
        copied[str(source)] = target
    return copied
