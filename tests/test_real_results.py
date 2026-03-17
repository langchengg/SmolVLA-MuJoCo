from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from portfolio_vla.benchmark_registry import (
    BENCHMARK_COLUMNS,
    empty_registry_frame,
    load_benchmark_registry,
    promote_eval_record,
    save_benchmark_registry,
    upsert_registry_rows,
)
from portfolio_vla.real_results import PromoteRealArgs, promote_real_results, render_real_results_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_eval_info(eval_dir: Path, pc_success: float, eval_ep_s: float, successes: list[int]) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregated": {
            "pc_success": pc_success,
            "eval_s": 12.0,
            "eval_ep_s": eval_ep_s,
        },
        "per_episode": {
            "success": successes,
            "sum_reward": [1.0 for _ in successes],
        },
        "video_paths": [str(eval_dir / "episode_0.mp4")],
    }
    (eval_dir / "eval_info.json").write_text(json.dumps(payload), encoding="utf-8")


class RealResultsTests(unittest.TestCase):
    def test_promote_eval_record_parses_eval_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = Path(tmpdir) / "eval_run"
            _write_eval_info(eval_dir, pc_success=0.75, eval_ep_s=11.2, successes=[1, 1, 0, 1])

            promoted = promote_eval_record(
                eval_dir=eval_dir,
                run_name="ft_nominal",
                experiment_family="generalization",
                model_variant="finetuned",
                task_suite="libero_10",
                language_variant="exact",
                spatial_variant="nominal",
                visual_variant="nominal",
                chunk_size=8,
                quantization="fp16",
                latency_ms=96.0,
                rollout_hz=10.4,
                trajectory_jerk=0.22,
            )

        row = promoted.row
        self.assertEqual(row["source"], "real")
        self.assertEqual(row["n_episodes"], 4)
        self.assertAlmostEqual(float(row["success_rate"]), 0.75)
        self.assertAlmostEqual(float(row["episode_time_s"]), 11.2)

    def test_registry_upsert_deduplicates_same_run_key(self) -> None:
        base = empty_registry_frame()
        row_a = pd.DataFrame(
            [
                {
                    "source": "real",
                    "run_name": "ft_nominal",
                    "policy_label": "ft",
                    "checkpoint_step": 1000,
                    "train_dir": "",
                    "eval_dir": "/tmp/run_a",
                    "policy_path": "",
                    "experiment_family": "generalization",
                    "model_variant": "finetuned",
                    "task_suite": "libero_10",
                    "language_variant": "exact",
                    "spatial_variant": "nominal",
                    "visual_variant": "nominal",
                    "chunk_size": 8,
                    "quantization": "fp16",
                    "n_episodes": 4,
                    "success_rate": 0.75,
                    "latency_ms": 96.0,
                    "rollout_hz": 10.4,
                    "trajectory_jerk": 0.22,
                    "episode_time_s": 11.2,
                }
            ],
            columns=BENCHMARK_COLUMNS,
        )
        row_b = row_a.copy()
        row_b.loc[0, "success_rate"] = 0.80

        registry = upsert_registry_rows(base, row_a)
        registry = upsert_registry_rows(registry, row_b)

        self.assertEqual(len(registry), 1)
        self.assertAlmostEqual(float(registry.iloc[0]["success_rate"]), 0.80)

    def test_promote_real_results_renders_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eval_dir = root / "eval"
            _write_eval_info(eval_dir, pc_success=0.75, eval_ep_s=11.2, successes=[1, 1, 0, 1])
            registry_path = root / "results" / "real" / "benchmark_registry.csv"
            results_dir = root / "results" / "real"
            docs_path = root / "docs" / "results" / "latest_real_results.md"
            examples_path = root / "examples" / "real_benchmark_results.csv"
            reports_dir = root / "reports" / "real"

            promote_real_results(
                PromoteRealArgs(
                    eval_dir=eval_dir,
                    run_name="ft_nominal",
                    experiment_family="generalization",
                    model_variant="finetuned",
                    task_suite="libero_10",
                    language_variant="exact",
                    spatial_variant="nominal",
                    visual_variant="nominal",
                    chunk_size=8,
                    quantization="fp16",
                    latency_ms=96.0,
                    rollout_hz=10.4,
                    trajectory_jerk=0.22,
                    registry_path=registry_path,
                    results_dir=results_dir,
                    docs_path=docs_path,
                    examples_path=examples_path,
                    reports_dir=reports_dir,
                )
            )

            registry = load_benchmark_registry(registry_path)
            self.assertEqual(len(registry), 1)
            self.assertTrue((results_dir / "figures" / "real_overview.png").exists())
            self.assertTrue((results_dir / "summary" / "summary.md").exists())
            self.assertTrue(docs_path.exists())
            self.assertTrue(examples_path.exists())
            self.assertTrue((reports_dir / "real_overview.png").exists())

    def test_render_real_results_bundle_accepts_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_path = root / "benchmark_registry.csv"
            frame = pd.DataFrame(
                [
                    {
                        "source": "real",
                        "run_name": "ft_nominal",
                        "policy_label": "smolvla-ft",
                        "checkpoint_step": 2000,
                        "train_dir": str(root / "train"),
                        "eval_dir": str(root / "eval"),
                        "policy_path": str(root / "policy"),
                        "experiment_family": "generalization",
                        "model_variant": "finetuned",
                        "task_suite": "libero_10",
                        "language_variant": "exact",
                        "spatial_variant": "nominal",
                        "visual_variant": "nominal",
                        "chunk_size": 8,
                        "quantization": "fp16",
                        "n_episodes": 4,
                        "success_rate": 0.75,
                        "latency_ms": 96.0,
                        "rollout_hz": 10.4,
                        "trajectory_jerk": 0.22,
                        "episode_time_s": 11.2,
                    }
                ],
                columns=BENCHMARK_COLUMNS,
            )
            save_benchmark_registry(frame, registry_path)

            render_real_results_bundle(
                registry_csv=registry_path,
                results_dir=root / "results" / "real",
                docs_path=root / "docs" / "results" / "latest_real_results.md",
                examples_path=root / "examples" / "real_benchmark_results.csv",
                reports_dir=root / "reports" / "real",
            )

            self.assertTrue((root / "results" / "real" / "figures" / "real_overview.png").exists())
            self.assertTrue((root / "docs" / "results" / "latest_real_results.md").exists())

    def test_shell_scripts_use_repo_owned_results_paths(self) -> None:
        train_text = (REPO_ROOT / "scripts" / "train_libero_smolvla.sh").read_text(encoding="utf-8")
        eval_text = (REPO_ROOT / "scripts" / "eval_libero_smolvla.sh").read_text(encoding="utf-8")
        self.assertIn('results/raw/train/$RUN_NAME', train_text)
        self.assertIn('results/raw/eval/$RUN_NAME', eval_text)
        self.assertNotIn('./outputs/libero_smolvla', train_text)
        self.assertNotIn('./eval_logs/libero_eval', eval_text)


if __name__ == "__main__":
    unittest.main()
