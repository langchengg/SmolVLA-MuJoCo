from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from portfolio_vla.benchmark_mock import BENCHMARK_COLUMNS, generate_mock_benchmark_frame, load_benchmark_config
from portfolio_vla.scene import DesktopSortingScene, build_layout
from portfolio_vla.showcase_analysis import summarize_showcase
from portfolio_vla.tasks import load_prompt_specs, parse_catalog_object, pick_prompt_subset, resolve_prompt_targets


REPO_ROOT = Path(__file__).resolve().parents[1]


class ShowcasePipelineTests(unittest.TestCase):
    def test_prompt_resolution(self) -> None:
        prompts = pick_prompt_subset(load_prompt_specs(REPO_ROOT / "templates" / "desktop_sorting_prompts.yaml"))
        prompt_map = {prompt.id: prompt for prompt in prompts}
        objects = [
            parse_catalog_object("red_cube"),
            parse_catalog_object("blue_cube"),
            parse_catalog_object("green_cylinder"),
            parse_catalog_object("yellow_block"),
            parse_catalog_object("orange_cylinder"),
        ]

        exact_targets = resolve_prompt_targets(prompt_map["exact_pick_red_cube_left_tray"], objects)
        self.assertEqual(exact_targets, {"red_cube": "left_tray"})

        compositional_targets = resolve_prompt_targets(prompt_map["comp_warm_left_cool_right"], objects)
        self.assertEqual(compositional_targets["red_cube"], "left_tray")
        self.assertEqual(compositional_targets["orange_cylinder"], "left_tray")
        self.assertEqual(compositional_targets["blue_cube"], "right_tray")
        self.assertEqual(compositional_targets["green_cylinder"], "right_tray")

    def test_scene_renders_rgb_frame(self) -> None:
        objects = [
            parse_catalog_object("red_cube"),
            parse_catalog_object("blue_cube"),
            parse_catalog_object("green_cylinder"),
            parse_catalog_object("yellow_block"),
        ]
        layout = build_layout(
            object_specs=objects,
            n_objects=4,
            seed=7,
            visual_variant="nominal",
            camera_yaw_deg=0,
        )
        scene = DesktopSortingScene(layout)
        frame = scene.render()

        self.assertEqual(frame.shape, (368, 640, 3))
        self.assertEqual(frame.dtype.name, "uint8")

    def test_showcase_summary_contains_expected_metrics(self) -> None:
        rows = [
            {
                "run_id": "showcase_zero_shot_s7",
                "timestamp": "2026-03-17T13:00:00+00:00",
                "model_variant": "zero_shot",
                "checkpoint": "mock/zero_shot",
                "scene_id": "scene_a",
                "task_family": "single_object_pick_place",
                "instruction_id": "exact_pick_red_cube_left_tray",
                "instruction_text": "Pick up the red cube and place it in the left tray.",
                "language_variant": "exact",
                "visual_variant": "nominal",
                "object_layout_seed": 7,
                "camera_variant": "front_yaw_0deg",
                "n_objects": 4,
                "task_success": True,
                "object_sort_accuracy": 1.0,
                "instruction_grounding_accuracy": 1.0,
                "completion_time_s": 10.8,
                "mean_policy_latency_ms": 68.0,
                "rollout_hz": 14.7,
                "collision_count": 0,
                "regrasp_count": 2,
                "trajectory_jerk": 0.34,
                "failure_tag": "",
                "video_path": "artifacts/showcase/videos/zero_shot_exact_pick_red_cube_left_tray.mp4",
                "notes": "Success under nominal conditions.",
            },
            {
                "run_id": "showcase_finetuned_s7",
                "timestamp": "2026-03-17T13:00:00+00:00",
                "model_variant": "finetuned",
                "checkpoint": "mock/finetuned",
                "scene_id": "scene_b",
                "task_family": "paraphrase_generalization",
                "instruction_id": "para_crimson_block_left_tray",
                "instruction_text": "Grasp the crimson block and drop it into the left tray.",
                "language_variant": "paraphrase",
                "visual_variant": "low_light",
                "object_layout_seed": 8,
                "camera_variant": "front_yaw_0deg",
                "n_objects": 5,
                "task_success": True,
                "object_sort_accuracy": 1.0,
                "instruction_grounding_accuracy": 0.93,
                "completion_time_s": 10.8,
                "mean_policy_latency_ms": 75.0,
                "rollout_hz": 13.3,
                "collision_count": 1,
                "regrasp_count": 1,
                "trajectory_jerk": 0.26,
                "failure_tag": "",
                "video_path": "artifacts/showcase/videos/finetuned_para_crimson_block_left_tray.mp4",
                "notes": "Success under perturbed conditions.",
            },
        ]
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episodes.jsonl"
            frame.to_json(path, orient="records", lines=True)
            summary = summarize_showcase(path)

        self.assertIn("success_rate", summary.summary_frame.columns)
        self.assertIn("Desktop Sorting Summary", summary.markdown)
        self.assertEqual(len(summary.summary_frame), 2)

    def test_benchmark_mock_schema(self) -> None:
        config = load_benchmark_config(REPO_ROOT / "configs" / "benchmark.yaml")
        frame = generate_mock_benchmark_frame(config)

        self.assertEqual(frame.columns.tolist(), BENCHMARK_COLUMNS)
        self.assertGreaterEqual(len(frame), 20)
        self.assertTrue(((frame["success_rate"] >= 0.0) & (frame["success_rate"] <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
