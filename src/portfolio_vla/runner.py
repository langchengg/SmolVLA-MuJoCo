from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

from portfolio_vla.policy import MockPolicy
from portfolio_vla.scene import HOME_TOOL_POS, DesktopSortingScene, build_layout
from portfolio_vla.showcase_analysis import summarize_showcase
from portfolio_vla.showcase_plotting import (
    save_failure_breakdown,
    save_latency_vs_success,
    save_success_by_language_variant,
    save_success_by_task_family,
)
from portfolio_vla.tasks import SceneObject, load_prompt_specs, parse_catalog_object, pick_prompt_subset, resolve_prompt_targets


@dataclass(frozen=True)
class ShowcaseConfig:
    object_catalog: tuple[str, ...]
    cameras: tuple[str, ...]
    primary_camera: str
    train_object_count: int
    eval_object_count: int


def load_showcase_config(path: str | Path) -> ShowcaseConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ShowcaseConfig(
        object_catalog=tuple(payload["task_family"]["objects"]),
        cameras=tuple(payload["cameras"]["views"]),
        primary_camera=payload["cameras"]["primary"],
        train_object_count=payload["task_family"]["object_count"]["train"],
        eval_object_count=payload["task_family"]["object_count"]["eval"],
    )


def ensure_output_tree(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    directories = {
        "root": root,
        "episodes": root / "episodes",
        "summary": root / "summary",
        "figures": root / "figures",
        "videos": root / "videos",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def overlay_text(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    width, _ = image.size
    draw.rectangle((0, 0, width, 46), fill=(18, 24, 38, 220))
    draw.text((14, 8), title, fill=(255, 255, 255), font=_font(22))
    draw.text((14, 26), subtitle, fill=(205, 219, 245), font=_font(14))
    return np.asarray(image)


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 12) -> None:
    with imageio.get_writer(path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(frame)


def write_gif(path: Path, frames: list[np.ndarray], fps: int = 8) -> None:
    duration_ms = int(1000 / fps)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


def _object_sort_accuracy(selected_targets: dict[str, str], expected_targets: dict[str, str]) -> float:
    if not expected_targets:
        return 0.0
    correct = 0
    for object_id, target in expected_targets.items():
        if selected_targets.get(object_id) == target:
            correct += 1
    return round(correct / len(expected_targets), 2)


def _completion_time(num_targets: int, success: bool, failure_tag: str) -> float:
    base = 8.8 + num_targets * 2.0
    if failure_tag == "timeout":
        return round(base + 5.5, 1)
    if not success:
        return round(base + 2.8, 1)
    return round(base, 1)


def _trajectory_jerk(success: bool, model_variant: str, visual_variant: str) -> float:
    jerk = 0.23 if model_variant == "finetuned" else 0.34
    if not success:
        jerk += 0.08
    if visual_variant != "nominal":
        jerk += 0.03
    return round(jerk, 2)


def _video_path_label(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def render_episode(
    scene: DesktopSortingScene,
    prompt_text: str,
    title_prefix: str,
    selected_targets: dict[str, str],
    expected_targets: dict[str, str],
) -> list[np.ndarray]:
    scene.reset()
    frames: list[np.ndarray] = []
    current = HOME_TOOL_POS.copy()
    placement_counts = {key: 0 for key in scene.layout.tray_slots}
    frames.extend(
        overlay_text(scene.render(), title_prefix, prompt_text)
        for _ in range(12)
    )

    for object_id, container in selected_targets.items():
        obj = next(item for item in scene.layout.objects if item.spec.object_id == object_id)
        object_height = obj.spec.size[2] if obj.spec.shape != "cylinder" else obj.spec.size[2] / 2
        object_pos = np.array([obj.position[0], obj.position[1], 0.04 + object_height], dtype=float)
        hover_object = object_pos + np.array([0.0, 0.0, 0.15], dtype=float)
        pick_pos = object_pos + np.array([0.0, 0.0, 0.04], dtype=float)
        slot_idx = placement_counts.setdefault(container, 0)
        place_pos = scene.container_slot(container, slot_idx)
        placement_counts[container] = slot_idx + 1
        hover_target = place_pos + np.array([0.0, 0.0, 0.18], dtype=float)

        for target, n_frames in [
            (hover_object, 10),
            (pick_pos, 8),
        ]:
            current = target
            frames.extend(
                overlay_text(frame, title_prefix, prompt_text)
                for frame in scene.move_tool_linear(target, n_frames)
            )

        scene.attach(object_id)
        frames.extend(overlay_text(scene.render(), title_prefix, prompt_text) for _ in range(4))

        for target, n_frames in [
            (hover_object, 8),
            (hover_target, 12),
            (place_pos + np.array([0.0, 0.0, 0.06], dtype=float), 8),
        ]:
            current = target
            frames.extend(
                overlay_text(frame, title_prefix, prompt_text)
                for frame in scene.move_tool_linear(target, n_frames)
            )

        final_pos = place_pos.copy()
        if object_id not in expected_targets:
            final_pos = final_pos + np.array([0.04, 0.0, 0.0], dtype=float)
        scene.release(object_id, final_pos)
        frames.extend(overlay_text(scene.render(), title_prefix, prompt_text) for _ in range(4))

        current = hover_target
        frames.extend(
            overlay_text(frame, title_prefix, prompt_text)
            for frame in scene.move_tool_linear(hover_target, 8)
        )

    frames.extend(
        overlay_text(frame, title_prefix, prompt_text)
        for frame in scene.move_tool_linear(HOME_TOOL_POS, 12)
    )
    frames.extend(overlay_text(scene.render(), title_prefix, prompt_text) for _ in range(10))
    return frames


def generate_showcase_artifacts(
    config_path: str | Path,
    prompts_path: str | Path,
    output_dir: str | Path,
    seed: int = 7,
) -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    showcase_config = load_showcase_config(config_path)
    prompt_specs = pick_prompt_subset(load_prompt_specs(prompts_path))
    directories = ensure_output_tree(output_dir)
    catalog = [parse_catalog_object(name) for name in showcase_config.object_catalog]

    episode_rows: list[dict[str, object]] = []
    hero_frames: list[np.ndarray] = []
    first_frame: np.ndarray | None = None

    for model_index, model_variant in enumerate(["zero_shot", "finetuned"]):
        policy = MockPolicy(model_variant=model_variant, seed=seed + model_index * 100)
        for prompt_index, prompt in enumerate(prompt_specs):
            layout_seed = seed + prompt_index * 11 + model_index * 101
            n_objects = (
                showcase_config.eval_object_count
                if prompt.task_family in {"shape_grouping", "compositional_sorting", "paraphrase_generalization"}
                else showcase_config.train_object_count
            )
            layout = build_layout(
                object_specs=catalog,
                n_objects=n_objects,
                seed=layout_seed,
                visual_variant=prompt.visual_variant,
                camera_yaw_deg=20 if prompt.visual_variant == "camera_yaw_20deg" else 0,
            )
            scene = DesktopSortingScene(
                layout,
                camera_yaw_deg=20 if prompt.visual_variant == "camera_yaw_20deg" else 0,
            )
            scene_objects = [entry.spec for entry in layout.objects]
            expected_targets = resolve_prompt_targets(prompt, scene_objects)
            outcome = policy.evaluate_prompt(prompt, expected_targets, scene_objects)
            frames = render_episode(
                scene=scene,
                prompt_text=prompt.text,
                title_prefix=f"{model_variant} | {prompt.task_family} | {prompt.visual_variant}",
                selected_targets=outcome.selected_targets,
                expected_targets=expected_targets,
            )

            video_path = directories["videos"] / f"{model_variant}_{prompt.id}.mp4"
            write_mp4(video_path, frames, fps=12)

            if first_frame is None:
                first_frame = frames[len(frames) // 2]

            if prompt.id in {"exact_sort_blue_right_tray", "para_crimson_block_left_tray", "robust_low_light_blue_sort"}:
                hero_frames.extend(frames[::2])

            accuracy = _object_sort_accuracy(outcome.selected_targets, expected_targets)
            row = {
                "run_id": f"showcase_{model_variant}_s{seed}",
                "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "model_variant": model_variant,
                "checkpoint": f"mock/{model_variant}",
                "scene_id": layout.scene_id,
                "task_family": prompt.task_family,
                "instruction_id": prompt.id,
                "instruction_text": prompt.text,
                "language_variant": prompt.language_variant,
                "visual_variant": prompt.visual_variant,
                "object_layout_seed": layout_seed,
                "camera_variant": layout.camera_variant,
                "n_objects": n_objects,
                "task_success": outcome.task_success,
                "object_sort_accuracy": accuracy,
                "instruction_grounding_accuracy": round(outcome.instruction_grounding_accuracy, 2),
                "completion_time_s": _completion_time(len(expected_targets), outcome.task_success, outcome.failure_tag),
                "mean_policy_latency_ms": outcome.mean_policy_latency_ms,
                "rollout_hz": outcome.rollout_hz,
                "collision_count": outcome.collision_count,
                "regrasp_count": outcome.regrasp_count,
                "trajectory_jerk": _trajectory_jerk(
                    success=outcome.task_success,
                    model_variant=model_variant,
                    visual_variant=prompt.visual_variant,
                ),
                "failure_tag": outcome.failure_tag,
                "video_path": _video_path_label(video_path, repo_root),
                "notes": (
                    "Success under nominal conditions."
                    if outcome.task_success
                    else f"Mock failure injected: {outcome.failure_tag}."
                ),
            }
            episode_rows.append(row)

    episode_frame = pd.DataFrame(episode_rows)
    episode_path = directories["episodes"] / "desktop_sorting_eval_log.jsonl"
    episode_frame.to_json(episode_path, orient="records", lines=True)

    showcase_summary = summarize_showcase(episode_path)
    summary_path = directories["summary"] / "desktop_sorting_eval_summary.csv"
    showcase_summary.summary_frame.to_csv(summary_path, index=False)
    (directories["summary"] / "summary.md").write_text(showcase_summary.markdown + "\n", encoding="utf-8")

    save_success_by_task_family(showcase_summary.summary_frame, directories["figures"] / "success_by_task_family.png")
    save_success_by_language_variant(
        showcase_summary.summary_frame,
        directories["figures"] / "success_by_language_variant.png",
    )
    save_latency_vs_success(showcase_summary.summary_frame, directories["figures"] / "latency_vs_success.png")
    save_failure_breakdown(episode_frame, directories["figures"] / "failure_breakdown.png")

    hero_path = directories["videos"] / "hero_showcase.mp4"
    write_mp4(hero_path, hero_frames, fps=10)

    gif_path = directories["videos"] / "hero_showcase.gif"
    write_gif(gif_path, hero_frames[::2], fps=6)

    if first_frame is not None:
        Image.fromarray(first_frame).save(directories["videos"] / "hero_thumbnail.png")

    return {
        "episode_log": episode_path,
        "summary_csv": summary_path,
        "hero_video": hero_path,
        "hero_gif": gif_path,
        "thumbnail": directories["videos"] / "hero_thumbnail.png",
        "figures_dir": directories["figures"],
    }
