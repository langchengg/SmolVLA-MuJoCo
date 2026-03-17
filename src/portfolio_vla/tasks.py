from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


WARM_COLORS = {"red", "orange", "yellow"}
COOL_COLORS = {"blue", "green"}


@dataclass(frozen=True)
class SelectorSpec:
    mode: str
    object: str | None = None
    color: str | None = None
    shape: str | None = None
    palette: str | None = None


@dataclass(frozen=True)
class PlacementRule:
    selector: SelectorSpec
    container: str


@dataclass(frozen=True)
class PlacementSpec:
    mode: str
    container: str | None = None
    rules: tuple[PlacementRule, ...] = ()


@dataclass(frozen=True)
class PromptSpec:
    id: str
    text: str
    task_family: str
    language_variant: str
    selector: SelectorSpec
    placement: PlacementSpec
    tags: tuple[str, ...] = ()

    @property
    def visual_variant(self) -> str:
        if "low_light" in self.tags:
            return "low_light"
        if "clutter_background" in self.tags:
            return "clutter_background"
        if "camera_yaw_20deg" in self.tags:
            return "camera_yaw_20deg"
        return "nominal"


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    color: str
    shape: str
    size: tuple[float, float, float]


def load_prompt_specs(path: str | Path) -> list[PromptSpec]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    prompts: list[PromptSpec] = []
    for _, entries in payload["prompt_sets"].items():
        for raw in entries:
            selector = SelectorSpec(**raw["selector"])
            placement = raw["placement"]
            rules = tuple(
                PlacementRule(selector=SelectorSpec(**rule["selector"]), container=rule["container"])
                for rule in placement.get("rules", [])
            )
            prompts.append(
                PromptSpec(
                    id=raw["id"],
                    text=raw["text"],
                    task_family=raw["task_family"],
                    language_variant=raw["language_variant"],
                    selector=selector,
                    placement=PlacementSpec(
                        mode=placement["mode"],
                        container=placement.get("container"),
                        rules=rules,
                    ),
                    tags=tuple(raw.get("tags", [])),
                )
            )
    return prompts


def parse_catalog_object(name: str) -> SceneObject:
    color, shape = name.split("_", 1)
    size_map = {
        "cube": (0.03, 0.03, 0.03),
        "block": (0.04, 0.025, 0.02),
        "cylinder": (0.024, 0.024, 0.05),
    }
    return SceneObject(object_id=name, color=color, shape=shape, size=size_map[shape])


def _matches_selector(selector: SelectorSpec, obj: SceneObject) -> bool:
    if selector.mode == "exact_object":
        return obj.object_id == selector.object
    if selector.mode == "by_color":
        return obj.color == selector.color
    if selector.mode == "by_shape":
        return obj.shape == selector.shape
    if selector.mode == "by_palette":
        palette = selector.palette or ""
        if palette == "warm":
            return obj.color in WARM_COLORS
        if palette == "cool":
            return obj.color in COOL_COLORS
        return False
    raise ValueError(f"Unsupported selector mode: {selector.mode}")


def resolve_prompt_targets(prompt: PromptSpec, objects: list[SceneObject]) -> dict[str, str]:
    targets: dict[str, str] = {}
    if prompt.placement.mode == "single_container":
        for obj in objects:
            if _matches_selector(prompt.selector, obj):
                targets[obj.object_id] = str(prompt.placement.container)
    elif prompt.placement.mode == "group_rules":
        for rule in prompt.placement.rules:
            for obj in objects:
                if _matches_selector(rule.selector, obj):
                    targets[obj.object_id] = rule.container
    else:
        raise ValueError(f"Unsupported placement mode: {prompt.placement.mode}")

    if not targets:
        raise ValueError(f"Prompt {prompt.id} did not resolve any target objects.")
    return targets


def pick_prompt_subset(prompts: list[PromptSpec]) -> list[PromptSpec]:
    """Stable subset used for sample artifacts and README media."""
    desired_ids = [
        "exact_pick_red_cube_left_tray",
        "exact_sort_blue_right_tray",
        "exact_sort_cylinders_back_bin",
        "comp_warm_left_cool_right",
        "para_crimson_block_left_tray",
        "robust_low_light_blue_sort",
        "robust_clutter_red_cube",
    ]
    prompt_map = {prompt.id: prompt for prompt in prompts}
    return [prompt_map[prompt_id] for prompt_id in desired_ids]
