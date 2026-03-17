from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from portfolio_vla.tasks import PromptSpec, SceneObject


@dataclass(frozen=True)
class PolicyOutcome:
    selected_targets: dict[str, str]
    failure_tag: str
    task_success: bool
    collision_count: int
    regrasp_count: int
    instruction_grounding_accuracy: float
    mean_policy_latency_ms: float
    rollout_hz: float


class MockPolicy:
    def __init__(self, model_variant: str, seed: int) -> None:
        self.model_variant = model_variant
        self.rng = np.random.default_rng(seed)

    def evaluate_prompt(
        self,
        prompt: PromptSpec,
        target_map: dict[str, str],
        objects: Iterable[SceneObject],
    ) -> PolicyOutcome:
        success_probability = self._success_probability(prompt)
        success = bool(self.rng.random() <= success_probability)
        failure_tag = ""
        selected = dict(target_map)
        grounding = 1.0
        collisions = 0
        regrasp = 1 if self.model_variant == "finetuned" else 2

        if not success:
            failure_tag = self._failure_tag(prompt)
            grounding = 0.0 if "paraphrase" in failure_tag else 0.5
            selected = self._inject_failure(prompt, selected, list(objects), failure_tag)
            collisions = 1 if failure_tag in {"collision_abort", "visual_perturbation_failure"} else 0
            regrasp = 2 if self.model_variant == "finetuned" else 3
        elif prompt.visual_variant != "nominal":
            grounding = 0.93 if self.model_variant == "finetuned" else 0.68
            collisions = 1 if prompt.visual_variant == "low_light" and self.model_variant == "finetuned" else 0

        latency = 68.0 if self.model_variant == "zero_shot" else 72.0
        latency += 3.0 if prompt.visual_variant == "low_light" else 0.0
        latency += 2.0 if prompt.language_variant == "paraphrase" else 0.0
        rollout_hz = round(1000.0 / latency, 1)

        return PolicyOutcome(
            selected_targets=selected,
            failure_tag=failure_tag,
            task_success=success,
            collision_count=collisions,
            regrasp_count=regrasp,
            instruction_grounding_accuracy=grounding,
            mean_policy_latency_ms=round(latency, 1),
            rollout_hz=rollout_hz,
        )

    def _success_probability(self, prompt: PromptSpec) -> float:
        score = 0.93 if self.model_variant == "finetuned" else 0.68
        if prompt.language_variant == "paraphrase":
            score -= 0.15 if self.model_variant == "finetuned" else 0.32
        if prompt.visual_variant == "low_light":
            score -= 0.12 if self.model_variant == "finetuned" else 0.24
        if prompt.visual_variant == "clutter_background":
            score -= 0.10 if self.model_variant == "finetuned" else 0.19
        if prompt.visual_variant == "camera_yaw_20deg":
            score -= 0.08 if self.model_variant == "finetuned" else 0.16
        if prompt.task_family == "compositional_sorting":
            score -= 0.08 if self.model_variant == "finetuned" else 0.15
        return float(np.clip(score, 0.15, 0.98))

    def _failure_tag(self, prompt: PromptSpec) -> str:
        if prompt.language_variant == "paraphrase":
            return "paraphrase_misunderstanding"
        if prompt.visual_variant != "nominal":
            return "visual_perturbation_failure"
        choices = ["wrong_object_selected", "wrong_container_selected", "grasp_failure", "timeout"]
        return str(self.rng.choice(choices))

    def _inject_failure(
        self,
        prompt: PromptSpec,
        selected: dict[str, str],
        objects: list[SceneObject],
        failure_tag: str,
    ) -> dict[str, str]:
        if failure_tag == "wrong_container_selected":
            swapped: dict[str, str] = {}
            for object_id, container in selected.items():
                swapped[object_id] = "right_tray" if container == "left_tray" else "left_tray"
            return swapped

        if failure_tag == "wrong_object_selected":
            wrong_candidate = next((obj.object_id for obj in objects if obj.object_id not in selected), None)
            if wrong_candidate and selected:
                first_container = next(iter(selected.values()))
                return {wrong_candidate: first_container}

        if failure_tag == "grasp_failure":
            return {}

        if failure_tag == "paraphrase_misunderstanding":
            wrong_candidate = next((obj.object_id for obj in objects if obj.color != "red"), None)
            if wrong_candidate:
                return {wrong_candidate: "left_tray"}

        if failure_tag == "visual_perturbation_failure":
            partial = dict(selected)
            if partial:
                first_key = next(iter(partial))
                partial[first_key] = "back_bin"
            return partial

        return dict(selected)
