from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from portfolio_vla.tasks import SceneObject


TABLE_TOP_Z = 0.04
GANTRY_BASE_Z = 0.55
HOME_TOOL_POS = np.array([-0.28, 0.0, 0.34], dtype=float)


COLOR_RGBA = {
    "red": (0.84, 0.19, 0.16, 1.0),
    "blue": (0.19, 0.35, 0.82, 1.0),
    "green": (0.20, 0.63, 0.34, 1.0),
    "yellow": (0.95, 0.82, 0.22, 1.0),
    "orange": (0.96, 0.53, 0.18, 1.0),
}


@dataclass(frozen=True)
class LayoutObject:
    spec: SceneObject
    position: tuple[float, float]


@dataclass(frozen=True)
class SceneLayout:
    scene_id: str
    objects: tuple[LayoutObject, ...]
    tray_slots: dict[str, tuple[tuple[float, float], ...]]
    visual_variant: str
    camera_variant: str
    background_rgba: tuple[float, float, float, float]
    light_diffuse: tuple[float, float, float]


def _camera_xyaxes(camera_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None) -> str:
    up_hint = np.array([0.0, 0.0, 1.0]) if up_hint is None else up_hint
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    xaxis = right
    yaxis = up / np.linalg.norm(up)
    return " ".join(f"{value:.4f}" for value in np.concatenate([xaxis, yaxis]))


def build_layout(
    object_specs: list[SceneObject],
    n_objects: int,
    seed: int,
    visual_variant: str,
    camera_yaw_deg: int,
) -> SceneLayout:
    rng = np.random.default_rng(seed)
    positions = [
        (-0.10, -0.12),
        (-0.02, 0.12),
        (0.06, -0.08),
        (0.02, 0.02),
        (-0.14, 0.08),
        (0.12, 0.10),
    ]
    rng.shuffle(positions)
    objects = tuple(
        LayoutObject(spec=spec, position=positions[index])
        for index, spec in enumerate(object_specs[:n_objects])
    )
    tray_slots = {
        "left_tray": ((0.22, 0.14), (0.16, 0.14), (0.22, 0.08), (0.16, 0.08)),
        "right_tray": ((0.22, -0.14), (0.16, -0.14), (0.22, -0.08), (0.16, -0.08)),
        "back_bin": ((-0.18, 0.04), (-0.18, -0.04), (-0.12, 0.00)),
    }

    if visual_variant == "low_light":
        background = (0.35, 0.36, 0.40, 1.0)
        light = (0.45, 0.45, 0.45)
    elif visual_variant == "clutter_background":
        background = (0.68, 0.73, 0.67, 1.0)
        light = (0.85, 0.82, 0.80)
    else:
        background = (0.87, 0.89, 0.92, 1.0)
        light = (0.92, 0.92, 0.92)

    return SceneLayout(
        scene_id=f"tabletop_two_trays_seed{seed}",
        objects=objects,
        tray_slots=tray_slots,
        visual_variant=visual_variant,
        camera_variant=f"front_yaw_{camera_yaw_deg}deg",
        background_rgba=background,
        light_diffuse=light,
    )


class DesktopSortingScene:
    def __init__(
        self,
        layout: SceneLayout,
        camera_yaw_deg: int = 0,
        width: int = 640,
        height: int = 368,
    ) -> None:
        self.layout = layout
        self.width = width
        self.height = height
        xml = self._build_xml(layout, camera_yaw_deg)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.object_qpos_adr = {
            obj.spec.object_id: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj.spec.object_id}_joint")
            ]
            for obj in layout.objects
        }
        self.tool_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
        self.current_tool_pos = HOME_TOOL_POS.copy()
        self.attached_object: str | None = None
        self.attachment_offset = np.array([0.0, 0.0, -0.035], dtype=float)
        self.reset()

    def reset(self) -> None:
        self.data.qpos[:] = 0.0
        self.current_tool_pos = HOME_TOOL_POS.copy()
        self._set_tool_pos(self.current_tool_pos)
        for obj in self.layout.objects:
            half_height = obj.spec.size[2] if obj.spec.shape != "cylinder" else obj.spec.size[2] / 2
            self._set_object_pose(
                obj.spec.object_id,
                np.array([obj.position[0], obj.position[1], TABLE_TOP_Z + half_height], dtype=float),
            )
        self.attached_object = None
        mujoco.mj_forward(self.model, self.data)

    def container_slot(self, container: str, index: int) -> np.ndarray:
        slots = self.layout.tray_slots[container]
        xy = slots[index % len(slots)]
        return np.array([xy[0], xy[1], TABLE_TOP_Z + 0.025], dtype=float)

    def move_tool_linear(self, target: np.ndarray, n_frames: int) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        start = self.current_tool_pos.copy()
        for step in range(1, n_frames + 1):
            alpha = step / n_frames
            pos = start * (1 - alpha) + target * alpha
            self._set_tool_pos(pos)
            frames.append(self.render())
        self.current_tool_pos = target.copy()
        return frames

    def attach(self, object_id: str) -> None:
        self.attached_object = object_id
        self._set_object_pose(object_id, self.current_tool_pos + self.attachment_offset)
        mujoco.mj_forward(self.model, self.data)

    def release(self, object_id: str, final_pos: np.ndarray) -> None:
        self._set_object_pose(object_id, final_pos)
        self.attached_object = None
        mujoco.mj_forward(self.model, self.data)

    def render(self, camera: str = "front_camera") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def _set_tool_pos(self, pos: np.ndarray) -> None:
        qpos = np.array([pos[0], pos[1], GANTRY_BASE_Z - pos[2]], dtype=float)
        self.data.qpos[0:3] = qpos
        if self.attached_object:
            self._set_object_pose(self.attached_object, pos + self.attachment_offset)
        mujoco.mj_forward(self.model, self.data)

    def _set_object_pose(self, object_id: str, pos: np.ndarray) -> None:
        adr = self.object_qpos_adr[object_id]
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _build_xml(self, layout: SceneLayout, camera_yaw_deg: int) -> str:
        front_target = np.array([0.02, 0.0, 0.14], dtype=float)
        radius = 0.95
        yaw = np.deg2rad(camera_yaw_deg)
        front_pos = np.array(
            [radius * np.cos(np.pi / 4 + yaw), radius * np.sin(yaw), 0.65],
            dtype=float,
        )
        front_xyaxes = _camera_xyaxes(front_pos, front_target)
        clutter = ""
        if layout.visual_variant == "clutter_background":
            clutter = """
      <geom type="box" pos="-0.34 0.24 0.14" size="0.04 0.04 0.14" rgba="0.58 0.51 0.42 1"/>
      <geom type="box" pos="-0.34 -0.20 0.10" size="0.05 0.03 0.10" rgba="0.42 0.53 0.61 1"/>
      <geom type="cylinder" pos="0.33 0.24 0.08" size="0.02 0.08" rgba="0.73 0.63 0.32 1"/>
"""

        object_entries: list[str] = []
        for obj in layout.objects:
            rgba = " ".join(f"{value:.2f}" for value in COLOR_RGBA[obj.spec.color])
            if obj.spec.shape == "cylinder":
                geom = (
                    f'<geom name="{obj.spec.object_id}_geom" type="cylinder" '
                    f'size="{obj.spec.size[0]:.3f} {obj.spec.size[2] / 2:.3f}" rgba="{rgba}"/>'
                )
            else:
                geom = (
                    f'<geom name="{obj.spec.object_id}_geom" type="box" '
                    f'size="{obj.spec.size[0] / 2:.3f} {obj.spec.size[1] / 2:.3f} {obj.spec.size[2] / 2:.3f}" '
                    f'rgba="{rgba}"/>'
                )
            object_entries.append(
                f"""
    <body name="{obj.spec.object_id}">
      <freejoint name="{obj.spec.object_id}_joint"/>
      {geom}
    </body>
"""
            )
        return f"""
<mujoco model="desktop_sorting">
  <compiler inertiafromgeom="true"/>
  <visual>
    <global offwidth="{self.width}" offheight="{self.height}"/>
  </visual>
  <option timestep="0.02" gravity="0 0 0"/>
  <worldbody>
    <geom name="background" type="plane" pos="0 0 -0.001" size="2 2 0.01" rgba="{' '.join(f'{v:.2f}' for v in layout.background_rgba)}"/>
    <light name="scene_light" pos="0 0 2.2" diffuse="{' '.join(f'{v:.2f}' for v in layout.light_diffuse)}"/>
    <camera name="front_camera" pos="{front_pos[0]:.3f} {front_pos[1]:.3f} {front_pos[2]:.3f}" xyaxes="{front_xyaxes}"/>
    <camera name="top_camera" pos="0 0 1.25" xyaxes="1 0 0 0 1 0"/>
    <body name="table" pos="0 0 0.02">
      <geom type="box" size="0.42 0.32 0.02" rgba="0.56 0.46 0.34 1"/>
      <geom type="box" pos="0.19 0.11 0.03" size="0.08 0.05 0.005" rgba="0.86 0.71 0.46 1"/>
      <geom type="box" pos="0.19 -0.11 0.03" size="0.08 0.05 0.005" rgba="0.48 0.73 0.88 1"/>
      <geom type="box" pos="-0.15 0.00 0.035" size="0.06 0.08 0.01" rgba="0.73 0.73 0.73 1"/>
      <geom type="box" pos="0.19 0.16 0.05" size="0.08 0.005 0.03" rgba="0.86 0.71 0.46 1"/>
      <geom type="box" pos="0.19 0.06 0.05" size="0.08 0.005 0.03" rgba="0.86 0.71 0.46 1"/>
      <geom type="box" pos="0.27 0.11 0.05" size="0.005 0.05 0.03" rgba="0.86 0.71 0.46 1"/>
      <geom type="box" pos="0.11 0.11 0.05" size="0.005 0.05 0.03" rgba="0.86 0.71 0.46 1"/>
      <geom type="box" pos="0.19 -0.06 0.05" size="0.08 0.005 0.03" rgba="0.48 0.73 0.88 1"/>
      <geom type="box" pos="0.19 -0.16 0.05" size="0.08 0.005 0.03" rgba="0.48 0.73 0.88 1"/>
      <geom type="box" pos="0.27 -0.11 0.05" size="0.005 0.05 0.03" rgba="0.48 0.73 0.88 1"/>
      <geom type="box" pos="0.11 -0.11 0.05" size="0.005 0.05 0.03" rgba="0.48 0.73 0.88 1"/>
      <geom type="box" pos="-0.09 0.08 0.07" size="0.005 0.08 0.05" rgba="0.73 0.73 0.73 1"/>
      <geom type="box" pos="-0.21 0.08 0.07" size="0.005 0.08 0.05" rgba="0.73 0.73 0.73 1"/>
      <geom type="box" pos="-0.15 0.16 0.07" size="0.06 0.005 0.05" rgba="0.73 0.73 0.73 1"/>
      <geom type="box" pos="-0.15 -0.08 0.07" size="0.06 0.005 0.05" rgba="0.73 0.73 0.73 1"/>
{clutter}
    </body>
    <body name="gantry_base" pos="-0.25 0.00 {GANTRY_BASE_Z:.3f}">
      <inertial pos="0 0 0.10" mass="2.0" diaginertia="0.02 0.02 0.02"/>
      <joint name="gantry_x" type="slide" axis="1 0 0" range="-0.12 0.55"/>
      <body name="gantry_y">
        <inertial pos="0 0 0.08" mass="1.4" diaginertia="0.01 0.01 0.01"/>
        <joint name="gantry_y" type="slide" axis="0 1 0" range="-0.24 0.24"/>
        <body name="gantry_z">
          <inertial pos="0 0 0.08" mass="0.9" diaginertia="0.008 0.008 0.008"/>
          <joint name="gantry_z" type="slide" axis="0 0 -1" range="0.08 0.44"/>
          <geom type="capsule" fromto="0 0 0.18 0 0 0.02" size="0.02" rgba="0.22 0.22 0.26 1"/>
          <geom type="capsule" fromto="0 0 0.02 0.03 0 0" size="0.01" rgba="0.18 0.18 0.20 1"/>
          <geom type="capsule" fromto="0 0 0.02 -0.03 0 0" size="0.01" rgba="0.18 0.18 0.20 1"/>
          <geom type="capsule" fromto="0.03 0 0 -0.03 0 0" size="0.008" rgba="0.75 0.75 0.78 1"/>
          <site name="tool_site" pos="0 0 0" size="0.01"/>
          <camera name="wrist_camera" pos="0.06 0 0.08" xyaxes="0 1 0 -0.4 0 1"/>
        </body>
      </body>
    </body>
    {''.join(object_entries)}
  </worldbody>
</mujoco>
"""
