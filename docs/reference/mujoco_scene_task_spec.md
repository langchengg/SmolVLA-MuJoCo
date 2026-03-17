# MuJoCo Scene Task Spec

This document describes the simulation-only desktop-sorting scene used for README media and logging examples.

## Scene

Workspace:

- tabletop surface
- `left_tray`
- `right_tray`
- `back_bin`

Robot:

- lightweight cartesian arm
- self-contained MuJoCo XML
- no external Panda asset dependency

Camera views:

- `front_camera`
- `wrist_camera`
- `top_camera`

## Objects

Default object set:

- `red_cube`
- `blue_cube`
- `green_cylinder`
- `yellow_block`
- `orange_cylinder`

Layouts:

- train-style layouts use 4 objects
- eval-style layouts use 5 objects

## Task families

- `single_object_pick_place`
- `color_grouping`
- `shape_grouping`
- `compositional_sorting`
- `paraphrase_generalization`

## Perturbations

- `nominal`
- `low_light`
- `clutter_background`
- `camera_yaw_20deg`

## Motion model

The demo path uses a scripted pick-and-place trajectory:

1. hover above object
2. descend
3. attach
4. lift
5. move to receptacle
6. descend
7. release
8. return home

## Output contract

Every showcase run produces:

- episode JSONL
- aggregate CSV
- summary markdown
- per-episode MP4 files
- a stitched hero MP4/GIF
- demo figures

This path is intentionally separate from the real LeRobot benchmark path.
