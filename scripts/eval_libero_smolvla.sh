#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-libero_eval_$(date +%Y%m%d_%H%M%S)}"

: "${POLICY_PATH:?Set POLICY_PATH to a HF policy id or a local checkpoint path.}"
: "${TASKS:=libero_10}"
: "${OUTPUT_DIR:=$ROOT_DIR/results/raw/eval/$RUN_NAME}"
: "${EVAL_BATCH_SIZE:=1}"
: "${EVAL_EPISODES:=3}"
: "${N_ACTION_STEPS:=10}"
: "${MAX_PARALLEL_TASKS:=1}"
: "${MUJOCO_GL:=egl}"

export MUJOCO_GL

mkdir -p "$OUTPUT_DIR"

WORK_DIR="${LEROBOT_DIR:-$ROOT_DIR}"

if [[ ! -d "$WORK_DIR" ]]; then
  echo "LEROBOT_DIR does not exist: $WORK_DIR" >&2
  exit 1
fi

cd "$WORK_DIR"

lerobot-eval \
  --output_dir="$OUTPUT_DIR" \
  --env.type=libero \
  --env.task="$TASKS" \
  --eval.batch_size="$EVAL_BATCH_SIZE" \
  --eval.n_episodes="$EVAL_EPISODES" \
  --policy.path="$POLICY_PATH" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --env.max_parallel_tasks="$MAX_PARALLEL_TASKS"
