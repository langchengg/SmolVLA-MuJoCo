#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$ROOT_DIR/third_party/lerobot}"

: "${HF_USER:=your_huggingface_username}"
: "${POLICY_REPO_ID:=$HF_USER/libero-smolvla-demo}"
: "${DATASET_REPO_ID:=HuggingFaceVLA/libero}"
: "${ENV_TASK:=libero_10}"
: "${OUTPUT_DIR:=./outputs/libero_smolvla}"
: "${STEPS:=20000}"
: "${BATCH_SIZE:=4}"
: "${EVAL_BATCH_SIZE:=1}"
: "${EVAL_EPISODES:=1}"
: "${EVAL_FREQ:=1000}"
: "${MUJOCO_GL:=egl}"

export MUJOCO_GL

cd "$LEROBOT_DIR"

# Official LIBERO docs show a 100k-step example. This template defaults to a
# smaller run so you can start with a portfolio-friendly experiment budget.
lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id="$POLICY_REPO_ID" \
  --policy.load_vlm_weights=true \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --env.type=libero \
  --env.task="$ENV_TASK" \
  --output_dir="$OUTPUT_DIR" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --eval.batch_size="$EVAL_BATCH_SIZE" \
  --eval.n_episodes="$EVAL_EPISODES" \
  --eval_freq="$EVAL_FREQ"
