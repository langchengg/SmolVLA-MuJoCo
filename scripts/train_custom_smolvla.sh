#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-custom_smolvla_$(date +%Y%m%d_%H%M%S)}"

: "${DATASET_REPO_ID:?Set DATASET_REPO_ID to your dataset on the Hugging Face Hub.}"
: "${OUTPUT_DIR:=$ROOT_DIR/results/raw/train/$RUN_NAME}"
: "${JOB_NAME:=custom_smolvla_training}"
: "${BATCH_SIZE:=32}"
: "${STEPS:=20000}"
: "${POLICY_DEVICE:=cuda}"
: "${WANDB_ENABLE:=false}"

export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$OUTPUT_DIR")"

WORK_DIR="${LEROBOT_DIR:-$ROOT_DIR}"

if [[ ! -d "$WORK_DIR" ]]; then
  echo "LEROBOT_DIR does not exist: $WORK_DIR" >&2
  exit 1
fi

cd "$WORK_DIR"

printf "Starting LeRobot training\n"
printf "  work_dir: %s\n" "$WORK_DIR"
printf "  output_dir: %s\n" "$OUTPUT_DIR"
printf "  dataset: %s\n" "$DATASET_REPO_ID"
printf "  steps: %s\n" "$STEPS"
printf "  batch_size: %s\n" "$BATCH_SIZE"
printf "  job_name: %s\n" "$JOB_NAME"

train_cmd=(
  lerobot-train
  --policy.path=lerobot/smolvla_base
  --dataset.repo_id="$DATASET_REPO_ID"
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --policy.device="$POLICY_DEVICE"
  --wandb.enable="$WANDB_ENABLE"
)

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${train_cmd[@]}"
else
  "${train_cmd[@]}"
fi
