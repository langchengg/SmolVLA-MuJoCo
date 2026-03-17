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

mkdir -p "$(dirname "$OUTPUT_DIR")"

WORK_DIR="${LEROBOT_DIR:-$ROOT_DIR}"

if [[ ! -d "$WORK_DIR" ]]; then
  echo "LEROBOT_DIR does not exist: $WORK_DIR" >&2
  exit 1
fi

cd "$WORK_DIR"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.device="$POLICY_DEVICE" \
  --wandb.enable="$WANDB_ENABLE"
