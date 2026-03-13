#!/bin/bash
# =============================================================================
# SmolVLA Fine-tuning Script
# =============================================================================
# Usage: bash scripts/run_finetune.sh [--max_steps N] [--no_wandb]

set -e

echo "🤖 SmolVLA Fine-tuning on LIBERO"
echo "================================"

# Default arguments
MODEL="HuggingFaceTB/SmolVLA-base"
DATASET="lerobot/libero_object_no_noops"
CONFIG="configs/finetune_libero.yaml"
MAX_STEPS=""
EXTRA_ARGS=""

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_steps)
            MAX_STEPS="--max_steps $2"
            shift 2
            ;;
        --no_wandb)
            EXTRA_ARGS="$EXTRA_ARGS --no_wandb"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "Model:   $MODEL"
echo "Dataset: $DATASET"
echo "Config:  $CONFIG"
echo ""

# Option 1: Use LeRobot CLI (recommended)
if command -v lerobot-train &> /dev/null; then
    echo "Using LeRobot CLI..."
    lerobot-train \
        --policy=smolvla \
        --dataset.repo_id=$DATASET \
        $EXTRA_ARGS
else
    # Option 2: Use custom training script
    echo "Using custom training script..."
    python -m src.training.finetune \
        --config $CONFIG \
        --model $MODEL \
        --dataset $DATASET \
        $MAX_STEPS \
        $EXTRA_ARGS
fi

echo ""
echo "✅ Fine-tuning complete!"
