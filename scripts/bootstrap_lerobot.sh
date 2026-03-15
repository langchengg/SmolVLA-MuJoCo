#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-$ROOT_DIR/third_party/lerobot}"

if [[ ! -d "$TARGET_DIR/.git" ]]; then
  git clone https://github.com/huggingface/lerobot.git "$TARGET_DIR"
fi

cd "$TARGET_DIR"
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[smolvla]"
python3 -m pip install -e ".[libero]"

printf "\nLeRobot is ready in %s\n" "$TARGET_DIR"
printf "Suggested next steps:\n"
printf "  export MUJOCO_GL=egl\n"
printf "  export HF_USER=your_huggingface_username\n"
printf "  export POLICY_REPO_ID=\$HF_USER/libero-smolvla-demo\n"
printf "  bash %s/scripts/train_libero_smolvla.sh\n" "$ROOT_DIR"
