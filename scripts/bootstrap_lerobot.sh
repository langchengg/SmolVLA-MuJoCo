#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-$ROOT_DIR/third_party/lerobot}"

install_libero_fallback() {
  printf "\nFalling back to notebook-friendly LIBERO install (skipping hf-egl-probe).\n"
  printf "This is useful on Colab/Kaggle when egl_probe wheel builds fail.\n"

  python3 -m pip install \
    "scipy>=1.14.0,<2.0.0" \
    "hydra-core>=1.2,<1.4" \
    "robomimic==0.2.0" \
    "robosuite==1.4.0" \
    "bddl==1.0.1" \
    "easydict" \
    "einops" \
    "thop" \
    "matplotlib>=3.5.3" \
    "cloudpickle>=2.0.0" \
    "opencv-python" \
    "wandb" \
    "gymnasium>=0.29.0" \
    "mujoco>=3.0.0" \
    "future>=0.18.2"

  # Source inspection of hf-libero 0.1.3 shows no direct runtime imports of
  # egl_probe in the Python package, so we install hf-libero without deps here
  # to avoid the unstable hf-egl-probe wheel build in notebook environments.
  python3 -m pip install --no-deps "hf-libero>=0.1.3,<0.2.0"
}

if [[ ! -d "$TARGET_DIR/.git" ]]; then
  git clone https://github.com/huggingface/lerobot.git "$TARGET_DIR"
fi

cd "$TARGET_DIR"
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[smolvla]"

if [[ "${LEROBOT_SKIP_EGL_PROBE:-0}" == "1" ]]; then
  install_libero_fallback
else
  if ! python3 -m pip install -e ".[libero]"; then
    install_libero_fallback
  fi
fi

printf "\nLeRobot is ready in %s\n" "$TARGET_DIR"
printf "Suggested next steps:\n"
printf "  export MUJOCO_GL=egl\n"
printf "  export HF_USER=your_huggingface_username\n"
printf "  export POLICY_REPO_ID=\$HF_USER/libero-smolvla-demo\n"
printf "  bash %s/scripts/train_libero_smolvla.sh\n" "$ROOT_DIR"
