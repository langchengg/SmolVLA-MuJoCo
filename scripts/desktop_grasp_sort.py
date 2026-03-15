#!/usr/bin/env python3
"""
VLA-Driven Desktop Grasping and Multi-Object Sorting
This script sets up a MuJoCo environment with SmolVLA and executes a 
sequence of language instructions for grasping and sorting objects.
"""

import argparse
import logging
import time
from pathlib import Path
import sys

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HUGGINGFACE_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import numpy as np
import torch

# Add src to python path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.env.mujoco_env import SmolVLAMuJoCoEnv, EnvConfig
from src.model.smolvla_wrapper import SmolVLAWrapper, SmolVLAConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="VLA-Driven Grasping and Sorting")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SmolVLA checkpoint. If None, uses mock weights for testing.")
    parser.add_argument("--env_type", type=str, default="libero", choices=["libero", "metaworld", "custom"], help="Environment type")
    parser.add_argument("--task_suite", type=str, default="libero_object", help="LIBERO task suite")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"], help="Render mode. Use 'human' for live visualization.")
    parser.add_argument("--max_steps_per_task", type=int, default=100, help="Max steps per instruction")
    parser.add_argument("--save_video", action="store_true", help="Whether to save the resulting video visualization (if not in human mode)")
    parser.add_argument("--output_dir", type=str, default="./results/sorting_demo", help="Output directory for rendering")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Initialize Model
    logger.info("Initializing SmolVLA model...")
    model_config = SmolVLAConfig(
        device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        dtype="float32",  # More compatible for fallback Macs
        use_lora=False     # Disable PEFT for simple demo if not fine-tuned
    )
    
    model = SmolVLAWrapper(model_config)
    model.load(checkpoint_path=args.checkpoint)
    model.eval_mode()
    
    # 2. Initialize Environment
    logger.info(f"Setting up {args.env_type} MuJoCo environment...")
    env_config = EnvConfig(
        env_type=args.env_type,
        task_suite=args.task_suite,
        render_mode=args.render_mode,
        max_episode_steps=args.max_steps_per_task * 4  # Enough for 4 tasks
    )
    env = SmolVLAMuJoCoEnv(env_config)
    
    # 3. Define Instruction Sequence for Grasping and Sorting
    task_instructions = [
        "grasp the red block",
        "put the red block in the blue bowl",
        "grasp the green block",
        "put the green block in the red basket"
    ]
    
    # 4. Execution Loop
    logger.info("Starting execution loop...")
    
    obs = env.reset()
    
    frames_for_video = []
    
    for task_idx, instruction in enumerate(task_instructions):
        logger.info(f"\n--- Task {task_idx + 1}/{len(task_instructions)}: '{instruction}' ---")
        
        for step in range(args.max_steps_per_task):
            # Formulate Images
            images = obs.get("image", {})
            
            # Predict Action
            action = model.predict_action(
                images=images,
                language_instruction=instruction,
                state=obs.get("state")
            )
            
            # Use the first chunk if chunking is returned
            if action.ndim > 1:
                action = action[0]
                
            # Execute step in Env
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render handling
            if args.render_mode == "human":
                env.render()
                time.sleep(0.02)  # Control framerate roughly
            elif args.save_video and images and "agentview" in images:
                frames_for_video.append(images["agentview"])
            
            if terminated:
                logger.info("Environment signaled termination.")
                
            # For demonstration, we break just on max steps, 
            # as our dummy model might not achieve success
        
        logger.info(f"Finished executing: '{instruction}'")
    
    # Optional: Save Video if not human render mode
    if args.save_video and args.render_mode != "human" and frames_for_video:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "sorting_demo.mp4"
        
        logger.info(f"Saving video to {video_path}")
        h, w = frames_for_video[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
        
        for frame in frames_for_video:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            
        writer.release()
        
    env.close()
    logger.info("Demo complete!")


if __name__ == "__main__":
    main()
