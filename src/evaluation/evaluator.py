"""
Core Evaluation Engine for SmolVLA in MuJoCo.

Orchestrates running evaluation episodes, collecting metrics,
and generating results for all innovation experiments.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    n_episodes: int = 50
    max_steps_per_episode: int = 300
    save_videos: bool = True
    video_fps: int = 30
    results_dir: str = "./results"
    seed: int = 42


@dataclass
class EpisodeResult:
    """Result from a single evaluation episode."""
    episode_id: int
    success: bool
    total_reward: float
    n_steps: int
    completion_time_s: float
    task_instruction: str
    actions: list[np.ndarray] = field(default_factory=list)
    images: list[np.ndarray] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class SmolVLAEvaluator:
    """
    Core evaluation engine for SmolVLA policies in MuJoCo environments.
    
    Handles:
    - Episode rollout with the VLA policy
    - Success rate computation
    - Video recording
    - Metrics aggregation
    
    Used by all 4 innovation-specific evaluators.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self._results_dir = Path(config.results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_policy(
        self,
        model,
        env,
        n_episodes: Optional[int] = None,
        language_instruction: Optional[str] = None,
        record_video: bool = False,
    ) -> list[EpisodeResult]:
        """
        Run evaluation episodes and collect results.
        
        Args:
            model: SmolVLAWrapper instance
            env: SmolVLAMuJoCoEnv instance
            n_episodes: Number of episodes (overrides config)
            language_instruction: Override language instruction
            record_video: Whether to save episode videos
            
        Returns:
            List of EpisodeResult for each episode
        """
        n_eps = n_episodes or self.config.n_episodes
        results = []
        
        model.eval_mode()
        
        for ep_idx in range(n_eps):
            result = self._run_episode(
                model, env, ep_idx, 
                language_instruction=language_instruction,
                record=record_video,
            )
            results.append(result)
            
            if (ep_idx + 1) % 10 == 0:
                sr = np.mean([r.success for r in results]) * 100
                logger.info(f"Episode {ep_idx+1}/{n_eps} | Success Rate: {sr:.1f}%")
        
        return results

    def _run_episode(
        self,
        model,
        env,
        episode_id: int,
        language_instruction: Optional[str] = None,
        record: bool = False,
    ) -> EpisodeResult:
        """Run a single evaluation episode."""
        obs = env.reset(seed=self.config.seed + episode_id)
        
        instruction = language_instruction or obs.get("language_instruction", "complete the task")
        
        total_reward = 0.0
        actions = []
        frames = []
        done = False
        
        start_time = time.perf_counter()
        
        for step in range(self.config.max_steps_per_episode):
            # Get action from model
            action = model.predict_action(
                images=obs["image"],
                language_instruction=instruction,
                state=obs.get("state"),
            )
            
            # Ensure action is 1D
            if action.ndim > 1:
                action = action[0]  # Take first action from chunk
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            actions.append(action)
            
            if record and "image" in obs:
                first_cam = list(obs["image"].values())[0]
                frames.append(first_cam)
            
            if terminated or truncated:
                done = True
                break
        
        elapsed = time.perf_counter() - start_time
        
        # Determine success
        success = self._check_success(info, total_reward, done)
        
        result = EpisodeResult(
            episode_id=episode_id,
            success=success,
            total_reward=total_reward,
            n_steps=step + 1,
            completion_time_s=elapsed,
            task_instruction=instruction,
            actions=actions,
            metadata={"info": info},
        )
        
        # Save video
        if record and frames:
            self._save_video(frames, episode_id, instruction)
        
        return result

    def _check_success(self, info: dict, total_reward: float, done: bool) -> bool:
        """Determine if the episode was successful."""
        # Check various success indicators
        if isinstance(info, dict):
            if "success" in info:
                return bool(info["success"])
            if "is_success" in info:
                return bool(info["is_success"])
            if "task_complete" in info:
                return bool(info["task_complete"])
        
        # Fallback: use reward threshold
        return total_reward > 0.5

    def _save_video(self, frames: list[np.ndarray], episode_id: int, instruction: str):
        """Save episode video."""
        try:
            import cv2
            
            video_dir = self._results_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            
            safe_name = instruction.replace(" ", "_")[:30]
            video_path = video_dir / f"ep{episode_id}_{safe_name}.mp4"
            
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, self.config.video_fps, (w, h))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            
            writer.release()
            logger.debug(f"Video saved: {video_path}")
            
        except ImportError:
            logger.warning("OpenCV not installed. Skipping video saving.")

    @staticmethod
    def compute_aggregate_metrics(results: list[EpisodeResult]) -> dict[str, float]:
        """Compute aggregate metrics from episode results."""
        if not results:
            return {}
        
        successes = [r.success for r in results]
        rewards = [r.total_reward for r in results]
        steps = [r.n_steps for r in results]
        times = [r.completion_time_s for r in results]
        
        metrics = {
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps)),
            "mean_completion_time_s": float(np.mean(times)),
            "n_episodes": len(results),
        }
        
        # Success rate confidence interval (Wilson score)
        n = len(successes)
        p = metrics["success_rate"]
        z = 1.96  # 95% CI
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
        metrics["success_rate_ci_low"] = float(max(0, center - spread))
        metrics["success_rate_ci_high"] = float(min(1, center + spread))
        
        return metrics

    @staticmethod
    def format_results_table(
        results_dict: dict[str, dict[str, float]],
        metric_key: str = "success_rate",
    ) -> str:
        """Format results as a markdown table."""
        lines = [
            f"| Condition | Success Rate | Mean Reward | Mean Steps |",
            f"|-----------|-------------|-------------|------------|",
        ]
        
        for name, metrics in results_dict.items():
            sr = metrics.get("success_rate", 0) * 100
            sr_ci_lo = metrics.get("success_rate_ci_low", 0) * 100
            sr_ci_hi = metrics.get("success_rate_ci_high", 0) * 100
            reward = metrics.get("mean_reward", 0)
            steps = metrics.get("mean_steps", 0)
            
            lines.append(
                f"| {name:<30} | {sr:.1f}% [{sr_ci_lo:.0f}-{sr_ci_hi:.0f}] | "
                f"{reward:.3f} | {steps:.0f} |"
            )
        
        return "\n".join(lines)

    def __repr__(self):
        return f"SmolVLAEvaluator(n_episodes={self.config.n_episodes})"
