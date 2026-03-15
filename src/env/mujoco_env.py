"""
MuJoCo Environment Wrapper for SmolVLA Evaluation.

Provides a unified interface wrapping LIBERO / custom MuJoCo environments
for VLA model training and evaluation with visual observation support.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Environment configuration."""
    env_type: str = "libero"
    task_suite: str = "libero_object"
    task_name: Optional[str] = None
    render_mode: str = "rgb_array"
    image_size: tuple[int, int] = (224, 224)
    max_episode_steps: int = 300
    camera_names: list[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    seed: int = 42


class SmolVLAMuJoCoEnv:
    """
    Unified MuJoCo environment wrapper for SmolVLA evaluation.
    
    Supports LIBERO benchmark tasks and custom MuJoCo manipulation scenes.
    Handles visual observation capture, action execution, and perturbation injection.
    
    Architecture:
        SmolVLA Model → Action → [SmolVLAMuJoCoEnv] → MuJoCo Physics → Obs (RGB + proprio)
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self._env = None
        self._perturbation_engine = None
        self._step_count = 0
        self._episode_count = 0
        self._setup_environment()

    def _setup_environment(self):
        """Initialize the underlying MuJoCo environment."""
        if self.config.env_type == "libero":
            self._setup_libero()
        elif self.config.env_type == "metaworld":
            self._setup_metaworld()
        else:
            self._setup_custom_mujoco()

    def _setup_libero(self):
        """Setup LIBERO benchmark environment via LeRobot."""
        try:
            # LeRobot provides LIBERO env integration
            from lerobot.common.envs.factory import make_env
            
            env_config = {
                "type": "libero",
                "task": self.config.task_suite,
                "obs_type": "pixels_agent_pos",
                "render_mode": self.config.render_mode,
            }
            self._env = make_env(env_config)
            logger.info(f"LIBERO environment created: {self.config.task_suite}")
            
        except ImportError:
            logger.warning("LeRobot not installed. Falling back to standalone LIBERO setup.")
            self._setup_libero_standalone()

    def _setup_libero_standalone(self):
        """Setup LIBERO without LeRobot dependency."""
        try:
            import libero.envs
            from libero.libero import benchmark
            
            bench = benchmark.get_benchmark(self.config.task_suite)
            task_id = 0 if self.config.task_name is None else bench.get_task_id(self.config.task_name)
            task = bench.get_task(task_id)
            
            env_args = {
                "bddl_file_name": task.bddl_file,
                "camera_heights": self.config.image_size[0],
                "camera_widths": self.config.image_size[1],
                "has_renderer": False,
                "has_offscreen_renderer": True,
                "camera_names": self.config.camera_names,
            }
            
            from libero.libero.envs import OffScreenRenderEnv
            self._env = OffScreenRenderEnv(**env_args)
            logger.info(f"LIBERO standalone env created: task_id={task_id}")
            
        except ImportError:
            logger.warning("LIBERO not installed. Creating minimal MuJoCo env.")
            self._setup_custom_mujoco()

    def _setup_metaworld(self):
        """Setup Meta-World environment."""
        try:
            import metaworld
            
            ml1 = metaworld.ML1(self.config.task_name or "reach-v2")
            env = ml1.train_classes[self.config.task_name or "reach-v2"]()
            task = ml1.train_tasks[0]
            env.set_task(task)
            self._env = env
            logger.info(f"MetaWorld environment created: {self.config.task_name}")
            
        except ImportError:
            logger.warning("MetaWorld not installed. Falling back to custom MuJoCo.")
            self._setup_custom_mujoco()

    def _setup_custom_mujoco(self):
        """Fallback: create a custom Gymnasium MuJoCo manipulation environment."""
        import mujoco
        
        # Use built-in Gymnasium Fetch environments as fallback
        env_id = "FetchReach-v3"
        try:
            self._env = gym.make(
                env_id,
                render_mode=self.config.render_mode,
                max_episode_steps=self.config.max_episode_steps,
            )
        except gym.error.NameNotFound:
            # Ultimate fallback: simple reacher
            self._env = gym.make(
                "Reacher-v4",
                render_mode=self.config.render_mode,
                max_episode_steps=self.config.max_episode_steps,
            )
        logger.info(f"Custom MuJoCo environment created.")

    def set_perturbation_engine(self, engine):
        """Attach a visual perturbation engine for robustness testing."""
        from .visual_perturbation import VisualPerturbationEngine
        assert isinstance(engine, VisualPerturbationEngine)
        self._perturbation_engine = engine

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        """
        Reset environment and return initial observation.
        
        Returns:
            dict with keys:
                - "image": dict of camera_name → (H, W, 3) uint8 arrays
                - "state": (D,) proprioceptive state vector
                - "language_instruction": str task description
        """
        self._step_count = 0
        self._episode_count += 1

        if hasattr(self._env, 'reset'):
            obs_raw = self._env.reset(seed=seed) if seed else self._env.reset()
            # Handle tuple return (obs, info)
            if isinstance(obs_raw, tuple):
                obs_raw, info = obs_raw
            else:
                info = {}
        else:
            obs_raw = self._env.reset()
            info = {}

        return self._process_observation(obs_raw, info)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: (action_dim,) array of joint commands + gripper
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        
        # Clip action to valid range
        if hasattr(self._env, 'action_space'):
            env_action_dim = self._env.action_space.shape[0]
            if action.shape[0] > env_action_dim:
                action = action[:env_action_dim]
            elif action.shape[0] < env_action_dim:
                padded_action = np.zeros(env_action_dim, dtype=action.dtype)
                padded_action[:action.shape[0]] = action
                action = padded_action
                
            action = np.clip(action, self._env.action_space.low, self._env.action_space.high)
        
        result = self._env.step(action)
        
        if len(result) == 5:
            obs_raw, reward, terminated, truncated, info = result
        else:
            obs_raw, reward, done, info = result
            terminated = done
            truncated = self._step_count >= self.config.max_episode_steps
        
        obs = self._process_observation(obs_raw, info)
        
        return obs, reward, terminated, truncated, info

    def _process_observation(self, obs_raw: Any, info: dict) -> dict[str, Any]:
        """Convert raw observation to standardized format with optional perturbation."""
        obs = {}
        
        # Extract images
        images = {}
        if isinstance(obs_raw, dict):
            for cam in self.config.camera_names:
                key_variants = [cam, f"{cam}_image", f"observation.images.{cam}"]
                for key in key_variants:
                    if key in obs_raw:
                        img = np.array(obs_raw[key], dtype=np.uint8)
                        if img.ndim == 3 and img.shape[-1] == 3:
                            images[cam] = img
                        break
            
            # Extract proprioceptive state
            state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                         "agent_pos", "observation.state", "proprio"]
            state_parts = []
            for key in state_keys:
                if key in obs_raw:
                    val = np.array(obs_raw[key], dtype=np.float32).flatten()
                    state_parts.append(val)
            
            obs["state"] = np.concatenate(state_parts) if state_parts else np.zeros(7, dtype=np.float32)
        else:
            # If obs is just an array (e.g., from simple Gym envs)
            obs["state"] = np.array(obs_raw, dtype=np.float32).flatten()
        
        # If no images from observation, try rendering
        if not images:
            try:
                img = self._env.render()
                if img is not None:
                    images["agentview"] = np.array(img, dtype=np.uint8)
            except Exception:
                # Generate a placeholder
                h, w = self.config.image_size
                images["agentview"] = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply visual perturbations if engine is set
        if self._perturbation_engine is not None:
            images = {k: self._perturbation_engine.apply(v) for k, v in images.items()}
        
        obs["image"] = images
        obs["language_instruction"] = info.get("language_instruction", 
                                                info.get("task_description", "manipulate the object"))
        
        return obs

    def render(self) -> Optional[np.ndarray]:
        """Render current frame."""
        if hasattr(self._env, 'render'):
            return self._env.render()
        return None

    def close(self):
        """Cleanup."""
        if self._env is not None:
            self._env.close()

    @property
    def action_space(self):
        return self._env.action_space if self._env else None

    @property
    def observation_space(self):
        return self._env.observation_space if self._env else None

    def get_task_description(self) -> str:
        """Return the natural language task description."""
        if hasattr(self._env, 'language_instruction'):
            return self._env.language_instruction
        return "complete the manipulation task"

    def __repr__(self):
        return (f"SmolVLAMuJoCoEnv(type={self.config.env_type}, "
                f"task={self.config.task_suite}, episodes={self._episode_count})")
