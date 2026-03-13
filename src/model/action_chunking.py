"""
Action Chunking Strategy for VLA Models.

Implements different action prediction strategies:
- Single-step prediction (chunk_size=1)
- Multi-step chunked prediction (chunk_size=4,8,16)
- Temporal ensemble of overlapping chunks

This is the core component of Innovation Point #3.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Action chunking configuration."""
    chunk_size: int = 1
    action_dim: int = 7
    temporal_ensemble: bool = False      # Average overlapping predictions
    ensemble_decay: float = 0.9          # Exponential decay for temporal ensemble
    interpolation: str = "linear"        # "linear", "cubic", "none"


class ActionChunkingStrategy:
    """
    Implements various action chunking strategies for VLA model output.
    
    Action chunking predicts multiple future actions at once instead of one,
    which can improve:
    - Motion smoothness (fewer jerky corrections)
    - Execution speed (amortized inference cost)
    - Temporal coherence (planned action sequences)
    
    But may hurt:
    - Reactivity (delayed response to perturbations)
    - Accuracy in contact-rich tasks
    
    This class systematically evaluates these trade-offs.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._action_buffer = []     # Buffer for temporal ensemble
        self._buffer_weights = []    # Weights for ensemble averaging
        self._step_in_chunk = 0      # Current position in active chunk
        self._current_chunk = None   # Currently active action chunk
        self._action_history = []    # Full execution history for analysis
        
    def reset(self):
        """Reset internal state for a new episode."""
        self._action_buffer.clear()
        self._buffer_weights.clear()
        self._step_in_chunk = 0
        self._current_chunk = None
        self._action_history.clear()

    def process_prediction(
        self, 
        predicted_actions: np.ndarray,
        model_predict_fn: callable = None,
        observation: dict = None,
    ) -> np.ndarray:
        """
        Process model prediction and return the next action to execute.
        
        Args:
            predicted_actions: (chunk_size, action_dim) or (action_dim,) array
            model_predict_fn: Optional callable to get new predictions
            observation: Optional current observation for re-prediction
            
        Returns:
            (action_dim,) action to execute at this timestep
        """
        # Reshape if single-step
        if predicted_actions.ndim == 1:
            predicted_actions = predicted_actions.reshape(1, -1)
        
        chunk_size = predicted_actions.shape[0]
        
        if self.config.temporal_ensemble and chunk_size > 1:
            action = self._temporal_ensemble_step(predicted_actions)
        elif chunk_size > 1:
            action = self._chunk_step(predicted_actions, model_predict_fn, observation)
        else:
            action = predicted_actions[0]
        
        # Record history
        self._action_history.append(action.copy())
        
        return action

    def _chunk_step(
        self,
        predicted_actions: np.ndarray,
        model_predict_fn: callable = None,
        observation: dict = None,
    ) -> np.ndarray:
        """Standard chunk execution: use all actions in sequence, then re-predict."""
        # Need new chunk?
        if self._current_chunk is None or self._step_in_chunk >= self.config.chunk_size:
            self._current_chunk = predicted_actions[:self.config.chunk_size]
            self._step_in_chunk = 0
        
        action = self._current_chunk[self._step_in_chunk]
        self._step_in_chunk += 1
        
        return action

    def _temporal_ensemble_step(self, predicted_actions: np.ndarray) -> np.ndarray:
        """
        Temporal ensemble: blend overlapping chunk predictions.
        
        At each timestep, we may have predictions from multiple previous chunks.
        We weight more recent predictions higher (exponential decay).
        """
        chunk_size = predicted_actions.shape[0]
        
        # Add new predictions to buffer
        for i in range(chunk_size):
            if len(self._action_buffer) <= i:
                self._action_buffer.append([])
                self._buffer_weights.append([])
            
            weight = self.config.ensemble_decay ** i  # Closer predictions get higher weight
            self._action_buffer[0].append(predicted_actions[i])
            self._buffer_weights[0].append(weight)
        
        # Get the current action by weighted average
        if self._action_buffer and self._action_buffer[0]:
            actions = np.array(self._action_buffer[0])
            weights = np.array(self._buffer_weights[0])
            weights = weights / weights.sum()
            action = np.average(actions, weights=weights, axis=0)
            
            # Shift buffer
            self._action_buffer.pop(0)
            self._buffer_weights.pop(0)
        else:
            action = predicted_actions[0]
        
        return action

    def needs_new_prediction(self) -> bool:
        """Check if a new model prediction is needed."""
        if self.config.chunk_size <= 1:
            return True
        
        if self.config.temporal_ensemble:
            return True  # Always predict for ensemble
        
        return (
            self._current_chunk is None or 
            self._step_in_chunk >= self.config.chunk_size
        )

    def compute_smoothness_metrics(self) -> dict[str, float]:
        """
        Compute action smoothness metrics from execution history.
        
        Returns:
            dict with:
                - jerk: Mean L2 norm of action jerk (3rd derivative)
                - acceleration: Mean L2 norm of action acceleration
                - velocity_variance: Variance of action velocities
                - direction_changes: Number of direction reversals
        """
        if len(self._action_history) < 4:
            return {
                "jerk": 0.0, "acceleration": 0.0,
                "velocity_variance": 0.0, "direction_changes": 0,
            }
        
        actions = np.array(self._action_history)
        
        # Velocity (1st derivative)
        velocity = np.diff(actions, axis=0)
        
        # Acceleration (2nd derivative)
        acceleration = np.diff(velocity, axis=0)
        
        # Jerk (3rd derivative)
        jerk = np.diff(acceleration, axis=0)
        
        # Direction changes
        signs = np.sign(velocity)
        direction_changes = np.sum(np.abs(np.diff(signs, axis=0)) > 0)
        
        metrics = {
            "jerk": float(np.mean(np.linalg.norm(jerk, axis=1))),
            "acceleration": float(np.mean(np.linalg.norm(acceleration, axis=1))),
            "velocity_variance": float(np.var(np.linalg.norm(velocity, axis=1))),
            "direction_changes": int(direction_changes),
            "path_length": float(np.sum(np.linalg.norm(velocity, axis=1))),
            "action_magnitude_mean": float(np.mean(np.linalg.norm(actions, axis=1))),
            "action_magnitude_std": float(np.std(np.linalg.norm(actions, axis=1))),
        }
        
        return metrics

    @staticmethod
    def get_ablation_configs() -> list[ChunkingConfig]:
        """Get all configurations for the ablation study."""
        configs = []
        
        # Standard chunking sweep
        for chunk_size in [1, 2, 4, 8, 16]:
            configs.append(ChunkingConfig(
                chunk_size=chunk_size,
                temporal_ensemble=False,
            ))
        
        # With temporal ensemble
        for chunk_size in [4, 8, 16]:
            configs.append(ChunkingConfig(
                chunk_size=chunk_size,
                temporal_ensemble=True,
                ensemble_decay=0.9,
            ))
        
        # Different ensemble decay rates
        for decay in [0.5, 0.7, 0.9, 0.95]:
            configs.append(ChunkingConfig(
                chunk_size=8,
                temporal_ensemble=True,
                ensemble_decay=decay,
            ))
        
        return configs

    def __repr__(self):
        return (
            f"ActionChunkingStrategy(chunk_size={self.config.chunk_size}, "
            f"ensemble={self.config.temporal_ensemble}, "
            f"history_len={len(self._action_history)})"
        )
