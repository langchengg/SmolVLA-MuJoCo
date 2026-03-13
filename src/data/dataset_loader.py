"""
Dataset Loader for LIBERO and LeRobot Datasets.

Handles downloading, loading, and preprocessing of robotic manipulation
datasets in LeRobot format for SmolVLA fine-tuning.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    repo_id: str = "lerobot/libero_object_no_noops"
    split: str = "train"
    batch_size: int = 32
    num_workers: int = 4
    image_size: tuple[int, int] = (224, 224)
    prefetch_factor: int = 2
    shuffle: bool = True
    max_episodes: Optional[int] = None
    cache_dir: Optional[str] = None


class LiberoDatasetLoader:
    """
    Loader for LIBERO datasets via the LeRobot/HuggingFace Hub.
    
    LIBERO provides 4 benchmark suites:
        - LIBERO-Spatial: spatial arrangement variations (10 tasks)
        - LIBERO-Object: object type variations (10 tasks)
        - LIBERO-Goal: goal variations (10 tasks)  
        - LIBERO-100: 100 diverse tasks for large-scale evaluation
    
    Each episode contains:
        - RGB images from workspace and wrist cameras
        - Robot proprioception (joint positions, EEF pose, gripper state)
        - Language instruction describing the task
        - Expert demonstration actions
    """

    # Available LIBERO datasets on HuggingFace Hub
    AVAILABLE_DATASETS = {
        "libero_spatial": "lerobot/libero_spatial_no_noops",
        "libero_object": "lerobot/libero_object_no_noops",
        "libero_goal": "lerobot/libero_goal_no_noops",
        "libero_10": "lerobot/libero_10_no_noops",
        "libero_90": "lerobot/libero_90_no_noops",
        # Additional LeRobot community datasets
        "aloha_mobile": "lerobot/aloha_mobile_cabinet",
        "pusht": "lerobot/pusht",
    }

    def __init__(self, config: DatasetConfig):
        self.config = config
        self._dataset = None

    def load(self) -> "LeRobotDatasetAdapter":
        """Load dataset from HuggingFace Hub."""
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            logger.info(f"Loading dataset: {self.config.repo_id}")
            
            kwargs = {"repo_id": self.config.repo_id}
            if self.config.split:
                kwargs["split"] = self.config.split
            
            self._dataset = LeRobotDataset(**kwargs)
            
            logger.info(
                f"Dataset loaded: {len(self._dataset)} samples, "
                f"features: {list(self._dataset.features.keys()) if hasattr(self._dataset, 'features') else 'N/A'}"
            )
            
            return LeRobotDatasetAdapter(self._dataset, self.config)
            
        except ImportError:
            logger.warning("LeRobot not installed. Using HuggingFace datasets fallback.")
            return self._load_hf_fallback()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.info("Creating synthetic demo dataset for testing...")
            return self._create_synthetic_dataset()

    def _load_hf_fallback(self) -> "LeRobotDatasetAdapter":
        """Fallback: load dataset directly from HuggingFace datasets library."""
        from datasets import load_dataset
        
        ds = load_dataset(self.config.repo_id, split=self.config.split)
        return LeRobotDatasetAdapter(ds, self.config)

    def _create_synthetic_dataset(self) -> "LeRobotDatasetAdapter":
        """Create a synthetic dataset for testing when real data is unavailable."""
        logger.info("Generating synthetic demonstration dataset...")
        
        n_episodes = 50
        episode_length = 100
        
        data = []
        for ep in range(n_episodes):
            for step in range(episode_length):
                sample = {
                    "observation.images.agentview": np.random.randint(
                        0, 255, (224, 224, 3), dtype=np.uint8
                    ),
                    "observation.images.robot0_eye_in_hand": np.random.randint(
                        0, 255, (224, 224, 3), dtype=np.uint8
                    ),
                    "observation.state": np.random.randn(7).astype(np.float32),
                    "action": np.random.randn(7).astype(np.float32) * 0.1,
                    "language_instruction": f"pick up the object {ep}",
                    "episode_index": ep,
                    "frame_index": step,
                }
                data.append(sample)
        
        return LeRobotDatasetAdapter(data, self.config, is_synthetic=True)

    def get_dataloader(self, dataset: Optional["LeRobotDatasetAdapter"] = None) -> DataLoader:
        """Create a PyTorch DataLoader."""
        if dataset is None:
            dataset = self.load()
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: list[dict]) -> dict:
        """Custom collate function for heterogeneous data."""
        collated = {}
        keys = batch[0].keys()
        
        for key in keys:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                collated[key] = torch.from_numpy(np.stack(values))
            elif isinstance(values[0], str):
                collated[key] = values  # Keep as list of strings
            elif isinstance(values[0], (int, float)):
                collated[key] = torch.tensor(values)
            else:
                collated[key] = values
        
        return collated

    @classmethod
    def list_available_datasets(cls) -> dict[str, str]:
        """List available pre-defined datasets."""
        return cls.AVAILABLE_DATASETS.copy()


class LeRobotDatasetAdapter(Dataset):
    """
    PyTorch Dataset adapter for LeRobot datasets.
    
    Handles image preprocessing, normalization, and format conversion
    to match SmolVLA's expected input format.
    """

    def __init__(self, raw_dataset, config: DatasetConfig, is_synthetic: bool = False):
        self._raw = raw_dataset
        self.config = config
        self.is_synthetic = is_synthetic
        
        # Image preprocessing
        self._image_transforms = self._build_transforms()

    def _build_transforms(self):
        """Build image preprocessing pipeline."""
        import torchvision.transforms as T
        
        return T.Compose([
            T.ToPILImage(),
            T.Resize(self.config.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        if isinstance(self._raw, list):
            return len(self._raw)
        return len(self._raw)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if isinstance(self._raw, list):
            sample = self._raw[idx]
        else:
            sample = self._raw[idx]
        
        processed = {}
        
        # Process images
        for key in list(sample.keys()):
            if "image" in key.lower() or key.startswith("observation.images"):
                img = sample[key]
                if isinstance(img, np.ndarray):
                    processed[key] = self._image_transforms(img)
                elif isinstance(img, torch.Tensor):
                    if img.dtype == torch.uint8:
                        processed[key] = self._image_transforms(img.numpy())
                    else:
                        processed[key] = img
                else:
                    processed[key] = img
        
        # Process state / proprioception
        for key in ["observation.state", "state", "robot0_eef_pos"]:
            if key in sample:
                val = sample[key]
                if isinstance(val, np.ndarray):
                    processed[key] = torch.from_numpy(val).float()
                elif isinstance(val, torch.Tensor):
                    processed[key] = val.float()
                else:
                    processed[key] = torch.tensor(val, dtype=torch.float32)
        
        # Process action
        if "action" in sample:
            action = sample["action"]
            if isinstance(action, np.ndarray):
                processed["action"] = torch.from_numpy(action).float()
            elif isinstance(action, torch.Tensor):
                processed["action"] = action.float()
            else:
                processed["action"] = torch.tensor(action, dtype=torch.float32)
        
        # Language instruction
        if "language_instruction" in sample:
            processed["language_instruction"] = str(sample["language_instruction"])
        elif "task" in sample:
            processed["language_instruction"] = str(sample["task"])
        else:
            processed["language_instruction"] = "complete the task"
        
        # Metadata
        for key in ["episode_index", "frame_index"]:
            if key in sample:
                processed[key] = int(sample[key])
        
        return processed

    def get_episode(self, episode_idx: int) -> list[dict]:
        """Get all frames for a specific episode."""
        frames = []
        for i in range(len(self)):
            sample = self[i]
            if sample.get("episode_index") == episode_idx:
                frames.append(sample)
        return frames

    def get_unique_instructions(self) -> list[str]:
        """Get unique language instructions in the dataset."""
        instructions = set()
        for i in range(min(len(self), 1000)):  # Sample first 1000
            sample = self[i]
            if "language_instruction" in sample:
                instructions.add(sample["language_instruction"])
        return sorted(instructions)

    def __repr__(self):
        return (f"LeRobotDatasetAdapter(repo_id={self.config.repo_id}, "
                f"len={len(self)}, synthetic={self.is_synthetic})")
