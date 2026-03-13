"""
SmolVLA Fine-tuning Pipeline.

Implements end-to-end fine-tuning of SmolVLA on LIBERO datasets using:
- LoRA parameter-efficient fine-tuning
- Multi-task training with configurable task sampling
- Language instruction augmentation
- WandB experiment tracking
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    """Fine-tuning configuration."""
    # Training
    num_epochs: int = 50
    max_steps: int = 20000
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Precision
    use_bf16: bool = True
    use_fp16: bool = False
    
    # Checkpointing
    save_steps: int = 2000
    eval_steps: int = 1000
    save_total_limit: int = 3
    output_dir: str = "./results/checkpoints"
    
    # Logging
    log_every_n_steps: int = 50
    use_wandb: bool = True
    wandb_project: str = "smolvla-finetune"
    wandb_run_name: Optional[str] = None
    
    # Data augmentation
    use_language_augmentation: bool = True
    language_aug_probability: float = 0.3
    
    # Multi-task
    multi_task: bool = True
    task_sampling: str = "uniform"


class SmolVLAFineTuner:
    """
    End-to-end fine-tuning pipeline for SmolVLA.
    
    Pipeline:
        1. Load base SmolVLA model (with optional LoRA)
        2. Load LIBERO dataset via LeRobot format
        3. Configure optimizer + scheduler
        4. Training loop with evaluation
        5. Save checkpoints + logs
    
    Usage (Script):
        >>> config = FineTuneConfig(max_steps=20000, use_wandb=True)
        >>> trainer = SmolVLAFineTuner(config)
        >>> trainer.setup(model_name="HuggingFaceTB/SmolVLA-base", dataset_repo="lerobot/libero_object_image")
        >>> trainer.train()
    
    Usage (LeRobot CLI):
        $ lerobot-train --policy=smolvla --dataset.repo_id=lerobot/libero_object_image
    """

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader = None
        self.eval_dataloader = None
        self._augmentor = None
        self._global_step = 0
        self._best_eval_loss = float("inf")
        self._wandb_run = None
        self._scaler = None

    def setup(
        self,
        model_name: str = "HuggingFaceTB/SmolVLA-base",
        dataset_repo: str = "lerobot/libero_object_image",
    ):
        """Initialize all components."""
        self._setup_model(model_name)
        self._setup_data(dataset_repo)
        self._setup_optimizer()
        self._setup_logging()
        self._setup_precision()
        
        logger.info("Fine-tuning pipeline setup complete.")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Dataset: {dataset_repo}")
        logger.info(f"  Max steps: {self.config.max_steps}")
        logger.info(f"  Batch size: {self.config.batch_size} × {self.config.gradient_accumulation_steps} accumulation")
        logger.info(f"  Learning rate: {self.config.learning_rate}")

    def _setup_model(self, model_name: str):
        """Load and prepare the model."""
        from ..model.smolvla_wrapper import SmolVLAWrapper, SmolVLAConfig
        
        model_config = SmolVLAConfig(
            model_name=model_name,
            use_lora=True,
        )
        
        self.model = SmolVLAWrapper(model_config)
        self.model.load()
        self.model.train_mode()

    def _setup_data(self, dataset_repo: str):
        """Load training and evaluation datasets."""
        from ..data.dataset_loader import LiberoDatasetLoader, DatasetConfig
        
        # Training data
        train_config = DatasetConfig(
            repo_id=dataset_repo,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        train_loader = LiberoDatasetLoader(train_config)
        self.dataloader = train_loader.get_dataloader()
        
        # Evaluation data (last 10% of training data)
        eval_config = DatasetConfig(
            repo_id=dataset_repo,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        eval_loader = LiberoDatasetLoader(eval_config)
        eval_dataset = eval_loader.load()
        
        # Simple split: use last 10% for eval
        n_eval = max(1, len(eval_dataset) // 10)
        eval_indices = list(range(len(eval_dataset) - n_eval, len(eval_dataset)))
        eval_subset = torch.utils.data.Subset(eval_dataset, eval_indices)
        self.eval_dataloader = torch.utils.data.DataLoader(
            eval_subset, batch_size=self.config.batch_size, shuffle=False
        )
        
        # Language augmentation
        if self.config.use_language_augmentation:
            from ..data.language_augmentation import LanguageAugmentor
            self._augmentor = LanguageAugmentor()

    def _setup_optimizer(self):
        """Configure optimizer and learning rate scheduler."""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Warmup + Cosine schedule
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps - self.config.warmup_steps,
            eta_min=1e-6,
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps],
        )

    def _setup_precision(self):
        """Setup mixed precision training."""
        if self.config.use_bf16 and torch.cuda.is_available():
            self._scaler = None  # bf16 doesn't need GradScaler
        elif self.config.use_fp16 and torch.cuda.is_available():
            self._scaler = torch.amp.GradScaler()
        else:
            self._scaler = None

    def _setup_logging(self):
        """Setup experiment tracking."""
        if self.config.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config={
                        "max_steps": self.config.max_steps,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate,
                        "use_lora": True,
                        "gradient_accumulation": self.config.gradient_accumulation_steps,
                    },
                )
            except ImportError:
                logger.warning("wandb not installed. Logging to console only.")
                self._wandb_run = None

    def train(self):
        """
        Main training loop.
        
        Returns:
            dict with training metrics
        """
        logger.info("=" * 60)
        logger.info("Starting SmolVLA fine-tuning")
        logger.info("=" * 60)
        
        metrics_history = []
        epoch = 0
        
        while self._global_step < self.config.max_steps:
            epoch += 1
            epoch_metrics = self._train_epoch(epoch)
            metrics_history.append(epoch_metrics)
            
            if self._global_step >= self.config.max_steps:
                break
        
        # Final save
        self._save_checkpoint("final")
        
        logger.info("=" * 60)
        logger.info(f"Training complete. Total steps: {self._global_step}")
        logger.info("=" * 60)
        
        if self._wandb_run:
            import wandb
            wandb.finish()
        
        return {
            "total_steps": self._global_step,
            "total_epochs": epoch,
            "best_eval_loss": self._best_eval_loss,
            "metrics_history": metrics_history,
        }

    def _train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train_mode()
        
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            if self._global_step >= self.config.max_steps:
                break
            
            # Apply language augmentation
            if self._augmentor and self.config.use_language_augmentation:
                batch = self._augment_batch(batch)
            
            # Forward pass
            loss = self._training_step(batch)
            
            if loss is not None:
                epoch_loss += loss
                n_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self._optimization_step()
                self._global_step += 1
                
                # Logging
                if self._global_step % self.config.log_every_n_steps == 0:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    lr = self.optimizer.param_groups[0]["lr"]
                    self._log_metrics({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step": self._global_step,
                    })
                    logger.info(
                        f"Step {self._global_step}/{self.config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )
                
                # Evaluation
                if self._global_step % self.config.eval_steps == 0:
                    eval_metrics = self._evaluate()
                    self._log_metrics(eval_metrics)
                
                # Checkpointing
                if self._global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self._global_step}")
        
        return {
            "epoch": epoch,
            "avg_loss": epoch_loss / max(n_batches, 1),
            "n_batches": n_batches,
        }

    def _training_step(self, batch: dict) -> Optional[float]:
        """Single training step."""
        try:
            # Determine precision context
            if self.config.use_bf16 and torch.cuda.is_available():
                ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            elif self.config.use_fp16 and torch.cuda.is_available():
                ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()
            
            with ctx:
                outputs = self.model(batch)
                
                if isinstance(outputs, dict):
                    loss = outputs.get("loss", outputs.get("total_loss"))
                elif isinstance(outputs, torch.Tensor):
                    loss = outputs
                else:
                    loss = None
                
                if loss is not None:
                    loss = loss / self.config.gradient_accumulation_steps
                    
                    if self._scaler:
                        self._scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    return loss.item()
        except Exception as e:
            logger.warning(f"Training step error: {e}")
        
        return None

    def _optimization_step(self):
        """Optimizer step with gradient clipping."""
        if self.config.max_grad_norm > 0:
            if self._scaler:
                self._scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
        
        if self._scaler:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def _evaluate(self) -> dict:
        """Run evaluation."""
        self.model.eval_mode()
        
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.eval_dataloader:
            try:
                outputs = self.model(batch)
                if isinstance(outputs, dict) and "loss" in outputs:
                    total_loss += outputs["loss"].item()
                    n_batches += 1
            except Exception:
                pass
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Track best
        if avg_loss < self._best_eval_loss:
            self._best_eval_loss = avg_loss
            self._save_checkpoint("best")
        
        self.model.train_mode()
        
        logger.info(f"Eval Loss: {avg_loss:.4f} (Best: {self._best_eval_loss:.4f})")
        
        return {
            "eval/loss": avg_loss,
            "eval/best_loss": self._best_eval_loss,
        }

    def _augment_batch(self, batch: dict) -> dict:
        """Apply language augmentation to batch."""
        if "language_instruction" not in batch:
            return batch
        
        import random
        augmented_instructions = []
        for inst in batch["language_instruction"]:
            if random.random() < self.config.language_aug_probability:
                variants = self._augmentor.generate_variants(inst, n=1)
                augmented_instructions.append(variants[0] if variants else inst)
            else:
                augmented_instructions.append(inst)
        
        batch["language_instruction"] = augmented_instructions
        return batch

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = Path(self.config.output_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_checkpoint(str(path))
        
        # Save training state
        state = {
            "global_step": self._global_step,
            "best_eval_loss": self._best_eval_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(state, path / "training_state.pt")
        
        logger.info(f"Checkpoint saved: {path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        output_dir = Path(self.config.output_dir)
        if not output_dir.exists():
            return
        
        checkpoints = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            old_ckpt = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_ckpt)
            logger.info(f"Removed old checkpoint: {old_ckpt}")

    def _log_metrics(self, metrics: dict):
        """Log metrics to all configured loggers."""
        if self._wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=self._global_step)
            except Exception:
                pass

    def __repr__(self):
        return (
            f"SmolVLAFineTuner(steps={self._global_step}/{self.config.max_steps}, "
            f"best_loss={self._best_eval_loss:.4f})"
        )


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    """CLI entry point for fine-tuning."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="SmolVLA Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/finetune_libero.yaml")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolVLA-base")
    parser.add_argument("--dataset", type=str, default="lerobot/libero_object_image")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    # Load config from YAML
    config_dict = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
            if "training" in yaml_config:
                config_dict.update(yaml_config["training"])
    
    # Override with CLI args
    if args.max_steps:
        config_dict["max_steps"] = args.max_steps
    if args.batch_size:
        config_dict["batch_size"] = args.batch_size
    if args.lr:
        config_dict["learning_rate"] = args.lr
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    if args.no_wandb:
        config_dict["use_wandb"] = False
    
    config = FineTuneConfig(**{k: v for k, v in config_dict.items() if hasattr(FineTuneConfig, k)})
    
    trainer = SmolVLAFineTuner(config)
    trainer.setup(model_name=args.model, dataset_repo=args.dataset)
    result = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"   Total steps: {result['total_steps']}")
    print(f"   Best eval loss: {result['best_eval_loss']:.4f}")


if __name__ == "__main__":
    main()
