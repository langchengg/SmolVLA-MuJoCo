"""
Innovation Point #3: Action Chunking Strategy Ablation Study.

Systematically evaluates the impact of different action chunk sizes on:
- Task success rate
- Motion smoothness (jerk, acceleration)
- Completion time
- Path efficiency
- Temporal ensemble vs standard execution

Output: Comprehensive ablation comparison across chunk sizes.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkingAblationConfig:
    """Chunking ablation study configuration."""
    chunk_sizes: list[int] = None
    n_episodes_per_config: int = 30
    results_dir: str = "./results/chunking_ablation"
    test_temporal_ensemble: bool = True

    def __post_init__(self):
        if self.chunk_sizes is None:
            self.chunk_sizes = [1, 2, 4, 8, 16]


class ChunkingAblation:
    """
    Action chunking ablation study.
    
    Evaluates the trade-off between:
    - Larger chunks → smoother, faster but less reactive
    - Smaller chunks → more reactive but potentially jerky
    
    Also compares temporal ensemble (blending overlapping predictions)
    vs standard chunk execution.
    """

    def __init__(self, config: ChunkingAblationConfig):
        self.config = config
        self._results_dir = Path(config.results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def run(self, model, env, evaluator) -> dict:
        """
        Run the chunking ablation study.
        
        Args:
            model: SmolVLAWrapper
            env: SmolVLAMuJoCoEnv
            evaluator: SmolVLAEvaluator
            
        Returns:
            Complete ablation results
        """
        from ..model.action_chunking import ActionChunkingStrategy, ChunkingConfig
        
        results = {}
        
        # 1. Standard chunking across different sizes
        logger.info("="*60)
        logger.info("Action Chunking Ablation Study")
        logger.info("="*60)
        
        for chunk_size in self.config.chunk_sizes:
            config_name = f"chunk_{chunk_size}"
            logger.info(f"\nTesting chunk_size={chunk_size}...")
            
            chunking = ActionChunkingStrategy(ChunkingConfig(
                chunk_size=chunk_size,
                temporal_ensemble=False,
            ))
            
            config_results = self._evaluate_with_chunking(
                model, env, evaluator, chunking, config_name
            )
            results[config_name] = config_results
            
            sr = config_results["metrics"]["success_rate"] * 100
            jerk = config_results["smoothness"]["jerk"]
            logger.info(f"  Success: {sr:.1f}% | Jerk: {jerk:.4f}")
        
        # 2. Temporal ensemble comparison
        if self.config.test_temporal_ensemble:
            logger.info("\n--- Temporal Ensemble Comparison ---")
            
            for chunk_size in [4, 8, 16]:
                config_name = f"chunk_{chunk_size}_ensemble"
                logger.info(f"\nTesting chunk_size={chunk_size} + temporal ensemble...")
                
                chunking = ActionChunkingStrategy(ChunkingConfig(
                    chunk_size=chunk_size,
                    temporal_ensemble=True,
                    ensemble_decay=0.9,
                ))
                
                config_results = self._evaluate_with_chunking(
                    model, env, evaluator, chunking, config_name
                )
                results[config_name] = config_results
                
                sr = config_results["metrics"]["success_rate"] * 100
                jerk = config_results["smoothness"]["jerk"]
                logger.info(f"  Success: {sr:.1f}% | Jerk: {jerk:.4f}")
        
        # 3. Analysis
        results["analysis"] = self._analyze_tradeoffs(results)
        
        # 4. Save
        self._save_results(results)
        
        return results

    def _evaluate_with_chunking(
        self, model, env, evaluator, chunking, config_name
    ) -> dict:
        """Evaluate with a specific chunking strategy."""
        from ..evaluation.evaluator import EpisodeResult
        
        episode_results = []
        all_smoothness = []
        
        model.eval_mode()
        
        for ep_idx in range(self.config.n_episodes_per_config):
            chunking.reset()
            obs = env.reset(seed=42 + ep_idx)
            
            instruction = obs.get("language_instruction", "complete the task")
            total_reward = 0.0
            done = False
            
            import time
            start_time = time.perf_counter()
            
            for step in range(300):  # max steps
                # Only predict if chunking needs it
                if chunking.needs_new_prediction():
                    raw_action = model.predict_action(
                        images=obs["image"],
                        language_instruction=instruction,
                        state=obs.get("state"),
                    )
                else:
                    raw_action = np.zeros((chunking.config.chunk_size, chunking.config.action_dim))
                
                # Process through chunking strategy
                action = chunking.process_prediction(raw_action)
                
                # Execute
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    done = True
                    break
            
            elapsed = time.perf_counter() - start_time
            
            # Check success
            success = False
            if isinstance(info, dict):
                success = info.get("success", info.get("is_success", total_reward > 0.5))
            
            episode_results.append(EpisodeResult(
                episode_id=ep_idx,
                success=bool(success),
                total_reward=total_reward,
                n_steps=step + 1,
                completion_time_s=elapsed,
                task_instruction=instruction,
            ))
            
            # Compute smoothness for this episode
            ep_smoothness = chunking.compute_smoothness_metrics()
            all_smoothness.append(ep_smoothness)
        
        # Aggregate metrics
        aggregate = evaluator.compute_aggregate_metrics(episode_results)
        
        # Aggregate smoothness
        avg_smoothness = {}
        if all_smoothness:
            for key in all_smoothness[0].keys():
                values = [s[key] for s in all_smoothness]
                avg_smoothness[key] = float(np.mean(values))
                avg_smoothness[f"{key}_std"] = float(np.std(values))
        
        return {
            "config_name": config_name,
            "chunk_size": chunking.config.chunk_size,
            "temporal_ensemble": chunking.config.temporal_ensemble,
            "metrics": aggregate,
            "smoothness": avg_smoothness,
            "n_model_calls": self._estimate_model_calls(
                chunking.config.chunk_size,
                aggregate.get("mean_steps", 300),
            ),
        }

    def _estimate_model_calls(self, chunk_size: int, mean_steps: float) -> float:
        """Estimate number of model forward passes per episode."""
        return mean_steps / chunk_size

    def _analyze_tradeoffs(self, results: dict) -> dict:
        """Analyze the success-vs-smoothness trade-off."""
        analysis = {
            "best_success": None,
            "best_smoothness": None,
            "best_balance": None,
            "pareto_optimal": [],
        }
        
        configs = {k: v for k, v in results.items() if k != "analysis" and isinstance(v, dict) and "metrics" in v}
        
        if not configs:
            return analysis
        
        # Best success rate
        best_sr_name = max(configs, key=lambda k: configs[k]["metrics"].get("success_rate", 0))
        analysis["best_success"] = {
            "config": best_sr_name,
            "success_rate": configs[best_sr_name]["metrics"]["success_rate"],
        }
        
        # Best smoothness (lowest jerk)
        best_smooth_name = min(
            configs, 
            key=lambda k: configs[k]["smoothness"].get("jerk", float("inf"))
        )
        analysis["best_smoothness"] = {
            "config": best_smooth_name,
            "jerk": configs[best_smooth_name]["smoothness"].get("jerk", 0),
        }
        
        # Best balance (maximize success * (1 - normalized_jerk))
        jerks = [v["smoothness"].get("jerk", 0) for v in configs.values()]
        max_jerk = max(jerks) if jerks else 1.0
        
        best_balanced = None
        best_score = -1
        
        for name, data in configs.items():
            sr = data["metrics"].get("success_rate", 0)
            jerk = data["smoothness"].get("jerk", 0)
            
            # Normalize jerk to [0, 1]
            norm_jerk = jerk / max_jerk if max_jerk > 0 else 0
            score = sr * (1 - 0.3 * norm_jerk)  # Weight smoothness 30%
            
            if score > best_score:
                best_score = score
                best_balanced = name
        
        analysis["best_balance"] = {
            "config": best_balanced,
            "score": best_score,
        }
        
        # Pareto front (no config is better in both metrics)
        for name, data in configs.items():
            sr = data["metrics"].get("success_rate", 0)
            jerk = data["smoothness"].get("jerk", float("inf"))
            
            is_pareto = True
            for other_name, other_data in configs.items():
                if other_name == name:
                    continue
                other_sr = other_data["metrics"].get("success_rate", 0)
                other_jerk = other_data["smoothness"].get("jerk", float("inf"))
                
                if other_sr >= sr and other_jerk <= jerk and (other_sr > sr or other_jerk < jerk):
                    is_pareto = False
                    break
            
            if is_pareto:
                analysis["pareto_optimal"].append(name)
        
        return analysis

    def _save_results(self, results: dict):
        """Save results and generate report."""
        output_path = self._results_dir / "chunking_ablation_results.json"
        serializable = self._make_serializable(results)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        
        self._generate_report(results)
        logger.info(f"Results saved to {self._results_dir}")

    def _generate_report(self, results: dict):
        """Generate markdown report."""
        report_path = self._results_dir / "chunking_ablation_report.md"
        
        lines = [
            "# Action Chunking Ablation Study\n",
            "## Results\n",
            "| Config | Chunk Size | Ensemble | Success Rate | Jerk ↓ | Steps | Model Calls |",
            "|--------|-----------|----------|-------------|--------|-------|-------------|",
        ]
        
        for name, data in results.items():
            if name == "analysis" or not isinstance(data, dict) or "metrics" not in data:
                continue
            
            cs = data.get("chunk_size", "?")
            ens = "✓" if data.get("temporal_ensemble", False) else "✗"
            sr = data["metrics"].get("success_rate", 0) * 100
            jerk = data["smoothness"].get("jerk", 0)
            steps = data["metrics"].get("mean_steps", 0)
            calls = data.get("n_model_calls", 0)
            
            lines.append(
                f"| {name:<25} | {cs:<9} | {ens:<8} | {sr:.1f}% | "
                f"{jerk:.4f} | {steps:.0f} | {calls:.0f} |"
            )
        
        # Analysis
        analysis = results.get("analysis", {})
        lines.extend([
            "\n## Analysis\n",
            f"- **Best success rate**: {analysis.get('best_success', {}).get('config', 'N/A')}",
            f"- **Smoothest execution**: {analysis.get('best_smoothness', {}).get('config', 'N/A')}",
            f"- **Best balance**: {analysis.get('best_balance', {}).get('config', 'N/A')}",
            f"- **Pareto optimal configs**: {', '.join(analysis.get('pareto_optimal', []))}",
        ])
        
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: ChunkingAblation._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ChunkingAblation._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def __repr__(self):
        return f"ChunkingAblation(sizes={self.config.chunk_sizes})"
