"""
Innovation Point #2: Visual Perturbation Robustness Analysis.

Systematically evaluates VLA model robustness under:
- Brightness variations
- Contrast changes
- Gaussian noise
- Camera viewpoint shifts
- Background texture changes
- Motion blur

Compares fine-tuned vs zero-shot model performance.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RobustnessConfig:
    """Robustness analysis configuration."""
    n_episodes_per_condition: int = 20
    results_dir: str = "./results/robustness"
    compare_zero_shot: bool = True


class RobustnessAnalyzer:
    """
    Visual perturbation robustness analyzer.
    
    Generates a comprehensive heatmap of model performance under
    different types and levels of visual perturbations.
    
    Output:
        - Perturbation robustness heatmap
        - Fine-tuned vs zero-shot comparison
        - Critical failure analysis
    """

    def __init__(self, config: RobustnessConfig):
        self.config = config
        self._results_dir = Path(config.results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        model,
        env,
        evaluator,
        zero_shot_model=None,
    ) -> dict:
        """
        Run complete robustness analysis.
        
        Args:
            model: Fine-tuned SmolVLAWrapper
            env: SmolVLAMuJoCoEnv
            evaluator: SmolVLAEvaluator
            zero_shot_model: Optional base (zero-shot) model for comparison
            
        Returns:
            Complete results dict
        """
        from ..env.visual_perturbation import VisualPerturbationEngine, PerturbationType
        
        perturbation_configs = VisualPerturbationEngine.get_all_perturbation_configs()
        
        results = {
            "finetuned": {},
            "zero_shot": {},
            "comparison": {},
        }
        
        # 1. Baseline (no perturbation)
        logger.info("Running baseline evaluation (no perturbation)...")
        engine = VisualPerturbationEngine()
        env.set_perturbation_engine(engine)
        
        baseline_results = evaluator.evaluate_policy(
            model, env, n_episodes=self.config.n_episodes_per_condition
        )
        baseline_metrics = evaluator.compute_aggregate_metrics(baseline_results)
        results["finetuned"]["baseline"] = baseline_metrics
        logger.info(f"  Baseline: {baseline_metrics['success_rate']*100:.1f}%")
        
        # 2. Test each perturbation
        for pconfig in perturbation_configs:
            name = pconfig["name"]
            ptype = pconfig["type"]
            level = pconfig["level"]
            
            logger.info(f"Testing perturbation: {name} (level={level})")
            
            # Configure perturbation engine
            engine = VisualPerturbationEngine()
            engine.add_perturbation(ptype, level=level)
            env.set_perturbation_engine(engine)
            
            # Fine-tuned model
            ft_results = evaluator.evaluate_policy(
                model, env, n_episodes=self.config.n_episodes_per_condition
            )
            ft_metrics = evaluator.compute_aggregate_metrics(ft_results)
            results["finetuned"][name] = ft_metrics
            
            # Zero-shot model (if provided)
            if zero_shot_model and self.config.compare_zero_shot:
                zs_results = evaluator.evaluate_policy(
                    zero_shot_model, env,
                    n_episodes=self.config.n_episodes_per_condition
                )
                zs_metrics = evaluator.compute_aggregate_metrics(zs_results)
                results["zero_shot"][name] = zs_metrics
                
                # Comparison
                results["comparison"][name] = {
                    "ft_success_rate": ft_metrics["success_rate"],
                    "zs_success_rate": zs_metrics["success_rate"],
                    "improvement": ft_metrics["success_rate"] - zs_metrics["success_rate"],
                }
                
                logger.info(
                    f"  FT: {ft_metrics['success_rate']*100:.1f}% | "
                    f"ZS: {zs_metrics['success_rate']*100:.1f}% | "
                    f"Δ: {(ft_metrics['success_rate']-zs_metrics['success_rate'])*100:+.1f}%"
                )
            else:
                degradation = baseline_metrics["success_rate"] - ft_metrics["success_rate"]
                logger.info(
                    f"  FT: {ft_metrics['success_rate']*100:.1f}% | "
                    f"Degradation: {degradation*100:.1f}%"
                )
        
        # Reset perturbation
        env.set_perturbation_engine(VisualPerturbationEngine())
        
        # 3. Analyze results
        results["analysis"] = self._analyze_robustness(results)
        
        # 4. Save
        self._save_results(results)
        
        return results

    def _analyze_robustness(self, results: dict) -> dict:
        """Analyze robustness patterns."""
        ft_results = results["finetuned"]
        baseline_sr = ft_results.get("baseline", {}).get("success_rate", 0)
        
        analysis = {
            "baseline_success_rate": baseline_sr,
            "perturbation_sensitivity": {},
            "most_robust_to": None,
            "most_sensitive_to": None,
            "critical_failures": [],
        }
        
        sensitivities = {}
        for name, metrics in ft_results.items():
            if name == "baseline":
                continue
            
            sr = metrics.get("success_rate", 0)
            degradation = baseline_sr - sr
            sensitivities[name] = degradation
            
            analysis["perturbation_sensitivity"][name] = {
                "success_rate": sr,
                "degradation": degradation,
                "relative_degradation": degradation / max(baseline_sr, 0.01),
            }
            
            # Critical failure: >50% degradation
            if degradation > 0.5 * baseline_sr:
                analysis["critical_failures"].append({
                    "perturbation": name,
                    "success_rate": sr,
                    "degradation": degradation,
                })
        
        if sensitivities:
            analysis["most_robust_to"] = min(sensitivities, key=sensitivities.get)
            analysis["most_sensitive_to"] = max(sensitivities, key=sensitivities.get)
        
        # Group by perturbation type
        type_groups = {}
        for name, deg in sensitivities.items():
            ptype = name.rsplit("_", 1)[0]  # e.g., "brightness" from "brightness_0.5"
            if ptype not in type_groups:
                type_groups[ptype] = []
            type_groups[ptype].append(deg)
        
        analysis["type_sensitivity"] = {
            ptype: {
                "mean_degradation": float(np.mean(degs)),
                "max_degradation": float(np.max(degs)),
            }
            for ptype, degs in type_groups.items()
        }
        
        return analysis

    def _save_results(self, results: dict):
        """Save results and generate report."""
        # JSON results
        output_path = self._results_dir / "robustness_results.json"
        serializable = self._make_serializable(results)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        
        # Markdown report
        self._generate_report(results)
        
        logger.info(f"Results saved to {self._results_dir}")

    def _generate_report(self, results: dict):
        """Generate markdown robustness report."""
        report_path = self._results_dir / "robustness_report.md"
        
        analysis = results.get("analysis", {})
        baseline_sr = analysis.get("baseline_success_rate", 0) * 100
        
        lines = [
            "# Visual Perturbation Robustness Analysis\n",
            f"**Baseline Success Rate**: {baseline_sr:.1f}%\n",
            "## Perturbation Results\n",
            "| Perturbation | Success Rate | Degradation | Relative Deg. |",
            "|-------------|-------------|-------------|---------------|",
        ]
        
        for name, sensitivity in analysis.get("perturbation_sensitivity", {}).items():
            sr = sensitivity["success_rate"] * 100
            deg = sensitivity["degradation"] * 100
            rel_deg = sensitivity["relative_degradation"] * 100
            
            lines.append(f"| {name:<20} | {sr:.1f}% | {deg:+.1f}% | {rel_deg:.1f}% |")
        
        # Summary
        lines.extend([
            f"\n## Key Findings\n",
            f"- **Most robust to**: {analysis.get('most_robust_to', 'N/A')}",
            f"- **Most sensitive to**: {analysis.get('most_sensitive_to', 'N/A')}",
            f"- **Critical failures**: {len(analysis.get('critical_failures', []))}",
        ])
        
        # Type-level summary
        lines.extend(["\n## Sensitivity by Perturbation Type\n",
                      "| Type | Mean Degradation | Max Degradation |",
                      "|------|-----------------|-----------------|"])
        
        for ptype, stats in analysis.get("type_sensitivity", {}).items():
            mean_deg = stats["mean_degradation"] * 100
            max_deg = stats["max_degradation"] * 100
            lines.append(f"| {ptype:<15} | {mean_deg:.1f}% | {max_deg:.1f}% |")
        
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: RobustnessAnalyzer._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [RobustnessAnalyzer._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def __repr__(self):
        return f"RobustnessAnalyzer(n_per_condition={self.config.n_episodes_per_condition})"
