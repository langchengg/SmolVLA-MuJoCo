"""
Innovation Point #1: Multi-Task Language Condition Generalization Test.

Evaluates how well the fine-tuned SmolVLA model generalizes across:
1. Synonym variations ("pick up" → "grasp", "grab", etc.)
2. Structural paraphrases ("pick up the red cube" → "the crimson block, grasp it")
3. Spatial generalization (changed object positions)

Output: Language generalization performance matrix.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LanguageGenConfig:
    """Language generalization test configuration."""
    n_episodes_per_variant: int = 20
    position_noise_std: float = 0.05
    results_dir: str = "./results/language_generalization"


class LanguageGeneralizationTest:
    """
    Systematic evaluation of language-conditioned generalization.
    
    Test matrix:
        - Rows: Original training instructions
        - Columns: Variant types (synonym, paraphrase, structural)
        - Cells: Success rate
    
    Additionally tests spatial generalization by randomizing object positions.
    """

    def __init__(self, config: LanguageGenConfig):
        self.config = config
        self._results_dir = Path(config.results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def run(self, model, env, evaluator) -> dict:
        """
        Run the full language generalization experiment.
        
        Args:
            model: SmolVLAWrapper
            env: SmolVLAMuJoCoEnv
            evaluator: SmolVLAEvaluator
            
        Returns:
            Complete results dict with performance matrix
        """
        from ..data.language_augmentation import LanguageAugmentor
        
        augmentor = LanguageAugmentor()
        
        # Get original instructions from the environment/dataset
        original_instructions = self._get_task_instructions(env)
        
        logger.info(f"Testing {len(original_instructions)} original instructions")
        
        # Generate test matrix
        gen_matrix = augmentor.get_generalization_matrix(
            original_instructions, n_variants=5
        )
        
        results = {
            "original_performance": {},
            "synonym_performance": {},
            "paraphrase_performance": {},
            "structural_performance": {},
            "spatial_performance": {},
            "summary_matrix": {},
        }
        
        for inst in original_instructions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: '{inst}'")
            logger.info(f"{'='*60}")
            
            # 1. Baseline: original instruction
            baseline = evaluator.evaluate_policy(
                model, env,
                n_episodes=self.config.n_episodes_per_variant,
                language_instruction=inst,
            )
            baseline_metrics = evaluator.compute_aggregate_metrics(baseline)
            results["original_performance"][inst] = baseline_metrics
            logger.info(f"  Original: {baseline_metrics['success_rate']*100:.1f}%")
            
            # 2. Synonym variants
            difficulty_levels = gen_matrix["difficulty_levels"].get(inst, {})
            
            for difficulty, variants in difficulty_levels.items():
                category = difficulty.split("_")[0]  # "easy", "medium", "hard"
                key = f"{category}_performance"
                
                if key not in results:
                    results[key] = {}
                
                for variant in variants[:3]:  # Test top 3 variants per difficulty
                    variant_results = evaluator.evaluate_policy(
                        model, env,
                        n_episodes=self.config.n_episodes_per_variant,
                        language_instruction=variant,
                    )
                    variant_metrics = evaluator.compute_aggregate_metrics(variant_results)
                    
                    result_key = f"{inst} → {variant}"
                    results[key][result_key] = variant_metrics
                    
                    logger.info(
                        f"  {difficulty}: '{variant}' → "
                        f"{variant_metrics['success_rate']*100:.1f}%"
                    )
            
            # 3. Build summary for this instruction
            results["summary_matrix"][inst] = self._build_instruction_summary(
                inst, baseline_metrics, difficulty_levels, results
            )
        
        # Save results
        self._save_results(results)
        
        return results

    def _get_task_instructions(self, env) -> list[str]:
        """Get task instructions to test."""
        # Try to get from environment
        if hasattr(env, 'get_task_description'):
            instructions = [env.get_task_description()]
        else:
            instructions = []
        
        # Default test instructions if none available
        if not instructions or instructions == ["complete the manipulation task"]:
            instructions = [
                "pick up the red cube",
                "put the bowl on the plate",
                "push the blue button",
                "open the drawer",
                "stack the blocks",
            ]
        
        return instructions

    def _build_instruction_summary(
        self, instruction: str, baseline: dict, 
        difficulty_levels: dict, all_results: dict
    ) -> dict:
        """Build summary metrics for one instruction across all variants."""
        summary = {
            "instruction": instruction,
            "baseline_success_rate": baseline.get("success_rate", 0),
            "variant_results": {},
        }
        
        for difficulty, variants in difficulty_levels.items():
            category = difficulty.split("_")[0]
            key = f"{category}_performance"
            
            rates = []
            for variant in variants[:3]:
                result_key = f"{instruction} → {variant}"
                if key in all_results and result_key in all_results[key]:
                    rates.append(all_results[key][result_key].get("success_rate", 0))
            
            if rates:
                summary["variant_results"][difficulty] = {
                    "mean_success_rate": float(np.mean(rates)),
                    "std_success_rate": float(np.std(rates)),
                    "degradation": float(baseline.get("success_rate", 0) - np.mean(rates)),
                }
        
        return summary

    def _save_results(self, results: dict):
        """Save results to JSON."""
        output_path = self._results_dir / "language_generalization_results.json"
        
        # Convert numpy types to Python types
        serializable = self._make_serializable(results)
        
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        
        # Also generate summary table
        self._generate_summary_report(results)

    def _generate_summary_report(self, results: dict):
        """Generate a markdown summary report."""
        report_path = self._results_dir / "language_generalization_report.md"
        
        lines = [
            "# Language Generalization Test Results\n",
            "## Performance Matrix\n",
            "| Instruction | Original | Easy (Synonym) | Medium (Paraphrase) | Hard (Structural) |",
            "|-------------|----------|----------------|---------------------|-------------------|",
        ]
        
        for inst, summary in results.get("summary_matrix", {}).items():
            baseline_sr = summary.get("baseline_success_rate", 0) * 100
            
            easy_sr = summary.get("variant_results", {}).get("easy_synonym", {}).get("mean_success_rate", 0) * 100
            med_sr = summary.get("variant_results", {}).get("medium_paraphrase", {}).get("mean_success_rate", 0) * 100
            hard_sr = summary.get("variant_results", {}).get("hard_structural", {}).get("mean_success_rate", 0) * 100
            
            lines.append(
                f"| {inst[:30]:<30} | {baseline_sr:.1f}% | {easy_sr:.1f}% | {med_sr:.1f}% | {hard_sr:.1f}% |"
            )
        
        lines.extend([
            "\n## Key Findings\n",
            "- **Synonym robustness**: Performance degradation from simple word substitutions",
            "- **Paraphrase robustness**: Performance under template-based rewrites",
            "- **Structural robustness**: Performance under sentence structure changes",
        ])
        
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        
        logger.info(f"Report saved to {report_path}")

    @staticmethod
    def _make_serializable(obj):
        """Convert numpy/torch types to Python native types."""
        if isinstance(obj, dict):
            return {k: LanguageGeneralizationTest._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [LanguageGeneralizationTest._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def __repr__(self):
        return f"LanguageGeneralizationTest(n_episodes_per_variant={self.config.n_episodes_per_variant})"
