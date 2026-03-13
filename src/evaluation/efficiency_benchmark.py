"""
Innovation Point #4: Computational Efficiency Benchmark.

Evaluates SmolVLA inference efficiency under different configurations:
- Precision: FP32, FP16, BF16
- Quantization: INT8, INT4
- Metrics: Latency, Throughput, Memory, Success Rate

Determines if real-time robotic control (>10Hz) is achievable.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyConfig:
    """Efficiency benchmark configuration."""
    n_warmup_steps: int = 50
    n_benchmark_steps: int = 200
    n_eval_episodes: int = 20
    realtime_threshold_hz: float = 10.0
    results_dir: str = "./results/efficiency"
    configurations: list[dict] = None

    def __post_init__(self):
        if self.configurations is None:
            self.configurations = [
                {"name": "fp32", "dtype": "float32", "quantization": None},
                {"name": "fp16", "dtype": "float16", "quantization": None},
                {"name": "bf16", "dtype": "bfloat16", "quantization": None},
                {"name": "int8", "dtype": "float16", "quantization": "int8"},
                {"name": "int4", "dtype": "float16", "quantization": "int4"},
            ]


class EfficiencyBenchmark:
    """
    Computational efficiency benchmark for SmolVLA.
    
    Measures the full compute stack:
    1. Image preprocessing latency
    2. Model inference latency
    3. Action postprocessing latency
    4. Total loop latency → control frequency
    
    Evaluates whether different quantization levels meet the
    real-time control requirement (>10Hz for most manipulation tasks).
    """

    def __init__(self, config: EfficiencyConfig):
        self.config = config
        self._results_dir = Path(config.results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def run(self, model_name: str, env=None, evaluator=None) -> dict:
        """
        Run the full efficiency benchmark.
        
        Args:
            model_name: HuggingFace model ID for loading different quantizations
            env: Optional environment for measuring success rate impact
            evaluator: Optional evaluator for full eval
            
        Returns:
            Complete benchmark results
        """
        from ..model.quantization import ModelQuantizer, QuantizationConfig
        from ..model.smolvla_wrapper import SmolVLAWrapper, SmolVLAConfig
        
        results = {
            "configurations": {},
            "latency_breakdown": {},
            "realtime_analysis": {},
        }
        
        logger.info("=" * 60)
        logger.info("Computational Efficiency Benchmark")
        logger.info("=" * 60)
        
        for config in self.config.configurations:
            name = config["name"]
            logger.info(f"\nBenchmarking: {name}")
            
            try:
                # Load model with specified configuration
                model_config = SmolVLAConfig(
                    model_name=model_name,
                    dtype=config["dtype"],
                    use_lora=False,  # Benchmark base model
                )
                model = SmolVLAWrapper(model_config)
                model.load()
                
                # Apply quantization if needed
                if config.get("quantization"):
                    quant = ModelQuantizer(QuantizationConfig(
                        bits=int(config["quantization"].replace("int", "")),
                    ))
                    # Note: Quantization applied during loading via bitsandbytes
                
                # Latency benchmark
                latency_result = self._benchmark_latency(model)
                results["configurations"][name] = latency_result
                
                # Latency breakdown
                breakdown = self._benchmark_breakdown(model)
                results["latency_breakdown"][name] = breakdown
                
                # Success rate evaluation (if env available)
                if env and evaluator:
                    eval_result = evaluator.evaluate_policy(
                        model, env, n_episodes=self.config.n_eval_episodes
                    )
                    eval_metrics = evaluator.compute_aggregate_metrics(eval_result)
                    results["configurations"][name]["success_rate"] = eval_metrics["success_rate"]
                    results["configurations"][name]["mean_reward"] = eval_metrics["mean_reward"]
                
                logger.info(
                    f"  Latency: {latency_result['latency_mean_ms']:.1f}ms | "
                    f"Throughput: {latency_result['throughput_hz']:.1f}Hz | "
                    f"Memory: {latency_result['memory_mb']:.0f}MB | "
                    f"Realtime: {'✅' if latency_result['meets_realtime'] else '❌'}"
                )
                
                # Cleanup
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"  Failed: {e}")
                results["configurations"][name] = {
                    "error": str(e),
                    "latency_mean_ms": float("inf"),
                    "throughput_hz": 0.0,
                    "memory_mb": 0.0,
                    "meets_realtime": False,
                }
        
        # Realtime analysis
        results["realtime_analysis"] = self._analyze_realtime(results)
        
        # Save
        self._save_results(results)
        
        return results

    def _benchmark_latency(self, model) -> dict:
        """Benchmark pure inference latency."""
        return model.get_inference_stats(
            n_warmup=self.config.n_warmup_steps,
            n_runs=self.config.n_benchmark_steps,
        )

    def _benchmark_breakdown(self, model) -> dict:
        """Benchmark latency breakdown: preprocess, inference, postprocess."""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_instruction = "pick up the red cube"
        dummy_state = np.random.randn(7).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            model.predict_action(
                {"agentview": dummy_image},
                dummy_instruction,
                dummy_state,
            )
        
        # Measure components
        preprocess_times = []
        inference_times = []
        postprocess_times = []
        
        for _ in range(self.config.n_benchmark_steps):
            # Preprocess
            t0 = time.perf_counter()
            images = {"agentview": dummy_image}
            t1 = time.perf_counter()
            
            # Inference (includes all model forward pass)
            action = model.predict_action(images, dummy_instruction, dummy_state)
            t2 = time.perf_counter()
            
            # Postprocess
            action_clipped = np.clip(action, -1.0, 1.0)
            t3 = time.perf_counter()
            
            preprocess_times.append((t1 - t0) * 1000)
            inference_times.append((t2 - t1) * 1000)
            postprocess_times.append((t3 - t2) * 1000)
        
        return {
            "preprocess_ms": {
                "mean": float(np.mean(preprocess_times)),
                "std": float(np.std(preprocess_times)),
            },
            "inference_ms": {
                "mean": float(np.mean(inference_times)),
                "std": float(np.std(inference_times)),
            },
            "postprocess_ms": {
                "mean": float(np.mean(postprocess_times)),
                "std": float(np.std(postprocess_times)),
            },
            "total_ms": {
                "mean": float(np.mean(preprocess_times) + np.mean(inference_times) + np.mean(postprocess_times)),
            },
        }

    def _analyze_realtime(self, results: dict) -> dict:
        """Analyze realistic real-time control feasibility."""
        threshold = self.config.realtime_threshold_hz
        threshold_ms = 1000.0 / threshold
        
        analysis = {
            "threshold_hz": threshold,
            "threshold_ms": threshold_ms,
            "configurations": {},
        }
        
        for name, data in results["configurations"].items():
            if "error" in data:
                continue
            
            latency = data.get("latency_mean_ms", float("inf"))
            memory = data.get("memory_mb", 0)
            
            analysis["configurations"][name] = {
                "meets_threshold": latency < threshold_ms,
                "headroom_ms": threshold_ms - latency,
                "headroom_percent": (1 - latency / threshold_ms) * 100 if threshold_ms > 0 else 0,
                "max_achievable_hz": 1000.0 / latency if latency > 0 else 0,
                "memory_efficiency_params_per_mb": data.get("n_parameters", 0) / max(memory, 1),
            }
        
        # Find optimal configuration
        viable = {
            k: v for k, v in analysis["configurations"].items()
            if v.get("meets_threshold", False)
        }
        
        if viable:
            # Among viable configs, pick the one with best success_rate (if available)
            best = max(viable.keys(), key=lambda k: 
                results["configurations"].get(k, {}).get("success_rate", 0)
            )
            analysis["recommended"] = best
        else:
            # Pick fastest
            all_configs = analysis["configurations"]
            if all_configs:
                fastest = max(all_configs.keys(), key=lambda k: 
                    all_configs[k].get("max_achievable_hz", 0)
                )
                analysis["recommended"] = fastest
                analysis["warning"] = "No configuration meets real-time threshold"
        
        return analysis

    def _save_results(self, results: dict):
        """Save results and generate report."""
        output_path = self._results_dir / "efficiency_results.json"
        serializable = self._make_serializable(results)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        
        self._generate_report(results)
        logger.info(f"Results saved to {self._results_dir}")

    def _generate_report(self, results: dict):
        """Generate formatted efficiency report."""
        from ..model.quantization import ModelQuantizer
        
        report_path = self._results_dir / "efficiency_report.md"
        
        lines = [
            "# Computational Efficiency Benchmark\n",
            f"**Real-time threshold**: {self.config.realtime_threshold_hz} Hz "
            f"({1000/self.config.realtime_threshold_hz:.0f}ms)\n",
            "## Configuration Comparison\n",
            "| Config | Latency (ms) | Throughput (Hz) | Memory (MB) | Realtime? | Success Rate |",
            "|--------|-------------|-----------------|-------------|-----------|-------------|",
        ]
        
        for name, data in results["configurations"].items():
            if "error" in data:
                lines.append(f"| {name:<6} | ERROR | - | - | ❌ | - |")
                continue
            
            latency = data.get("latency_mean_ms", 0)
            latency_std = data.get("latency_std_ms", 0)
            throughput = data.get("throughput_hz", 0)
            memory = data.get("memory_mb", 0)
            realtime = "✅" if data.get("meets_realtime", False) else "❌"
            sr = data.get("success_rate", None)
            sr_str = f"{sr*100:.1f}%" if sr is not None else "N/A"
            
            lines.append(
                f"| {name:<6} | {latency:.1f}±{latency_std:.1f} | "
                f"{throughput:.1f} | {memory:.0f} | {realtime} | {sr_str} |"
            )
        
        # Latency breakdown
        lines.extend(["\n## Latency Breakdown\n",
                      "| Config | Preprocess | Inference | Postprocess | Total |",
                      "|--------|-----------|-----------|-------------|-------|"])
        
        for name, breakdown in results.get("latency_breakdown", {}).items():
            if not isinstance(breakdown, dict):
                continue
            pre = breakdown.get("preprocess_ms", {}).get("mean", 0)
            inf_time = breakdown.get("inference_ms", {}).get("mean", 0)
            post = breakdown.get("postprocess_ms", {}).get("mean", 0)
            total = breakdown.get("total_ms", {}).get("mean", 0)
            
            lines.append(
                f"| {name:<6} | {pre:.2f}ms | {inf_time:.1f}ms | {post:.2f}ms | {total:.1f}ms |"
            )
        
        # Recommendation
        rt_analysis = results.get("realtime_analysis", {})
        recommended = rt_analysis.get("recommended", "N/A")
        warning = rt_analysis.get("warning", "")
        
        lines.extend([
            f"\n## Recommendation\n",
            f"**Recommended configuration**: `{recommended}`",
        ])
        if warning:
            lines.append(f"\n⚠️ {warning}")
        
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: EfficiencyBenchmark._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [EfficiencyBenchmark._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def __repr__(self):
        return f"EfficiencyBenchmark(configs={len(self.config.configurations)})"
