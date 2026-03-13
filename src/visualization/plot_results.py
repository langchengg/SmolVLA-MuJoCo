"""
Result Visualization Tools.

Generates publication-quality figures for all 4 innovation experiments:
1. Language generalization performance matrix (heatmap)
2. Visual robustness sensitivity chart
3. Action chunking trade-off plots
4. Computational efficiency comparison bars
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """
    Generate publication-quality visualizations for SmolVLA experiments.
    
    All plots follow academic paper conventions:
    - Clean, readable fonts (12pt+)
    - Color-blind friendly palettes
    - Error bars where applicable
    - Saved as both PNG (for README) and PDF (for papers)
    """

    # Color palette (color-blind friendly)
    COLORS = {
        "primary": "#2196F3",
        "secondary": "#FF9800",
        "success": "#4CAF50",
        "danger": "#F44336",
        "neutral": "#9E9E9E",
        "palette": ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0", 
                     "#00BCD4", "#FFC107", "#795548"],
    }

    def __init__(self, results_dir: str = "./results", style: str = "seaborn-v0_8-paper"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._style = style

    def _setup_matplotlib(self):
        """Configure matplotlib for publication-quality output."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        try:
            plt.style.use(self._style)
        except Exception:
            plt.style.use("seaborn-v0_8")
        
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox_inches": "tight",
        })
        
        return plt

    # ─── Innovation 1: Language Generalization ──────────────────────────────

    def plot_language_generalization_matrix(self, results: dict, save: bool = True):
        """
        Plot language generalization performance as a heatmap.
        
        Rows = Original instructions
        Columns = Variant difficulty levels
        Cells = Success rate (color-coded)
        """
        plt = self._setup_matplotlib()
        import seaborn as sns
        
        summary = results.get("summary_matrix", {})
        if not summary:
            logger.warning("No summary matrix data to plot")
            return
        
        # Build matrix
        instructions = list(summary.keys())
        difficulty_levels = ["baseline", "easy_synonym", "medium_paraphrase", "hard_structural"]
        
        matrix = np.zeros((len(instructions), len(difficulty_levels)))
        
        for i, inst in enumerate(instructions):
            data = summary[inst]
            matrix[i, 0] = data.get("baseline_success_rate", 0) * 100
            
            for j, diff in enumerate(difficulty_levels[1:], 1):
                variant_data = data.get("variant_results", {}).get(diff, {})
                matrix[i, j] = variant_data.get("mean_success_rate", 0) * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(4, len(instructions) * 0.8)))
        
        short_instructions = [inst[:25] + "..." if len(inst) > 25 else inst for inst in instructions]
        short_difficulties = ["Original", "Synonym\n(Easy)", "Paraphrase\n(Medium)", "Structural\n(Hard)"]
        
        heatmap = sns.heatmap(
            matrix, annot=True, fmt=".1f", cmap="RdYlGn",
            xticklabels=short_difficulties,
            yticklabels=short_instructions,
            vmin=0, vmax=100,
            cbar_kws={"label": "Success Rate (%)"},
            ax=ax,
        )
        
        ax.set_title("Language Generalization Performance Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Instruction Variant Difficulty")
        ax.set_ylabel("Task Instruction")
        
        plt.tight_layout()
        
        if save:
            for ext in ["png", "pdf"]:
                fig.savefig(self.figures_dir / f"language_generalization_matrix.{ext}")
        
        plt.close(fig)
        logger.info("Language generalization matrix plot saved.")

    # ─── Innovation 2: Visual Robustness ────────────────────────────────────

    def plot_robustness_heatmap(self, results: dict, save: bool = True):
        """
        Plot visual robustness as grouped bar chart and sensitivity heatmap.
        """
        plt = self._setup_matplotlib()
        import seaborn as sns
        
        analysis = results.get("analysis", {})
        sensitivity = analysis.get("perturbation_sensitivity", {})
        
        if not sensitivity:
            logger.warning("No sensitivity data to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- Left: Success rate bar chart ---
        names = list(sensitivity.keys())
        success_rates = [sensitivity[n]["success_rate"] * 100 for n in names]
        degradations = [sensitivity[n]["degradation"] * 100 for n in names]
        
        x = np.arange(len(names))
        bars = axes[0].bar(x, success_rates, color=self.COLORS["palette"][:len(names)], alpha=0.8)
        
        # Add baseline line
        baseline_sr = analysis.get("baseline_success_rate", 0) * 100
        axes[0].axhline(y=baseline_sr, color="red", linestyle="--", alpha=0.7, label=f"Baseline ({baseline_sr:.1f}%)")
        
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Success Rate (%)")
        axes[0].set_title("Success Rate Under Perturbations")
        axes[0].legend()
        axes[0].set_ylim(0, 105)
        
        # --- Right: Type-level sensitivity ---
        type_sens = analysis.get("type_sensitivity", {})
        if type_sens:
            types = list(type_sens.keys())
            mean_degs = [type_sens[t]["mean_degradation"] * 100 for t in types]
            max_degs = [type_sens[t]["max_degradation"] * 100 for t in types]
            
            x2 = np.arange(len(types))
            width = 0.35
            axes[1].bar(x2 - width/2, mean_degs, width, label="Mean Degradation",
                       color=self.COLORS["secondary"], alpha=0.8)
            axes[1].bar(x2 + width/2, max_degs, width, label="Max Degradation",
                       color=self.COLORS["danger"], alpha=0.8)
            
            axes[1].set_xticks(x2)
            axes[1].set_xticklabels(types, rotation=30, ha="right")
            axes[1].set_ylabel("Performance Degradation (%)")
            axes[1].set_title("Sensitivity by Perturbation Type")
            axes[1].legend()
        
        plt.suptitle("Visual Perturbation Robustness Analysis", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        if save:
            for ext in ["png", "pdf"]:
                fig.savefig(self.figures_dir / f"robustness_analysis.{ext}")
        
        plt.close(fig)
        logger.info("Robustness analysis plot saved.")

    # ─── Innovation 3: Action Chunking ──────────────────────────────────────

    def plot_chunking_ablation(self, results: dict, save: bool = True):
        """
        Plot chunking ablation: success vs smoothness trade-off.
        """
        plt = self._setup_matplotlib()
        
        configs = {k: v for k, v in results.items() 
                   if k != "analysis" and isinstance(v, dict) and "metrics" in v}
        
        if not configs:
            logger.warning("No chunking data to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Separate standard and ensemble configs
        standard = {k: v for k, v in configs.items() if not v.get("temporal_ensemble", False)}
        ensemble = {k: v for k, v in configs.items() if v.get("temporal_ensemble", False)}
        
        # --- Left: Success rate vs chunk size ---
        for group, label, color in [(standard, "Standard", self.COLORS["primary"]),
                                      (ensemble, "Temporal Ensemble", self.COLORS["secondary"])]:
            if not group:
                continue
            sizes = [v["chunk_size"] for v in group.values()]
            srs = [v["metrics"]["success_rate"] * 100 for v in group.values()]
            
            sorted_pairs = sorted(zip(sizes, srs))
            sizes, srs = zip(*sorted_pairs)
            
            axes[0].plot(sizes, srs, "o-", color=color, label=label, linewidth=2, markersize=8)
        
        axes[0].set_xlabel("Chunk Size")
        axes[0].set_ylabel("Success Rate (%)")
        axes[0].set_title("Success Rate vs Chunk Size")
        axes[0].legend()
        axes[0].set_xticks([1, 2, 4, 8, 16])
        
        # --- Middle: Smoothness (jerk) vs chunk size ---
        for group, label, color in [(standard, "Standard", self.COLORS["primary"]),
                                      (ensemble, "Temporal Ensemble", self.COLORS["secondary"])]:
            if not group:
                continue
            sizes = [v["chunk_size"] for v in group.values()]
            jerks = [v["smoothness"].get("jerk", 0) for v in group.values()]
            
            sorted_pairs = sorted(zip(sizes, jerks))
            sizes, jerks = zip(*sorted_pairs)
            
            axes[1].plot(sizes, jerks, "s-", color=color, label=label, linewidth=2, markersize=8)
        
        axes[1].set_xlabel("Chunk Size")
        axes[1].set_ylabel("Jerk (lower = smoother)")
        axes[1].set_title("Motion Smoothness vs Chunk Size")
        axes[1].legend()
        axes[1].set_xticks([1, 2, 4, 8, 16])
        
        # --- Right: Pareto front ---
        all_srs = []
        all_jerks = []
        all_names = []
        
        for name, data in configs.items():
            all_srs.append(data["metrics"]["success_rate"] * 100)
            all_jerks.append(data["smoothness"].get("jerk", 0))
            all_names.append(name)
        
        pareto = results.get("analysis", {}).get("pareto_optimal", [])
        
        for i, (sr, jerk, name) in enumerate(zip(all_srs, all_jerks, all_names)):
            is_pareto = name in pareto
            color = self.COLORS["success"] if is_pareto else self.COLORS["neutral"]
            marker = "*" if is_pareto else "o"
            size = 200 if is_pareto else 80
            
            axes[2].scatter(jerk, sr, c=color, marker=marker, s=size, 
                          edgecolors="black", linewidths=0.5, zorder=3)
            axes[2].annotate(name.replace("chunk_", "c"), (jerk, sr), 
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
        
        axes[2].set_xlabel("Jerk (lower = smoother) →")
        axes[2].set_ylabel("Success Rate (%) ↑")
        axes[2].set_title("Pareto Front: Success vs Smoothness")
        
        plt.suptitle("Action Chunking Ablation Study", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        if save:
            for ext in ["png", "pdf"]:
                fig.savefig(self.figures_dir / f"chunking_ablation.{ext}")
        
        plt.close(fig)
        logger.info("Chunking ablation plot saved.")

    # ─── Innovation 4: Computational Efficiency ─────────────────────────────

    def plot_efficiency_benchmark(self, results: dict, save: bool = True):
        """
        Plot efficiency benchmark: latency, memory, and throughput comparison.
        """
        plt = self._setup_matplotlib()
        
        configs = results.get("configurations", {})
        configs = {k: v for k, v in configs.items() if "error" not in v}
        
        if not configs:
            logger.warning("No efficiency data to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        names = list(configs.keys())
        x = np.arange(len(names))
        
        # --- Left: Latency ---
        latencies = [configs[n].get("latency_mean_ms", 0) for n in names]
        latency_stds = [configs[n].get("latency_std_ms", 0) for n in names]
        
        bars = axes[0].bar(x, latencies, yerr=latency_stds, 
                          color=self.COLORS["palette"][:len(names)], 
                          alpha=0.8, capsize=5)
        
        threshold_ms = 1000 / results.get("realtime_analysis", {}).get("threshold_hz", 10)
        axes[0].axhline(y=threshold_ms, color="red", linestyle="--", alpha=0.7,
                       label=f"Realtime ({threshold_ms:.0f}ms)")
        
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names)
        axes[0].set_ylabel("Latency (ms)")
        axes[0].set_title("Inference Latency")
        axes[0].legend()
        
        # --- Middle: Throughput ---
        throughputs = [configs[n].get("throughput_hz", 0) for n in names]
        
        bars = axes[1].bar(x, throughputs, color=self.COLORS["palette"][:len(names)], alpha=0.8)
        
        threshold_hz = results.get("realtime_analysis", {}).get("threshold_hz", 10)
        axes[1].axhline(y=threshold_hz, color="red", linestyle="--", alpha=0.7,
                       label=f"Realtime ({threshold_hz:.0f}Hz)")
        
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names)
        axes[1].set_ylabel("Throughput (Hz)")
        axes[1].set_title("Control Frequency")
        axes[1].legend()
        
        # --- Right: Memory ---
        memories = [configs[n].get("memory_mb", 0) for n in names]
        success_rates = [configs[n].get("success_rate", 0) * 100 for n in names]
        
        bars = axes[2].bar(x, memories, color=self.COLORS["palette"][:len(names)], alpha=0.8)
        
        # Overlay success rate as line if available
        if any(sr > 0 for sr in success_rates):
            ax2 = axes[2].twinx()
            ax2.plot(x, success_rates, "ro-", linewidth=2, markersize=8, label="Success Rate")
            ax2.set_ylabel("Success Rate (%)", color="red")
            ax2.legend(loc="upper right")
        
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names)
        axes[2].set_ylabel("Memory (MB)")
        axes[2].set_title("Memory Usage")
        
        plt.suptitle("Computational Efficiency Analysis", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        if save:
            for ext in ["png", "pdf"]:
                fig.savefig(self.figures_dir / f"efficiency_benchmark.{ext}")
        
        plt.close(fig)
        logger.info("Efficiency benchmark plot saved.")

    # ─── Combined Dashboard ─────────────────────────────────────────────────

    def generate_all_plots(self, results_dir: Optional[str] = None):
        """Load all result files and generate all plots."""
        rdir = Path(results_dir) if results_dir else self.results_dir
        
        # Language generalization
        lg_path = rdir / "language_generalization" / "language_generalization_results.json"
        if lg_path.exists():
            with open(lg_path) as f:
                self.plot_language_generalization_matrix(json.load(f))
        
        # Robustness
        rob_path = rdir / "robustness" / "robustness_results.json"
        if rob_path.exists():
            with open(rob_path) as f:
                self.plot_robustness_heatmap(json.load(f))
        
        # Chunking
        chunk_path = rdir / "chunking_ablation" / "chunking_ablation_results.json"
        if chunk_path.exists():
            with open(chunk_path) as f:
                self.plot_chunking_ablation(json.load(f))
        
        # Efficiency
        eff_path = rdir / "efficiency" / "efficiency_results.json"
        if eff_path.exists():
            with open(eff_path) as f:
                self.plot_efficiency_benchmark(json.load(f))
        
        logger.info(f"All plots saved to {self.figures_dir}")

    def __repr__(self):
        return f"ResultVisualizer(figures_dir={self.figures_dir})"
