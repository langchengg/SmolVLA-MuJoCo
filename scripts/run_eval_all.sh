#!/bin/bash
# =============================================================================
# Run All Evaluation Experiments
# =============================================================================
# Usage: bash scripts/run_eval_all.sh [--smoke_test] [--checkpoint PATH]

set -e

echo "🔬 SmolVLA Evaluation Pipeline"
echo "=============================="

CHECKPOINT="./results/checkpoints/best"
SMOKE_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke_test)
            SMOKE_TEST=true
            shift
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ "$SMOKE_TEST" = true ]; then
    N_EPISODES=3
    echo "⚡ Running in SMOKE TEST mode (3 episodes per condition)"
else
    N_EPISODES=20
fi

echo "Checkpoint: $CHECKPOINT"
echo ""

# Run all 4 innovation experiments
python -c "
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')

from src.env.mujoco_env import SmolVLAMuJoCoEnv, EnvConfig
from src.model.smolvla_wrapper import SmolVLAWrapper, SmolVLAConfig
from src.evaluation.evaluator import SmolVLAEvaluator, EvalConfig

# Setup
env_config = EnvConfig(env_type='libero')
env = SmolVLAMuJoCoEnv(env_config)

model_config = SmolVLAConfig(model_name='${CHECKPOINT}')
model = SmolVLAWrapper(model_config)
model.load()

eval_config = EvalConfig(n_episodes=${N_EPISODES})
evaluator = SmolVLAEvaluator(eval_config)

print()
print('=' * 60)
print('Innovation 1: Language Generalization')
print('=' * 60)
from src.evaluation.language_generalization import LanguageGeneralizationTest, LanguageGenConfig
lg_config = LanguageGenConfig(n_episodes_per_variant=${N_EPISODES})
lg_test = LanguageGeneralizationTest(lg_config)
lg_results = lg_test.run(model, env, evaluator)

print()
print('=' * 60)
print('Innovation 2: Visual Robustness')
print('=' * 60)
from src.evaluation.robustness_analysis import RobustnessAnalyzer, RobustnessConfig
rob_config = RobustnessConfig(n_episodes_per_condition=${N_EPISODES}, compare_zero_shot=False)
rob_analyzer = RobustnessAnalyzer(rob_config)
rob_results = rob_analyzer.run(model, env, evaluator)

print()
print('=' * 60)
print('Innovation 3: Action Chunking Ablation')
print('=' * 60)
from src.evaluation.chunking_ablation import ChunkingAblation, ChunkingAblationConfig
chunk_config = ChunkingAblationConfig(n_episodes_per_config=${N_EPISODES})
chunk_study = ChunkingAblation(chunk_config)
chunk_results = chunk_study.run(model, env, evaluator)

print()
print('=' * 60)
print('Innovation 4: Computational Efficiency')
print('=' * 60)
from src.evaluation.efficiency_benchmark import EfficiencyBenchmark, EfficiencyConfig
eff_config = EfficiencyConfig(n_eval_episodes=${N_EPISODES})
eff_bench = EfficiencyBenchmark(eff_config)
eff_results = eff_bench.run('${CHECKPOINT}', env, evaluator)

# Generate all plots
print()
print('=' * 60)
print('Generating Visualizations')
print('=' * 60)
from src.visualization.plot_results import ResultVisualizer
viz = ResultVisualizer()
viz.plot_language_generalization_matrix(lg_results)
viz.plot_robustness_heatmap(rob_results)
viz.plot_chunking_ablation(chunk_results)
viz.plot_efficiency_benchmark(eff_results)

print()
print('✅ All evaluations complete!')
print(f'Results saved to: ./results/')
print(f'Figures saved to: ./results/figures/')

env.close()
"

echo ""
echo "✅ All evaluation experiments complete!"
echo "📊 Check ./results/ for detailed results and figures."
