# Run Real LIBERO Training and Promotion

This tutorial covers the real benchmark path in this repository:

1. bootstrap LeRobot
2. train or load a SmolVLA policy
3. evaluate it on LIBERO
4. promote one eval run into the repository benchmark registry
5. regenerate committed benchmark figures, docs, and README assets

## 1. Install and bootstrap

```bash
python3 -m pip install -e .
bash scripts/bootstrap_lerobot.sh
```

The bootstrap script installs LeRobot under `third_party/lerobot`, but canonical outputs are not written there.

## 2. Train

```bash
RUN_NAME=smolvla_ft \
HF_USER=your_hf_user \
POLICY_REPO_ID=your_hf_user/libero-smolvla-demo \
MUJOCO_GL=egl \
bash scripts/train_libero_smolvla.sh
```

Default raw training output:

- `results/raw/train/<run_name>/`

With the example above:

- `results/raw/train/smolvla_ft/`

## 3. Evaluate

```bash
RUN_NAME=smolvla_eval \
POLICY_PATH=your_hf_user/libero-smolvla-demo \
TASKS=libero_10 \
MUJOCO_GL=egl \
bash scripts/eval_libero_smolvla.sh
```

Default raw eval output:

- `results/raw/eval/<run_name>/`

Expected LeRobot eval artifact:

- `results/raw/eval/<run_name>/eval_info.json`

The promotion pipeline reads `eval_info.json` and uses it as the source of success rate, episode count, and episode time.

## 4. Promote one eval run

Use `portfolio-vla-promote-real` to normalize one eval run into the benchmark registry:

```bash
portfolio-vla-promote-real \
  --eval-dir results/raw/eval/smolvla_eval \
  --run-name ft_nominal \
  --experiment-family generalization \
  --model-variant finetuned \
  --task-suite libero_10 \
  --language-variant exact \
  --spatial-variant nominal \
  --visual-variant nominal \
  --chunk-size 8 \
  --quantization fp16 \
  --policy-label smolvla_finetuned \
  --checkpoint-step 20000 \
  --policy-path your_hf_user/libero-smolvla-demo \
  --latency-ms 96 \
  --rollout-hz 10.4 \
  --trajectory-jerk 0.22
```

Why these explicit fields are required:

- `eval_info.json` contains success and timing data
- benchmark condition metadata such as language/spatial/visual variants must still be declared by the caller
- latency, rollout rate, and jerk are not guaranteed to exist in raw LeRobot eval output, so this repository accepts them as explicit promotion inputs

## 5. Published outputs refreshed by promotion

Promotion updates:

- `results/real/benchmark_registry.csv`
- `results/real/summary/summary.md`
- `results/real/figures/*.png`
- `examples/real_benchmark_results.csv`
- `reports/real/*.png`
- `docs/results/latest_real_results.md`

After that, refresh the compact README assets:

```bash
python3 scripts/export_readme_assets.py \
  --showcase-dir artifacts/showcase \
  --real-dir results/real \
  --output assets/readme
```

## 6. Coverage model

Promotion is incremental.

One promoted run produces one normalized registry row. To fill the benchmark charts, you need multiple promoted rows covering:

- generalization: exact/paraphrase and spatial variants
- visual robustness: nominal and perturbation variants
- chunking: multiple chunk sizes
- latency: multiple quantization settings

If a chart has partial coverage, the repository still renders it using only the rows currently present in the registry.
