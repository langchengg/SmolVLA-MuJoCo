# Run the Mock MuJoCo Showcase

This repository keeps a separate simulation-only demo for visual storytelling and JSONL/CSV examples.

It is intentionally separate from the real benchmark publication path.

## 1. Generate the showcase

```bash
python3 -m pip install -e .
python3 scripts/run_showcase_demo.py --output artifacts/showcase
```

Outputs:

- `artifacts/showcase/episodes/desktop_sorting_eval_log.jsonl`
- `artifacts/showcase/summary/desktop_sorting_eval_summary.csv`
- `artifacts/showcase/summary/summary.md`
- `artifacts/showcase/figures/*.png`
- `artifacts/showcase/videos/*.mp4`
- `artifacts/showcase/videos/hero_showcase.gif`

## 2. Generate the mock benchmark sample

```bash
python3 scripts/run_mock_benchmark.py --output artifacts/mock_benchmark
```

Outputs:

- `artifacts/mock_benchmark/summary/mock_benchmark_results.csv`
- `artifacts/mock_benchmark/summary/summary.md`
- `artifacts/mock_benchmark/figures/*.png`

## 3. Export README demo assets

```bash
python3 scripts/export_readme_assets.py \
  --showcase-dir artifacts/showcase \
  --real-dir results/real \
  --output assets/readme
```

The export step copies:

- real benchmark overview/charts from `results/real`
- demo GIF/video and demo figures from `artifacts/showcase`

## 4. Committed mock aliases

The repository keeps lightweight committed mock/demo aliases here:

- `examples/mock_desktop_sorting_eval_log.jsonl`
- `examples/mock_desktop_sorting_eval_summary.csv`
- `examples/mock_benchmark_results.csv`
- `reports/mock/*.png`

These are useful for:

- showcasing the JSONL/CSV schema
- README demo media
- regression tests for the simulation-only path
