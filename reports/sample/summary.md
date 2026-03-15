# Experiment Summary

## Key findings

- Fine-tuning improves nominal LIBERO success from 72% to 89%.
- Under the hardest language+spatial conditions, zero-shot averages 34% while fine-tuned averages 70%.
- Visual perturbation drop shrinks from 21% to 11% after fine-tuning.
- Best realtime chunk size is 8, with 89% success at 10.4 Hz.
- Best quantization setting above 10 Hz is int8, with 87% success at 14.7 Hz.

## Suggested resume framing

- Built a simulation-first benchmark for SmolVLA on LIBERO Panda tasks, covering language, spatial, and visual generalization.
- Quantified the accuracy-latency trade-off of action chunking and model quantization for realtime robotic control.
