# Project Overview

## What This Project Is

This project studies multimodal hallucination alignment. The basic problem is
that vision-language models can answer questions about images, but they can
also say things that are not actually supported by the image.

The research question is:

> Can image-aware preference tuning make a vision-language model more grounded
> in the image?

The base model is `Qwen2.5-VL-3B-Instruct`. I compare two LoRA fine-tuned
versions of the same model:

- `standard_dpo`: normal Direct Preference Optimization.
- `image_aware_dpo`: a modified DPO-style method that also uses mismatched
  images to pressure the model to care about the visual input.

## Why This Matters

Standard preference tuning can teach a model which answer sounds better, but in
multimodal settings the better answer should also be supported by the image.
This project tests whether adding image awareness to the preference objective
changes the model behavior in the intended direction.

## Data And Evaluation

Training uses `trl-lib/rlaif-v` preference examples. The full pilot used:

- 8000 training examples
- 1000 validation examples
- 1000 optimizer steps for each adapter
- LoRA/QLoRA-style training on Colab GPUs

Evaluation uses three benchmarks:

- `ChartQA`: chart question answering.
- `HallusionBench`: multimodal hallucination and visual illusion.
- `POPE`: object hallucination / object existence questions.

Each evaluation sample is tested with:

- the original image
- a blank image
- a mismatched image

This lets the project measure both normal benchmark performance and whether the
model reacts when the image changes.

## Final Result

| Benchmark | Metric | standard_dpo | image_aware_dpo |
| --- | --- | ---: | ---: |
| ChartQA | relaxed accuracy | 0.3797 | 0.4063 |
| HallusionBench | accuracy | 0.5722 | 0.5740 |
| POPE | accuracy | 0.8733 | 0.8718 |
| POPE | F1 | 0.8830 | 0.8814 |

The result is mixed. Image-aware DPO helps most on ChartQA. HallusionBench is a
near tie. POPE slightly favors standard DPO.

The statistical analysis supports the same interpretation:

- ChartQA delta is `+0.0266`, with a bootstrap interval above zero.
- HallusionBench delta is `+0.0018`, with an interval crossing zero.
- POPE delta is about `-0.0016`, so standard DPO is slightly ahead.

## Final Claim

The strongest honest claim is:

> Image-aware DPO gives modest grounding-related gains, strongest on ChartQA,
> while staying close to standard DPO on hallucination benchmarks.

It should not be described as a universal win or a solved hallucination method.
It is better framed as a careful pilot study with useful evidence and clear
limitations.

## Main Contributions

- Built a full data preparation, training, evaluation, and dashboard pipeline.
- Trained standard DPO and image-aware DPO adapters on Qwen2.5-VL-3B.
- Evaluated both models on three benchmarks with three image variants.
- Added resumable prediction caching because Colab runs were long and unstable.
- Fixed benchmark scoring/resume issues found during evaluation.
- Added bootstrap confidence intervals and paired win/loss analysis.
- Built a dashboard that tells the final result visually.
