# Final Video Story

This is the simple story I should tell in the final video.

## 1. Problem

Vision language models can answer questions about images, but they can also
hallucinate. The answer can sound confident even when the image does not support
it.

The project asks whether preference tuning can make the model more grounded in
the image.

## 2. Method

I compare two versions of Qwen2.5-VL-3B-Instruct.

- `standard_dpo`: normal preference tuning.
- `image_aware_dpo`: preference tuning with an extra image mismatch pressure.

The idea is simple. A good answer should not just sound good. It should depend
on the right image.

## 3. Setup

The training data comes from RLAIF-V. The full pilot uses 8000 preference
examples and LoRA adapters, so I am not training the whole model from scratch.

The evaluation uses three benchmarks:

- ChartQA for chart question answering.
- HallusionBench for multimodal hallucination.
- POPE for object existence hallucination.

Each benchmark is evaluated with original, blank, and mismatched images.

## 4. Main Results

The result is mixed, not a clean sweep.

- ChartQA improves from 0.3797 to 0.4063.
- HallusionBench is basically tied, 0.5722 vs 0.5740.
- POPE slightly favors standard DPO, 0.8733 vs 0.8718.

So the best claim is not that image-aware DPO wins everywhere. The best claim is
that it gives modest grounding-related gains, strongest on ChartQA.

## 5. Statistical Evidence

The bootstrap analysis makes the story clearer.

- ChartQA has a positive delta of +0.0266 with a confidence interval from
  +0.0177 to +0.0359.
- HallusionBench has a very small delta and the interval crosses zero.
- POPE slightly favors standard DPO.

The paired win/loss counts also support this:

- ChartQA: image-aware wins 64 cases and standard wins 13.
- HallusionBench: image-aware wins 16 and standard wins 14.
- POPE: image-aware wins 27 and standard wins 41.

## 6. Dashboard Demo Path

Open the dashboard and show:

1. `Story Map`: explain the whole project quickly.
2. `Evidence`: show confidence intervals and paired win/loss counts.
3. `Dependence`: show blank and mismatched image sensitivity.
4. `Examples`: show a few concrete qualitative examples.

## 7. Limitations

This is a pilot study. It uses one base model, one final seed, and expensive
Colab GPU runs. Colab also made the project hard to rerun because GPU
availability and runtime length were not guaranteed.

The image-aware method is also simple. It uses mismatched images, but it does
not solve hallucination by itself.

## 8. Final Takeaway

The project worked end to end. It prepared data, trained two adapters, evaluated
them on three benchmarks, built a dashboard, and added statistical evidence.

The final answer is:

Image-aware DPO helps in some ways, especially on ChartQA, but it is not a full
solution. It gives a small grounding benefit while staying close to standard DPO
on the other benchmarks.
