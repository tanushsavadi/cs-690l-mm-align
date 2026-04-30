# Final Pilot Results

This note records the corrected full pilot results after both evaluation runs
finished and dashboard artifacts were rebuilt.

## Runs

- `2026-04-08-standard_dpo-pilot-7`
- `2026-04-08-image_aware_dpo-pilot-7`

Both runs completed:

- full pilot training with 8000 training examples
- evaluation on HallusionBench, POPE, and ChartQA
- original, blank-image, and mismatched-image variants
- dashboard artifact generation

## Evaluation Sanity Check

The corrected image-aware prediction file has the expected count:

| Benchmark | Original predictions |
| --- | ---: |
| HallusionBench | 1129 |
| POPE | 9000 |
| ChartQA | 1920 |

The full prediction count is `36147`, equal to:

```text
(1129 + 9000 + 1920) * 3 image variants
```

This matters because an earlier resume bug caused HallusionBench duplicate
sample IDs to be skipped incorrectly. That bug was fixed before these final
numbers were recorded.

## Headline Metrics

| Benchmark | Metric | standard_dpo | image_aware_dpo | Direction |
| --- | --- | ---: | ---: | --- |
| ChartQA | relaxed accuracy | 0.3797 | 0.4063 | image-aware higher |
| HallusionBench | accuracy | 0.5722 | 0.5740 | image-aware slightly higher |
| POPE | accuracy | 0.8733 | 0.8718 | standard slightly higher |
| POPE | F1 | 0.8830 | 0.8814 | standard slightly higher |

## ChartQA

Image-aware DPO improves ChartQA:

| Split | standard_dpo | image_aware_dpo |
| --- | ---: | ---: |
| augmented relaxed accuracy | 0.4990 | 0.5292 |
| human relaxed accuracy | 0.2604 | 0.2833 |
| overall relaxed accuracy | 0.3797 | 0.4063 |

This is the clearest positive result for the image-aware objective.

## HallusionBench

Image-aware DPO is slightly higher overall:

| Metric | standard_dpo | image_aware_dpo |
| --- | ---: | ---: |
| overall accuracy | 0.5722 | 0.5740 |
| category VD accuracy | 0.5550 | 0.5550 |
| category VS accuracy | 0.5911 | 0.5948 |

The difference is very small, so this should be described as roughly tied with
a slight image-aware advantage, not as a large improvement.

## POPE

Standard DPO is slightly higher on POPE:

| Metric | standard_dpo | image_aware_dpo |
| --- | ---: | ---: |
| accuracy | 0.8733 | 0.8718 |
| precision | 0.8650 | 0.8624 |
| recall | 0.9018 | 0.9014 |
| F1 | 0.8830 | 0.8814 |
| yes ratio | 0.5202 | 0.5216 |

The difference is small. This is best interpreted as no meaningful POPE gain
from image-aware DPO in this pilot.

## Dependence Metrics

| Metric | standard_dpo | image_aware_dpo |
| --- | ---: | ---: |
| blank changed rate | 0.9667 | 0.9705 |
| mismatch changed rate | 0.9285 | 0.9361 |
| blank score drop mean | 0.4272 | 0.4200 |
| mismatch score drop mean | 0.3508 | 0.3508 |

Image-aware DPO changes answers slightly more often when the image is blanked
or mismatched. That supports the idea that it made the model a bit more
responsive to visual perturbations. However, the score drop metrics do not show
a large advantage, so the grounding improvement is modest.

## Final Interpretation

The hypothesis is partially supported.

Image-aware DPO improves ChartQA and is slightly higher on HallusionBench. It
also increases answer sensitivity under blank and mismatched images. On POPE,
standard DPO is slightly better, but the gap is very small.

The clean conclusion is:

> Image-aware DPO produced modest grounding-related gains in this pilot,
> especially on ChartQA, while staying roughly competitive with standard DPO on
> hallucination benchmarks. It did not dominate standard DPO across all metrics.

This is still a useful result because it shows that the image-aware objective
changes behavior in the intended direction, but the effect size is small.

## Dashboard Notes

The dashboard is now meant to be part of the final presentation, not just a
debugging tool. The strongest page to open first is `Story Map`. It gives a
quick visual explanation of the project using:

- a method-to-benchmark network graph
- a Sankey style flow from preference data to model outcomes
- a metric delta heatmap
- a dependence tree
- a radar style dependence fingerprint

The dashboard also has separate pages for detailed comparison, dependence
analysis, training curves, qualitative examples, preference examples, and
failure tags.

The dashboard does not run live model inference locally. That is intentional.
The model work was done on Colab because the Qwen2.5-VL model and adapters are
too heavy for a normal laptop workflow. The local dashboard reads cached
artifacts and is used to explain the completed experiment.

## Reporting Notes

- Do not use the old image-aware HallusionBench value of `0.1250`.
- Do not use the old POPE zero result from before yes/no normalization was fixed.
- Use the corrected run IDs listed above for all final tables.
- The dashboard artifacts should be regenerated after any evaluation refresh.
