# Dashboard Guide

## Launching The Dashboard

From the repo root:

```bash
cd "/Users/tanushsavadi/Documents/CS 690L"
bash scripts/run_dashboard.sh
```

The script sets `PYTHONPATH` and uses the Python environment that has the
project dependencies installed. This avoids the common issue where `streamlit`
comes from a different Python install than `plotly`, `pandas`, or the local
package.

## What The Dashboard Does

The dashboard is not a live model demo. It does not load Qwen2.5-VL or run
inference on the laptop. That is intentional because the model is too heavy for
normal local demo use.

Instead, the dashboard reads cached artifacts from completed Colab runs:

- `metrics.json`
- `predictions.jsonl`
- `dependence.jsonl`
- `dashboard_summary.parquet`
- `dashboard_examples.parquet`
- statistical evidence parquet files

This makes the dashboard reliable for presentation.

## Recommended Presentation Flow

For the final video, use this order:

1. `Story Map`
2. `Evidence`
3. `Dependence`
4. optional: `Examples`

Avoid spending time on `Preferences` in the recorded video because some raw
training images may not exist locally. The page is still useful for inspecting
text examples and training margins, but it is not the strongest presentation
page.

## Page Guide

### Overview

Shows metrics for one selected run. Useful for quick inspection, but not the
main final story.

### Story Map

This is the best opening page. It gives the visual argument:

- ChartQA is the clearest gain.
- HallusionBench is nearly tied.
- POPE slightly favors standard DPO.
- The final result is mixed but useful.

Use this page to explain the project at a high level.

### Evidence

This is the proof page. It contains:

- bootstrap confidence intervals
- bootstrap delta distributions
- paired win/loss matrix
- dependence evidence
- representative cases

This page is important because it prevents overclaiming. It shows that ChartQA
has the strongest evidence, while HallusionBench and POPE are near ties or
slight standard-DPO advantages.

### Dependence

This page shows whether answers changed under blank and mismatched images. It
supports the claim that image-aware DPO made the model a little more sensitive
to visual perturbations.

### Examples

This page gives concrete qualitative examples. Use it only if there is enough
time in the presentation.

### Comparison

Shows side-by-side metrics and deltas. It is useful for debugging and checking
numbers, but the `Evidence` page is stronger for the final presentation.

### Training

Shows training curves and fingerprints. It is useful for proving both training
runs completed, but it should not be the main result page.

## Common Dashboard Issues

### `ModuleNotFoundError: No module named 'app'`

Run the dashboard from the repo root using:

```bash
bash scripts/run_dashboard.sh
```

### `ModuleNotFoundError: No module named 'plotly'`

This usually means Streamlit is using a different Python environment. Use the
launcher script instead of calling `streamlit` directly.

### Missing raw image warning

Some image paths point to Colab or Google Drive paths. The dashboard handles
this by showing a warning instead of crashing. The final dashboard story does
not depend on local raw image availability.
