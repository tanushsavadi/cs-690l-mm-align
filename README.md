# Multimodal Hallucination Alignment Project

This repository implements a greenfield research scaffold for studying whether
image-aware preference optimization reduces hallucinations in multimodal
language models.

The codebase is organized around three model variants:

- `base`
- `standard_dpo`
- `image_aware_dpo`

The default model target is `Qwen2.5-VL-3B-Instruct`. The default preference
dataset is `trl-lib/rlaif-v`, while `HallusionBench`, `POPE`, and `ChartQA`
cover evaluation and capability-retention checks. A Streamlit dashboard loads
cached experiment artifacts and never serves live model inference.

## Current Full Pilot Result

The corrected full pilot comparison uses these two completed runs:

- `2026-04-08-standard_dpo-pilot-7`
- `2026-04-08-image_aware_dpo-pilot-7`

Both runs finished full pilot training, full benchmark evaluation, and dashboard
artifact generation. The final headline metrics are:

| Benchmark | Metric | standard_dpo | image_aware_dpo |
| --- | --- | ---: | ---: |
| ChartQA | relaxed accuracy | 0.3797 | 0.4063 |
| HallusionBench | accuracy | 0.5722 | 0.5740 |
| POPE | accuracy | 0.8733 | 0.8718 |
| POPE | F1 | 0.8830 | 0.8814 |

The result is mixed but useful. Image-aware DPO improves ChartQA and is slightly
higher on HallusionBench, while standard DPO is slightly higher on POPE. The
dependence analysis also shows image-aware DPO changes answers slightly more
often when images are blanked or mismatched. The effect is modest rather than a
clear overall win.

See `reports/final_results.md` for the corrected result table, dependence
summary, and interpretation notes. The final submission report is available at
`reports/final_report.pdf`, with source in `reports/final_report.tex`.

## Quick Start

Create an environment, install the package, and prepare data:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m mm_align.cli prepare-data --config configs/smoke.yaml
```

Run training on a CUDA-backed environment such as Google Colab Pro:

```bash
bash scripts/colab_setup.sh
bash scripts/colab_smoke_run.sh
# after smoke succeeds, use the default full pilot:
python -m mm_align.cli prepare-data --config configs/pilot.yaml
python -m mm_align.cli train-dpo --config configs/pilot.yaml
python -m mm_align.cli train-imgaware --config configs/pilot.yaml
python -m mm_align.cli evaluate --config configs/pilot.yaml --run 2026-04-08-standard_dpo-pilot-7
python -m mm_align.cli evaluate --config configs/pilot.yaml --run 2026-04-08-image_aware_dpo-pilot-7
python -m mm_align.cli build-dashboard-data --run 2026-04-08-standard_dpo-pilot-7
python -m mm_align.cli build-dashboard-data --run 2026-04-08-image_aware_dpo-pilot-7
```

If you need a smaller fallback for a T4 or an unstable runtime, use
`configs/pilot_t4.yaml`. `configs/pilot_full.yaml` remains available as an
explicit alias of the original larger pilot configuration.

Launch the dashboard locally:

```bash
bash scripts/run_dashboard.sh
```

If launching manually, prefer `python3 -m streamlit run app/dashboard.py`
instead of `streamlit run app/dashboard.py`. This avoids accidentally using a
different Homebrew Python environment from the one where the project was
installed.

The dashboard pages are:

- `Overview`: metrics for one selected run.
- `Story Map`: a professor-facing visual story with a method/benchmark network,
  Sankey flow, delta heatmap, dependence tree, and radar fingerprint.
- `Examples`: qualitative predictions for original, blank, and mismatched
  image cases.
- `Preferences`: saved preference examples and training margins.
- `Failures`: failure-tagged prediction examples.
- `Comparison`: side-by-side metric comparison between completed runs, including
  delta bars, a heatmap, and a metric tree.
- `Dependence`: blank-image and mismatched-image sensitivity summaries, with a
  dependence tree and perturbation example explorer.
- `Training`: cached training curves from `metrics.json`, including a phase
  space view and final training fingerprint.

The dashboard intentionally does not run live VLM inference on the laptop. Live
testing would require loading the Qwen2.5-VL model and LoRA adapters, which is
why the heavy model work is done on Colab and the local dashboard reads saved
artifacts.

## Key Design Choices

- The MacBook is treated as a development and dashboard machine, not a serious
  multimodal training target.
- The training path uses the current official TRL multimodal DPO stack.
- VLM DPO runs set `max_length: null` by default to avoid truncating image
  tokens.
- Every experiment writes a stable artifact bundle under
  `artifacts/runs/<run_id>/`.
- Colab runs can redirect raw data, processed data, and artifacts to Drive by
  setting `MM_ALIGN_RAW_DIR`, `MM_ALIGN_PROCESSED_DIR`, and
  `MM_ALIGN_ARTIFACTS_DIR` before calling the CLI.

## Project Layout

```text
app/                  Streamlit multipage dashboard
configs/              Smoke, pilot, and main experiment configs
notebooks/colab/      Colab setup and workflow scaffolding
scripts/              Small helper scripts
src/mm_align/         Package code
tests/                Unit and integration tests
```

## Data Expectations

`prepare-data` normalizes datasets into a shared schema with these core fields:

- `sample_id`
- `dataset`
- `split`
- `image_path`
- `prompt`
- `chosen`
- `rejected`
- `ground_truth`
- `metadata`
- `mismatch_image_path`

Processed outputs are cached to `data/processed/<dataset>/<split>.parquet`.

## Notes

- Training commands fail fast when CUDA is missing.
- The image-aware trainer is intentionally marked as mDPO-inspired, not a
  paper-faithful reimplementation.
- If benchmark raw files are not present locally, the configuration can point
  to local raw paths or supported Hugging Face sources where available.
