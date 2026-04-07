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
# after smoke succeeds, use the reduced T4-safe pilot:
python -m mm_align.cli prepare-data --config configs/pilot.yaml
python -m mm_align.cli train-dpo --config configs/pilot.yaml
python -m mm_align.cli train-imgaware --config configs/pilot.yaml
python -m mm_align.cli evaluate --config configs/pilot.yaml --run 2026-04-06-standard_dpo-pilot-7
python -m mm_align.cli build-dashboard-data --run 2026-04-06-standard_dpo-pilot-7
```

If you have a stronger GPU or a more reliable runtime, `configs/pilot_full.yaml`
preserves the original larger pilot configuration.

Launch the dashboard locally:

```bash
streamlit run app/dashboard.py
```

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
