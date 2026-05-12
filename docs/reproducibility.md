# Reproducibility And Colab Runbook

## Important Reality

This project is reproducible in code, but the full run is not lightweight. The
training and evaluation were done through Google Colab over several weeks. The
MacBook was used for development, analysis, report writing, and dashboard
presentation, not for heavy VLM training.

The larger pilot runs needed stronger Colab GPU runtimes when available, such
as G4, A100, and H100 class machines. Even then, the full pipeline took hours.
Evaluation was especially slow because every benchmark sample is tested with
three image variants.

Because of that, the project is designed around Drive-backed artifacts and
resume-safe evaluation.

## Local Setup

Use this for local development and dashboard work:

```bash
cd "/Users/tanushsavadi/Documents/CS 690L"
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On this Mac, the dashboard script prefers the Python 3.12 framework install if
it exists:

```bash
bash scripts/run_dashboard.sh
```

## Colab Setup

In Colab, mount Drive first:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then clone or enter the repo:

```python
from pathlib import Path

REPO_ROOT = Path("/content/CS 690L")
if REPO_ROOT.exists():
    %cd "/content/CS 690L"
    !git pull origin main
else:
    !git clone https://github.com/tanushsavadi/cs-690l-mm-align.git "/content/CS 690L"
    %cd "/content/CS 690L"
```

Set Drive-backed paths:

```python
import os

os.environ["MM_ALIGN_RAW_DIR"] = "/content/drive/MyDrive/mm-align/data/raw"
os.environ["MM_ALIGN_PROCESSED_DIR"] = "/content/drive/MyDrive/mm-align/data/processed"
os.environ["MM_ALIGN_ARTIFACTS_DIR"] = "/content/drive/MyDrive/mm-align/artifacts/runs"
```

Install the package:

```python
%cd "/content/CS 690L"
!bash scripts/colab_setup.sh
!python -m mm_align.cli --help
```

## Recommended Run Order

Always run smoke first:

```python
%cd "/content/CS 690L"
!bash scripts/colab_smoke_run.sh
```

Then run the pilot:

```python
%cd "/content/CS 690L"
!python -m mm_align.cli prepare-data --config configs/pilot.yaml
!python -m mm_align.cli train-dpo --config configs/pilot.yaml
!python -m mm_align.cli train-imgaware --config configs/pilot.yaml
```

The final run ids from the completed project were:

```python
RUN_STD = "2026-04-08-standard_dpo-pilot-7"
RUN_IMG = "2026-04-08-image_aware_dpo-pilot-7"
```

Evaluate:

```python
%cd "/content/CS 690L"
!python -m mm_align.cli evaluate --config configs/pilot.yaml --run {RUN_STD}
!python -m mm_align.cli evaluate --config configs/pilot.yaml --run {RUN_IMG}
```

Build dashboard data:

```python
%cd "/content/CS 690L"
!python -m mm_align.cli build-dashboard-data --run {RUN_STD} --artifacts-dir "$MM_ALIGN_ARTIFACTS_DIR"
!python -m mm_align.cli build-dashboard-data --run {RUN_IMG} --artifacts-dir "$MM_ALIGN_ARTIFACTS_DIR"
```

Build statistical evidence:

```python
%cd "/content/CS 690L"
!python -m mm_align.cli build-statistical-report \
  --artifacts-dir "$MM_ALIGN_ARTIFACTS_DIR" \
  --reports-dir reports
```

## Resume Behavior

Evaluation writes predictions while it runs. If Colab disconnects, the next
evaluation command can resume from the cached predictions instead of starting
from zero.

This mattered a lot in practice because evaluation took several hours and
runtime disconnects were possible.

## Validation Check

After evaluation, verify the image-aware prediction counts:

```python
import pandas as pd
from pathlib import Path

ART = Path("/content/drive/MyDrive/mm-align/artifacts/runs")
pred = pd.read_json(ART / "2026-04-08-image_aware_dpo-pilot-7" / "predictions.jsonl", lines=True)

print("total predictions:", len(pred))
for bench in ["hallusionbench", "pope", "chartqa"]:
    original = pred[(pred["benchmark"] == bench) & (pred["image_variant"] == "original")]
    print(bench, "original:", len(original))
```

Expected:

```text
total predictions: 36147
hallusionbench original: 1129
pope original: 9000
chartqa original: 1920
```

## What Not To Rerun Unless Necessary

Do not rerun full training or full evaluation unless something is actually
broken. The completed artifacts are already the final source of truth. Full
runs are expensive and can consume many hours of Colab time.
