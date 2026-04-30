# Colab Workflow

Use Google Colab Pro for all training and bulk evaluation runs.

## Recommended Sequence

1. Mount Google Drive.
2. Point `MM_ALIGN_RAW_DIR`, `MM_ALIGN_PROCESSED_DIR`, and `MM_ALIGN_ARTIFACTS_DIR`
   at a Drive-backed location.
3. Run `bash scripts/colab_setup.sh`.
4. Run the smoke workflow first:

```python
!bash scripts/colab_smoke_run.sh
```

5. Only after both smoke runs pass, move to pilot. The default `pilot.yaml`
   is the full pilot configuration. If you are on a T4 or an unstable runtime,
   use `pilot_t4.yaml` instead:

```python
!python -m mm_align.cli prepare-data --config configs/pilot.yaml
!python -m mm_align.cli train-dpo --config configs/pilot.yaml
!python -m mm_align.cli train-imgaware --config configs/pilot.yaml
```

Reduced T4-safe fallback:

```python
!python -m mm_align.cli prepare-data --config configs/pilot_t4.yaml
!python -m mm_align.cli train-dpo --config configs/pilot_t4.yaml
!python -m mm_align.cli train-imgaware --config configs/pilot_t4.yaml
```

6. Evaluate each finished pilot run and build dashboard artifacts:

```python
RUN_STD = "2026-04-08-standard_dpo-pilot-7"
RUN_IMG = "2026-04-08-image_aware_dpo-pilot-7"

!python -m mm_align.cli evaluate --config configs/pilot.yaml --run {RUN_STD}
!python -m mm_align.cli evaluate --config configs/pilot.yaml --run {RUN_IMG}
!python -m mm_align.cli build-dashboard-data --run {RUN_STD} --artifacts-dir "$MM_ALIGN_ARTIFACTS_DIR"
!python -m mm_align.cli build-dashboard-data --run {RUN_IMG} --artifacts-dir "$MM_ALIGN_ARTIFACTS_DIR"
```

The Colab notebook should set the environment variables inside Python before
calling shell commands so the Drive-backed paths persist across cells.

## Completed Full Pilot Runs

The final full pilot artifacts are:

- `2026-04-08-standard_dpo-pilot-7`
- `2026-04-08-image_aware_dpo-pilot-7`

Both should contain `metrics.json`, `predictions.jsonl`, `dependence.jsonl`,
`dashboard_examples.parquet`, and `dashboard_summary.parquet`.

If a rerun resumes from cached predictions, verify that the image-aware run has
the full expected prediction count:

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

Expected output:

```text
total predictions: 36147
hallusionbench original: 1129
pope original: 9000
chartqa original: 1920
```
