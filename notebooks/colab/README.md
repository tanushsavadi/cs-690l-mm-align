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

5. Only after both smoke runs pass, move to pilot:

```python
!python -m mm_align.cli prepare-data --config configs/pilot.yaml
!python -m mm_align.cli train-dpo --config configs/pilot.yaml
!python -m mm_align.cli train-imgaware --config configs/pilot.yaml
```

6. Evaluate each finished pilot run and build dashboard artifacts:

```python
!python -m mm_align.cli evaluate --config configs/pilot.yaml --run <run_id>
!python -m mm_align.cli build-dashboard-data --run <run_id>
```

The Colab notebook should set the environment variables inside Python before
calling shell commands so the Drive-backed paths persist across cells.
