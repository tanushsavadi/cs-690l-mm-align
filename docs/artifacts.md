# Artifact Guide

## Why Artifacts Matter

The project is artifact-driven because full training and evaluation are slow.
The heavy work was done in Colab, and the local dashboard/report read saved
outputs.

This makes the project easier to resume, debug, and present.

## Final Run Folders

Each final run lives under:

```text
artifacts/runs/<run_id>/
```

The final run ids are:

```text
2026-04-08-standard_dpo-pilot-7
2026-04-08-image_aware_dpo-pilot-7
```

## Expected Files Per Run

Each completed run should contain:

```text
adapter/
config.yaml
dashboard_examples.parquet
dashboard_summary.parquet
dependence.jsonl
env.json
metrics.json
predictions.jsonl
preferences.parquet
```

The most important files are:

- `adapter/`: LoRA adapter weights.
- `metrics.json`: training and final metric records.
- `predictions.jsonl`: all benchmark predictions.
- `dependence.jsonl`: original/blank/mismatch comparison data.
- `dashboard_summary.parquet`: dashboard metric summaries.
- `dashboard_examples.parquet`: dashboard qualitative examples.

## Expected Prediction Counts

Each complete run should have:

- `36147` total predictions
- `1129` original HallusionBench predictions
- `9000` original POPE predictions
- `1920` original ChartQA predictions

The total is larger than the benchmark sample count because each sample is
evaluated with three image variants.

## Statistical Evidence Artifacts

The statistical report command writes files used by the final report and
dashboard:

```text
reports/statistical_summary.md
reports/statistical_summary.csv
reports/paired_case_counts.csv
reports/representative_failure_cases.csv
```

It also writes dashboard-ready evidence parquet files under `artifacts/runs/`
or the configured artifact directory:

```text
evidence_summary.parquet
evidence_bootstrap_deltas.parquet
evidence_paired_cases.parquet
evidence_failure_cases.parquet
```

## Validation Commands

Check run files:

```bash
ls artifacts/runs/2026-04-08-standard_dpo-pilot-7
ls artifacts/runs/2026-04-08-image_aware_dpo-pilot-7
```

Build dashboard data:

```bash
python -m mm_align.cli build-dashboard-data --run 2026-04-08-standard_dpo-pilot-7
python -m mm_align.cli build-dashboard-data --run 2026-04-08-image_aware_dpo-pilot-7
```

Build statistical evidence:

```bash
python -m mm_align.cli build-statistical-report \
  --artifacts-dir artifacts/runs \
  --reports-dir reports
```

## GitHub Note

Large raw datasets and model checkpoints are not all stored directly in GitHub.
That is intentional. The repo stores source code, configs, docs, reports, and
small final analysis artifacts. Heavy raw data and adapter outputs can live in
Google Drive or a local artifact directory.
