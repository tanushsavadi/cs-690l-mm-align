from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mm_align.eval.metrics import tag_failure
from mm_align.utils.io import read_json, read_jsonl


def build_dashboard_artifacts(artifacts_dir: Path, run_id: str) -> None:
    run_dir = artifacts_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    metrics = read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    predictions = pd.DataFrame(read_jsonl(run_dir / "predictions.jsonl")) if (run_dir / "predictions.jsonl").exists() else pd.DataFrame()
    dependence = pd.DataFrame(read_jsonl(run_dir / "dependence.jsonl")) if (run_dir / "dependence.jsonl").exists() else pd.DataFrame()
    preferences = pd.read_parquet(run_dir / "preferences.parquet") if (run_dir / "preferences.parquet").exists() else pd.DataFrame()

    examples = _build_examples(predictions, dependence)
    summary = _build_summary(metrics, run_id)

    examples.to_parquet(run_dir / "dashboard_examples.parquet", index=False)
    summary.to_parquet(run_dir / "dashboard_summary.parquet", index=False)
    if not preferences.empty and not (run_dir / "preferences.parquet").exists():
        preferences.to_parquet(run_dir / "preferences.parquet", index=False)


def _build_examples(predictions: pd.DataFrame, dependence: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    originals = predictions[predictions["image_variant"] == "original"].copy()
    originals = originals.rename(
        columns={
            "prediction": "prediction_original",
            "is_correct": "is_correct_original",
        }
    )
    merged = originals.merge(
        dependence,
        on=["run_id", "model_variant", "benchmark", "sample_id", "prompt", "ground_truth", "image_path", "metadata"],
        how="left",
        suffixes=("", "_dependence"),
    )
    merged["failure_tag"] = merged.apply(tag_failure, axis=1)
    return merged


def _build_summary(metrics: dict[str, Any], run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evaluation = metrics.get("evaluation", {})
    for benchmark, payload in evaluation.get("benchmarks", {}).items():
        for key, value in payload.items():
            if isinstance(value, dict):
                for subkey, subvalue in _flatten_nested_metrics(f"{key}", value):
                    rows.append({"run_id": run_id, "benchmark": benchmark, "metric": subkey, "value": subvalue})
            else:
                rows.append({"run_id": run_id, "benchmark": benchmark, "metric": key, "value": value})
    if not rows:
        return pd.DataFrame(columns=["run_id", "benchmark", "metric", "value"])
    return pd.DataFrame(rows)


def _flatten_nested_metrics(prefix: str, payload: dict[str, Any]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for key, value in payload.items():
        name = f"{prefix}.{key}"
        if isinstance(value, dict):
            rows.extend(_flatten_nested_metrics(name, value))
        else:
            rows.append((name, float(value)))
    return rows
