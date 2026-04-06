from __future__ import annotations

import json
import math
import re
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def normalize_yes_no(value: Any) -> str:
    text = normalize_text(value)
    if text in {"1", "true", "yes"}:
        return "yes"
    if text in {"0", "false", "no"}:
        return "no"
    return text


def exact_match(prediction: Any, ground_truth: Any) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def relaxed_chartqa_match(prediction: Any, ground_truth: Any, tolerance: float = 0.05) -> bool:
    pred_text = normalize_text(prediction)
    gt_text = normalize_text(ground_truth)
    if pred_text == gt_text:
        return True

    pred_number = _extract_first_number(pred_text)
    gt_number = _extract_first_number(gt_text)
    if pred_number is None or gt_number is None:
        return False
    if gt_number == 0:
        return abs(pred_number) <= tolerance
    return abs(pred_number - gt_number) / abs(gt_number) <= tolerance


def pope_metrics(frame: pd.DataFrame) -> dict[str, float]:
    pred = frame["prediction"].map(normalize_yes_no)
    gt = frame["ground_truth"].map(normalize_yes_no)
    tp = int(((pred == "yes") & (gt == "yes")).sum())
    fp = int(((pred == "yes") & (gt == "no")).sum())
    fn = int(((pred == "no") & (gt == "yes")).sum())
    tn = int(((pred == "no") & (gt == "no")).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    accuracy = (tp + tn) / max(1, len(frame))
    yes_ratio = (pred == "yes").mean() if len(frame) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "yes_ratio": float(yes_ratio),
    }


def build_dependence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    wide = frame.pivot_table(
        index=["run_id", "model_variant", "benchmark", "sample_id", "prompt", "ground_truth", "image_path", "metadata"],
        columns="image_variant",
        values=["prediction", "is_correct"],
        aggfunc="first",
    )
    wide.columns = ["_".join(str(part) for part in column if part).strip("_") for column in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    if "prediction_original" in wide.columns and "prediction_blank-image" in wide.columns:
        wide["blank_changed"] = wide["prediction_original"] != wide["prediction_blank-image"]
        wide["blank_score_drop"] = wide["is_correct_original"].astype(int) - wide["is_correct_blank-image"].astype(int)
    else:
        wide["blank_changed"] = False
        wide["blank_score_drop"] = 0
    if "prediction_mismatched-image" in wide.columns:
        wide["mismatch_changed"] = wide["prediction_original"] != wide["prediction_mismatched-image"]
        wide["mismatch_score_drop"] = wide["is_correct_original"].astype(int) - wide["is_correct_mismatched-image"].astype(int)
    else:
        wide["mismatch_changed"] = False
        wide["mismatch_score_drop"] = 0
    return wide


def aggregate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {"benchmarks": {}}
    originals = frame[frame["image_variant"] == "original"].copy()
    if originals.empty:
        return metrics

    for benchmark, benchmark_frame in originals.groupby("benchmark"):
        benchmark_metrics: dict[str, Any] = {}
        if benchmark == "pope":
            benchmark_metrics.update(pope_metrics(benchmark_frame))
            by_variant = benchmark_frame["metadata"].map(lambda value: json.loads(value).get("variant", "unknown"))
            benchmark_metrics["by_variant"] = {}
            for variant, variant_frame in benchmark_frame.groupby(by_variant):
                benchmark_metrics["by_variant"][variant] = pope_metrics(variant_frame)
        elif benchmark == "chartqa":
            benchmark_metrics["relaxed_accuracy"] = float(
                benchmark_frame.apply(lambda row: relaxed_chartqa_match(row["prediction"], row["ground_truth"]), axis=1).mean()
            )
            subsets = benchmark_frame["metadata"].map(lambda value: json.loads(value).get("subset", "default"))
            benchmark_metrics["by_subset"] = {}
            for subset, subset_frame in benchmark_frame.groupby(subsets):
                benchmark_metrics["by_subset"][subset] = {
                    "relaxed_accuracy": float(
                        subset_frame.apply(lambda row: relaxed_chartqa_match(row["prediction"], row["ground_truth"]), axis=1).mean()
                    )
                }
        else:
            benchmark_metrics["accuracy"] = float(benchmark_frame["is_correct"].mean())
            categories = benchmark_frame["metadata"].map(lambda value: json.loads(value))
            if benchmark == "hallusionbench":
                benchmark_metrics["by_group"] = {}
                for field in ("category", "subcategory", "visual_input"):
                    groups = categories.map(lambda value: value.get(field, "unknown"))
                    benchmark_metrics["by_group"][field] = {
                        group: {"accuracy": float(group_frame["is_correct"].mean())}
                        for group, group_frame in benchmark_frame.groupby(groups)
                    }
        metrics["benchmarks"][benchmark] = benchmark_metrics
    return metrics


def tag_failure(row: pd.Series) -> str:
    if not bool(row.get("is_correct_original", True)):
        prediction = normalize_text(row.get("prediction_original", ""))
        gt = normalize_text(row.get("ground_truth", ""))
        if row.get("benchmark") == "pope" and prediction == "yes" and gt == "no":
            return "unsupported_object_mention"
        if row.get("mismatch_changed") and row.get("mismatch_score_drop", 0) > 0:
            return "image_dependence_regression"
        return "incorrect_original"
    if row.get("blank_changed") or row.get("mismatch_changed"):
        return "grounded_response"
    return "stable_response"


def _extract_first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    value = float(match.group(0))
    if math.isfinite(value):
        return value
    return None
