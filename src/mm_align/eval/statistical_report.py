from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from mm_align.eval.metrics import pope_metrics
from mm_align.utils.io import read_jsonl


FINAL_BASELINE_RUN = "2026-04-08-standard_dpo-pilot-7"
FINAL_CANDIDATE_RUN = "2026-04-08-image_aware_dpo-pilot-7"

PREDICTION_KEYS = [
    "benchmark",
    "sample_id",
    "prompt",
    "ground_truth",
    "image_path",
    "metadata",
    "image_variant",
]
DEPENDENCE_KEYS = [
    "benchmark",
    "sample_id",
    "prompt",
    "ground_truth",
    "image_path",
    "metadata",
]


@dataclass(frozen=True)
class StatisticalReportPaths:
    summary_csv: Path
    summary_md: Path
    paired_cases_csv: Path
    failure_cases_csv: Path
    evidence_summary_parquet: Path
    bootstrap_deltas_parquet: Path
    paired_cases_parquet: Path
    failure_cases_parquet: Path


def build_statistical_report(
    artifacts_dir: Path,
    reports_dir: Path,
    baseline_run: str = FINAL_BASELINE_RUN,
    candidate_run: str = FINAL_CANDIDATE_RUN,
    bootstrap_samples: int = 1000,
    seed: int = 7,
) -> StatisticalReportPaths:
    """Build artifact-only evidence files for the final comparison."""

    artifacts_dir = artifacts_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline_predictions = _load_predictions(artifacts_dir, baseline_run)
    candidate_predictions = _load_predictions(artifacts_dir, candidate_run)
    baseline_dependence = _load_dependence(artifacts_dir, baseline_run)
    candidate_dependence = _load_dependence(artifacts_dir, candidate_run)

    paired_originals = _paired_original_predictions(baseline_predictions, candidate_predictions)
    paired_dependence = _paired_dependence(baseline_dependence, candidate_dependence)

    prediction_summary, prediction_bootstrap = _prediction_metric_summary(
        paired_originals,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    dependence_summary, dependence_bootstrap = _dependence_metric_summary(
        paired_dependence,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 17,
    )
    summary = pd.concat([prediction_summary, dependence_summary], ignore_index=True)
    bootstrap_deltas = pd.concat([prediction_bootstrap, dependence_bootstrap], ignore_index=True)
    paired_cases = _paired_case_counts(paired_originals)
    failure_cases = _representative_failure_cases(paired_originals, paired_dependence)

    summary_csv = reports_dir / "statistical_summary.csv"
    summary_md = reports_dir / "statistical_summary.md"
    paired_cases_csv = reports_dir / "paired_case_counts.csv"
    failure_cases_csv = reports_dir / "representative_failure_cases.csv"
    evidence_summary_parquet = artifacts_dir / "evidence_summary.parquet"
    bootstrap_deltas_parquet = artifacts_dir / "evidence_bootstrap_deltas.parquet"
    paired_cases_parquet = artifacts_dir / "evidence_paired_cases.parquet"
    failure_cases_parquet = artifacts_dir / "evidence_failure_cases.parquet"

    summary.to_csv(summary_csv, index=False)
    paired_cases.to_csv(paired_cases_csv, index=False)
    failure_cases.to_csv(failure_cases_csv, index=False)
    summary.to_parquet(evidence_summary_parquet, index=False)
    bootstrap_deltas.to_parquet(bootstrap_deltas_parquet, index=False)
    paired_cases.to_parquet(paired_cases_parquet, index=False)
    failure_cases.to_parquet(failure_cases_parquet, index=False)
    summary_md.write_text(
        _render_markdown_report(summary, paired_cases, failure_cases, baseline_run, candidate_run),
        encoding="utf-8",
    )

    return StatisticalReportPaths(
        summary_csv=summary_csv,
        summary_md=summary_md,
        paired_cases_csv=paired_cases_csv,
        failure_cases_csv=failure_cases_csv,
        evidence_summary_parquet=evidence_summary_parquet,
        bootstrap_deltas_parquet=bootstrap_deltas_parquet,
        paired_cases_parquet=paired_cases_parquet,
        failure_cases_parquet=failure_cases_parquet,
    )


def _load_predictions(artifacts_dir: Path, run_id: str) -> pd.DataFrame:
    path = artifacts_dir / run_id / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    frame = pd.DataFrame(read_jsonl(path))
    if frame.empty:
        raise ValueError(f"No predictions found in {path}")
    return frame


def _load_dependence(artifacts_dir: Path, run_id: str) -> pd.DataFrame:
    path = artifacts_dir / run_id / "dependence.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dependence file: {path}")
    frame = pd.DataFrame(read_jsonl(path))
    if frame.empty:
        raise ValueError(f"No dependence rows found in {path}")
    return frame


def _paired_original_predictions(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    baseline_original = baseline[baseline["image_variant"] == "original"].copy()
    candidate_original = candidate[candidate["image_variant"] == "original"].copy()
    paired = candidate_original.merge(
        baseline_original,
        on=PREDICTION_KEYS,
        how="inner",
        suffixes=("_candidate", "_baseline"),
    )
    expected = min(len(baseline_original), len(candidate_original))
    if len(paired) != expected:
        raise ValueError(f"Could only pair {len(paired)} original predictions; expected {expected}.")
    return paired


def _paired_dependence(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    paired = candidate.merge(
        baseline,
        on=DEPENDENCE_KEYS,
        how="inner",
        suffixes=("_candidate", "_baseline"),
    )
    expected = min(len(baseline), len(candidate))
    if len(paired) != expected:
        raise ValueError(f"Could only pair {len(paired)} dependence rows; expected {expected}.")
    return paired


def _prediction_metric_summary(paired: pd.DataFrame, bootstrap_samples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_specs: list[tuple[str, str, Callable[[pd.DataFrame, str], float]]] = [
        ("chartqa", "relaxed_accuracy", _mean_correct),
        ("hallusionbench", "accuracy", _mean_correct),
        ("pope", "accuracy", _pope_accuracy),
        ("pope", "f1", _pope_f1),
    ]
    rows = []
    bootstrap_rows = []
    rng = np.random.default_rng(seed)
    for benchmark, metric, metric_fn in metric_specs:
        benchmark_frame = paired[paired["benchmark"] == benchmark].copy()
        if benchmark_frame.empty:
            continue
        baseline_value = metric_fn(benchmark_frame, "baseline")
        candidate_value = metric_fn(benchmark_frame, "candidate")
        deltas = _bootstrap_deltas(benchmark_frame, metric_fn, bootstrap_samples, rng)
        bootstrap_rows.extend(_bootstrap_rows("benchmark", benchmark, metric, deltas))
        rows.append(
            _summary_row(
                source="benchmark",
                benchmark=benchmark,
                metric=metric,
                n=len(benchmark_frame),
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                deltas=deltas,
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def _dependence_metric_summary(paired: pd.DataFrame, bootstrap_samples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_specs = [
        ("all", "blank_changed_rate", "blank_changed"),
        ("all", "mismatch_changed_rate", "mismatch_changed"),
        ("all", "blank_score_drop_mean", "blank_score_drop"),
        ("all", "mismatch_score_drop_mean", "mismatch_score_drop"),
    ]
    rows = []
    bootstrap_rows = []
    rng = np.random.default_rng(seed)
    for benchmark, metric, column in metric_specs:
        baseline_column = f"{column}_baseline"
        candidate_column = f"{column}_candidate"
        baseline_value = float(paired[baseline_column].mean())
        candidate_value = float(paired[candidate_column].mean())
        deltas = _bootstrap_column_deltas(paired, baseline_column, candidate_column, bootstrap_samples, rng)
        bootstrap_rows.extend(_bootstrap_rows("dependence", benchmark, metric, deltas))
        rows.append(
            _summary_row(
                source="dependence",
                benchmark=benchmark,
                metric=metric,
                n=len(paired),
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                deltas=deltas,
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def _bootstrap_rows(source: str, benchmark: str, metric: str, deltas: np.ndarray) -> list[dict[str, float | int | str]]:
    return [
        {
            "source": source,
            "benchmark": benchmark,
            "metric": metric,
            "sample": int(index),
            "delta": float(delta),
        }
        for index, delta in enumerate(deltas)
    ]


def _mean_correct(frame: pd.DataFrame, side: str) -> float:
    return float(frame[f"is_correct_{side}"].mean())


def _pope_accuracy(frame: pd.DataFrame, side: str) -> float:
    pope_frame = _pope_frame_for_side(frame, side)
    return float(pope_metrics(pope_frame)["accuracy"])


def _pope_f1(frame: pd.DataFrame, side: str) -> float:
    pope_frame = _pope_frame_for_side(frame, side)
    return float(pope_metrics(pope_frame)["f1"])


def _pope_frame_for_side(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction": frame[f"prediction_{side}"],
            "ground_truth": frame["ground_truth"],
        }
    )


def _bootstrap_deltas(
    frame: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame, str], float],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    deltas = np.empty(bootstrap_samples, dtype=float)
    row_count = len(frame)
    for index in range(bootstrap_samples):
        sample_positions = rng.integers(0, row_count, row_count)
        sample = frame.iloc[sample_positions]
        deltas[index] = metric_fn(sample, "candidate") - metric_fn(sample, "baseline")
    return deltas


def _bootstrap_column_deltas(
    frame: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    deltas = np.empty(bootstrap_samples, dtype=float)
    row_count = len(frame)
    for index in range(bootstrap_samples):
        sample_positions = rng.integers(0, row_count, row_count)
        sample = frame.iloc[sample_positions]
        deltas[index] = float(sample[candidate_column].mean() - sample[baseline_column].mean())
    return deltas


def _summary_row(
    source: str,
    benchmark: str,
    metric: str,
    n: int,
    baseline_value: float,
    candidate_value: float,
    deltas: np.ndarray,
) -> dict[str, float | int | str]:
    delta = candidate_value - baseline_value
    return {
        "source": source,
        "benchmark": benchmark,
        "metric": metric,
        "n": n,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "delta": delta,
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "prob_candidate_higher": float((deltas > 0).mean()),
    }


def _paired_case_counts(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark, group in paired.groupby("benchmark"):
        candidate_correct = group["is_correct_candidate"].astype(bool)
        baseline_correct = group["is_correct_baseline"].astype(bool)
        rows.extend(
            [
                _case_row(benchmark, "image_aware_only_correct", candidate_correct & ~baseline_correct),
                _case_row(benchmark, "standard_only_correct", baseline_correct & ~candidate_correct),
                _case_row(benchmark, "both_correct", baseline_correct & candidate_correct),
                _case_row(benchmark, "both_wrong", ~baseline_correct & ~candidate_correct),
            ]
        )
    return pd.DataFrame(rows)


def _case_row(benchmark: str, case_type: str, mask: pd.Series) -> dict[str, float | int | str]:
    count = int(mask.sum())
    total = int(len(mask))
    return {
        "benchmark": benchmark,
        "case_type": case_type,
        "count": count,
        "rate": count / max(1, total),
        "total": total,
    }


def _representative_failure_cases(paired: pd.DataFrame, dependence: pd.DataFrame, per_type: int = 8) -> pd.DataFrame:
    examples = []
    candidate_correct = paired["is_correct_candidate"].astype(bool)
    baseline_correct = paired["is_correct_baseline"].astype(bool)
    case_masks = {
        "image_aware_win": candidate_correct & ~baseline_correct,
        "standard_win": baseline_correct & ~candidate_correct,
        "both_fail": ~baseline_correct & ~candidate_correct,
        "both_correct": baseline_correct & candidate_correct,
    }
    for case_type, mask in case_masks.items():
        examples.append(_sample_cases(paired[mask], case_type, per_type))
    grounded = dependence[
        (dependence["blank_changed_candidate"].astype(bool) | dependence["mismatch_changed_candidate"].astype(bool))
        & dependence["is_correct_original_candidate"].astype(bool)
    ].copy()
    if not grounded.empty:
        grounded = grounded.rename(
            columns={
                "prediction_original_candidate": "prediction_candidate",
                "prediction_original_baseline": "prediction_baseline",
                "is_correct_original_candidate": "is_correct_candidate",
                "is_correct_original_baseline": "is_correct_baseline",
            }
        )
        examples.append(_sample_cases(grounded, "grounded_perturbation_sensitive", per_type))
    if not examples:
        return pd.DataFrame()
    return pd.concat([frame for frame in examples if not frame.empty], ignore_index=True)


def _sample_cases(frame: pd.DataFrame, case_type: str, limit: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    selected = frame.sort_values(["benchmark", "sample_id"]).head(limit).copy()
    selected["case_type"] = case_type
    columns = [
        "case_type",
        "benchmark",
        "sample_id",
        "prompt",
        "ground_truth",
        "prediction_baseline",
        "prediction_candidate",
        "is_correct_baseline",
        "is_correct_candidate",
    ]
    return selected[[column for column in columns if column in selected.columns]]


def _render_markdown_report(
    summary: pd.DataFrame,
    paired_cases: pd.DataFrame,
    failure_cases: pd.DataFrame,
    baseline_run: str,
    candidate_run: str,
) -> str:
    headline = summary[summary["source"] == "benchmark"].copy()
    dependence = summary[summary["source"] == "dependence"].copy()
    lines = [
        "# Statistical Evidence Summary",
        "",
        "This file is generated from cached artifacts only. No model was rerun.",
        "",
        f"- Baseline run: `{baseline_run}`",
        f"- Candidate run: `{candidate_run}`",
        "",
        "## Bootstrap Metric Deltas",
        "",
        _markdown_table(headline),
        "",
        "The delta column is `image_aware_dpo - standard_dpo`. A confidence interval crossing zero means the result should be described as a small or uncertain effect, not a clean win.",
        "",
        "## Dependence Deltas",
        "",
        _markdown_table(dependence),
        "",
        "These rows measure how much answers change when images are blanked or mismatched. Higher changed rates suggest more visual sensitivity, but score-drop metrics still need to be interpreted carefully.",
        "",
        "## Paired Case Counts",
        "",
        _markdown_table(paired_cases),
        "",
        "## Representative Case Types",
        "",
        _markdown_table(failure_cases.head(20)),
        "",
        "## Short Interpretation",
        "",
        "The strongest result is still ChartQA, where image-aware DPO is higher. HallusionBench is basically a tie with a tiny image-aware edge. POPE slightly favors standard DPO. The honest conclusion is that image-aware DPO gives modest grounding-related gains, not a universal improvement.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    columns = [str(column) for column in display.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in display.iterrows():
        values = [_escape_markdown_cell(row[column]) for column in display.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _escape_markdown_cell(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")
