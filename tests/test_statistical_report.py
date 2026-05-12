from pathlib import Path

from mm_align.eval.statistical_report import build_statistical_report
from mm_align.utils.io import write_jsonl


def _prediction(run_id: str, variant: str, sample_id: str, prediction: str, is_correct: bool) -> dict:
    return {
        "run_id": run_id,
        "model_variant": variant,
        "benchmark": "pope",
        "sample_id": sample_id,
        "prompt": f"Is object {sample_id} present?",
        "ground_truth": "yes",
        "image_path": f"{sample_id}.jpg",
        "metadata": '{"variant": "random"}',
        "image_variant": "original",
        "prediction": prediction,
        "is_correct": is_correct,
    }


def _dependence(run_id: str, variant: str, sample_id: str, blank_changed: bool) -> dict:
    return {
        "run_id": run_id,
        "model_variant": variant,
        "benchmark": "pope",
        "sample_id": sample_id,
        "prompt": f"Is object {sample_id} present?",
        "ground_truth": "yes",
        "image_path": f"{sample_id}.jpg",
        "metadata": '{"variant": "random"}',
        "prediction_original": "yes",
        "is_correct_original": True,
        "prediction_blank-image": "no" if blank_changed else "yes",
        "is_correct_blank-image": not blank_changed,
        "prediction_mismatched-image": "yes",
        "is_correct_mismatched-image": True,
        "blank_changed": blank_changed,
        "blank_score_drop": int(blank_changed),
        "mismatch_changed": False,
        "mismatch_score_drop": 0,
    }


def test_build_statistical_report_outputs_evidence_files(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    baseline = "2026-04-08-standard_dpo-pilot-7"
    candidate = "2026-04-08-image_aware_dpo-pilot-7"
    (artifacts_dir / baseline).mkdir(parents=True)
    (artifacts_dir / candidate).mkdir(parents=True)

    write_jsonl(
        artifacts_dir / baseline / "predictions.jsonl",
        [
            _prediction(baseline, "standard_dpo", "s1", "yes", True),
            _prediction(baseline, "standard_dpo", "s2", "no", False),
        ],
    )
    write_jsonl(
        artifacts_dir / candidate / "predictions.jsonl",
        [
            _prediction(candidate, "image_aware_dpo", "s1", "yes", True),
            _prediction(candidate, "image_aware_dpo", "s2", "yes", True),
        ],
    )
    write_jsonl(
        artifacts_dir / baseline / "dependence.jsonl",
        [_dependence(baseline, "standard_dpo", "s1", False), _dependence(baseline, "standard_dpo", "s2", False)],
    )
    write_jsonl(
        artifacts_dir / candidate / "dependence.jsonl",
        [
            _dependence(candidate, "image_aware_dpo", "s1", True),
            _dependence(candidate, "image_aware_dpo", "s2", False),
        ],
    )

    paths = build_statistical_report(
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        baseline_run=baseline,
        candidate_run=candidate,
        bootstrap_samples=25,
        seed=3,
    )

    assert paths.summary_md.exists()
    assert paths.summary_csv.exists()
    assert paths.paired_cases_csv.exists()
    assert paths.failure_cases_csv.exists()
    assert paths.evidence_summary_parquet.exists()
    assert paths.bootstrap_deltas_parquet.exists()
    assert paths.paired_cases_parquet.exists()
    assert paths.failure_cases_parquet.exists()
    assert "image-aware DPO" in paths.summary_md.read_text()
