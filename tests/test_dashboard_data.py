from pathlib import Path

import pandas as pd

from mm_align.eval.dashboard_data import build_dashboard_artifacts
from mm_align.utils.io import write_json, write_jsonl


def test_build_dashboard_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "2026-04-06-standard_dpo-pilot-7"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "metrics.json",
        {
            "evaluation": {
                "benchmarks": {
                    "hallusionbench": {"accuracy": 0.5},
                    "pope": {"accuracy": 0.75, "precision": 0.8},
                }
            }
        },
    )
    write_jsonl(
        run_dir / "predictions.jsonl",
        [
            {
                "run_id": run_dir.name,
                "model_variant": "standard_dpo",
                "benchmark": "hallusionbench",
                "sample_id": "s1",
                "prompt": "prompt",
                "ground_truth": "yes",
                "image_path": "image.png",
                "metadata": "{}",
                "image_variant": "original",
                "prediction": "yes",
                "is_correct": True,
            }
        ],
    )
    write_jsonl(
        run_dir / "dependence.jsonl",
        [
            {
                "run_id": run_dir.name,
                "model_variant": "standard_dpo",
                "benchmark": "hallusionbench",
                "sample_id": "s1",
                "prompt": "prompt",
                "ground_truth": "yes",
                "image_path": "image.png",
                "metadata": "{}",
                "prediction_original": "yes",
                "is_correct_original": True,
                "prediction_blank-image": "no",
                "is_correct_blank-image": False,
                "prediction_mismatched-image": "yes",
                "is_correct_mismatched-image": True,
                "blank_changed": True,
                "blank_score_drop": 1,
                "mismatch_changed": False,
                "mismatch_score_drop": 0,
            }
        ],
    )
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "image_path": "image.png",
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected",
                "matched_margin": 0.4,
                "mismatched_margin": 0.1,
            }
        ]
    ).to_parquet(run_dir / "preferences.parquet", index=False)

    build_dashboard_artifacts(tmp_path / "runs", run_dir.name)

    assert (run_dir / "dashboard_examples.parquet").exists()
    assert (run_dir / "dashboard_summary.parquet").exists()
