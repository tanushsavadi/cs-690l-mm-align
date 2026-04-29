from pathlib import Path

import pandas as pd
from PIL import Image

from mm_align.config import ProjectConfig
from mm_align.eval.runner import run_evaluation
from mm_align.utils.io import read_json, read_jsonl, write_jsonl
from mm_align.utils.images import load_image


def test_load_image_retries_transient_oserror(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)

    import mm_align.utils.images as image_utils

    real_open = image_utils.Image.open
    attempts = {"count": 0}

    def flaky_open(path):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError("transient drive error")
        return real_open(path)

    monkeypatch.setattr(image_utils.Image, "open", flaky_open)
    monkeypatch.setattr(image_utils.time, "sleep", lambda _: None)

    image = load_image(image_path)

    assert image.mode == "RGB"
    assert attempts["count"] == 3


def test_run_evaluation_skips_unreadable_samples(tmp_path: Path, monkeypatch) -> None:
    config = ProjectConfig.model_validate(
        {
            "runtime": {
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "artifacts_dir": str(tmp_path / "artifacts"),
            }
        }
    )
    run_id = "2026-04-09-standard_dpo-pilot-7"

    frame = pd.DataFrame(
        [
            {
                "sample_id": "good",
                "prompt": "good prompt",
                "ground_truth": "yes",
                "image_path": "good.png",
                "mismatch_image_path": "",
                "metadata": "{}",
            },
            {
                "sample_id": "bad",
                "prompt": "bad prompt",
                "ground_truth": "yes",
                "image_path": "bad.png",
                "mismatch_image_path": "",
                "metadata": "{}",
            },
        ]
    )

    monkeypatch.setattr("mm_align.eval.runner.load_model_for_evaluation", lambda config, adapter_dir: ("model", "processor"))
    monkeypatch.setattr("mm_align.eval.runner._load_benchmark_frames", lambda config: {"hallusionbench": frame})

    def fake_generate_prediction(**kwargs):
        if kwargs["prompt"] == "bad prompt":
            raise OSError("drive read failed")
        return "yes"

    monkeypatch.setattr("mm_align.eval.runner.generate_prediction", fake_generate_prediction)

    run_evaluation(config, run_id)

    run_dir = config.runtime.artifacts_dir / run_id
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    metrics = read_json(run_dir / "metrics.json")

    assert all(row["sample_id"] == "good" for row in predictions)
    assert len(predictions) == 3
    assert metrics["evaluation"]["benchmarks"]["hallusionbench"]["accuracy"] == 1.0


def test_run_evaluation_reuses_cached_predictions_without_loading_model(tmp_path: Path, monkeypatch) -> None:
    config = ProjectConfig.model_validate(
        {
            "runtime": {
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "artifacts_dir": str(tmp_path / "artifacts"),
            }
        }
    )
    run_id = "2026-04-09-standard_dpo-pilot-7"
    run_dir = config.runtime.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        [
            {
                "sample_id": "pope-1",
                "prompt": "Is there a snowboard in the image?",
                "ground_truth": "yes",
                "image_path": "good.png",
                "mismatch_image_path": "other.png",
                "metadata": '{"variant":"adversarial"}',
            }
        ]
    )
    write_jsonl(
        run_dir / "predictions.jsonl",
        [
            {
                "run_id": run_id,
                "model_variant": "standard_dpo",
                "benchmark": "pope",
                "sample_id": "pope-1",
                "prompt": "Is there a snowboard in the image?",
                "ground_truth": "yes",
                "image_path": "good.png",
                "metadata": '{"variant":"adversarial"}',
                "image_variant": variant,
                "prediction": "Yes, there is a snowboard in the image.",
                "is_correct": False,
            }
            for variant in ("original", "blank-image", "mismatched-image")
        ],
    )

    monkeypatch.setattr("mm_align.eval.runner._load_benchmark_frames", lambda config: {"pope": frame})

    def fail_load_model(config, adapter_dir):
        raise AssertionError("evaluation should not reload the model when predictions are already cached")

    monkeypatch.setattr("mm_align.eval.runner.load_model_for_evaluation", fail_load_model)

    run_evaluation(config, run_id)

    predictions = read_jsonl(run_dir / "predictions.jsonl")
    metrics = read_json(run_dir / "metrics.json")
    dependence = read_jsonl(run_dir / "dependence.jsonl")

    assert len(predictions) == 3
    assert all(row["is_correct"] for row in predictions)
    assert metrics["evaluation"]["benchmarks"]["pope"]["accuracy"] == 1.0
    assert len(dependence) == 1


def test_run_evaluation_resume_distinguishes_duplicate_sample_ids(tmp_path: Path, monkeypatch) -> None:
    config = ProjectConfig.model_validate(
        {
            "runtime": {
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "artifacts_dir": str(tmp_path / "artifacts"),
            }
        }
    )
    run_id = "2026-04-09-image_aware_dpo-pilot-7"
    run_dir = config.runtime.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        [
            {
                "sample_id": "hallusionbench-0",
                "prompt": "First question?",
                "ground_truth": "yes",
                "image_path": "image-1.png",
                "mismatch_image_path": "",
                "metadata": '{"figure_id":"0"}',
            },
            {
                "sample_id": "hallusionbench-0",
                "prompt": "Second question?",
                "ground_truth": "no",
                "image_path": "image-2.png",
                "mismatch_image_path": "",
                "metadata": '{"figure_id":"1"}',
            },
        ]
    )
    write_jsonl(
        run_dir / "predictions.jsonl",
        [
            {
                "run_id": run_id,
                "model_variant": "image_aware_dpo",
                "benchmark": "hallusionbench",
                "sample_id": "hallusionbench-0",
                "prompt": "First question?",
                "ground_truth": "yes",
                "image_path": "image-1.png",
                "metadata": '{"figure_id":"0"}',
                "image_variant": variant,
                "prediction": "yes",
                "is_correct": True,
            }
            for variant in ("original", "blank-image", "mismatched-image")
        ],
    )

    monkeypatch.setattr("mm_align.eval.runner.load_model_for_evaluation", lambda config, adapter_dir: ("model", "processor"))
    monkeypatch.setattr("mm_align.eval.runner._load_benchmark_frames", lambda config: {"hallusionbench": frame})
    monkeypatch.setattr("mm_align.eval.runner.generate_prediction", lambda **kwargs: "no")

    run_evaluation(config, run_id)

    predictions = read_jsonl(run_dir / "predictions.jsonl")
    originals = [row for row in predictions if row["image_variant"] == "original"]

    assert len(predictions) == 6
    assert len(originals) == 2
    assert {row["prompt"] for row in originals} == {"First question?", "Second question?"}
