from pathlib import Path

import pandas as pd
from PIL import Image

from mm_align.config import ProjectConfig
from mm_align.eval.runner import run_evaluation
from mm_align.utils.io import read_json, read_jsonl
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
