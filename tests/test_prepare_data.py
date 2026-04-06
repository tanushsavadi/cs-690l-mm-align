from pathlib import Path

import pandas as pd
from PIL import Image

from mm_align.config import ProjectConfig
from mm_align.data.preparation import prepare_all_datasets


def test_prepare_data_from_local_parquet(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for index in range(6):
        image_path = image_dir / f"{index}.png"
        Image.new("RGB", (8, 8), color=(index * 10, index * 10, index * 10)).save(image_path)
        rows.append(
            {
                "id": f"sample-{index}",
                "question": f"question {index}",
                "chosen": f"chosen {index}",
                "rejected": f"rejected {index}",
                "image_path": str(image_path),
            }
        )
    training_parquet = tmp_path / "training.parquet"
    pd.DataFrame(rows).to_parquet(training_parquet, index=False)

    config = ProjectConfig.model_validate(
        {
            "runtime": {
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "artifacts_dir": str(tmp_path / "artifacts"),
            },
            "datasets": {
                "training": {"source": "local-parquet", "path": str(training_parquet), "split": "train"},
                "training_fallback": {"source": "local-parquet", "path": str(training_parquet), "split": "train", "enabled": False},
                "vlfeedback": {"source": "local-parquet", "path": str(training_parquet), "split": "train", "enabled": False},
                "hallusionbench": {"source": "local-json", "path": str(tmp_path / "missing.json"), "split": "validation"},
                "pope": {"source": "local-dir", "path": str(tmp_path / "missing-pope"), "split": "validation"},
                "chartqa": {"source": "local-dir", "path": str(tmp_path / "missing-chartqa"), "split": "val"},
            },
            "training": {"subset_name": "smoke", "subset_size": 4, "val_size": 2},
        }
    )
    outputs = prepare_all_datasets(config)
    train_path = outputs["rlaif-v"]["train"]
    frame = pd.read_parquet(train_path)
    assert "mismatch_image_path" in frame.columns
    assert len(frame) == 4

    val_path = outputs["rlaif-v"]["val"]
    val_frame = pd.read_parquet(val_path)
    assert len(val_frame) == 2


def test_prepare_data_only_materializes_requested_subset(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for index in range(12):
        image_path = image_dir / f"{index}.png"
        Image.new("RGB", (8, 8), color=(index * 5, index * 5, index * 5)).save(image_path)
        rows.append(
            {
                "id": f"pilot-{index}",
                "question": f"question {index}",
                "chosen": f"chosen {index}",
                "rejected": f"rejected {index}",
                "image_path": str(image_path),
            }
        )
    training_parquet = tmp_path / "pilot-training.parquet"
    pd.DataFrame(rows).to_parquet(training_parquet, index=False)

    config = ProjectConfig.model_validate(
        {
            "runtime": {
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "artifacts_dir": str(tmp_path / "artifacts"),
            },
            "datasets": {
                "training": {"source": "local-parquet", "path": str(training_parquet), "split": "train"},
                "training_fallback": {"source": "local-parquet", "path": str(training_parquet), "split": "train", "enabled": False},
                "vlfeedback": {"source": "local-parquet", "path": str(training_parquet), "split": "train", "enabled": False},
                "hallusionbench": {"source": "local-json", "path": str(tmp_path / "missing.json"), "split": "validation"},
                "pope": {"source": "local-dir", "path": str(tmp_path / "missing-pope"), "split": "validation"},
                "chartqa": {"source": "local-dir", "path": str(tmp_path / "missing-chartqa"), "split": "val"},
            },
            "training": {"subset_name": "pilot", "subset_size": 8, "val_size": 2},
        }
    )

    outputs = prepare_all_datasets(config)["rlaif-v"]
    assert outputs["train"].name == "train.parquet"
    assert outputs["pilot"].name == "pilot.parquet"
    assert outputs["smoke"].name == "smoke.parquet"
    assert outputs["val"].name == "val.parquet"
    assert "main" not in outputs

    pilot_frame = pd.read_parquet(outputs["pilot"])
    smoke_frame = pd.read_parquet(outputs["smoke"])
    val_frame = pd.read_parquet(outputs["val"])
    assert len(pilot_frame) == 8
    assert len(smoke_frame) == 8
    assert len(val_frame) == 2
