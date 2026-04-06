import json
from pathlib import Path

import pandas as pd

from mm_align.config import ProjectConfig
from mm_align.data.hallusionbench import prepare_hallusionbench


def test_prepare_hallusionbench_uses_placeholder_for_missing_filename(tmp_path: Path) -> None:
    payload = [
        {
            "category": "VS",
            "subcategory": "chart",
            "visual_input": "0",
            "set_id": "0",
            "figure_id": "0",
            "sample_note": "example",
            "question_id": "0",
            "question": "Is the answer yes?",
            "gt_answer": "1",
            "filename": None,
        }
    ]
    source_path = tmp_path / "HallusionBench.json"
    source_path.write_text(json.dumps(payload), encoding="utf-8")

    config = ProjectConfig.model_validate(
        {
            "runtime": {
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "artifacts_dir": str(tmp_path / "artifacts"),
            },
            "datasets": {
                "hallusionbench": {"source": "local-json", "path": str(source_path), "split": "validation"},
            },
        }
    )

    output_path = prepare_hallusionbench(config)
    assert output_path is not None
    frame = pd.read_parquet(output_path)
    blank_image_path = tmp_path / "blank.png"
    assert blank_image_path.exists()
    assert frame.iloc[0]["image_path"] == str(blank_image_path.resolve())
    metadata = json.loads(frame.iloc[0]["metadata"])
    assert metadata["uses_placeholder_image"] is True
