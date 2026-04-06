from __future__ import annotations

import json
from pathlib import Path

from mm_align.config import DatasetSourceConfig, ProjectConfig
from mm_align.data.common import add_mismatch_paths, finalize_frame, json_dumps, normalize_yes_no, resolve_image_path
from mm_align.utils.images import make_blank_image, save_image
from mm_align.utils.io import write_parquet


def prepare_hallusionbench(config: ProjectConfig) -> Path | None:
    dataset_cfg = config.datasets.hallusionbench
    source_path = Path(dataset_cfg.path)
    if not source_path.exists():
        return None

    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    base_dir = source_path.parent
    blank_image_path = _ensure_blank_image(base_dir)
    rows = []
    for index, item in enumerate(payload):
        sample_id = f"hallusionbench-{item.get('question_id', index)}"
        filename = item.get("filename")
        if filename:
            image_path = resolve_image_path(base_dir, filename, image_root=dataset_cfg.image_root)
        else:
            image_path = blank_image_path
        metadata = {
            "category": item.get("category"),
            "subcategory": item.get("subcategory"),
            "visual_input": item.get("visual_input"),
            "figure_id": item.get("figure_id"),
            "set_id": item.get("set_id"),
            "sample_note": item.get("sample_note"),
            "uses_placeholder_image": not bool(filename),
        }
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": "hallusionbench",
                "split": dataset_cfg.split,
                "image_path": str(image_path),
                "prompt": item.get("question", ""),
                "chosen": "",
                "rejected": "",
                "ground_truth": normalize_yes_no(item.get("gt_answer", "")),
                "metadata": json_dumps(metadata),
                "mismatch_image_path": "",
            }
        )

    frame = add_mismatch_paths(finalize_frame(rows))
    output_path = config.runtime.processed_dir / "hallusionbench" / f"{dataset_cfg.split}.parquet"
    write_parquet(output_path, frame)
    return output_path


def _ensure_blank_image(base_dir: Path) -> Path:
    blank_path = base_dir / "blank.png"
    if blank_path.exists():
        return blank_path.resolve()
    return save_image(make_blank_image(), blank_path).resolve()
