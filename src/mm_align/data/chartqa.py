from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mm_align.config import DatasetSourceConfig, ProjectConfig
from mm_align.data.common import add_mismatch_paths, finalize_frame, json_dumps, resolve_image_path
from mm_align.utils.io import write_parquet


def prepare_chartqa(config: ProjectConfig) -> dict[str, Path]:
    dataset_cfg = config.datasets.chartqa
    root = Path(dataset_cfg.path)
    if not root.exists():
        return {}

    outputs: dict[str, Path] = {}
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        rows = []
        for json_file in sorted(split_dir.glob("*.json")):
            with json_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            split_name = split_dir.name
            subset_name = _subset_name(json_file.name)
            for index, item in enumerate(payload):
                image_field = item.get("imgname") or item.get("image") or item.get("image_path")
                if not image_field:
                    continue
                image_path = resolve_image_path(
                    base_dir=split_dir / "png",
                    image_path=image_field,
                    image_root=dataset_cfg.image_root or split_dir / "png",
                )
                answer = item.get("answer", item.get("label", item.get("answers", "")))
                rows.append(
                    {
                        "sample_id": f"chartqa-{split_name}-{subset_name}-{index}",
                        "dataset": "chartqa",
                        "split": split_name,
                        "image_path": str(image_path),
                        "prompt": item.get("question", item.get("query", "")),
                        "chosen": "",
                        "rejected": "",
                        "ground_truth": _serialize_answer(answer),
                        "metadata": json_dumps({"subset": subset_name, "source_file": json_file.name}),
                        "mismatch_image_path": "",
                    }
                )

        if not rows:
            continue
        frame = add_mismatch_paths(finalize_frame(rows))
        output_path = config.runtime.processed_dir / "chartqa" / f"{split_dir.name}.parquet"
        write_parquet(output_path, frame)
        outputs[split_dir.name] = output_path
    return outputs


def _subset_name(filename: str) -> str:
    lowered = filename.lower()
    if "human" in lowered:
        return "human"
    if "augmented" in lowered:
        return "augmented"
    return "default"


def _serialize_answer(answer: Any) -> str:
    if isinstance(answer, list):
        return " || ".join(str(item) for item in answer)
    return str(answer)
