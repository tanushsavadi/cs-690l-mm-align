from __future__ import annotations

import json
from pathlib import Path

from mm_align.config import DatasetSourceConfig, ProjectConfig
from mm_align.data.common import add_mismatch_paths, finalize_frame, json_dumps, normalize_yes_no, resolve_image_path
from mm_align.utils.io import write_parquet


def prepare_pope(config: ProjectConfig) -> Path | None:
    dataset_cfg = config.datasets.pope
    root = Path(dataset_cfg.path)
    if not root.exists():
        return None

    rows = []
    json_files = sorted(root.rglob("*.json"))
    for json_file in json_files:
        variant = _detect_variant(json_file.name)
        payload = _load_pope_payload(json_file)
        for index, item in enumerate(payload):
            sample_id = f"pope-{variant}-{item.get('question_id', index)}"
            image_path = resolve_image_path(
                base_dir=root,
                image_path=item.get("image", ""),
                image_root=dataset_cfg.image_root,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": "pope",
                    "split": dataset_cfg.split,
                    "image_path": str(image_path),
                    "prompt": item.get("text", item.get("question", "")),
                    "chosen": "",
                    "rejected": "",
                    "ground_truth": normalize_yes_no(item.get("label", "")),
                    "metadata": json_dumps({"variant": variant, "source_file": json_file.name}),
                    "mismatch_image_path": "",
                }
            )

    if not rows:
        return None
    frame = add_mismatch_paths(finalize_frame(rows))
    output_path = config.runtime.processed_dir / "pope" / f"{dataset_cfg.split}.parquet"
    write_parquet(output_path, frame)
    return output_path


def _detect_variant(filename: str) -> str:
    lowered = filename.lower()
    for variant in ("random", "popular", "adversarial"):
        if variant in lowered:
            return variant
    return "unknown"


def _load_pope_payload(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        return payload if isinstance(payload, list) else [payload]
    return [json.loads(line) for line in text.splitlines() if line.strip()]
