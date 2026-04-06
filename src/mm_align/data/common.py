from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from mm_align.utils.images import load_image, save_image

INTERNAL_COLUMNS = [
    "sample_id",
    "dataset",
    "split",
    "image_path",
    "prompt",
    "chosen",
    "rejected",
    "ground_truth",
    "metadata",
    "mismatch_image_path",
]


def extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            parts.append(extract_text(item))
        return " ".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if "text" in content and content.get("type", "text") == "text":
            return str(content.get("text") or "").strip()
        if "content" in content:
            return extract_text(content["content"])
        values = [extract_text(value) for value in content.values()]
        return " ".join(value for value in values if value).strip()
    return str(content).strip()


def json_dumps(payload: Any) -> str:
    return json.dumps(payload or {}, ensure_ascii=True, sort_keys=True)


def resolve_image_path(base_dir: Path, image_path: str | Path, image_root: Path | None = None) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    if image_root is not None:
        rooted = (image_root / candidate).resolve()
        if rooted.exists():
            return rooted
    return (base_dir / candidate).resolve()


def coerce_and_save_image(
    image_value: Any,
    destination_dir: Path,
    sample_id: str,
    image_root: Path | None = None,
    base_dir: Path | None = None,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image_value, Image.Image):
        image = image_value.convert("RGB")
    elif isinstance(image_value, dict) and image_value.get("path"):
        path = Path(image_value["path"])
        if path.exists():
            return path.resolve()
        if base_dir is not None:
            path = resolve_image_path(base_dir, image_value["path"], image_root=image_root)
            if path.exists():
                return path
        raise FileNotFoundError(f"Image path not found for sample {sample_id}: {image_value['path']}")
    elif isinstance(image_value, (str, Path)):
        path = resolve_image_path(base_dir or Path.cwd(), image_value, image_root=image_root)
        if path.exists():
            return path
        raise FileNotFoundError(f"Image path not found for sample {sample_id}: {image_value}")
    else:
        image = image_value.convert("RGB")

    return save_image(image, destination_dir / f"{sample_id}.png")


def add_mismatch_paths(frame: pd.DataFrame) -> pd.DataFrame:
    updated = frame.copy()
    if updated.empty:
        updated["mismatch_image_path"] = pd.Series(dtype="string")
        return updated
    paths = updated["image_path"].astype(str).tolist()
    if len(paths) == 1:
        updated["mismatch_image_path"] = paths
        return updated
    offset = max(1, len(paths) // 2)
    mismatch = [paths[(index + offset) % len(paths)] for index in range(len(paths))]
    updated["mismatch_image_path"] = mismatch
    return updated


def finalize_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in INTERNAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[INTERNAL_COLUMNS].copy()


def normalize_yes_no(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "yes", "true"}:
        return "yes"
    if text in {"0", "no", "false"}:
        return "no"
    return text


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def open_image_for_record(path: str | Path) -> Image.Image:
    return load_image(path)
