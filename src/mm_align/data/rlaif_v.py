from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mm_align.config import DatasetSourceConfig, ProjectConfig
from mm_align.data.common import add_mismatch_paths, coerce_and_save_image, extract_text, finalize_frame, json_dumps
from mm_align.utils.io import write_parquet


def prepare_training_preferences(config: ProjectConfig) -> dict[str, Path]:
    dataset_cfg = config.datasets.training
    if not dataset_cfg.enabled:
        raise ValueError("Primary training dataset is disabled.")

    raw = _load_dataset(dataset_cfg)

    requested_train_size = config.training.subset_size
    total_needed = requested_train_size + config.training.val_size
    if isinstance(raw, pd.DataFrame):
        raw = raw.sample(frac=1.0, random_state=config.runtime.seed).reset_index(drop=True)
        select_count = min(total_needed, len(raw))
        raw_rows = raw.iloc[:select_count].to_dict(orient="records")
    else:
        raw = raw.shuffle(seed=config.runtime.seed)
        select_count = min(total_needed, len(raw))
        raw_rows = raw.select(range(select_count))

    image_dir = config.runtime.raw_dir / "rlaif-v" / dataset_cfg.split / "images"
    base_dir = Path(dataset_cfg.path) if dataset_cfg.source != "huggingface" else config.runtime.raw_dir
    rows = [_normalize_row(example, image_dir, index, dataset_cfg, base_dir) for index, example in enumerate(raw_rows)]
    frame = finalize_frame(rows)

    train_size = min(requested_train_size, len(frame))
    val_size = min(config.training.val_size, max(0, len(frame) - train_size))
    train_frame = add_mismatch_paths(frame.iloc[:train_size].reset_index(drop=True))
    val_frame = add_mismatch_paths(frame.iloc[train_size : train_size + val_size].reset_index(drop=True))
    subset_frames = {
        "smoke": add_mismatch_paths(train_frame.iloc[: min(1000, len(train_frame))].reset_index(drop=True)),
    }
    if len(train_frame) >= 8000 or config.training.subset_name in {"pilot", "main"}:
        subset_frames["pilot"] = add_mismatch_paths(train_frame.iloc[: min(8000, len(train_frame))].reset_index(drop=True))
    if len(train_frame) >= 24000 or config.training.subset_name == "main":
        subset_frames["main"] = add_mismatch_paths(train_frame.iloc[: min(24000, len(train_frame))].reset_index(drop=True))

    target_dir = config.runtime.processed_dir / "rlaif-v"
    active_train_frame = subset_frames[config.training.subset_name]
    outputs = {
        "train": target_dir / "train.parquet",
        "val": target_dir / "val.parquet",
        config.training.subset_name: target_dir / f"{config.training.subset_name}.parquet",
    }
    write_parquet(outputs["train"], active_train_frame)
    for subset_name, subset_frame in subset_frames.items():
        subset_path = target_dir / f"{subset_name}.parquet"
        write_parquet(subset_path, subset_frame)
        outputs[subset_name] = subset_path
    write_parquet(outputs["val"], val_frame)
    return outputs


def _load_dataset(dataset_cfg: DatasetSourceConfig):
    if dataset_cfg.source == "huggingface":
        from datasets import load_dataset

        return load_dataset(dataset_cfg.path, split=dataset_cfg.split)
    if dataset_cfg.source == "local-parquet":
        return pd.read_parquet(dataset_cfg.path)
    raise ValueError(f"Unsupported training dataset source: {dataset_cfg.source}")


def _normalize_row(
    example: dict[str, Any],
    image_dir: Path,
    index: int,
    dataset_cfg: DatasetSourceConfig,
    base_dir: Path,
) -> dict[str, Any]:
    sample_id = str(example.get("id") or example.get("idx") or f"rlaif-v-{index:06d}")
    prompt = extract_text(example.get("prompt") or example.get("question"))
    chosen = extract_text(example.get("chosen"))
    rejected = extract_text(example.get("rejected"))

    image_value = None
    if example.get("images"):
        images = example["images"]
        image_value = images[0] if isinstance(images, list) else images
    elif example.get("image") is not None:
        image_value = example["image"]
    elif example.get("image_path") is not None:
        image_value = example["image_path"]
    else:
        raise ValueError(f"Training example {sample_id} does not contain an image field.")

    image_path = coerce_and_save_image(
        image_value=image_value,
        destination_dir=image_dir,
        sample_id=sample_id,
        image_root=dataset_cfg.image_root,
        base_dir=base_dir,
    )
    metadata = {
        "origin_dataset": example.get("origin_dataset"),
        "origin_split": example.get("origin_split"),
        "extra": example.get("extra"),
    }
    return {
        "sample_id": sample_id,
        "dataset": "rlaif-v",
        "split": dataset_cfg.split,
        "image_path": str(image_path),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "ground_truth": "",
        "metadata": json_dumps(metadata),
        "mismatch_image_path": "",
    }
