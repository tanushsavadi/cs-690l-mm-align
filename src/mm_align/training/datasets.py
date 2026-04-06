from __future__ import annotations

from pathlib import Path

import pandas as pd

from mm_align.config import ProjectConfig


def load_training_frame(config: ProjectConfig, subset_name: str | None = None) -> pd.DataFrame:
    subset = subset_name or config.training.subset_name
    path = config.runtime.processed_dir / "rlaif-v" / f"{subset}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed training subset: {path}. Run prepare-data first.")
    return pd.read_parquet(path)


def load_validation_frame(config: ProjectConfig) -> pd.DataFrame:
    path = config.runtime.processed_dir / "rlaif-v" / "val.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def frame_to_hf_dataset(frame: pd.DataFrame):
    from datasets import Dataset

    records = frame.to_dict(orient="records")
    return Dataset.from_list(records)


def preview_frame(frame: pd.DataFrame, limit: int = 128) -> pd.DataFrame:
    return frame.iloc[: min(limit, len(frame))].reset_index(drop=True)


def adapter_path_for_run(artifacts_dir: Path, run_id: str) -> Path:
    return artifacts_dir / run_id / "adapter"
