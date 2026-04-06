from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

from mm_align.config import ProjectConfig, dump_config
from mm_align.utils.io import write_json, write_jsonl


@dataclass(frozen=True)
class RunPaths:
    root: Path
    config_path: Path
    env_path: Path
    metrics_path: Path
    predictions_path: Path
    dependence_path: Path
    preferences_path: Path
    dashboard_examples_path: Path
    dashboard_summary_path: Path


def build_run_id(model_variant: str, subset_name: str, seed: int, today: date | None = None) -> str:
    stamp = (today or date.today()).isoformat()
    return f"{stamp}-{model_variant}-{subset_name}-{seed}"


def ensure_run_paths(base_dir: Path, run_id: str) -> RunPaths:
    root = base_dir / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        config_path=root / "config.yaml",
        env_path=root / "env.json",
        metrics_path=root / "metrics.json",
        predictions_path=root / "predictions.jsonl",
        dependence_path=root / "dependence.jsonl",
        preferences_path=root / "preferences.parquet",
        dashboard_examples_path=root / "dashboard_examples.parquet",
        dashboard_summary_path=root / "dashboard_summary.parquet",
    )


def write_run_metadata(run_paths: RunPaths, config: ProjectConfig, extra_env: dict[str, Any] | None = None) -> None:
    dump_config(config, run_paths.config_path)
    env_manifest = collect_environment(extra=extra_env)
    write_json(run_paths.env_path, env_manifest)


def collect_environment(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    packages = [
        "accelerate",
        "datasets",
        "pandas",
        "peft",
        "plotly",
        "pyarrow",
        "pydantic",
        "streamlit",
        "torch",
        "transformers",
        "trl",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    if extra:
        versions.update({str(key): str(value) for key, value in extra.items()})
    return versions


def append_predictions(run_paths: RunPaths, rows: list[dict[str, Any]]) -> None:
    write_jsonl(run_paths.predictions_path, rows)


def append_dependence(run_paths: RunPaths, rows: list[dict[str, Any]]) -> None:
    write_jsonl(run_paths.dependence_path, rows)
