from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PathModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class RuntimeConfig(PathModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts/runs")
    num_proc: int = 4
    seed: int = 7


class DatasetSourceConfig(PathModel):
    source: Literal["huggingface", "local-json", "local-parquet", "local-dir"] = "huggingface"
    path: str
    split: str = "train"
    image_root: Optional[Path] = None
    subset_size: Optional[int] = None
    enabled: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_root", mode="before")
    @classmethod
    def _convert_path(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        return Path(value)


class DatasetCollectionConfig(PathModel):
    training: DatasetSourceConfig = Field(
        default_factory=lambda: DatasetSourceConfig(path="trl-lib/rlaif-v", split="train")
    )
    training_fallback: DatasetSourceConfig = Field(
        default_factory=lambda: DatasetSourceConfig(
            path="openbmb/RLAIF-V-Dataset",
            split="train",
            enabled=False,
        )
    )
    vlfeedback: DatasetSourceConfig = Field(
        default_factory=lambda: DatasetSourceConfig(
            path="MMInstruction/VLFeedback",
            split="train",
            enabled=False,
        )
    )
    hallusionbench: DatasetSourceConfig = Field(
        default_factory=lambda: DatasetSourceConfig(
            source="local-json",
            path="data/raw/hallusionbench/HallusionBench.json",
            split="validation",
        )
    )
    pope: DatasetSourceConfig = Field(
        default_factory=lambda: DatasetSourceConfig(
            source="local-dir",
            path="data/raw/pope",
            split="validation",
        )
    )
    chartqa: DatasetSourceConfig = Field(
        default_factory=lambda: DatasetSourceConfig(
            source="local-dir",
            path="data/raw/chartqa",
            split="validation",
        )
    )


class ModelConfig(BaseModel):
    base_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor_name: Optional[str] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = "sdpa"
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    max_new_tokens: int = 128


class LoraConfigModel(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    bias: str = "none"


class TrainingConfig(BaseModel):
    subset_name: Literal["smoke", "pilot", "main"] = "pilot"
    subset_size: int = 8000
    val_size: int = 1000
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    max_length: Optional[int] = None
    gradient_checkpointing: bool = True
    precision: Literal["bf16", "fp16"] = "bf16"
    beta: float = 0.1
    lambda_gap: float = 0.2
    margin: float = 0.1
    lambda_anchor: float = 0.05
    precompute_ref_log_probs: bool = False

    @model_validator(mode="after")
    def _sync_subset(self) -> "TrainingConfig":
        defaults = {"smoke": 1000, "pilot": 8000, "main": 24000}
        if self.subset_size <= 0:
            self.subset_size = defaults[self.subset_name]
        return self


class EvaluationConfig(BaseModel):
    evaluate_variants: list[str] = Field(
        default_factory=lambda: ["base", "standard_dpo", "image_aware_dpo"]
    )
    dependence_variants: list[str] = Field(
        default_factory=lambda: ["original", "blank-image", "mismatched-image"]
    )
    blank_image_size: tuple[int, int] = (448, 448)
    yes_tokens: tuple[str, ...] = ("yes", "true", "1")
    no_tokens: tuple[str, ...] = ("no", "false", "0")


class DashboardConfig(BaseModel):
    title: str = "Multimodal Alignment Dashboard"
    default_run: Optional[str] = None
    page_size: int = 50


class ProjectConfig(PathModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    datasets: DatasetCollectionConfig = Field(default_factory=DatasetCollectionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    def apply_env_overrides(self) -> "ProjectConfig":
        path_overrides = {
            "MM_ALIGN_RAW_DIR": "raw_dir",
            "MM_ALIGN_PROCESSED_DIR": "processed_dir",
            "MM_ALIGN_ARTIFACTS_DIR": "artifacts_dir",
        }
        for env_name, attr_name in path_overrides.items():
            value = os.getenv(env_name)
            if value:
                setattr(self.runtime, attr_name, Path(value))

        num_proc = os.getenv("MM_ALIGN_NUM_PROC")
        if num_proc:
            self.runtime.num_proc = int(num_proc)

        seed = os.getenv("MM_ALIGN_SEED")
        if seed:
            self.runtime.seed = int(seed)
        return self

    def resolve_paths(self, repo_root: Path) -> "ProjectConfig":
        def maybe_resolve(value: Path) -> Path:
            return value if value.is_absolute() else (repo_root / value).resolve()

        def rebase_data_path(value: Path) -> Path:
            if value.is_absolute():
                return value
            rebases = {
                ("data", "raw"): self.runtime.raw_dir,
                ("data", "processed"): self.runtime.processed_dir,
                ("artifacts", "runs"): self.runtime.artifacts_dir,
            }
            for prefix, root in rebases.items():
                if value.parts[: len(prefix)] == prefix:
                    suffix_parts = value.parts[len(prefix) :]
                    suffix = Path(*suffix_parts) if suffix_parts else Path()
                    return (root / suffix).resolve()
            return (repo_root / value).resolve()

        self.runtime.raw_dir = maybe_resolve(self.runtime.raw_dir)
        self.runtime.processed_dir = maybe_resolve(self.runtime.processed_dir)
        self.runtime.artifacts_dir = maybe_resolve(self.runtime.artifacts_dir)
        for dataset in self.datasets.model_dump().keys():
            cfg = getattr(self.datasets, dataset)
            if cfg.image_root is not None:
                cfg.image_root = rebase_data_path(cfg.image_root)
            if cfg.source != "huggingface":
                cfg.path = str(rebase_data_path(Path(cfg.path)))
        return self


def load_config(path: str | Path, repo_root: Optional[Path] = None) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = ProjectConfig.model_validate(payload).apply_env_overrides()
    return config.resolve_paths(repo_root or config_path.parent.parent.resolve())


def dump_config(config: ProjectConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
