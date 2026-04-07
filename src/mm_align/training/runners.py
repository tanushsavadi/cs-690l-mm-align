from __future__ import annotations

import json
import logging

from mm_align.artifacts import build_run_id, ensure_run_paths, write_run_metadata
from mm_align.config import ProjectConfig
from mm_align.training.datasets import load_training_frame
from mm_align.training.env import assert_supported_versions, require_cuda_for_training
from mm_align.training.image_aware import ImageAwareDPOTrainer, StandardDPOTrainer, materialize_preference_preview
from mm_align.training.modeling import load_trainable_models

LOGGER = logging.getLogger(__name__)


def run_standard_dpo(config: ProjectConfig) -> str:
    assert_supported_versions()
    require_cuda_for_training()

    model_variant = "standard_dpo"
    run_id = build_run_id(model_variant, config.training.subset_name, config.runtime.seed)
    run_paths = ensure_run_paths(config.runtime.artifacts_dir, run_id)
    write_run_metadata(run_paths, config, extra_env={"model_variant": model_variant})

    train_frame = load_training_frame(config)
    LOGGER.info("Starting standard DPO run %s with %s training records", run_id, len(train_frame))

    model, ref_model, processor = load_trainable_models(config)
    trainer = StandardDPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        train_records=train_frame.to_dict(orient="records"),
        config=config,
        output_dir=run_paths.root / "adapter",
    )
    train_result = trainer.train()
    LOGGER.info("Training finished for %s; writing metrics and preview artifacts", run_id)
    metrics = {
        "run_id": run_id,
        "model_variant": model_variant,
        **train_result,
    }
    run_paths.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    materialize_preference_preview(train_frame, processor, trainer.model, ref_model, config, run_paths.preferences_path)
    print(run_id)
    return run_id


def run_image_aware_dpo(config: ProjectConfig) -> str:
    assert_supported_versions()
    require_cuda_for_training()

    model_variant = "image_aware_dpo"
    run_id = build_run_id(model_variant, config.training.subset_name, config.runtime.seed)
    run_paths = ensure_run_paths(config.runtime.artifacts_dir, run_id)
    write_run_metadata(run_paths, config, extra_env={"model_variant": model_variant})

    train_frame = load_training_frame(config)
    LOGGER.info("Starting image-aware DPO run %s with %s training records", run_id, len(train_frame))
    model, ref_model, processor = load_trainable_models(config)
    trainer = ImageAwareDPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        train_records=train_frame.to_dict(orient="records"),
        config=config,
        output_dir=run_paths.root / "adapter",
    )
    metrics = trainer.train()
    LOGGER.info("Training finished for %s; writing metrics and preview artifacts", run_id)
    run_paths.metrics_path.write_text(json.dumps({"run_id": run_id, "model_variant": model_variant, **metrics}, indent=2), encoding="utf-8")
    materialize_preference_preview(train_frame, processor, model, ref_model, config, run_paths.preferences_path)
    print(run_id)
    return run_id
