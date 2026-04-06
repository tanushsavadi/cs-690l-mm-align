from __future__ import annotations

import json
from pathlib import Path

from mm_align.artifacts import build_run_id, ensure_run_paths, write_run_metadata
from mm_align.config import ProjectConfig
from mm_align.training.datasets import frame_to_hf_dataset, load_training_frame, load_validation_frame
from mm_align.training.env import assert_supported_versions, require_cuda_for_training
from mm_align.training.image_aware import ImageAwareDPOTrainer, materialize_preference_preview
from mm_align.training.modeling import load_trainable_models


def run_standard_dpo(config: ProjectConfig) -> str:
    assert_supported_versions()
    require_cuda_for_training()

    from trl import DPOConfig, DPOTrainer

    from mm_align.training.collators import PathAwareVisionPreferenceCollator

    model_variant = "standard_dpo"
    run_id = build_run_id(model_variant, config.training.subset_name, config.runtime.seed)
    run_paths = ensure_run_paths(config.runtime.artifacts_dir, run_id)
    write_run_metadata(run_paths, config, extra_env={"model_variant": model_variant})

    train_frame = load_training_frame(config)
    val_frame = load_validation_frame(config)
    train_dataset = frame_to_hf_dataset(train_frame)
    eval_dataset = frame_to_hf_dataset(val_frame) if not val_frame.empty else None

    model, ref_model, processor = load_trainable_models(config)
    collator = PathAwareVisionPreferenceCollator(
        processor=processor,
        max_length=config.training.max_length,
        use_mismatch_images=False,
    )
    trainer_args = DPOConfig(
        output_dir=str(run_paths.root / "checkpoints"),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        num_train_epochs=config.training.num_train_epochs,
        max_length=config.training.max_length,
        beta=config.training.beta,
        bf16=config.training.precision == "bf16",
        fp16=config.training.precision == "fp16",
        gradient_checkpointing=config.training.gradient_checkpointing,
        precompute_ref_log_probs=config.training.precompute_ref_log_probs,
        remove_unused_columns=False,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="no",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=collator,
    )
    train_result = trainer.train()
    adapter_dir = run_paths.root / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(adapter_dir)
    metrics = {
        "run_id": run_id,
        "model_variant": model_variant,
        "train_runtime": getattr(train_result, "metrics", {}),
        "log_history": trainer.state.log_history,
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
    run_paths.metrics_path.write_text(json.dumps({"run_id": run_id, "model_variant": model_variant, **metrics}, indent=2), encoding="utf-8")
    materialize_preference_preview(train_frame, processor, model, ref_model, config, run_paths.preferences_path)
    print(run_id)
    return run_id
