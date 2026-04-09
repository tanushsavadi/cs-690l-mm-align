from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mm_align.artifacts import ensure_run_paths
from mm_align.config import ProjectConfig
from mm_align.eval.metrics import (
    aggregate_metrics,
    build_dependence_summary,
    exact_match,
    normalize_yes_no,
    relaxed_chartqa_match,
)
from mm_align.training.modeling import load_model_for_evaluation
from mm_align.utils.images import load_image, make_blank_image
from mm_align.utils.io import read_json, write_jsonl


def run_evaluation(config: ProjectConfig, run_id: str) -> None:
    run_dir = config.runtime.artifacts_dir / run_id
    model_variant = _infer_model_variant(run_id)
    run_paths = ensure_run_paths(config.runtime.artifacts_dir, run_id)

    model, processor = load_model_for_evaluation(config, run_dir if run_dir.exists() and model_variant != "base" else None)

    rows: list[dict[str, Any]] = []
    skipped_predictions = 0
    for benchmark_name, frame in _load_benchmark_frames(config).items():
        if frame.empty:
            continue
        records = frame.to_dict(orient="records")
        logging.info(
            "Evaluating benchmark %s with %s samples and %s image variants",
            benchmark_name,
            len(records),
            len(config.evaluation.dependence_variants),
        )
        progress_every = max(1, len(records) // 10)
        for index, record in enumerate(records, start=1):
            if index == 1 or index % progress_every == 0 or index == len(records):
                logging.info("Evaluation progress for %s: %s/%s samples", benchmark_name, index, len(records))
            for image_variant in config.evaluation.dependence_variants:
                try:
                    prediction = generate_prediction(
                        model=model,
                        processor=processor,
                        prompt=record["prompt"],
                        image_variant=image_variant,
                        image_path=record["image_path"],
                        mismatch_image_path=record.get("mismatch_image_path"),
                        max_new_tokens=config.model.max_new_tokens,
                        blank_size=config.evaluation.blank_image_size,
                    )
                except OSError as error:
                    skipped_predictions += 1
                    logging.warning(
                        "Skipping %s sample %s variant %s because the image could not be read: %s",
                        benchmark_name,
                        record["sample_id"],
                        image_variant,
                        error,
                    )
                    continue
                is_correct = _score_prediction(benchmark_name, prediction, record["ground_truth"])
                rows.append(
                    {
                        "run_id": run_id,
                        "model_variant": model_variant,
                        "benchmark": benchmark_name,
                        "sample_id": record["sample_id"],
                        "prompt": record["prompt"],
                        "ground_truth": record["ground_truth"],
                        "image_path": record["image_path"],
                        "metadata": record["metadata"],
                        "image_variant": image_variant,
                        "prediction": prediction,
                        "is_correct": bool(is_correct),
                    }
                )

    prediction_frame = pd.DataFrame(rows)
    if prediction_frame.empty:
        raise RuntimeError("No benchmark predictions were produced. Check that processed benchmark parquet files exist.")

    dependence_frame = build_dependence_summary(prediction_frame)
    metrics = aggregate_metrics(prediction_frame)
    if skipped_predictions:
        logging.warning("Skipped %s benchmark predictions due to unreadable images", skipped_predictions)

    existing_metrics = read_json(run_paths.metrics_path) if run_paths.metrics_path.exists() else {}
    existing_metrics["evaluation"] = metrics
    run_paths.metrics_path.write_text(json.dumps(existing_metrics, indent=2), encoding="utf-8")
    write_jsonl(run_paths.predictions_path, rows)
    if not dependence_frame.empty:
        write_jsonl(run_paths.dependence_path, dependence_frame.to_dict(orient="records"))


def generate_prediction(
    model: Any,
    processor: Any,
    prompt: str,
    image_variant: str,
    image_path: str,
    mismatch_image_path: str | None,
    max_new_tokens: int,
    blank_size: tuple[int, int],
) -> str:
    import torch

    image = _resolve_image_for_variant(image_variant, image_path, mismatch_image_path, blank_size)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip()


def _resolve_image_for_variant(
    variant: str,
    image_path: str,
    mismatch_image_path: str | None,
    blank_size: tuple[int, int],
):
    if variant == "blank-image":
        return make_blank_image(size=blank_size)
    if variant == "mismatched-image" and mismatch_image_path:
        return load_image(mismatch_image_path)
    return load_image(image_path)


def _load_benchmark_frames(config: ProjectConfig) -> dict[str, pd.DataFrame]:
    datasets = {}
    for benchmark, split in (("hallusionbench", "validation"), ("pope", "validation"), ("chartqa", "val")):
        path = config.runtime.processed_dir / benchmark / f"{split}.parquet"
        if path.exists():
            datasets[benchmark] = pd.read_parquet(path)
    return datasets


def _infer_model_variant(run_id: str) -> str:
    for variant in ("image_aware_dpo", "standard_dpo", "base"):
        if variant in run_id:
            return variant
    return "base"


def _score_prediction(benchmark: str, prediction: str, ground_truth: str) -> bool:
    if benchmark == "pope":
        return normalize_yes_no(prediction) == normalize_yes_no(ground_truth)
    if benchmark == "chartqa":
        return relaxed_chartqa_match(prediction, ground_truth)
    return exact_match(normalize_yes_no(prediction), normalize_yes_no(ground_truth))
