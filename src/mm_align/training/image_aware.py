from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mm_align.config import ProjectConfig
from mm_align.training.collators import PathAwareVisionPreferenceCollator


def _tensor_inputs(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    inputs: dict[str, Any] = {}
    for key, value in batch.items():
        if key in {"completion_mask", "sample_id"}:
            continue
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    return inputs


def _sequence_logps(model: Any, batch: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch

    device = next(model.parameters()).device
    inputs = _tensor_inputs(batch, device)
    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    labels = inputs["input_ids"][:, 1:]
    mask = batch["completion_mask"][:, 1:].to(device)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    sequence_logps = (token_logps * mask).sum(dim=-1)
    token_counts = mask.sum(dim=-1).clamp(min=1)
    return sequence_logps, token_counts, outputs


def _dpo_components(policy_model: Any, ref_model: Any, batch: dict[str, Any], beta: float) -> dict[str, Any]:
    import torch

    policy_logps, token_counts, _ = _sequence_logps(policy_model, batch)
    with torch.no_grad():
        ref_logps, _, _ = _sequence_logps(ref_model, batch)

    batch_size = len(batch["sample_id"])
    policy_chosen = policy_logps[:batch_size]
    policy_rejected = policy_logps[batch_size:]
    ref_chosen = ref_logps[:batch_size]
    ref_rejected = ref_logps[batch_size:]

    normalized_margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    dpo_loss = -torch.nn.functional.logsigmoid(beta * normalized_margin).mean()
    anchor_loss = -(policy_chosen / token_counts[:batch_size]).mean()
    return {
        "dpo_loss": dpo_loss,
        "anchor_loss": anchor_loss,
        "normalized_margin": normalized_margin,
        "policy_chosen": policy_chosen,
        "policy_rejected": policy_rejected,
        "ref_chosen": ref_chosen,
        "ref_rejected": ref_rejected,
    }


def compute_image_aware_loss(
    policy_model: Any,
    ref_model: Any,
    matched_batch: dict[str, Any],
    mismatched_batch: dict[str, Any],
    config: ProjectConfig,
) -> tuple[Any, dict[str, float]]:
    import torch

    matched = _dpo_components(policy_model, ref_model, matched_batch, beta=config.training.beta)
    mismatched = _dpo_components(policy_model, ref_model, mismatched_batch, beta=config.training.beta)

    gap_loss = torch.relu(
        torch.tensor(config.training.margin, device=matched["normalized_margin"].device)
        - (matched["normalized_margin"] - mismatched["normalized_margin"])
    ).mean()
    total = (
        matched["dpo_loss"]
        + config.training.lambda_gap * gap_loss
        + config.training.lambda_anchor * matched["anchor_loss"]
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "dpo_loss": float(matched["dpo_loss"].detach().cpu()),
        "gap_loss": float(gap_loss.detach().cpu()),
        "anchor_loss": float(matched["anchor_loss"].detach().cpu()),
        "matched_margin": float(matched["normalized_margin"].mean().detach().cpu()),
        "mismatched_margin": float(mismatched["normalized_margin"].mean().detach().cpu()),
    }
    return total, metrics


@dataclass
class ImageAwareDPOTrainer:
    model: Any
    ref_model: Any
    processor: Any
    train_records: list[dict[str, Any]]
    config: ProjectConfig
    output_dir: Path

    def train(self) -> dict[str, Any]:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_loader = DataLoader(
            self.train_records,
            batch_size=self.config.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,
        )
        matched_collator = PathAwareVisionPreferenceCollator(
            processor=self.processor,
            max_length=self.config.training.max_length,
            use_mismatch_images=False,
            include_sample_ids=True,
        )
        mismatched_collator = PathAwareVisionPreferenceCollator(
            processor=self.processor,
            max_length=self.config.training.max_length,
            use_mismatch_images=True,
            include_sample_ids=True,
        )

        steps_per_epoch = max(1, math.ceil(len(self.train_records) / self.config.training.per_device_train_batch_size))
        optimizer_steps = max(
            1,
            math.ceil(steps_per_epoch * self.config.training.num_train_epochs / self.config.training.gradient_accumulation_steps),
        )
        optimizer = AdamW((param for param in self.model.parameters() if param.requires_grad), lr=self.config.training.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=max(1, int(optimizer_steps * self.config.training.warmup_ratio)),
            num_training_steps=optimizer_steps,
        )

        device_type = "cuda"
        dtype = torch.bfloat16 if self.config.training.precision == "bf16" else torch.float16
        metrics_history: list[dict[str, float]] = []
        global_step = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(max(1, int(math.ceil(self.config.training.num_train_epochs)))):
            for batch_index, examples in enumerate(raw_loader):
                matched_batch = matched_collator(examples)
                mismatched_batch = mismatched_collator(examples)
                with torch.autocast(device_type=device_type, dtype=dtype):
                    loss, metrics = compute_image_aware_loss(
                        policy_model=self.model,
                        ref_model=self.ref_model,
                        matched_batch=matched_batch,
                        mismatched_batch=mismatched_batch,
                        config=self.config,
                    )
                    loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()
                if (batch_index + 1) % self.config.training.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    metrics["step"] = float(global_step)
                    metrics["epoch"] = float(epoch)
                    metrics_history.append(metrics)

        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        return {
            "global_step": global_step,
            "optimizer_steps": optimizer_steps,
            "last_metrics": metrics_history[-1] if metrics_history else {},
            "log_history": metrics_history,
        }


@dataclass
class StandardDPOTrainer:
    model: Any
    ref_model: Any
    processor: Any
    train_records: list[dict[str, Any]]
    config: ProjectConfig
    output_dir: Path

    def train(self) -> dict[str, Any]:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_loader = DataLoader(
            self.train_records,
            batch_size=self.config.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,
        )
        collator = PathAwareVisionPreferenceCollator(
            processor=self.processor,
            max_length=self.config.training.max_length,
            include_sample_ids=True,
        )

        steps_per_epoch = max(1, math.ceil(len(self.train_records) / self.config.training.per_device_train_batch_size))
        optimizer_steps = max(
            1,
            math.ceil(steps_per_epoch * self.config.training.num_train_epochs / self.config.training.gradient_accumulation_steps),
        )
        optimizer = AdamW((param for param in self.model.parameters() if param.requires_grad), lr=self.config.training.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=max(1, int(optimizer_steps * self.config.training.warmup_ratio)),
            num_training_steps=optimizer_steps,
        )

        device_type = "cuda"
        dtype = torch.bfloat16 if self.config.training.precision == "bf16" else torch.float16
        metrics_history: list[dict[str, float]] = []
        global_step = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(max(1, int(math.ceil(self.config.training.num_train_epochs)))):
            for batch_index, examples in enumerate(raw_loader):
                batch = collator(examples)
                with torch.autocast(device_type=device_type, dtype=dtype):
                    components = _dpo_components(self.model, self.ref_model, batch, beta=self.config.training.beta)
                    loss = components["dpo_loss"]
                    scaled_loss = loss / self.config.training.gradient_accumulation_steps
                scaled_loss.backward()
                if (batch_index + 1) % self.config.training.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    metrics_history.append(
                        {
                            "step": float(global_step),
                            "epoch": float(epoch),
                            "loss": float(loss.detach().cpu()),
                            "dpo_loss": float(components["dpo_loss"].detach().cpu()),
                            "matched_margin": float(components["normalized_margin"].mean().detach().cpu()),
                        }
                    )

        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        return {
            "global_step": global_step,
            "optimizer_steps": optimizer_steps,
            "last_metrics": metrics_history[-1] if metrics_history else {},
            "log_history": metrics_history,
        }


def materialize_preference_preview(
    frame: pd.DataFrame,
    processor: Any,
    model: Any,
    ref_model: Any,
    config: ProjectConfig,
    output_path: Path,
) -> None:
    from torch.utils.data import DataLoader

    preview_records = frame.iloc[: min(64, len(frame))].to_dict(orient="records")
    if not preview_records:
        pd.DataFrame(
            columns=[
                "sample_id",
                "image_path",
                "prompt",
                "chosen",
                "rejected",
                "matched_margin",
                "mismatched_margin",
            ]
        ).to_parquet(output_path, index=False)
        return

    raw_loader = DataLoader(preview_records, batch_size=4, shuffle=False, collate_fn=lambda batch: batch)
    matched_collator = PathAwareVisionPreferenceCollator(
        processor=processor,
        max_length=config.training.max_length,
        include_sample_ids=True,
    )
    mismatched_collator = PathAwareVisionPreferenceCollator(
        processor=processor,
        max_length=config.training.max_length,
        use_mismatch_images=True,
        include_sample_ids=True,
    )

    rows: list[dict[str, Any]] = []
    for examples in raw_loader:
        matched = _dpo_components(model, ref_model, matched_collator(examples), beta=config.training.beta)
        mismatched = _dpo_components(model, ref_model, mismatched_collator(examples), beta=config.training.beta)
        for index, example in enumerate(examples):
            rows.append(
                {
                    "sample_id": example["sample_id"],
                    "image_path": example["image_path"],
                    "prompt": example["prompt"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"],
                    "matched_margin": float(matched["normalized_margin"][index].detach().cpu()),
                    "mismatched_margin": float(mismatched["normalized_margin"][index].detach().cpu()),
                }
            )
    pd.DataFrame(rows).to_parquet(output_path, index=False)
