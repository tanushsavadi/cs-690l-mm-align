from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from mm_align.utils.images import load_image


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_to_text(item) for item in value if item is not None).strip()
    if isinstance(value, dict):
        if "text" in value:
            return _to_text(value["text"])
        if "content" in value:
            return _to_text(value["content"])
    return str(value)


def _build_prompt_messages(prompt: Any) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _to_text(prompt)},
            ],
        }
    ]


def _build_answer_message(text: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": _to_text(text)}],
    }


def _processor_call(
    processor: Any,
    texts: list[str],
    images: list[Any],
    max_length: int | None,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {
        "text": texts,
        "images": images,
        "padding": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
        kwargs["truncation"] = True
    return processor(**kwargs)


def _build_completion_mask(attention_mask: torch.Tensor, prompt_lengths: list[int]) -> torch.Tensor:
    completion_mask = torch.zeros_like(attention_mask)
    for index, prompt_length in enumerate(prompt_lengths):
        seq_len = int(attention_mask[index].sum().item())
        start = min(int(prompt_length), seq_len)
        if start < seq_len:
            completion_mask[index, start:seq_len] = 1
    return completion_mask


def _pad_tensor_for_concat(tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[1] == target_length:
        return tensor
    return F.pad(tensor, (0, target_length - tensor.shape[1]), value=pad_value)


def _concat_feature_batches(
    chosen_batch: dict[str, torch.Tensor],
    rejected_batch: dict[str, torch.Tensor],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    combined: dict[str, torch.Tensor] = {}
    pad_values = {
        "input_ids": pad_token_id,
        "attention_mask": 0,
        "completion_mask": 0,
        "token_type_ids": 0,
        "labels": -100,
    }
    shared_keys = set(chosen_batch) & set(rejected_batch)
    for key in shared_keys:
        chosen_value = chosen_batch[key]
        rejected_value = rejected_batch[key]
        if not isinstance(chosen_value, torch.Tensor) or not isinstance(rejected_value, torch.Tensor):
            continue
        if chosen_value.ndim >= 2 and rejected_value.ndim >= 2 and chosen_value.shape[1] != rejected_value.shape[1]:
            target_length = max(chosen_value.shape[1], rejected_value.shape[1])
            pad_value = pad_values.get(key, 0)
            chosen_value = _pad_tensor_for_concat(chosen_value, target_length, pad_value)
            rejected_value = _pad_tensor_for_concat(rejected_value, target_length, pad_value)
        combined[key] = torch.cat([chosen_value, rejected_value], dim=0)
    return combined


class PathAwareVisionPreferenceCollator:
    def __init__(
        self,
        processor: Any,
        max_length: int | None = None,
        use_mismatch_images: bool = False,
        include_sample_ids: bool = False,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.use_mismatch_images = use_mismatch_images
        self.include_sample_ids = include_sample_ids
        tokenizer = getattr(processor, "tokenizer", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images: list[Any] = []
        prompt_texts: list[str] = []
        chosen_texts: list[str] = []
        rejected_texts: list[str] = []
        sample_ids: list[str] = []

        for example in examples:
            image_key = "mismatch_image_path" if self.use_mismatch_images and example.get("mismatch_image_path") else "image_path"
            image = load_image(example[image_key])
            prompt_messages = _build_prompt_messages(example["prompt"])

            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
            chosen_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages + [_build_answer_message(example["chosen"])],
                    add_generation_prompt=False,
                    tokenize=False,
                )
            )
            rejected_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages + [_build_answer_message(example["rejected"])],
                    add_generation_prompt=False,
                    tokenize=False,
                )
            )
            images.append(image)
            sample_ids.append(str(example.get("sample_id", "")))

        prompt_batch = _processor_call(self.processor, prompt_texts, images, self.max_length)
        chosen_batch = _processor_call(self.processor, chosen_texts, images, self.max_length)
        rejected_batch = _processor_call(self.processor, rejected_texts, images, self.max_length)

        prompt_lengths = [int(length) for length in prompt_batch["attention_mask"].sum(dim=1).tolist()]
        chosen_batch["completion_mask"] = _build_completion_mask(chosen_batch["attention_mask"], prompt_lengths)
        rejected_batch["completion_mask"] = _build_completion_mask(rejected_batch["attention_mask"], prompt_lengths)

        batch = _concat_feature_batches(chosen_batch, rejected_batch, self.pad_token_id)
        if self.include_sample_ids:
            batch["sample_id"] = sample_ids
        return batch
