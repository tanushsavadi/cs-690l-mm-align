from pathlib import Path

import torch
import pandas as pd
from PIL import Image

from mm_align.training.collators import PathAwareVisionPreferenceCollator, _processor_call
from mm_align.training.datasets import frame_to_hf_dataset
from mm_align.training.image_aware import _slice_batch


class _MockProcessor:
    class _Tokenizer:
        pad_token_id = 0

    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.last_call_kwargs = None

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        parts = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "image":
                        parts.append("<image>")
                    elif item["type"] == "text":
                        parts.append(item["text"])
            else:
                parts.append(str(content))
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join(parts)

    def __call__(self, text, images, padding=True, return_tensors="pt", max_length=None, truncation=False):
        self.last_call_kwargs = {
            "padding": padding,
            "return_tensors": return_tensors,
            "max_length": max_length,
            "truncation": truncation,
        }
        tokenized = []
        for index, item in enumerate(text):
            tokens = list(range(1, len(item.split()) + 1))
            if max_length is not None and truncation:
                tokens = tokens[:max_length]
            tokenized.append(tokens)
        max_len = max(len(tokens) for tokens in tokenized)
        input_ids = []
        attention_mask = []
        for tokens in tokenized:
            pad_len = max_len - len(tokens)
            input_ids.append(tokens + [0] * pad_len)
            attention_mask.append([1] * len(tokens) + [0] * pad_len)
        pixel_values = torch.stack([torch.full((3, 4, 4), fill_value=index + 1, dtype=torch.float32) for index, _ in enumerate(images)])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 2, 2] for _ in images], dtype=torch.long),
        }


def test_path_aware_vision_preference_collator_builds_completion_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)

    collator = PathAwareVisionPreferenceCollator(
        processor=_MockProcessor(),
        include_sample_ids=True,
    )
    batch = collator(
        [
            {
                "sample_id": "sample-1",
                "image_path": str(image_path),
                "mismatch_image_path": str(image_path),
                "prompt": "Describe the chart",
                "chosen": "The chart trends upward",
                "rejected": "It is a dog",
            }
        ]
    )

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["completion_mask"].shape
    assert batch["completion_mask"][0].sum().item() > 0
    assert batch["completion_mask"][1].sum().item() > 0
    assert batch["pixel_values"].shape[0] == 2
    assert batch["image_grid_thw"].shape == (2, 3)
    assert batch["sample_id"] == ["sample-1"]


def test_frame_to_hf_dataset_can_materialize_images(tmp_path: Path) -> None:
    image_path = tmp_path / "dataset.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)

    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "dataset": "rlaif-v",
                "split": "train",
                "image_path": str(image_path),
                "prompt": "Describe the chart",
                "chosen": "The line increases",
                "rejected": "There is a dog",
                "ground_truth": "",
                "metadata": "{}",
                "mismatch_image_path": str(image_path),
            }
        ]
    )

    dataset = frame_to_hf_dataset(frame, include_images=True)
    record = dataset[0]
    assert "images" in record
    assert record["prompt"] == "Describe the chart"


def test_slice_batch_respects_qwen_flattened_pixel_values() -> None:
    batch = {
        "sample_id": ["sample-1", "sample-2"],
        "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.long),
        "completion_mask": torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]], dtype=torch.long),
        "pixel_values": torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3),
    }

    chosen = _slice_batch(batch, 0, 2)
    rejected = _slice_batch(batch, 2, 4)

    assert chosen["sample_id"] == ["sample-1", "sample-2"]
    assert rejected["sample_id"] == []
    assert chosen["pixel_values"].shape[0] == 8
    assert rejected["pixel_values"].shape[0] == 8


def test_processor_call_never_truncates_multimodal_sequences() -> None:
    processor = _MockProcessor()
    image = Image.new("RGB", (8, 8), color=(255, 255, 255))

    _processor_call(processor, ["<image> describe this"], [image], max_length=128)

    assert processor.last_call_kwargs is not None
    assert processor.last_call_kwargs["max_length"] is None
    assert processor.last_call_kwargs["truncation"] is False
