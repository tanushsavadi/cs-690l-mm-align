from __future__ import annotations

from typing import Any

from mm_align.utils.images import load_image


def as_prompt_messages(prompt: Any) -> Any:
    if isinstance(prompt, list):
        return prompt
    return [{"role": "user", "content": str(prompt)}]


def as_completion_messages(text: Any) -> Any:
    if isinstance(text, list):
        return text
    return [{"role": "assistant", "content": str(text)}]


class PathAwareVisionPreferenceCollator:
    def __init__(self, processor: Any, max_length: int | None = None, use_mismatch_images: bool = False) -> None:
        from trl.trainer.dpo_trainer import DataCollatorForVisionPreference

        self.base_collator = DataCollatorForVisionPreference(processor=processor, max_length=max_length)
        self.use_mismatch_images = use_mismatch_images

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prepared = []
        sample_ids: list[str] = []
        for example in examples:
            image_key = "mismatch_image_path" if self.use_mismatch_images and example.get("mismatch_image_path") else "image_path"
            image = load_image(example[image_key])
            prepared.append(
                {
                    "images": [image],
                    "prompt": as_prompt_messages(example["prompt"]),
                    "chosen": as_completion_messages(example["chosen"]),
                    "rejected": as_completion_messages(example["rejected"]),
                }
            )
            sample_ids.append(str(example.get("sample_id", "")))
        batch = self.base_collator(prepared)
        batch["sample_id"] = sample_ids
        return batch
