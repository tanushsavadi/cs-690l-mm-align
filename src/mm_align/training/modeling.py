from __future__ import annotations

from pathlib import Path
from typing import Any

from mm_align.config import ProjectConfig


def _dtype_for_config(config: ProjectConfig):
    import torch

    return torch.bfloat16 if config.training.precision == "bf16" else torch.float16


def load_processor(config: ProjectConfig) -> Any:
    from transformers import AutoProcessor

    processor_name = config.model.processor_name or config.model.base_model_name
    kwargs: dict[str, Any] = {}
    if config.model.min_pixels is not None:
        kwargs["min_pixels"] = config.model.min_pixels
    if config.model.max_pixels is not None:
        kwargs["max_pixels"] = config.model.max_pixels
    return AutoProcessor.from_pretrained(processor_name, **kwargs)


def load_trainable_models(config: ProjectConfig) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    dtype = _dtype_for_config(config)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    common_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": dtype,
        "trust_remote_code": config.model.trust_remote_code,
        "quantization_config": quantization_config,
    }
    if config.model.attn_implementation:
        common_kwargs["attn_implementation"] = config.model.attn_implementation

    policy_model = AutoModelForImageTextToText.from_pretrained(config.model.base_model_name, **common_kwargs)
    ref_model = AutoModelForImageTextToText.from_pretrained(config.model.base_model_name, **common_kwargs)
    policy_model = prepare_model_for_kbit_training(policy_model, use_gradient_checkpointing=config.training.gradient_checkpointing)
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(policy_model, lora_config)
    if config.training.gradient_checkpointing and hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()
    policy_model.train()
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad = False
    processor = load_processor(config)
    return policy_model, ref_model, processor


def load_model_for_evaluation(config: ProjectConfig, run_dir: Path | None = None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText

    dtype = _dtype_for_config(config)
    model = AutoModelForImageTextToText.from_pretrained(
        config.model.base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype if torch.cuda.is_available() else None,
        trust_remote_code=config.model.trust_remote_code,
        attn_implementation=config.model.attn_implementation,
    )
    if run_dir is not None and (run_dir / "adapter").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, run_dir / "adapter")
    processor = load_processor(config)
    model.eval()
    return model, processor
