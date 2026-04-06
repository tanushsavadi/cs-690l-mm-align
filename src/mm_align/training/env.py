from __future__ import annotations

from importlib import metadata

from packaging.version import Version


def assert_supported_versions() -> None:
    transformers_version = metadata.version("transformers")
    trl_version = metadata.version("trl")
    if Version(transformers_version) < Version("4.57.0"):
        raise RuntimeError(
            f"transformers>={Version('4.57.0')} is required for the current Qwen2.5-VL workflow; found {transformers_version}."
        )
    if Version(trl_version) < Version("0.24.0"):
        raise RuntimeError(
            f"trl>={Version('0.24.0')} is required for the current multimodal DPO workflow; found {trl_version}."
        )


def require_cuda_for_training() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training commands. Use Google Colab Pro or another GPU-backed environment."
        )
