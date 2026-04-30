#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements-colab.txt
python3 -m pip install -e ".[dev]"
# Colab may include an older torchao. PEFT probes optional torchao support
# while loading LoRA adapters, and incompatible versions can abort evaluation.
python3 -m pip uninstall -y torchao >/dev/null 2>&1 || true

for env_name in MM_ALIGN_RAW_DIR MM_ALIGN_PROCESSED_DIR MM_ALIGN_ARTIFACTS_DIR HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE; do
  env_value="${!env_name-}"
  if [[ -n "${env_value}" ]]; then
    mkdir -p "${env_value}"
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

python3 - <<'PY'
from importlib import metadata

packages = [
    "accelerate",
    "datasets",
    "peft",
    "streamlit",
    "torch",
    "transformers",
    "trl",
]
for package in packages:
    try:
        print(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package} not installed")

try:
    import torch

    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
except Exception as exc:  # pragma: no cover - best effort environment report
    print(f"torch_cuda_probe_failed={exc}")
PY
