#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/smoke.yaml}"

echo "[smoke] prepare-data ${CONFIG_PATH}"
python3 -u -m mm_align.cli prepare-data --config "${CONFIG_PATH}"

echo "[smoke] train-dpo ${CONFIG_PATH}"
python3 -u -m mm_align.cli train-dpo --config "${CONFIG_PATH}"

echo "[smoke] train-imgaware ${CONFIG_PATH}"
python3 -u -m mm_align.cli train-imgaware --config "${CONFIG_PATH}"
