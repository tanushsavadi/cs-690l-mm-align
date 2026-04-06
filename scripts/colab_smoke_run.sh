#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/smoke.yaml}"

python3 -m mm_align.cli prepare-data --config "${CONFIG_PATH}"
python3 -m mm_align.cli train-dpo --config "${CONFIG_PATH}"
python3 -m mm_align.cli train-imgaware --config "${CONFIG_PATH}"
