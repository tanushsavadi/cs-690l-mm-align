from __future__ import annotations

import logging

from mm_align.config import ProjectConfig
from mm_align.data.chartqa import prepare_chartqa
from mm_align.data.hallusionbench import prepare_hallusionbench
from mm_align.data.pope import prepare_pope
from mm_align.data.rlaif_v import prepare_training_preferences

LOGGER = logging.getLogger(__name__)


def prepare_all_datasets(config: ProjectConfig) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.runtime.raw_dir.mkdir(parents=True, exist_ok=True)
    config.runtime.processed_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, object] = {}
    LOGGER.info("Preparing training preferences from %s", config.datasets.training.path)
    outputs["rlaif-v"] = prepare_training_preferences(config)

    LOGGER.info("Preparing HallusionBench if raw files are present")
    hallusion_path = prepare_hallusionbench(config)
    if hallusion_path is not None:
        outputs["hallusionbench"] = hallusion_path
    else:
        LOGGER.warning("Skipping HallusionBench because %s does not exist", config.datasets.hallusionbench.path)

    LOGGER.info("Preparing POPE if raw files are present")
    pope_path = prepare_pope(config)
    if pope_path is not None:
        outputs["pope"] = pope_path
    else:
        LOGGER.warning("Skipping POPE because %s does not exist", config.datasets.pope.path)

    LOGGER.info("Preparing ChartQA if raw files are present")
    chartqa_paths = prepare_chartqa(config)
    if chartqa_paths:
        outputs["chartqa"] = chartqa_paths
    else:
        LOGGER.warning("Skipping ChartQA because %s does not exist", config.datasets.chartqa.path)
    return outputs
