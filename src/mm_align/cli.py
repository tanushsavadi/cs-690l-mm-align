from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from mm_align.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal hallucination alignment CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Normalize and cache datasets.")
    prepare.add_argument("--config", required=True, help="Path to experiment YAML.")

    train_dpo = subparsers.add_parser("train-dpo", help="Run standard DPO training.")
    train_dpo.add_argument("--config", required=True, help="Path to experiment YAML.")

    train_imgaware = subparsers.add_parser("train-imgaware", help="Run image-aware DPO training.")
    train_imgaware.add_argument("--config", required=True, help="Path to experiment YAML.")

    evaluate = subparsers.add_parser("evaluate", help="Run benchmark evaluation.")
    evaluate.add_argument("--config", required=True, help="Path to experiment YAML.")
    evaluate.add_argument("--run", required=True, help="Run ID to evaluate.")

    dashboard = subparsers.add_parser(
        "build-dashboard-data",
        help="Assemble Streamlit-ready dashboard artifacts from a finished run.",
    )
    dashboard.add_argument("--run", required=True, help="Run ID to materialize.")
    dashboard.add_argument(
        "--artifacts-dir",
        default=os.getenv("MM_ALIGN_ARTIFACTS_DIR", "artifacts/runs"),
        help="Artifact base directory.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-dashboard-data":
        from mm_align.eval.dashboard_data import build_dashboard_artifacts

        build_dashboard_artifacts(Path(args.artifacts_dir), args.run)
        return

    config = load_config(args.config)

    if args.command == "prepare-data":
        from mm_align.data.preparation import prepare_all_datasets

        prepare_all_datasets(config)
    elif args.command == "train-dpo":
        from mm_align.training.runners import run_standard_dpo

        run_standard_dpo(config)
    elif args.command == "train-imgaware":
        from mm_align.training.runners import run_image_aware_dpo

        run_image_aware_dpo(config)
    elif args.command == "evaluate":
        from mm_align.eval.runner import run_evaluation

        run_evaluation(config, args.run)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
