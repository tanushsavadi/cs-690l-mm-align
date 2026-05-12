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

    stats = subparsers.add_parser(
        "build-statistical-report",
        help="Build artifact-only statistical evidence for two completed runs.",
    )
    stats.add_argument("--baseline-run", default="2026-04-08-standard_dpo-pilot-7", help="Baseline run ID.")
    stats.add_argument("--candidate-run", default="2026-04-08-image_aware_dpo-pilot-7", help="Candidate run ID.")
    stats.add_argument(
        "--artifacts-dir",
        default=os.getenv("MM_ALIGN_ARTIFACTS_DIR", "artifacts/runs"),
        help="Artifact base directory.",
    )
    stats.add_argument("--reports-dir", default="reports", help="Directory for markdown and CSV report outputs.")
    stats.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap resamples per metric.")
    stats.add_argument("--seed", type=int, default=7, help="Bootstrap random seed.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-dashboard-data":
        from mm_align.eval.dashboard_data import build_dashboard_artifacts

        build_dashboard_artifacts(Path(args.artifacts_dir), args.run)
        return
    if args.command == "build-statistical-report":
        from mm_align.eval.statistical_report import build_statistical_report

        paths = build_statistical_report(
            artifacts_dir=Path(args.artifacts_dir),
            reports_dir=Path(args.reports_dir),
            baseline_run=args.baseline_run,
            candidate_run=args.candidate_run,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        print(paths.summary_md)
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
