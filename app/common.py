from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("MM_ALIGN_ARTIFACTS_DIR", REPO_ROOT / "artifacts" / "runs")).resolve()
COLAB_PROJECT_ROOT = "/content/drive/MyDrive/mm-align"
DEFAULT_FINAL_RUNS = ["2026-04-08-standard_dpo-pilot-7", "2026-04-08-image_aware_dpo-pilot-7"]


def init_state() -> None:
    st.session_state.setdefault("selected_run", None)


def list_runs() -> list[str]:
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(
        [path.name for path in ARTIFACTS_DIR.iterdir() if path.is_dir()],
        reverse=True,
    )


def default_final_runs(runs: list[str], minimum: int = 2) -> list[str]:
    selected = [run for run in DEFAULT_FINAL_RUNS if run in runs]
    if len(selected) >= minimum:
        return selected
    return runs[:minimum]


def model_label(run_id: str) -> str:
    if "image_aware" in run_id:
        return "image_aware_dpo"
    if "standard_dpo" in run_id:
        return "standard_dpo"
    if "smoke" in run_id:
        return f"smoke: {run_id}"
    return run_id


def sidebar_run_selector() -> str | None:
    init_state()
    runs = list_runs()
    with st.sidebar:
        st.header("Run Selection")
        if not runs:
            st.info("No run artifacts found yet.")
            st.session_state["selected_run"] = None
            return None
        default_index = 0
        if st.session_state["selected_run"] in runs:
            default_index = runs.index(st.session_state["selected_run"])
        st.session_state["selected_run"] = st.selectbox("Run", runs, index=default_index)
        st.caption(str(ARTIFACTS_DIR))
    return st.session_state["selected_run"]


def selected_run_dir() -> Path | None:
    run = st.session_state.get("selected_run")
    if not run:
        return None
    return ARTIFACTS_DIR / run


@st.cache_data
def load_examples(run_id: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / run_id / "dashboard_examples.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_summary(run_id: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / run_id / "dashboard_summary.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_preferences(run_id: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / run_id / "preferences.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_dependence(run_id: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / run_id / "dependence.jsonl"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


@st.cache_data
def load_metrics(run_id: str) -> dict:
    path = ARTIFACTS_DIR / run_id / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def load_training_history(run_id: str) -> pd.DataFrame:
    metrics = load_metrics(run_id)
    history = metrics.get("log_history", [])
    if not history:
        return pd.DataFrame()
    frame = pd.DataFrame(history)
    frame["run_id"] = run_id
    frame["model"] = metrics.get("model_variant") or model_label(run_id)
    return frame


def load_selected_summaries(run_ids: list[str]) -> pd.DataFrame:
    frames = []
    for run_id in run_ids:
        frame = load_summary(run_id).copy()
        if frame.empty:
            continue
        frame["model"] = model_label(run_id)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_selected_dependence(run_ids: list[str]) -> pd.DataFrame:
    frames = []
    for run_id in run_ids:
        frame = load_dependence(run_id).copy()
        if frame.empty:
            continue
        frame["model"] = model_label(run_id)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def headline_metric_rows(summary: pd.DataFrame) -> pd.DataFrame:
    headline_metrics = {
        ("chartqa", "relaxed_accuracy"),
        ("hallusionbench", "accuracy"),
        ("pope", "accuracy"),
        ("pope", "f1"),
    }
    if summary.empty:
        return pd.DataFrame()
    return summary[
        summary.apply(lambda row: (row["benchmark"], row["metric"]) in headline_metrics, axis=1)
    ].copy()


def comparison_delta(summary: pd.DataFrame, baseline_run: str, candidate_run: str) -> pd.DataFrame:
    baseline = summary[summary["run_id"] == baseline_run][["benchmark", "metric", "value"]]
    candidate = summary[summary["run_id"] == candidate_run][["benchmark", "metric", "value"]]
    delta = candidate.merge(
        baseline,
        on=["benchmark", "metric"],
        suffixes=("_candidate", "_baseline"),
    )
    delta["delta"] = delta["value_candidate"] - delta["value_baseline"]
    delta["baseline_run"] = baseline_run
    delta["candidate_run"] = candidate_run
    delta["baseline_model"] = model_label(baseline_run)
    delta["candidate_model"] = model_label(candidate_run)
    return delta


def dependence_summary(dependence: pd.DataFrame) -> pd.DataFrame:
    if dependence.empty:
        return pd.DataFrame()
    rows = []
    for (run_id, model), group in dependence.groupby(["run_id", "model"], dropna=False):
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "samples": len(group),
                "blank_changed_rate": group["blank_changed"].mean(),
                "mismatch_changed_rate": group["mismatch_changed"].mean(),
                "blank_score_drop_mean": group["blank_score_drop"].mean(),
                "mismatch_score_drop_mean": group["mismatch_score_drop"].mean(),
            }
        )
    return pd.DataFrame(rows)


def inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 4rem;
            max-width: 1500px;
        }
        .story-hero {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 24px;
            padding: 1.25rem 1.4rem;
            background:
                radial-gradient(circle at 10% 10%, rgba(20, 184, 166, 0.22), transparent 28%),
                radial-gradient(circle at 90% 0%, rgba(245, 158, 11, 0.20), transparent 30%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.97), rgba(17, 24, 39, 0.92));
            box-shadow: 0 18px 60px rgba(2, 6, 23, 0.32);
            margin-bottom: 1.1rem;
        }
        .story-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.25rem;
            letter-spacing: -0.04em;
        }
        .story-hero p {
            color: rgba(226, 232, 240, 0.86);
            font-size: 1rem;
            margin: 0;
        }
        .story-card {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 1rem;
            background: rgba(15, 23, 42, 0.55);
        }
        .small-note {
            color: rgba(148, 163, 184, 0.95);
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def require_run() -> str | None:
    run_id = sidebar_run_selector()
    if not run_id:
        st.warning("No run selected.")
        return None
    return run_id


def resolve_dashboard_image_path(image_path: object) -> Path | None:
    if image_path is None or pd.isna(image_path):
        return None
    path_text = str(image_path)
    candidates = [Path(path_text).expanduser()]
    if path_text.startswith(COLAB_PROJECT_ROOT):
        candidates.append(REPO_ROOT / path_text.removeprefix(COLAB_PROJECT_ROOT).lstrip("/"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def render_dashboard_image(image_path: object, caption: object) -> None:
    resolved = resolve_dashboard_image_path(image_path)
    if resolved is None:
        st.warning("Image file is not available on this machine.")
        if image_path is not None and not pd.isna(image_path):
            st.caption(str(image_path))
        return
    try:
        st.image(str(resolved), caption=str(caption), width="stretch")
    except Exception as exc:
        st.warning(f"Could not render image: {exc}")
        st.caption(str(resolved))
