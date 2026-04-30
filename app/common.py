from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("MM_ALIGN_ARTIFACTS_DIR", REPO_ROOT / "artifacts" / "runs")).resolve()
COLAB_PROJECT_ROOT = "/content/drive/MyDrive/mm-align"


def init_state() -> None:
    st.session_state.setdefault("selected_run", None)


def list_runs() -> list[str]:
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(
        [path.name for path in ARTIFACTS_DIR.iterdir() if path.is_dir()],
        reverse=True,
    )


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
