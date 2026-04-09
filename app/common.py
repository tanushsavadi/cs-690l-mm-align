from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("MM_ALIGN_ARTIFACTS_DIR", REPO_ROOT / "artifacts" / "runs")).resolve()


def init_state() -> None:
    st.session_state.setdefault("selected_run", None)


def list_runs() -> list[str]:
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(
        [path.name for path in ARTIFACTS_DIR.iterdir() if path.is_dir()],
        reverse=True,
    )


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


def require_run() -> str | None:
    run_id = sidebar_run_selector()
    if not run_id:
        st.warning("No run selected.")
        return None
    return run_id
