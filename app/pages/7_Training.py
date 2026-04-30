from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import list_runs, load_metrics, load_training_history, model_label

st.title("Training")
st.caption("Training curves from saved metrics.json files. This is cached training history, not a live run.")

runs = list_runs()
if not runs:
    st.warning("No run artifacts found.")
    st.stop()

default_runs = [
    run
    for run in runs
    if run in {"2026-04-08-standard_dpo-pilot-7", "2026-04-08-image_aware_dpo-pilot-7"}
]
if not default_runs:
    default_runs = runs[:2]

selected_runs = st.multiselect("Runs", runs, default=default_runs)
if not selected_runs:
    st.stop()

histories = []
last_rows = []
for run_id in selected_runs:
    history = load_training_history(run_id)
    if not history.empty:
        histories.append(history)
    metrics = load_metrics(run_id)
    last = metrics.get("last_metrics", {}).copy()
    if last:
        last["run_id"] = run_id
        last["model"] = metrics.get("model_variant") or model_label(run_id)
        last_rows.append(last)

if last_rows:
    st.subheader("Last Logged Metrics")
    st.dataframe(pd.DataFrame(last_rows), width="stretch", hide_index=True)

if not histories:
    st.info("No training history found for selected runs.")
    st.stop()

history = pd.concat(histories, ignore_index=True)
metric_options = [
    metric
    for metric in ["loss", "dpo_loss", "gap_loss", "anchor_loss", "matched_margin", "mismatched_margin"]
    if metric in history.columns
]
selected_metric = st.selectbox("Training metric", metric_options)

st.plotly_chart(
    px.line(
        history,
        x="step",
        y=selected_metric,
        color="model",
        title=f"{selected_metric} by step",
    ),
    theme="streamlit",
)

st.subheader("Raw Training History")
columns = ["run_id", "model", "step"] + metric_options
st.dataframe(history[[column for column in columns if column in history.columns]], width="stretch", hide_index=True)
