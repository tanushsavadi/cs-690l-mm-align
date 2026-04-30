from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import default_final_runs, list_runs, load_metrics, load_training_history, model_label

st.title("Training")
st.caption("Training curves from saved metrics.json files. This is cached training history, not a live run.")

runs = list_runs()
if not runs:
    st.warning("No run artifacts found.")
    st.stop()

default_runs = default_final_runs(runs)

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

if {"loss", "matched_margin"}.issubset(history.columns):
    st.subheader("Training Phase Space")
    phase = history.copy()
    if "gap_loss" in phase:
        phase["gap_loss_display"] = phase["gap_loss"].fillna(0).clip(lower=0)
    else:
        phase["gap_loss_display"] = 0.0
    fig = px.scatter(
        phase,
        x="matched_margin",
        y="loss",
        color="model",
        size="gap_loss_display",
        hover_data=["step", "run_id"],
        title="Loss versus matched margin. Larger bubbles mean larger image-aware gap loss.",
        color_discrete_sequence=["#14b8a6", "#f59e0b", "#ef4444"],
    )
    st.plotly_chart(fig, theme="streamlit")

if {"loss", "dpo_loss"}.issubset(history.columns):
    st.subheader("Final Training Snapshot")
    final_points = history.sort_values("step").groupby("model", as_index=False).tail(1)
    fig = go.Figure()
    for _, row in final_points.iterrows():
        labels = ["loss", "dpo_loss", "matched_margin"]
        values = [row.get(label, 0) or 0 for label in labels]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name=row["model"],
            )
        )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig, theme="streamlit")

st.subheader("Raw Training History")
columns = ["run_id", "model", "step"] + metric_options
st.dataframe(history[[column for column in columns if column in history.columns]], width="stretch", hide_index=True)
