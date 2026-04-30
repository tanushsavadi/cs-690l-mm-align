from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import list_runs, load_selected_summaries, model_label

st.title("Comparison")
st.caption(
    "Side by side comparison of completed cached runs. This dashboard does not run live model inference."
)

runs = list_runs()
if len(runs) < 2:
    st.info("At least two run artifact folders are needed for comparison.")
    st.stop()

default_runs = [
    run
    for run in runs
    if run in {"2026-04-08-standard_dpo-pilot-7", "2026-04-08-image_aware_dpo-pilot-7"}
]
if len(default_runs) < 2:
    default_runs = runs[:2]

selected_runs = st.multiselect("Runs", runs, default=default_runs)
if len(selected_runs) < 2:
    st.warning("Select at least two runs.")
    st.stop()

summary = load_selected_summaries(selected_runs)
if summary.empty:
    st.info("No dashboard summary artifacts found for the selected runs.")
    st.stop()

headline_metrics = {
    ("chartqa", "relaxed_accuracy"),
    ("hallusionbench", "accuracy"),
    ("pope", "accuracy"),
    ("pope", "f1"),
}
headline = summary[
    summary.apply(lambda row: (row["benchmark"], row["metric"]) in headline_metrics, axis=1)
].copy()

st.subheader("Headline Metrics")
if not headline.empty:
    pivot = headline.pivot_table(
        index=["benchmark", "metric"],
        columns="model",
        values="value",
        aggfunc="first",
    )
    st.dataframe(pivot, width="stretch")
else:
    st.info("No headline metrics found.")

baseline_run = st.selectbox("Baseline run", selected_runs, index=0)
candidate_run = st.selectbox("Candidate run", selected_runs, index=min(1, len(selected_runs) - 1))

if baseline_run != candidate_run:
    baseline = summary[summary["run_id"] == baseline_run][["benchmark", "metric", "value"]]
    candidate = summary[summary["run_id"] == candidate_run][["benchmark", "metric", "value"]]
    delta = candidate.merge(
        baseline,
        on=["benchmark", "metric"],
        suffixes=("_candidate", "_baseline"),
    )
    delta["delta"] = delta["value_candidate"] - delta["value_baseline"]
    delta["comparison"] = f"{model_label(candidate_run)} minus {model_label(baseline_run)}"

    st.subheader("Metric Deltas")
    focused_delta = delta[
        delta.apply(lambda row: (row["benchmark"], row["metric"]) in headline_metrics, axis=1)
    ].copy()
    if focused_delta.empty:
        focused_delta = delta.copy()
    st.plotly_chart(
        px.bar(
            focused_delta,
            x="metric",
            y="delta",
            color="benchmark",
            title=delta["comparison"].iloc[0],
        ),
        theme="streamlit",
    )
    st.dataframe(delta.sort_values(["benchmark", "metric"]), width="stretch", hide_index=True)

st.subheader("Full Metric Table")
full = summary.pivot_table(
    index=["benchmark", "metric"],
    columns="model",
    values="value",
    aggfunc="first",
)
st.dataframe(full, width="stretch")
