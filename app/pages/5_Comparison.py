from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import comparison_delta, default_final_runs, headline_metric_rows, list_runs, load_selected_summaries, model_label

st.title("Comparison")
st.caption(
    "Side by side comparison of completed cached runs. This dashboard does not run live model inference."
)

runs = list_runs()
if len(runs) < 2:
    st.info("At least two run artifact folders are needed for comparison.")
    st.stop()

default_runs = default_final_runs(runs)

selected_runs = st.multiselect("Runs", runs, default=default_runs)
if len(selected_runs) < 2:
    st.warning("Select at least two runs.")
    st.stop()

summary = load_selected_summaries(selected_runs)
if summary.empty:
    st.info("No dashboard summary artifacts found for the selected runs.")
    st.stop()

headline = headline_metric_rows(summary)

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
    delta = comparison_delta(summary, baseline_run, candidate_run)
    delta["comparison"] = f"{model_label(candidate_run)} minus {model_label(baseline_run)}"

    st.subheader("Metric Deltas")
    focused_delta = headline_metric_rows(delta.rename(columns={"value_candidate": "value"}))
    if not focused_delta.empty:
        focused_delta = delta.merge(
            focused_delta[["benchmark", "metric"]],
            on=["benchmark", "metric"],
            how="inner",
        )
    if focused_delta.empty:
        focused_delta = delta.copy()

    tabs = st.tabs(["Delta Bars", "Delta Heatmap", "Metric Tree"])
    with tabs[0]:
        st.plotly_chart(
            px.bar(
                focused_delta,
                x="metric",
                y="delta",
                color="benchmark",
                title=delta["comparison"].iloc[0],
                color_discrete_sequence=["#14b8a6", "#f59e0b", "#ef4444"],
            ),
            theme="streamlit",
        )
    with tabs[1]:
        heat = focused_delta.pivot_table(index="benchmark", columns="metric", values="delta", aggfunc="first").fillna(0)
        fig = px.imshow(
            heat,
            text_auto=".4f",
            aspect="auto",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Candidate minus baseline",
        )
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, theme="streamlit")
    with tabs[2]:
        tree = delta.copy()
        tree["abs_delta"] = tree["delta"].abs()
        tree["direction"] = tree["delta"].map(lambda value: "candidate higher" if value > 0 else "baseline higher")
        st.plotly_chart(
            px.treemap(
                tree,
                path=["benchmark", "direction", "metric"],
                values="abs_delta",
                color="delta",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                title="Where the comparison changes most",
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
