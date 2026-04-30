from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import default_final_runs, dependence_summary, list_runs, load_selected_dependence

st.title("Dependence")
st.caption(
    "Blank and mismatched image checks. Higher changed rates mean the answer is more sensitive to the image perturbation."
)

runs = list_runs()
if not runs:
    st.warning("No run artifacts found.")
    st.stop()

default_runs = default_final_runs(runs)

selected_runs = st.multiselect("Runs", runs, default=default_runs)
if not selected_runs:
    st.stop()

dependence = load_selected_dependence(selected_runs)
if dependence.empty:
    st.info("No dependence artifacts found for the selected runs.")
    st.stop()

summary = dependence_summary(dependence)

st.subheader("Overall Dependence Summary")
st.dataframe(summary, width="stretch", hide_index=True)

long_summary = summary.melt(
    id_vars=["run_id", "model", "samples"],
    value_vars=[
        "blank_changed_rate",
        "mismatch_changed_rate",
        "blank_score_drop_mean",
        "mismatch_score_drop_mean",
    ],
    var_name="metric",
    value_name="value",
)
st.plotly_chart(
    px.bar(long_summary, x="metric", y="value", color="model", barmode="group"),
    theme="streamlit",
)

st.subheader("Dependence Tree")
tree = long_summary.copy()
tree["metric_family"] = tree["metric"].str.replace("_mean", "", regex=False).str.replace("_rate", "", regex=False)
st.plotly_chart(
    px.treemap(
        tree,
        path=["model", "metric_family", "metric"],
        values="value",
        color="value",
        color_continuous_scale="Teal",
        title="How each model reacts to image perturbations",
    ),
    theme="streamlit",
)

st.subheader("By Benchmark")
by_benchmark = (
    dependence.groupby(["model", "benchmark"], dropna=False)
    .agg(
        samples=("sample_id", "count"),
        blank_changed_rate=("blank_changed", "mean"),
        mismatch_changed_rate=("mismatch_changed", "mean"),
        blank_score_drop_mean=("blank_score_drop", "mean"),
        mismatch_score_drop_mean=("mismatch_score_drop", "mean"),
    )
    .reset_index()
)
st.dataframe(by_benchmark, width="stretch", hide_index=True)

st.subheader("Perturbation Example Explorer")
benchmark_options = ["All"] + sorted(dependence["benchmark"].dropna().unique())
selected_benchmark = st.selectbox("Benchmark", benchmark_options)
filtered = dependence if selected_benchmark == "All" else dependence[dependence["benchmark"] == selected_benchmark]

changed_only = st.checkbox("Only examples that changed under blank or mismatch", value=True)
if changed_only:
    filtered = filtered[filtered["blank_changed"] | filtered["mismatch_changed"]]

sample_options = filtered["sample_id"].dropna().astype(str).head(200).tolist()
if not sample_options:
    st.info("No examples match these filters.")
    st.stop()

selected_sample = st.selectbox("Sample", sample_options)
row = filtered[filtered["sample_id"].astype(str) == selected_sample].iloc[0]

st.markdown(f"**Prompt**\n\n{row['prompt']}")
st.markdown(f"**Ground truth**\n\n{row['ground_truth']}")

cols = st.columns(3)
with cols[0]:
    st.markdown("**Original image answer**")
    st.write(row.get("prediction_original", ""))
    st.caption(f"Correct: {bool(row.get('is_correct_original', False))}")
with cols[1]:
    st.markdown("**Blank image answer**")
    st.write(row.get("prediction_blank-image", ""))
    st.caption(f"Changed: {bool(row.get('blank_changed', False))}")
with cols[2]:
    st.markdown("**Mismatched image answer**")
    st.write(row.get("prediction_mismatched-image", ""))
    st.caption(f"Changed: {bool(row.get('mismatch_changed', False))}")
