from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import list_runs, load_dependence, model_label

st.title("Dependence")
st.caption(
    "Blank and mismatched image checks. Higher changed rates mean the answer is more sensitive to the image perturbation."
)

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

frames = []
for run_id in selected_runs:
    frame = load_dependence(run_id).copy()
    if frame.empty:
        continue
    frame["model"] = model_label(run_id)
    frames.append(frame)

if not frames:
    st.info("No dependence artifacts found for the selected runs.")
    st.stop()

dependence = pd.concat(frames, ignore_index=True)

summary_rows = []
for (run_id, model), group in dependence.groupby(["run_id", "model"], dropna=False):
    summary_rows.append(
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
summary = pd.DataFrame(summary_rows)

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
