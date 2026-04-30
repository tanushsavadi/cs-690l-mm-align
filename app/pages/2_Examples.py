from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import load_examples, render_dashboard_image, require_run

st.title("Examples")
run_id = require_run()
if not run_id:
    st.stop()

examples = load_examples(run_id)
if examples.empty:
    st.info("Example artifacts are missing. Run build-dashboard-data first.")
    st.stop()

benchmarks = ["All"] + sorted(examples["benchmark"].unique())
failure_tags = ["All"] + sorted(examples["failure_tag"].dropna().unique())
selected_benchmark = st.selectbox("Benchmark", benchmarks)
selected_tag = st.selectbox("Failure Tag", failure_tags)

filtered = examples.copy()
if selected_benchmark != "All":
    filtered = filtered[filtered["benchmark"] == selected_benchmark]
if selected_tag != "All":
    filtered = filtered[filtered["failure_tag"] == selected_tag]

st.dataframe(
    filtered[
        [
            "sample_id",
            "benchmark",
            "failure_tag",
            "prompt",
            "ground_truth",
            "prediction_original",
        ]
    ],
    width="stretch",
    hide_index=True,
)

sample_ids = filtered["sample_id"].tolist()
selected_sample = st.selectbox("Sample", sample_ids) if sample_ids else None
if not selected_sample:
    st.stop()

row = filtered[filtered["sample_id"] == selected_sample].iloc[0]
image_col, text_col = st.columns([1, 2])
with image_col:
    render_dashboard_image(row.get("image_path"), row.get("sample_id", "sample"))
with text_col:
    st.markdown(f"**Prompt**\n\n{row['prompt']}")
    st.markdown(f"**Ground Truth**\n\n{row['ground_truth']}")
    tabs = st.tabs(["Original", "Blank Image", "Mismatched Image"])
    with tabs[0]:
        st.write(row.get("prediction_original", ""))
        st.caption(f"Correct: {bool(row.get('is_correct_original', False))}")
    with tabs[1]:
        st.write(row.get("prediction_blank-image", ""))
        st.caption(f"Changed: {bool(row.get('blank_changed', False))}")
    with tabs[2]:
        st.write(row.get("prediction_mismatched-image", ""))
        st.caption(f"Changed: {bool(row.get('mismatch_changed', False))}")
