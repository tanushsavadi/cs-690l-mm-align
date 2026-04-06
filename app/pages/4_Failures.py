from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.common import load_examples, require_run

st.title("Failures")
run_id = require_run()
if not run_id:
    st.stop()

examples = load_examples(run_id)
if examples.empty:
    st.info("Example artifacts are missing. Run build-dashboard-data first.")
    st.stop()

failures = examples[examples["failure_tag"] != "stable_response"].copy()
if failures.empty:
    st.success("No failure-tagged examples found for this run.")
    st.stop()

counts = failures.groupby(["benchmark", "failure_tag"]).size().reset_index(name="count")
st.plotly_chart(
    px.bar(counts, x="failure_tag", y="count", color="benchmark", barmode="group"),
    use_container_width=True,
    theme="streamlit",
)
st.dataframe(failures, use_container_width=True, hide_index=True)
