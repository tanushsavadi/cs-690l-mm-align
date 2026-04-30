from __future__ import annotations

from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import load_summary, require_run

st.title("Overview")
run_id = require_run()
if not run_id:
    st.stop()

summary = load_summary(run_id)
if summary.empty:
    st.info("Dashboard summary artifacts are missing. Run build-dashboard-data first.")
    st.stop()

benchmarks = sorted(summary["benchmark"].unique())
selected_benchmark = st.selectbox("Benchmark", ["All"] + benchmarks)
view = summary if selected_benchmark == "All" else summary[summary["benchmark"] == selected_benchmark]

metrics = view[~view["metric"].str.contains(r"\.", regex=True)]
if not metrics.empty:
    columns = st.columns(min(4, max(1, len(metrics))))
    for column, (_, row) in zip(columns, metrics.iterrows()):
        column.metric(label=f"{row['benchmark']}:{row['metric']}", value=f"{row['value']:.3f}")

st.subheader("Metric Table")
st.dataframe(view, width="stretch", hide_index=True)

st.subheader("Metric Chart")
figure = px.bar(view, x="metric", y="value", color="benchmark", barmode="group")
st.plotly_chart(figure, theme="streamlit")
