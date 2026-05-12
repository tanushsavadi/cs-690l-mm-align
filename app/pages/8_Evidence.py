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

from app.common import (
    ARTIFACTS_DIR,
    load_evidence_bootstrap,
    load_evidence_failure_cases,
    load_evidence_paired_cases,
    load_evidence_summary,
)


def _claim_strength(row: pd.Series) -> tuple[str, str]:
    delta = float(row["delta"])
    ci_low = float(row["ci_low"])
    ci_high = float(row["ci_high"])
    if ci_low > 0:
        return "moderate positive evidence", "#14b8a6"
    if ci_high < 0:
        return "standard slightly favored", "#f97316"
    if abs(delta) < 0.003:
        return "near tie", "#94a3b8"
    return "uncertain small effect", "#f59e0b"


def _evidence_card(label: str, value: str, note: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="story-card" style="border-color:{color};">
            <div class="small-note">{label}</div>
            <h2 style="margin:0.25rem 0;color:{color};">{value}</h2>
            <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_label(row: pd.Series) -> str:
    if row["source"] == "dependence":
        return f"dependence / {row['metric']}"
    return f"{row['benchmark']} / {row['metric']}"


def _build_ci_figure(summary: pd.DataFrame) -> go.Figure:
    plot_frame = summary.copy()
    plot_frame["label"] = plot_frame.apply(_metric_label, axis=1)
    plot_frame["claim"], plot_frame["color"] = zip(*plot_frame.apply(_claim_strength, axis=1))
    fig = go.Figure()
    for _, row in plot_frame.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["ci_low"], row["ci_high"]],
                y=[row["label"], row["label"]],
                mode="lines",
                line=dict(color=row["color"], width=8),
                hovertemplate="95% CI: %{x:.4f}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["delta"]],
                y=[row["label"]],
                mode="markers+text",
                marker=dict(color=row["color"], size=13, line=dict(color="white", width=1)),
                text=[f"{row['delta']:+.4f}"],
                textposition="middle right",
                hovertemplate="delta: %{x:.4f}<extra></extra>",
                showlegend=False,
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(148,163,184,0.9)")
    fig.update_layout(
        title="Bootstrap intervals for image-aware minus standard",
        xaxis_title="Delta",
        yaxis_title="",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


st.markdown(
    """
    <div class="story-hero">
        <h1>Evidence Board</h1>
        <p>
        This page is the proof layer for the final project. It uses cached
        predictions only, compares image-aware DPO against standard DPO, and
        separates strong claims from small or uncertain effects.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

summary = load_evidence_summary()
bootstrap = load_evidence_bootstrap()
paired_cases = load_evidence_paired_cases()
failure_cases = load_evidence_failure_cases()

if summary.empty:
    st.warning(
        "No statistical evidence artifacts found. Run "
        "`python3 -m mm_align.cli build-statistical-report --artifacts-dir artifacts/runs --reports-dir reports`."
    )
    st.caption(str(ARTIFACTS_DIR))
    st.stop()

benchmark_summary = summary[summary["source"] == "benchmark"].copy()
dependence_summary = summary[summary["source"] == "dependence"].copy()

headline_rows = {
    (row["benchmark"], row["metric"]): row for _, row in benchmark_summary.iterrows()
}
cards = st.columns(4)
for column, key, label in [
    (cards[0], ("chartqa", "relaxed_accuracy"), "ChartQA"),
    (cards[1], ("hallusionbench", "accuracy"), "HallusionBench"),
    (cards[2], ("pope", "accuracy"), "POPE accuracy"),
    (cards[3], ("pope", "f1"), "POPE F1"),
]:
    row = headline_rows.get(key)
    with column:
        if row is None:
            _evidence_card(label, "n/a", "missing evidence row", "#94a3b8")
            continue
        claim, color = _claim_strength(row)
        _evidence_card(label, f"{row['delta']:+.4f}", claim, color)

st.subheader("1. Claim Strength")
st.plotly_chart(_build_ci_figure(summary), theme="streamlit", width="stretch")

left, right = st.columns([1, 1])
with left:
    st.subheader("2. Bootstrap Delta Distributions")
    if bootstrap.empty:
        st.info("No bootstrap samples found.")
    else:
        selected = st.multiselect(
            "Metrics",
            sorted(bootstrap["metric"].unique()),
            default=["relaxed_accuracy", "accuracy", "f1"],
        )
        plot_data = bootstrap[bootstrap["metric"].isin(selected)].copy()
        plot_data["label"] = plot_data["benchmark"] + " / " + plot_data["metric"]
        fig = px.violin(
            plot_data,
            x="delta",
            y="label",
            color="source",
            box=True,
            points=False,
            title="Bootstrap samples of the metric delta",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(148,163,184,0.9)")
        st.plotly_chart(fig, theme="streamlit", width="stretch")

with right:
    st.subheader("3. Paired Win/Loss Matrix")
    if paired_cases.empty:
        st.info("No paired case counts found.")
    else:
        matrix = paired_cases.pivot_table(index="benchmark", columns="case_type", values="count", aggfunc="sum").fillna(0)
        fig = px.imshow(
            matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Tealgrn",
            title="How often each model wins on the exact same samples",
        )
        st.plotly_chart(fig, theme="streamlit", width="stretch")

st.subheader("4. Dependence Evidence")
if dependence_summary.empty:
    st.info("No dependence summary found.")
else:
    dep = dependence_summary.copy()
    dep["claim"], dep["color"] = zip(*dep.apply(_claim_strength, axis=1))
    st.dataframe(
        dep[["metric", "baseline_value", "candidate_value", "delta", "ci_low", "ci_high", "claim"]],
        width="stretch",
        hide_index=True,
    )

st.subheader("5. Representative Cases")
if failure_cases.empty:
    st.info("No representative cases found.")
else:
    case_type = st.selectbox("Case type", sorted(failure_cases["case_type"].unique()))
    filtered = failure_cases[failure_cases["case_type"] == case_type]
    st.dataframe(
        filtered[
            [
                "benchmark",
                "sample_id",
                "prompt",
                "ground_truth",
                "prediction_baseline",
                "prediction_candidate",
                "is_correct_baseline",
                "is_correct_candidate",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

st.subheader("6. Final Takeaway")
st.markdown(
    """
    The clean reading is not that image-aware DPO wins everywhere. The stronger
    and more honest claim is that it gives a clear ChartQA gain, nearly ties
    HallusionBench, slightly trails standard DPO on POPE, and shows a small
    increase in visual perturbation sensitivity. That is enough to support the
    project as a serious pilot study, while still being honest about limits.
    """
)
