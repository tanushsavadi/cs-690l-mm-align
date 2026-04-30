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
    comparison_delta,
    default_final_runs,
    dependence_summary,
    headline_metric_rows,
    list_runs,
    load_selected_dependence,
    load_selected_summaries,
    model_label,
)


def _delta_value(frame: pd.DataFrame, benchmark: str, metric: str) -> float | None:
    match = frame[(frame["benchmark"] == benchmark) & (frame["metric"] == metric)]
    if match.empty:
        return None
    return float(match["delta"].iloc[0])


def _metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="story-card">
            <div class="small-note">{label}</div>
            <h2 style="margin:0.2rem 0 0.2rem 0;">{value}</h2>
            <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_performance_sankey(headline: pd.DataFrame) -> go.Figure:
    labels = ["Preference Data", "standard_dpo", "image_aware_dpo", "ChartQA", "HallusionBench", "POPE"]
    label_to_index = {label: index for index, label in enumerate(labels)}
    sources = [0, 0]
    targets = [1, 2]
    values = [80, 80]
    colors = ["rgba(20,184,166,0.35)", "rgba(245,158,11,0.35)"]

    metric_names = {
        ("chartqa", "relaxed_accuracy"): "ChartQA",
        ("hallusionbench", "accuracy"): "HallusionBench",
        ("pope", "accuracy"): "POPE",
    }
    for _, row in headline.iterrows():
        target = metric_names.get((row["benchmark"], row["metric"]))
        if not target:
            continue
        sources.append(label_to_index[row["model"]])
        targets.append(label_to_index[target])
        values.append(max(float(row["value"]) * 100, 1))
        colors.append("rgba(20,184,166,0.45)" if row["model"] == "image_aware_dpo" else "rgba(99,102,241,0.42)")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=22,
                    thickness=20,
                    line=dict(color="rgba(226,232,240,0.5)", width=0.6),
                    label=labels,
                    color=["#0f172a", "#6366f1", "#14b8a6", "#f59e0b", "#22c55e", "#ef4444"],
                ),
                link=dict(source=sources, target=targets, value=values, color=colors),
            )
        ]
    )
    fig.update_layout(
        title="Performance flow from preference data to evaluation tasks",
        font=dict(size=13),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def _build_story_network(delta: pd.DataFrame) -> go.Figure:
    focused = delta[
        delta.apply(
            lambda row: (row["benchmark"], row["metric"])
            in {
                ("chartqa", "relaxed_accuracy"),
                ("hallusionbench", "accuracy"),
                ("pope", "accuracy"),
                ("pope", "f1"),
            },
            axis=1,
        )
    ].copy()
    if focused.empty:
        focused = delta.copy().head(8)

    nodes = [
        {"id": "standard_dpo", "x": 0.05, "y": 0.65, "size": 34, "color": "#6366f1"},
        {"id": "image_aware_dpo", "x": 0.05, "y": 0.35, "size": 34, "color": "#14b8a6"},
    ]
    benchmarks = sorted(focused["benchmark"].unique())
    for index, benchmark in enumerate(benchmarks):
        nodes.append(
            {
                "id": benchmark,
                "x": 0.48,
                "y": 0.82 - index * (0.62 / max(len(benchmarks) - 1, 1)),
                "size": 28,
                "color": "#f59e0b",
            }
        )
    for _, row in focused.iterrows():
        winner = "image-aware higher" if row["delta"] > 0 else "standard higher"
        nodes.append(
            {
                "id": f"{row['benchmark']} {row['metric']}\\n{winner} {row['delta']:+.4f}",
                "x": 0.9,
                "y": 0.82 - benchmarks.index(row["benchmark"]) * (0.62 / max(len(benchmarks) - 1, 1)),
                "size": 18 + min(abs(row["delta"]) * 600, 14),
                "color": "#22c55e" if row["delta"] > 0 else "#ef4444",
                "benchmark": row["benchmark"],
            }
        )

    node_lookup = {node["id"]: node for node in nodes}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for benchmark in benchmarks:
        for model in ["standard_dpo", "image_aware_dpo"]:
            edge_x.extend([node_lookup[model]["x"], node_lookup[benchmark]["x"], None])
            edge_y.extend([node_lookup[model]["y"], node_lookup[benchmark]["y"], None])
    for node in nodes:
        if "benchmark" not in node:
            continue
        benchmark_node = node_lookup[node["benchmark"]]
        edge_x.extend([benchmark_node["x"], node["x"], None])
        edge_y.extend([benchmark_node["y"], node["y"], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.6, color="rgba(148,163,184,0.38)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[node["x"] for node in nodes],
            y=[node["y"] for node in nodes],
            mode="markers+text",
            marker=dict(
                size=[node["size"] for node in nodes],
                color=[node["color"] for node in nodes],
                line=dict(width=1.4, color="rgba(255,255,255,0.75)"),
            ),
            text=[node["id"] for node in nodes],
            textposition="bottom center",
            hovertext=[node["id"] for node in nodes],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Result graph: methods, benchmarks, and where the story changes",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=45, b=10),
        height=560,
    )
    return fig


st.markdown(
    """
    <div class="story-hero">
        <h1>Multimodal Alignment Story Map</h1>
        <p>
        This page turns the final cached experiment into a visual argument:
        image-aware DPO helped most on ChartQA, tied HallusionBench closely,
        and did not clearly beat standard DPO on POPE.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

runs = list_runs()
if len(runs) < 2:
    st.info("At least two completed run folders are needed for this story view.")
    st.stop()

default_runs = default_final_runs(runs)
with st.sidebar:
    st.header("Story Controls")
    selected_runs = st.multiselect("Runs in story", runs, default=default_runs)
if len(selected_runs) < 2:
    st.warning("Select at least two runs.")
    st.stop()

summary = load_selected_summaries(selected_runs)
if summary.empty:
    st.info("No dashboard summary artifacts found for selected runs.")
    st.stop()

with st.sidebar:
    baseline_run = st.selectbox("Baseline", selected_runs, index=0)
    candidate_run = st.selectbox("Candidate", selected_runs, index=min(1, len(selected_runs) - 1))
if baseline_run == candidate_run:
    st.warning("Pick two different runs.")
    st.stop()

headline = headline_metric_rows(summary)
delta = comparison_delta(summary, baseline_run, candidate_run)
focused_delta = headline_metric_rows(delta.rename(columns={"value_candidate": "value"}))
if not focused_delta.empty:
    focused_delta = delta.merge(focused_delta[["benchmark", "metric"]], on=["benchmark", "metric"], how="inner")

chartqa_delta = _delta_value(focused_delta, "chartqa", "relaxed_accuracy")
hallusion_delta = _delta_value(focused_delta, "hallusionbench", "accuracy")
pope_delta = _delta_value(focused_delta, "pope", "accuracy")

cards = st.columns(4)
with cards[0]:
    _metric_card("Best clear gain", f"{chartqa_delta:+.4f}" if chartqa_delta is not None else "n/a", "ChartQA relaxed accuracy")
with cards[1]:
    _metric_card("Hallucination benchmark", f"{hallusion_delta:+.4f}" if hallusion_delta is not None else "n/a", "HallusionBench overall accuracy")
with cards[2]:
    _metric_card("Object hallucination", f"{pope_delta:+.4f}" if pope_delta is not None else "n/a", "POPE accuracy")
with cards[3]:
    _metric_card("Main conclusion", "mixed", "small but useful image-aware signal")

st.subheader("1. Network View")
st.plotly_chart(_build_story_network(delta), theme="streamlit")

left, right = st.columns([1.05, 1])
with left:
    st.subheader("2. Performance Flow")
    st.plotly_chart(_build_performance_sankey(headline), theme="streamlit")

with right:
    st.subheader("3. Delta Heatmap")
    heat_source = focused_delta if not focused_delta.empty else delta
    heat = heat_source.pivot_table(index="benchmark", columns="metric", values="delta", aggfunc="first").fillna(0)
    heat_fig = px.imshow(
        heat,
        text_auto=".4f",
        aspect="auto",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title=f"{model_label(candidate_run)} minus {model_label(baseline_run)}",
    )
    heat_fig.update_xaxes(side="top")
    st.plotly_chart(heat_fig, theme="streamlit")

st.subheader("4. Dependence Terrain")
dependence = load_selected_dependence(selected_runs)
if dependence.empty:
    st.info("No dependence artifacts found for selected runs.")
else:
    dep_summary = dependence_summary(dependence)
    dep_long = dep_summary.melt(
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
    dep_long["metric_family"] = dep_long["metric"].str.replace("_mean", "", regex=False).str.replace(
        "_rate", "", regex=False
    )
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(
            px.treemap(
                dep_long,
                path=["model", "metric_family", "metric"],
                values="value",
                color="value",
                color_continuous_scale="Viridis",
                title="Grounding sensitivity tree",
            ),
            theme="streamlit",
        )
    with col_b:
        radar = go.Figure()
        radar_metrics = [
            "blank_changed_rate",
            "mismatch_changed_rate",
            "blank_score_drop_mean",
            "mismatch_score_drop_mean",
        ]
        for _, row in dep_summary.iterrows():
            values = [float(row[metric]) for metric in radar_metrics]
            radar.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself",
                    name=row["model"],
                )
            )
        radar.update_layout(
            title="Dependence fingerprint",
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
        )
        st.plotly_chart(radar, theme="streamlit")

with st.expander("How to explain this page in the final demo"):
    st.markdown(
        """
        - The network graph shows the full story in one place: methods, benchmarks, and which side is higher.
        - The Sankey chart shows that both models come from the same preference data, but their benchmark flows differ.
        - The heatmap makes the direction of each metric change easy to see.
        - The dependence tree and radar chart show whether answers react when the image is blanked or mismatched.
        - This is cached analysis, not live model inference. The heavy model work was already done on Colab.
        """
    )
