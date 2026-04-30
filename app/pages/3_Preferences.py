from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import load_preferences, require_run

st.title("Preferences")
run_id = require_run()
if not run_id:
    st.stop()

preferences = load_preferences(run_id)
if preferences.empty:
    st.info("No preference preview artifact found for this run.")
    st.stop()

st.dataframe(preferences, use_container_width=True, hide_index=True)
sample_ids = preferences["sample_id"].tolist()
selected_sample = st.selectbox("Preference Sample", sample_ids)
row = preferences[preferences["sample_id"] == selected_sample].iloc[0]

left, right = st.columns([1, 2])
with left:
    st.image(row["image_path"], caption=row["sample_id"], use_container_width=True)
with right:
    st.markdown(f"**Prompt**\n\n{row['prompt']}")
    st.markdown(f"**Chosen**\n\n{row['chosen']}")
    st.markdown(f"**Rejected**\n\n{row['rejected']}")
    st.metric("Matched Margin", f"{row.get('matched_margin', 0.0):.4f}")
    st.metric("Mismatched Margin", f"{row.get('mismatched_margin', 0.0):.4f}")
