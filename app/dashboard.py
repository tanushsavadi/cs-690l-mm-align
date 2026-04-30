from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import inject_dashboard_css

st.set_page_config(page_title="Multimodal Alignment Dashboard", layout="wide")
inject_dashboard_css()

pages_root = Path(__file__).resolve().parent / "pages"
navigation = st.navigation(
    [
        st.Page(pages_root / "1_Overview.py", title="Overview", icon=":material/dashboard:"),
        st.Page(pages_root / "0_Story_Map.py", title="Story Map", icon=":material/account_tree:"),
        st.Page(pages_root / "2_Examples.py", title="Examples", icon=":material/image:"),
        st.Page(pages_root / "3_Preferences.py", title="Preferences", icon=":material/tune:"),
        st.Page(pages_root / "4_Failures.py", title="Failures", icon=":material/error:"),
        st.Page(pages_root / "5_Comparison.py", title="Comparison", icon=":material/compare_arrows:"),
        st.Page(pages_root / "6_Dependence.py", title="Dependence", icon=":material/hub:"),
        st.Page(pages_root / "7_Training.py", title="Training", icon=":material/monitoring:"),
    ]
)
navigation.run()
