"""Streamlit entry point for the Shill Bidding Detection dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make project packages importable regardless of the working directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import about, backlog, dataset, detection, overview, performance  # noqa: E402

st.set_page_config(
    page_title="Shill Bidding Detection",
    page_icon=":material/gavel:",
    layout="wide",
)

pages = [
    st.Page(overview.render, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(dataset.render, title="Dataset Explorer", icon=":material/table_chart:", url_path="dataset"),
    st.Page(performance.render, title="Model Performance", icon=":material/insights:", url_path="performance"),
    st.Page(detection.render, title="Fraud Detection", icon=":material/policy:", url_path="detection"),
    st.Page(backlog.render, title="Fraud Backlog", icon=":material/inventory:", url_path="backlog"),
    st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
]

with st.sidebar:
    st.markdown("### Shill Bidding Detection")
    st.caption("UCI Shill Bidding Dataset (ID 562) - CC BY 4.0")

st.navigation(pages).run()
