"""Shared, cached resources and UI helpers for the Streamlit dashboard."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from config import settings
from src.backlog.manager import BacklogManager
from src.explainability.explain import ModelExplainer
from src.prediction.predict import ModelNotTrainedError, load_bundle

COLORS = {"normal": "#4C78A8", "shill": "#E45756", "accent": "#72B7B2"}
STATUS_COLORS = {
    "Pending Review": "#F2B701",
    "Under Investigation": "#3969AC",
    "Confirmed Suspicious": "#E73F74",
    "False Positive": "#7F3C8D",
    "Closed": "#80BA5A",
}


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict | None:
    try:
        return load_bundle()
    except ModelNotTrainedError:
        return None


@st.cache_resource(show_spinner=False)
def get_explainer() -> ModelExplainer | None:
    bundle = get_bundle()
    return ModelExplainer(bundle) if bundle else None


@st.cache_resource(show_spinner=False)
def _manager_for(db_path: str) -> BacklogManager:
    return BacklogManager(db_path)


def get_manager() -> BacklogManager:
    """Backlog manager for the configured database (path resolved at call time)."""
    return _manager_for(str(settings.BACKLOG_DB_FILE))


@st.cache_data(show_spinner=False)
def get_clean_data() -> pd.DataFrame | None:
    if not settings.CLEAN_DATA_FILE.exists():
        return None
    df = pd.read_csv(settings.CLEAN_DATA_FILE, dtype={settings.BIDDER_ID: str})
    if settings.SPLIT_FILE.exists():
        split = pd.read_csv(settings.SPLIT_FILE)
        df = df.merge(split, on=settings.RECORD_ID, how="left")
    return df


@st.cache_data(show_spinner=False)
def get_results() -> dict | None:
    if not settings.EVALUATION_RESULTS_FILE.exists():
        return None
    return json.loads(settings.EVALUATION_RESULTS_FILE.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_eda() -> dict | None:
    if not settings.EDA_SUMMARY_FILE.exists():
        return None
    return json.loads(settings.EDA_SUMMARY_FILE.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def get_curves() -> dict | None:
    import joblib

    if not settings.CURVES_FILE.exists():
        return None
    return joblib.load(settings.CURVES_FILE)


def require_artifacts() -> bool:
    """Show a helpful message and return False if the pipeline has not been run."""
    if get_bundle() is None or get_results() is None or get_clean_data() is None:
        st.warning(
            "No trained model or processed data found. Run the offline pipeline first:\n\n"
            "```\npython run_pipeline.py\n```\n\nthen reload this page."
        )
        return False
    return True


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def fmt_pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x * 100:.{digits}f}%"


def disclaimer() -> None:
    st.info(
        "A positive prediction marks a **potential shill-bidding case that requires investigation**. "
        "It is a statistical signal from a model trained on labelled historical data, not proof "
        "of fraud or intent.",
        icon=":material/policy:",
    )
