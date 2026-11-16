"""Dashboard overview page: headline numbers and historical detection statistics."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from config import settings
from dashboard.common import (
    COLORS,
    STATUS_COLORS,
    fmt_pct,
    get_bundle,
    get_clean_data,
    get_manager,
    get_results,
    page_header,
    require_artifacts,
)


def render() -> None:
    page_header(
        "Shill Bidding Detection - Overview",
        "Machine-learning analysis of suspicious bidding behaviour in online auctions "
        "(UCI Shill Bidding Dataset, ID 562).",
    )
    if not require_artifacts():
        return
    bundle, results, df = get_bundle(), get_results(), get_clean_data()
    stats = get_manager().statistics()

    st.subheader("Model activity (prediction log)")
    c = st.columns(5)
    c[0].metric("Records analysed", f"{stats['records_analysed']:,}")
    c[1].metric("Predicted normal", f"{stats['predicted_normal']:,}")
    c[2].metric("Potential shill bidding", f"{stats['predicted_shill']:,}")
    c[3].metric("Flag rate", fmt_pct(stats["detection_rate"]))
    c[4].metric("Pending investigations", f"{stats['pending']:,}",
                help="Backlog entries with status 'Pending Review'.")
    if stats["records_analysed"] == 0:
        st.caption("No records scored yet - use **Fraud Detection** to analyse records or run a batch scan.")

    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Dataset (ground-truth labels)")
        n = len(df)
        shill = int(df[settings.TARGET].sum())
        d = st.columns(3)
        d[0].metric("Records", f"{n:,}")
        d[1].metric("Normal (label 0)", f"{n - shill:,}")
        d[2].metric("Shill (label 1)", f"{shill:,}", fmt_pct(shill / n), delta_color="off")
        d2 = st.columns(3)
        d2[0].metric("Auctions", f"{df[settings.AUCTION_ID].nunique():,}")
        d2[1].metric("Masked bidder IDs", f"{df[settings.BIDDER_ID].nunique():,}")
        d2[2].metric("Behavioural features", len(settings.BEHAVIOURAL_FEATURES))
    with right:
        st.subheader("Deployed model")
        tm = bundle["test_metrics"]
        st.markdown(
            f"**{bundle['model_name']}** - version `{bundle['model_version']}`  \n"
            f"Decision threshold **{bundle['threshold']:.2f}** - trained {bundle['trained_at']}"
        )
        m = st.columns(4)
        m[0].metric("Recall", fmt_pct(tm["recall"]))
        m[1].metric("Precision", fmt_pct(tm["precision"]))
        m[2].metric("F1", f"{tm['f1']:.3f}")
        m[3].metric("PR-AUC", f"{tm['pr_auc']:.3f}")
        st.caption(f"Measured on {results['split']['test_rows']:,} records from "
                   f"{results['split']['test_auctions']} held-out auctions never seen during training.")

    st.divider()
    st.subheader("Investigation backlog")
    b = st.columns(4)
    b[0].metric("Backlog entries", stats["backlog_total"])
    b[1].metric("Open cases", stats["open_cases"], help="Pending Review + Under Investigation")
    b[2].metric("Confirmed suspicious", stats["status_counts"]["Confirmed Suspicious"])
    b[3].metric("Confirmation rate", fmt_pct(stats["confirmation_rate"]),
                help="Confirmed / (Confirmed + False Positive) among reviewed cases.")

    if stats["backlog_total"]:
        g1, g2 = st.columns(2)
        with g1:
            sc = {k: v for k, v in stats["status_counts"].items() if v}
            fig = px.pie(names=list(sc), values=list(sc.values()), hole=0.5,
                         color=list(sc), color_discrete_map=STATUS_COLORS, title="Cases by review status")
            st.plotly_chart(fig, width="stretch")
        with g2:
            daily = stats["daily_flags"]
            fig = px.bar(daily, x="date", y="flags", title="Flagged records over time",
                         color_discrete_sequence=[COLORS["shill"]])
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, width="stretch")
    else:
        st.caption("The backlog is empty. Flagged predictions saved from **Fraud Detection** appear here.")
