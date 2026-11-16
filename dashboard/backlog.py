"""Fraud backlog page: browse, filter, review and export flagged cases."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings
from dashboard.common import STATUS_COLORS, fmt_pct, get_bundle, get_manager, page_header

LIST_COLUMNS = ["backlog_id", "record_id", "auction_id", "bidder_id", "fraud_probability",
                "prediction_label", "review_status", "model_version", "prediction_timestamp", "updated_at"]


def _case_detail(manager, backlog_id: int) -> None:
    case = manager.get_case(backlog_id)
    if case is None:
        st.warning("Case not found.")
        return
    st.markdown(f"### Case #{case['backlog_id']}")
    c = st.columns(4)
    c[0].metric("Fraud probability", f"{case['fraud_probability']:.3f}")
    c[1].metric("Model prediction", "Potential shill" if case["predicted_class"] else "Normal")
    c[2].metric("Review status", case["review_status"])
    c[3].metric("Threshold", f"{case['threshold']:.2f}")
    st.markdown(
        f"**Record** `{case['record_id']}` - **Auction** `{case['auction_id']}` - **Bidder** `{case['bidder_id']}`  \n"
        f"Predicted {case['prediction_timestamp']} by model `{case['model_version']}` (source: {case['source']}). "
        f"Last updated {case['updated_at']}."
    )
    st.caption("The model prediction is fixed; the review status records the investigator's conclusion.")

    left, right = st.columns(2)
    with left:
        if case["top_features"]:
            st.markdown("**Main behaviours behind the flag:** " + ", ".join(f"`{f}`" for f in case["top_features"]))
        if case["feature_values"]:
            fv = pd.Series(case["feature_values"], name="value").to_frame()
            bundle = get_bundle()
            if bundle:
                ref = bundle["reference_stats"]
                fv["typical normal"] = [ref["median_normal"].get(f) for f in fv.index]
                fv["typical shill"] = [ref["median_shill"].get(f) for f in fv.index]
            st.dataframe(fv.round(4), width="stretch")
    with right:
        st.markdown("**Investigation notes**")
        st.text(case["investigation_notes"] or "No notes yet.")
        st.markdown("**Status history**")
        st.dataframe(case["history"][["changed_at", "old_status", "new_status", "note", "changed_by"]],
                     hide_index=True, width="stretch")

    with st.form(f"update_{backlog_id}", clear_on_submit=True):
        st.markdown("**Update investigation**")
        f1, f2 = st.columns([1, 1])
        new_status = f1.selectbox("Review status", settings.REVIEW_STATUSES,
                                  index=settings.REVIEW_STATUSES.index(case["review_status"]))
        author = f2.text_input("Investigator", value="investigator")
        note = st.text_area("Note", placeholder="Findings, evidence checked, next steps...")
        if st.form_submit_button("Save update", type="primary"):
            try:
                if new_status != case["review_status"]:
                    manager.update_status(backlog_id, new_status, note=note, changed_by=author or "investigator")
                elif note.strip():
                    manager.add_note(backlog_id, note, author=author or "investigator")
                else:
                    st.info("Nothing to save - change the status or write a note.")
                    return
                st.success("Case updated.")
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc))


def _statistics(manager) -> None:
    stats = manager.statistics()
    c = st.columns(5)
    c[0].metric("Backlog entries", stats["backlog_total"])
    c[1].metric("Open cases", stats["open_cases"])
    c[2].metric("Confirmed suspicious", stats["status_counts"]["Confirmed Suspicious"])
    c[3].metric("False positives", stats["status_counts"]["False Positive"])
    c[4].metric("Confirmation rate", fmt_pct(stats["confirmation_rate"]))
    if not stats["backlog_total"]:
        st.info("No historical data yet.")
        return
    data = manager.all_entries()
    g1, g2 = st.columns(2)
    sc = pd.DataFrame({"status": list(stats["status_counts"]), "cases": list(stats["status_counts"].values())})
    g1.plotly_chart(px.bar(sc, x="status", y="cases", color="status", color_discrete_map=STATUS_COLORS,
                           title="Cases by review status"), width="stretch")
    g2.plotly_chart(px.histogram(data, x="fraud_probability", nbins=20, color="review_status",
                                 color_discrete_map=STATUS_COLORS, title="Fraud probability of backlog cases"),
                    width="stretch")
    g3, g4 = st.columns(2)
    daily = stats["daily_flags"]
    g3.plotly_chart(px.line(daily, x="date", y="flags", markers=True, title="Flags per day"), width="stretch")
    g4.plotly_chart(px.bar(stats["top_bidders"], x="bidder_id", y="flags", title="Most frequently flagged bidder IDs")
                    .update_xaxes(type="category"), width="stretch")
    st.caption("Masked bidder IDs may combine several real bidders; treat repeat counts as leads, not identities.")


def render() -> None:
    page_header("Fraud Backlog", "Historical record of potential shill-bidding cases and their investigation outcome.")
    manager = get_manager()
    tabs = st.tabs(["Cases", "Case details & review", "Statistics"])

    with tabs[0]:
        f1, f2, f3 = st.columns([1.2, 1.6, 1.2])
        search = f1.text_input("Search bidder / auction / record ID")
        statuses = f2.multiselect("Review status", settings.REVIEW_STATUSES)
        prob = f3.slider("Fraud probability", 0.0, 1.0, (0.0, 1.0), 0.01)
        data = manager.query(search=search or None, statuses=statuses or None,
                             min_probability=prob[0], max_probability=prob[1])
        st.caption(f"{len(data)} matching cases")
        st.dataframe(data[LIST_COLUMNS] if len(data) else data, hide_index=True, width="stretch", height=420,
                     column_config={"fraud_probability": st.column_config.ProgressColumn(
                         "fraud_probability", min_value=0.0, max_value=1.0, format="%.3f")})
        c1, c2 = st.columns(2)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        c1.download_button("Export filtered cases (CSV)", manager.export_csv(frame=data),
                           file_name=f"fraud_backlog_filtered_{stamp}.csv", disabled=data.empty)
        c2.download_button("Export full backlog (CSV)", manager.export_csv(),
                           file_name=f"fraud_backlog_{stamp}.csv")

    with tabs[1]:
        entries = manager.all_entries()
        if entries.empty:
            st.info("The backlog is empty. Flag records on the **Fraud Detection** page and save them here.")
        else:
            labels = {int(r.backlog_id): f"#{r.backlog_id} - bidder {r.bidder_id}, auction {r.auction_id} "
                                         f"- p={r.fraud_probability:.3f} - {r.review_status}"
                      for r in entries.itertuples()}
            chosen = st.selectbox("Select case", list(labels), format_func=labels.get)
            _case_detail(manager, int(chosen))

    with tabs[2]:
        _statistics(manager)
