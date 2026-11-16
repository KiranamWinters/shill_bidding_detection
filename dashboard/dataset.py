"""Dataset explorer page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings
from dashboard.common import COLORS, get_clean_data, get_eda, page_header, require_artifacts

LABEL_MAP = {0: "Normal", 1: "Shill"}
COLOR_MAP = {"Normal": COLORS["normal"], "Shill": COLORS["shill"]}


def render() -> None:
    page_header("Dataset Explorer", f"Official source: {settings.UCI_DATASET_URL} (DOI {settings.UCI_DOI})")
    if not require_artifacts():
        return
    df = get_clean_data()
    eda = get_eda() or {}
    plot_df = df.assign(Label=df[settings.TARGET].map(LABEL_MAP))

    tabs = st.tabs(["Preview", "Statistics", "Feature guide", "Class distribution",
                    "Feature explorer", "Correlation", "Groups & leakage"])

    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        label = c1.selectbox("Class", ["All", "Normal", "Shill"])
        auction = c2.text_input("Auction ID contains")
        bidder = c3.text_input("Bidder ID contains")
        view = plot_df
        if label != "All":
            view = view[view["Label"] == label]
        if auction:
            view = view[view[settings.AUCTION_ID].astype(str).str.contains(auction, regex=False)]
        if bidder:
            view = view[view[settings.BIDDER_ID].str.contains(bidder, regex=False)]
        st.caption(f"{len(view):,} of {len(df):,} records")
        st.dataframe(view.drop(columns=["Label"]), width="stretch", height=420, hide_index=True)

    with tabs[1]:
        c = st.columns(4)
        c[0].metric("Rows", f"{df.shape[0]:,}")
        c[1].metric("Columns", settings.EXPECTED_COLUMNS.__len__())
        c[2].metric("Missing values", eda.get("total_missing", int(df.isna().sum().sum())))
        c[3].metric("Duplicate rows", eda.get("duplicate_rows", int(df.duplicated().sum())))
        st.markdown("**Descriptive statistics**")
        st.dataframe(df[settings.BEHAVIOURAL_FEATURES].describe().T.round(4), width="stretch")
        st.markdown("**Data types and unique values**")
        st.dataframe(pd.DataFrame({
            "dtype": df[settings.EXPECTED_COLUMNS].dtypes.astype(str),
            "unique": df[settings.EXPECTED_COLUMNS].nunique(),
            "missing": df[settings.EXPECTED_COLUMNS].isna().sum(),
        }), width="stretch")
        if eda.get("class_difference_tests"):
            st.markdown("**Normal vs shill: Mann-Whitney U test and rank-biserial effect size**")
            st.dataframe(pd.DataFrame(eda["class_difference_tests"]).round(4), width="stretch", hide_index=True)

    with tabs[2]:
        leak = {r["feature"]: r for r in eda.get("leakage_assessment", [])}
        rows = []
        for col in settings.EXPECTED_COLUMNS:
            role = ("Identifier" if col in settings.IDENTIFIER_COLUMNS
                    else "Target" if col == settings.TARGET else "Behavioural predictor")
            rows.append({
                "column": col, "role": role, "description": settings.FEATURE_DESCRIPTIONS[col],
                "measured at": settings.FEATURE_LEVEL.get(col, ""),
                "available": leak.get(col, {}).get("available_at", ""),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Descriptions follow UCI's variable information. In the data, shill records have higher values "
            "of almost every indicator (including Winning_Ratio and Starting_Price_Average), which suggests "
            "the features are encoded as suspicion scores - see the dataset analysis report."
        )

    with tabs[3]:
        counts = plot_df["Label"].value_counts().reset_index()
        counts.columns = ["Label", "Records"]
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(counts, names="Label", values="Records", hole=0.45, color="Label",
                               color_discrete_map=COLOR_MAP, title="Class share"), width="stretch")
        c2.plotly_chart(px.bar(counts, x="Label", y="Records", color="Label", text="Records",
                               color_discrete_map=COLOR_MAP, title="Records per class"), width="stretch")
        if eda:
            st.markdown(f"Imbalance ratio **{eda['imbalance_ratio']:.2f} : 1** (normal : shill). "
                        f"A classifier that always predicts *normal* would be "
                        f"{eda['class_share']['0'] * 100:.1f}% accurate but useless - hence the focus on "
                        "precision, recall and PR-AUC.")

    with tabs[4]:
        feat = st.selectbox("Feature", settings.BEHAVIOURAL_FEATURES, index=2)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(plot_df, x=feat, color="Label", barmode="overlay", histnorm="probability density",
                                     nbins=40, color_discrete_map=COLOR_MAP, opacity=0.65,
                                     title=f"{feat} distribution by class"), width="stretch")
        c2.plotly_chart(px.box(plot_df, x="Label", y=feat, color="Label", color_discrete_map=COLOR_MAP,
                               title=f"{feat} by class"), width="stretch")
        st.caption(settings.FEATURE_DESCRIPTIONS[feat])
        x2 = st.selectbox("Compare against", [f for f in settings.BEHAVIOURAL_FEATURES if f != feat], index=0)
        st.plotly_chart(px.scatter(plot_df.sample(frac=1, random_state=1), x=feat, y=x2, color="Label",
                                   opacity=0.5, color_discrete_map=COLOR_MAP, title=f"{feat} vs {x2}"),
                        width="stretch")

    with tabs[5]:
        method = st.radio("Method", ["spearman", "pearson"], horizontal=True)
        corr = df[settings.BEHAVIOURAL_FEATURES + [settings.TARGET]].corr(method=method).round(2)
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  aspect="auto", title=f"{method.title()} correlation"), width="stretch")
        pairs = pd.DataFrame(eda.get("correlated_pairs", []))
        if len(pairs):
            st.markdown(f"**Highly correlated pairs (|Spearman| >= {settings.CORRELATION_THRESHOLD})**")
            st.dataframe(pairs, hide_index=True)
            st.caption(f"Last_Bidding equals Early_Bidding exactly in "
                       f"{eda['last_equals_early_share'] * 100:.1f}% of records (single-bid bidders).")

    with tabs[6]:
        gs = eda.get("group_structure", {})
        if gs:
            c = st.columns(4)
            c[0].metric("Auctions", gs["n_auctions"])
            c[1].metric("Masked bidder IDs", gs["n_bidder_ids"])
            c[2].metric("Auctions with shill records", gs["auctions_with_shill"])
            c[3].metric("...of which also contain normal", gs["auctions_mixed_classes"])
            st.markdown(
                "- Auction-level features are constant within each auction: "
                + ", ".join(f"`{k}` {v:.0%}" for k, v in gs["auction_feature_constant_within_auction"].items())
                + " - a random row split would leak auction context into the test set, so evaluation is "
                "**grouped by Auction ID**.\n"
                "- Bidder-level indicators are constant within a masked bidder ID in only "
                + ", ".join(f"`{k}` {v:.0%}" for k, v in gs["bidder_feature_constant_within_bidder_id"].items())
                + " of cases, so masked IDs are not reliable unique bidder keys."
            )
        by_auction = df.groupby(settings.AUCTION_ID)[settings.TARGET].agg(["size", "mean"]).reset_index()
        st.plotly_chart(px.histogram(by_auction, x="mean", nbins=20, title="Share of shill records per auction",
                                     color_discrete_sequence=[COLORS["shill"]]), width="stretch")
        if eda.get("leakage_assessment"):
            st.markdown("**When is each feature available?**")
            st.dataframe(pd.DataFrame(eda["leakage_assessment"]), hide_index=True, width="stretch")
            st.warning("Several features need the auction to have ended, so the system is a **retrospective** "
                       "detector for post-auction investigation, not a real-time prevention tool.")
