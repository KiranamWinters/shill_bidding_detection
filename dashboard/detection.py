"""Fraud detection page: score a dataset record, a manual entry or an uploaded CSV."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import settings
from dashboard.common import disclaimer, get_bundle, get_clean_data, get_explainer, get_manager, page_header, require_artifacts
from src.prediction.predict import predict_records


def _gauge(prob: float, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%"},
        title={"text": "Fraud probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#E45756" if prob >= threshold else "#4C78A8"},
            "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
            "steps": [{"range": [0, threshold * 100], "color": "#E8EEF6"},
                      {"range": [threshold * 100, 100], "color": "#FBE3E3"}],
        },
    ))
    fig.update_layout(height=260, margin=dict(t=50, b=10, l=30, r=30))
    return fig


def _analyse_single(record: pd.DataFrame, source: str) -> None:
    """Score one record, log it and keep the result in session state."""
    bundle, explainer = get_bundle(), get_explainer()
    pred = predict_records(record, bundle)
    explanation = explainer.explain_record(record)
    get_manager().log_predictions(pred, source=source)
    st.session_state["single_result"] = {
        "pred": pred, "explanation": explanation, "record": record, "source": source, "saved": False,
    }


def _show_single_result(source: str) -> None:
    res = st.session_state.get("single_result")
    if not res or res["source"] != source:
        return
    bundle = get_bundle()
    pred = res["pred"].iloc[0]
    exp = res["explanation"]
    st.divider()
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(_gauge(pred["fraud_probability"], pred["threshold"]), width="stretch")
        if pred["predicted_class"] == 1:
            st.error(f"**{pred['prediction']}** - probability {pred['fraud_probability']:.3f} "
                     f">= threshold {pred['threshold']:.2f}. Recommended action: review this case.")
        else:
            st.success(f"**{pred['prediction']} bidding behaviour** - probability "
                       f"{pred['fraud_probability']:.3f} < threshold {pred['threshold']:.2f}.")
        st.caption(f"Model `{bundle['model_version']}` - explanation method: {exp['method']}")
        rec = res["record"].iloc[0]
        ids = {k: rec.get(k) for k in settings.IDENTIFIER_COLUMNS if k in res["record"].columns}
        ids = {k: (v.item() if hasattr(v, "item") else v) for k, v in ids.items() if pd.notna(v)}
        if ids:
            st.markdown("  \n".join(f"**{k}:** `{v}`" for k, v in ids.items()))
    with c2:
        contrib = exp["contributions"]
        fig = px.bar(contrib.iloc[::-1], x="contribution", y="feature", orientation="h",
                     color="direction", color_discrete_map={"towards shill": "#E45756", "towards normal": "#4C78A8"},
                     title="Feature contributions to this prediction")
        fig.update_layout(height=360, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(exp["unit"])
    st.markdown("**Explanation**")
    for line in exp["narrative"]:
        st.markdown(f"- {line}")
    with st.expander("Feature values compared with typical training records"):
        st.dataframe(exp["contributions"].round(4), hide_index=True, width="stretch")

    if pred["predicted_class"] == 1:
        if res["saved"]:
            st.success("Saved to the fraud backlog.")
        elif st.button("Save to fraud backlog", type="primary", icon=":material/flag:"):
            top = exp["contributions"].query("contribution > 0").head(3)["feature"].tolist()
            ids = get_manager().add_flagged(
                res["pred"], features=res["record"][bundle["features"]],
                explanations={res["pred"].index[0]: top}, source=res["source"],
            )
            res["saved"] = True
            if ids:
                st.success(f"Created backlog entry #{ids[0]} (status: Pending Review).")
            else:
                st.info("This record is already in the backlog for the current model version.")
    else:
        st.caption("Only records flagged as potential shill bidding can be added to the backlog.")


def _dataset_tab(df: pd.DataFrame) -> None:
    only_test = st.toggle("Only held-out test records (not used for training)", value=True)
    pool = df[df["partition"] == "test"] if only_test and "partition" in df else df
    options = pool[settings.RECORD_ID].tolist()
    c1, c2 = st.columns([2, 1])
    # Widget state must be changed before the widget is created in this run.
    if c2.button("Random record with successive outbidding"):
        cand = pool[pool["Successive_Outbidding"] > 0]
        st.session_state["record_pick"] = int(cand.sample(1)[settings.RECORD_ID].iloc[0])
    if st.session_state.get("record_pick") not in options:
        st.session_state.pop("record_pick", None)
    record_id = c1.selectbox("Record ID", options, key="record_pick")
    row = pool[pool[settings.RECORD_ID] == record_id]
    st.dataframe(row[settings.IDENTIFIER_COLUMNS + settings.BEHAVIOURAL_FEATURES], hide_index=True, width="stretch")
    if not only_test:
        st.warning("Training records were seen by the model; their predictions are optimistic.")
    if st.button("Analyse record", type="primary", key="analyse_dataset"):
        # The ground-truth label is NOT passed to the model.
        _analyse_single(row.drop(columns=[settings.TARGET, "partition"], errors="ignore"), "dataset_record")
        st.session_state["single_truth"] = int(row[settings.TARGET].iloc[0])
    res = st.session_state.get("single_result")
    if res and res["source"] == "dataset_record" and "single_truth" in st.session_state:
        st.caption(f"For reference only (not used by the model): the dataset label of this record is "
                   f"**{settings.CLASS_NAMES[st.session_state['single_truth']]}**.")


def _manual_tab(bundle: dict) -> None:
    ref = bundle["reference_stats"]
    st.caption("Values default to the training median of normal records. Eight indicators are on a 0-1 scale.")
    with st.form("manual_form"):
        ids = st.columns(3)
        rec_id = ids[0].text_input("Record ID (optional)", value="MANUAL-1")
        auction = ids[1].text_input("Auction ID (optional)", value="")
        bidder = ids[2].text_input("Bidder ID (optional)", value="")
        cols = st.columns(3)
        values = {}
        for i, f in enumerate(bundle["features"]):
            default = float(ref["median_normal"][f])
            if f == "Successive_Outbidding":
                values[f] = cols[i % 3].select_slider(f, options=[0.0, 0.5, 1.0], value=default, help=settings.FEATURE_DESCRIPTIONS[f])
            elif f == "Auction_Duration":
                values[f] = cols[i % 3].selectbox(f, [1, 3, 5, 7, 10], index=[1, 3, 5, 7, 10].index(int(default))
                                                  if int(default) in [1, 3, 5, 7, 10] else 0,
                                                  help=settings.FEATURE_DESCRIPTIONS[f])
            else:
                values[f] = cols[i % 3].slider(f, 0.0, 1.0, round(default, 3), 0.001, help=settings.FEATURE_DESCRIPTIONS[f])
        submitted = st.form_submit_button("Analyse", type="primary")
    if submitted:
        record = pd.DataFrame([{settings.RECORD_ID: rec_id or None, settings.AUCTION_ID: auction or None,
                                settings.BIDDER_ID: bidder or None, **values}])
        st.session_state.pop("single_truth", None)
        _analyse_single(record, "manual_entry")


def _batch_tab(df: pd.DataFrame, bundle: dict) -> None:
    st.markdown("Upload a CSV with the nine behavioural columns (identifier columns optional; a `Class` "
                "column, if present, is ignored by the model).")
    template = df[df["partition"] == "test"].head(20)[settings.IDENTIFIER_COLUMNS + settings.BEHAVIOURAL_FEATURES]
    st.download_button("Download a template (20 test records)", template.to_csv(index=False),
                       file_name="shill_bidding_template.csv")
    upload = st.file_uploader("CSV file", type=["csv"])
    run_test_scan = st.button("...or scan all held-out test auctions",
                              help="Scores every record of the auctions held out from training.")
    batch = None
    if upload is not None and st.button("Score uploaded file", type="primary"):
        try:
            batch = pd.read_csv(upload)
            source = f"upload:{upload.name}"
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
    elif run_test_scan:
        batch = df[df["partition"] == "test"].drop(columns=["partition"])
        source = "test_set_scan"
    if batch is not None:
        try:
            preds = predict_records(batch, bundle)
        except ValueError as exc:
            st.error(str(exc))
            return
        get_manager().log_predictions(preds, source=source)
        st.session_state["batch_result"] = {"pred": preds, "features": batch[bundle["features"]],
                                            "source": source, "saved": False,
                                            "labels": batch[settings.TARGET] if settings.TARGET in batch else None}

    res = st.session_state.get("batch_result")
    if not res:
        return
    preds = res["pred"]
    n_flag = int(preds["predicted_class"].sum())
    c = st.columns(3)
    c[0].metric("Records scored", len(preds))
    c[1].metric("Potential shill bidding", n_flag)
    c[2].metric("Flag rate", f"{n_flag / len(preds):.1%}")
    st.plotly_chart(px.histogram(preds, x="fraud_probability", nbins=40, title="Fraud probability distribution",
                                 color="prediction", color_discrete_map={settings.CLASS_NAMES[1]: "#E45756",
                                                                          settings.CLASS_NAMES[0]: "#4C78A8"}),
                    width="stretch")
    st.dataframe(preds.sort_values("fraud_probability", ascending=False), hide_index=True, width="stretch", height=320)
    if res["labels"] is not None:
        from sklearn.metrics import precision_score, recall_score
        y = res["labels"].astype(int).values
        st.caption(f"Labels were present in the file and are shown for reference only: precision "
                   f"{precision_score(y, preds['predicted_class']):.3f}, recall {recall_score(y, preds['predicted_class']):.3f}.")
    st.download_button("Download predictions CSV", preds.to_csv(index=False), file_name="predictions.csv")
    if n_flag and not res["saved"]:
        if st.button(f"Save {n_flag} flagged records to backlog", type="primary", icon=":material/flag:"):
            explainer = get_explainer()
            flagged_idx = preds.index[preds["predicted_class"] == 1]
            feats = res["features"].loc[flagged_idx]
            with st.spinner("Computing explanations..."):
                expl = {}
                for idx in flagged_idx:
                    table = explainer.explain_record(feats.loc[[idx]])["contributions"]
                    expl[idx] = table.query("contribution > 0").head(3)["feature"].tolist()
            ids = get_manager().add_flagged(preds, features=res["features"], explanations=expl, source=res["source"])
            res["saved"] = True
            st.success(f"Created {len(ids)} backlog entries ({n_flag - len(ids)} already present).")
    elif res["saved"]:
        st.success("Flagged records saved to the backlog.")


def render() -> None:
    page_header("Fraud Detection", "Score bidding records with the trained model and explain the result.")
    if not require_artifacts():
        return
    disclaimer()
    bundle, df = get_bundle(), get_clean_data()
    tab_choice = st.segmented_control("Input", ["Dataset record", "Manual entry", "Batch / CSV"],
                                      default="Dataset record", key="detect_mode")
    if tab_choice == "Dataset record":
        _dataset_tab(df)
        _show_single_result("dataset_record")
    elif tab_choice == "Manual entry":
        _manual_tab(bundle)
        _show_single_result("manual_entry")
    else:
        _batch_tab(df, bundle)
