"""Model performance page: comparison, curves, thresholds, importance."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import settings
from dashboard.common import get_curves, get_results, page_header, require_artifacts

METRIC_COLS = ["precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy"]


def _confusion_fig(row: pd.Series) -> go.Figure:
    z = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    fig = px.imshow(z, text_auto=True, color_continuous_scale="Blues",
                    x=["Predicted normal", "Predicted shill"], y=["Actual normal", "Actual shill"])
    fig.update_layout(coloraxis_showscale=False, height=330, margin=dict(t=30, b=10),
                      title=f"{row['model']} (threshold {row['threshold']:.2f})")
    return fig


def render() -> None:
    page_header("Model Performance",
                "All numbers are measured by the training pipeline; the test set contains only auctions "
                "never seen during training or model selection.")
    if not require_artifacts():
        return
    r = get_results()
    curves = get_curves() or {}
    comp = pd.DataFrame(r["comparison"])
    sel = comp[comp["selected"]].iloc[0]

    st.success(f"**Selected model: {r['selected_model_name']}** - decision threshold {r['threshold']:.2f}")
    c = st.columns(6)
    for col, m in zip(c, ["recall", "precision", "f1", "pr_auc", "roc_auc", "accuracy"]):
        col.metric(m.upper().replace("_", "-") if "auc" in m else m.title(), f"{sel[m]:.4f}")
    st.caption(f"Test confusion matrix: TP {int(sel.tp)}, FP {int(sel.fp)}, FN {int(sel.fn)}, TN {int(sel.tn)}.")

    with st.expander("Why this model? (selection methodology)", expanded=True):
        top = pd.DataFrame(r["top_cv_models"]).round(4)
        st.markdown(
            f"""
* **Split:** test set = one fold of a `StratifiedGroupKFold` grouped by Auction ID
  ({r['split']['test_rows']} records, {r['split']['test_auctions']} auctions). Auction-ID overlap with training: 0.
* **Tuning and selection:** 5-fold auction-grouped cross-validation on the training split, optimising
  PR-AUC (average precision) - appropriate for a {r['split']['train_shill_share']:.1%} minority class.
* **Rule:** {r['selection_rule']}
* **Threshold:** {r['threshold_rule']} Missing a shill bidder (false negative) lets fraud go unreviewed;
  a false positive costs investigator time - the precision floor keeps the review queue manageable.
* **Features:** {r['feature_decision']}
"""
        )
        st.dataframe(top, hide_index=True, width="stretch")

    tabs = st.tabs(["Comparison", "Confusion matrices", "ROC / PR curves", "Threshold",
                    "Validation strategy", "Feature importance", "Reports & rules"])

    with tabs[0]:
        show = comp[["model", "cv_pr_auc_mean", "cv_pr_auc_std", "threshold"] + METRIC_COLS
                    + ["fp", "fn", "f1@0.5", "best_params"]].round(4)
        st.dataframe(show, hide_index=True, width="stretch")
        metrics = st.multiselect("Metrics", METRIC_COLS, default=["precision", "recall", "f1", "pr_auc"])
        long = comp.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
        fig = px.bar(long, x="model", y="score", color="metric", barmode="group",
                     title="Held-out test metrics (shill class)")
        fig.update_yaxes(range=[max(0, long["score"].min() - 0.05), 1.0])
        st.plotly_chart(fig, width="stretch")
        cv_vs_test = comp[~comp["is_baseline"]][["model", "cv_pr_auc_mean", "pr_auc"]].rename(
            columns={"cv_pr_auc_mean": "CV PR-AUC (train)", "pr_auc": "Test PR-AUC"})
        st.plotly_chart(px.bar(cv_vs_test.melt(id_vars="model"), x="model", y="value", color="variable",
                               barmode="group", title="Cross-validation vs test PR-AUC").update_yaxes(range=[0.9, 1.0]),
                        width="stretch")

    with tabs[1]:
        cols = st.columns(3)
        for i, (_, row) in enumerate(comp.iterrows()):
            cols[i % 3].plotly_chart(_confusion_fig(row), width="stretch")

    with tabs[2]:
        test_curves = curves.get("test_curves", {})
        chosen = st.multiselect("Models", list(test_curves), default=list(test_curves))
        c1, c2 = st.columns(2)
        roc = go.Figure()
        pr = go.Figure()
        for name in chosen:
            cv = test_curves[name]
            roc.add_trace(go.Scatter(x=cv["fpr"], y=cv["tpr"], name=f"{name} ({cv['roc_auc']:.3f})", mode="lines"))
            pr.add_trace(go.Scatter(x=cv["recall"], y=cv["precision"], name=f"{name} ({cv['pr_auc']:.3f})", mode="lines"))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False))
        roc.update_layout(title="ROC curves (test)", xaxis_title="False positive rate", yaxis_title="Recall", height=480)
        pr.update_layout(title="Precision-recall curves (test)", xaxis_title="Recall", yaxis_title="Precision", height=480)
        c1.plotly_chart(roc, width="stretch")
        c2.plotly_chart(pr, width="stretch")

    with tabs[3]:
        sweep = curves.get("threshold_sweep")
        if sweep is not None:
            long = sweep.melt(id_vars="threshold", value_vars=["precision", "recall", "f1", "fbeta"])
            fig = px.line(long, x="threshold", y="value", color="variable",
                          title=f"Threshold sweep on out-of-fold training predictions - {r['selected_model_name']}")
            fig.add_vline(x=r["threshold"], line_dash="dash", annotation_text=f"chosen {r['threshold']:.2f}")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"`fbeta` is F{settings.THRESHOLD_BETA:g}. The test set was not used to choose the threshold.")
            st.dataframe(sweep.round(4), hide_index=True, height=260)

    with tabs[4]:
        val = pd.DataFrame(r["validation_study"])
        st.plotly_chart(px.bar(val, x="model", y="pr_auc_mean", color="cv_strategy", barmode="group",
                               error_y="pr_auc_std", title="CV PR-AUC by splitting strategy (untuned models)")
                        .update_yaxes(range=[max(0, val["pr_auc_mean"].min() - 0.05), 1.0]), width="stretch")
        st.dataframe(val.round(4), hide_index=True, width="stretch")
        st.markdown(
            "* **row** - stratified K-fold over records (records of one auction can be in train and validation).\n"
            "* **auction** - grouped by Auction ID: estimates performance on *new auctions* (the intended use).\n"
            "* **bidder** - grouped by masked Bidder ID; masked IDs are not unique, so this is approximate."
        )
        fs = pd.DataFrame(r["feature_set_study"])
        st.markdown("**Feature-set study (auction-grouped CV)**")
        st.dataframe(fs.round(4), hide_index=True, width="stretch")
        bg = r.get("bidder_generalisation", {})
        if bg:
            st.markdown("**Selected model on test records with seen vs unseen masked bidder IDs**")
            st.dataframe(pd.DataFrame(bg).T[["rows", "shill", "precision", "recall", "f1"]]
                         .apply(pd.to_numeric, errors="coerce").round(4), width="stretch")

    with tabs[5]:
        gi = r["global_importance"]
        perm = pd.DataFrame(gi["permutation"]).sort_values("importance_mean")
        st.plotly_chart(px.bar(perm, x="importance_mean", y="feature", error_x="importance_std", orientation="h",
                               title="Permutation importance on test auctions (drop in average precision)"),
                        width="stretch")
        c1, c2 = st.columns(2)
        if gi.get("native"):
            nat = pd.Series(gi["native"]).sort_values().reset_index()
            nat.columns = ["feature", "importance"]
            c1.plotly_chart(px.bar(nat, x="importance", y="feature", orientation="h",
                                   title="Model-native importance (impurity)"), width="stretch")
        if gi.get("mean_abs_shap"):
            sh = pd.Series(gi["mean_abs_shap"]).sort_values().reset_index()
            sh.columns = ["feature", "mean |SHAP|"]
            c2.plotly_chart(px.bar(sh, x="mean |SHAP|", y="feature", orientation="h",
                                   title="Mean |SHAP| value (test sample)"), width="stretch")
        shap_png = settings.FIGURES_DIR / "explain_shap_summary.png"
        if shap_png.exists():
            st.image(str(shap_png), caption="SHAP summary: each dot is a test record; colour = feature value.")
        st.caption("Importance measures how much the model relies on a behaviour, not causation or intent.")

    with tabs[6]:
        model = st.selectbox("Classification report for", list(r["classification_reports"]),
                             index=list(r["classification_reports"]).index(r["selected_model_name"]))
        st.code(r["classification_reports"][model])
        st.markdown("**Interpretable rules learned by the tuned Decision Tree (first levels)**")
        st.code(r["decision_tree_rules"])
        for path in (settings.MODEL_REPORT_MD, settings.MODEL_COMPARISON_CSV):
            if path.exists():
                st.download_button(f"Download {path.name}", path.read_bytes(), file_name=path.name)
