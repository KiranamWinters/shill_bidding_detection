"""End-to-end experiment: split, feature study, tuning, selection, evaluation.

Order of operations (the test set is used exactly once, after selection):

1. Grouped hold-out split by Auction ID (test auctions never seen in training).
2. Feature analysis and feature-set study on the training split.
3. Validation-strategy study (row vs auction vs bidder grouped CV).
4. Hyperparameter tuning of every candidate with auction-grouped CV.
5. Threshold optimisation per model on out-of-fold training predictions.
6. Model selection from CV results only.
7. Test-set evaluation of all models (reporting) and explanation of the
   selected model; artefacts and reports are saved.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

from config import settings
from src.evaluation import evaluate as ev
from src.explainability.explain import (
    ModelExplainer,
    global_permutation_importance,
    make_global_figures,
    native_importance,
)
from src.features.feature_selection import (
    find_correlated_pairs,
    mutual_information_ranking,
    propose_reduced_feature_set,
)
from src.training.compare_models import (
    feature_set_study,
    get_candidates,
    select_best_model,
    validation_strategy_study,
)
from src.training.splitting import group_overlap, grouped_holdout_split
from src.training.tune import tune_candidate
from src.utils.logging_config import get_logger

logger = get_logger("training.train")


def _reference_stats(train_df: pd.DataFrame, features: list[str]) -> dict:
    y = train_df[settings.TARGET]
    X = train_df[features]
    return {
        "median": X.median().to_dict(),
        "min": X.min().to_dict(),
        "max": X.max().to_dict(),
        "median_normal": X[y == 0].median().to_dict(),
        "median_shill": X[y == 1].median().to_dict(),
    }


def _choose_feature_set(study: pd.DataFrame, full: list[str], reduced: list[str]) -> tuple[list[str], str]:
    """Use the reduced set only if it is not worse than the full set (parsimony)."""
    if reduced == full:
        return full, "No highly correlated pairs - all behavioural features kept."
    means = study.groupby("feature_set")["pr_auc_mean"].mean()
    diff = means["reduced"] - means["all_features"]
    if diff >= -settings.SELECTION_TOLERANCE:
        return reduced, (f"Reduced set kept: mean CV PR-AUC change {diff:+.4f} is within the "
                         f"{settings.SELECTION_TOLERANCE} tolerance, so the simpler set is preferred.")
    return full, (f"All features kept: removing correlated features lowered mean CV PR-AUC by "
                  f"{-diff:.4f} (> tolerance {settings.SELECTION_TOLERANCE}).")


def run_training(df: pd.DataFrame) -> dict:
    """Run the complete model experiment on a cleaned dataset."""
    settings.ensure_directories()
    cands = get_candidates()
    feats_all = list(settings.BEHAVIOURAL_FEATURES)

    # 1. Grouped hold-out split ------------------------------------------------
    train_idx, test_idx = grouped_holdout_split(df)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    overlap = group_overlap(train_df, test_df)
    assert overlap[settings.AUCTION_ID]["shared_ids"] == 0, "Auction leakage between train and test"
    assert overlap[settings.RECORD_ID]["shared_ids"] == 0, "Record leakage between train and test"
    pd.DataFrame({
        settings.RECORD_ID: pd.concat([train_df[settings.RECORD_ID], test_df[settings.RECORD_ID]]),
        "partition": ["train"] * len(train_df) + ["test"] * len(test_df),
    }).to_csv(settings.SPLIT_FILE, index=False)
    split_info = {
        "train_rows": len(train_df), "test_rows": len(test_df),
        "train_shill_share": float(train_df[settings.TARGET].mean()),
        "test_shill_share": float(test_df[settings.TARGET].mean()),
        "train_auctions": int(train_df[settings.AUCTION_ID].nunique()),
        "test_auctions": int(test_df[settings.AUCTION_ID].nunique()),
        "overlap": overlap,
    }
    logger.info("Split: %s", {k: v for k, v in split_info.items() if k != "overlap"})

    # 2. Feature analysis (training data only) -----------------------------------
    X_tr = train_df[feats_all]
    y_tr = train_df[settings.TARGET]
    mi = mutual_information_ranking(X_tr, y_tr)
    corr_pairs = find_correlated_pairs(X_tr)
    reduced, drop_log = propose_reduced_feature_set(X_tr, y_tr)
    no_so = [f for f in feats_all if f != "Successive_Outbidding"]
    feature_sets = {"all_features": feats_all, "reduced": reduced,
                    "ablation_without_successive_outbidding": no_so}
    fs_study = feature_set_study(train_df, feature_sets, keys=["logreg", "rf", "hgb"])
    selectable = fs_study[fs_study["feature_set"].isin(["all_features", "reduced"])]
    features, feature_decision = _choose_feature_set(selectable, feats_all, reduced)
    logger.info("Feature decision: %s -> %s", feature_decision, features)

    # 3. Validation strategy study --------------------------------------------------
    val_study = validation_strategy_study(train_df, keys=["logreg", "rf", "hgb"])
    val_study.to_csv(settings.VALIDATION_STUDY_CSV, index=False)

    # 4-5. Tuning + per-model threshold on OOF predictions ---------------------------
    tuned, sweeps = {}, {}
    for key, spec in cands.items():
        res = tune_candidate(spec, train_df, features)
        thr, sweep = ev.optimise_threshold(y_tr, res["oof_proba"])
        if spec.is_baseline:
            thr = 0.5  # baselines keep their defined decision rule
        res["threshold"] = thr
        res["oof_metrics"] = ev.compute_metrics(y_tr, res["oof_proba"], thr)
        tuned[key] = res
        sweeps[key] = sweep

    cv_table = pd.DataFrame([{
        "key": k, "model": cands[k].display_name, "is_baseline": cands[k].is_baseline,
        "complexity": cands[k].complexity, "cv_pr_auc_mean": r["cv_pr_auc_mean"],
        "cv_pr_auc_std": r["cv_pr_auc_std"], "threshold": r["threshold"],
        "oof_precision": r["oof_metrics"]["precision"], "oof_recall": r["oof_metrics"]["recall"],
        "oof_f1": r["oof_metrics"]["f1"], "best_params": json.dumps(r["best_params"], default=str),
    } for k, r in tuned.items()])

    # 6. Selection (CV only, before looking at the test set) ------------------------
    best_key = select_best_model(cv_table)
    best_cv = cv_table.set_index("key").loc[best_key]
    top_cv = cv_table[~cv_table.is_baseline].sort_values("cv_pr_auc_mean", ascending=False)
    logger.info("Selected model: %s (CV PR-AUC %.4f)", best_key, best_cv["cv_pr_auc_mean"])

    # 7. Test evaluation of every model ------------------------------------------------
    y_te = test_df[settings.TARGET]
    test_rows, curves, cms, reports = [], {}, {}, {}
    for key, res in tuned.items():
        proba = res["pipeline"].predict_proba(test_df)[:, 1]
        m_opt = ev.compute_metrics(y_te, proba, res["threshold"])
        m_05 = ev.compute_metrics(y_te, proba, 0.5)
        name = cands[key].display_name
        curves[name] = ev.curve_data(y_te, proba)
        cms[name] = m_opt
        reports[name] = ev.text_classification_report(y_te, proba, res["threshold"])
        test_rows.append({
            "key": key, "model": name, **{k: v for k, v in m_opt.items()},
            "precision@0.5": m_05["precision"], "recall@0.5": m_05["recall"], "f1@0.5": m_05["f1"],
        })
    test_table = pd.DataFrame(test_rows)
    comparison = cv_table.drop(columns=["threshold"]).merge(test_table.drop(columns=["model"]), on="key")
    comparison["selected"] = comparison["key"] == best_key
    comparison = comparison.sort_values(["is_baseline", "cv_pr_auc_mean"], ascending=[True, False])
    comparison.to_csv(settings.MODEL_COMPARISON_CSV, index=False)

    # Generalisation to unseen (masked) bidder IDs within test auctions
    best = tuned[best_key]
    best_proba = best["pipeline"].predict_proba(test_df)[:, 1]
    seen_bidder = test_df[settings.BIDDER_ID].isin(set(train_df[settings.BIDDER_ID]))
    bidder_generalisation = {}
    for label, mask in (("bidder_id_seen_in_training", seen_bidder), ("bidder_id_unseen", ~seen_bidder)):
        if mask.sum() and y_te[mask].nunique() > 1:
            m = ev.compute_metrics(y_te[mask], best_proba[mask], best["threshold"])
            bidder_generalisation[label] = {"rows": int(mask.sum()), "shill": int(y_te[mask].sum()), **m}
        else:
            bidder_generalisation[label] = {"rows": int(mask.sum()), "shill": int(y_te[mask].sum())}

    # Figures ------------------------------------------------------------------------
    figures = [
        ev.plot_confusion_matrices(cms),
        *ev.plot_roc_pr(curves),
        ev.plot_model_comparison(comparison),
        ev.plot_threshold_sweep(sweeps[best_key], best["threshold"], cands[best_key].display_name),
        ev.plot_validation_study(val_study),
    ]

    # Model bundle ---------------------------------------------------------------------
    now = datetime.now()
    version = f"{best_key}-{now:%Y%m%d-%H%M%S}"
    background = train_df[features].sample(min(200, len(train_df)), random_state=settings.RANDOM_SEED)
    bundle = {
        "pipeline": best["pipeline"],
        "model_key": best_key,
        "model_name": cands[best_key].display_name,
        "model_version": version,
        "features": features,
        "threshold": best["threshold"],
        "best_params": best["best_params"],
        "trained_at": now.isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_seed": settings.RANDOM_SEED,
        "background": background,
        "reference_stats": _reference_stats(train_df, features),
        "cv_pr_auc_mean": float(best_cv["cv_pr_auc_mean"]),
        "test_metrics": cms[cands[best_key].display_name],
        "training_rows": len(train_df),
    }

    # Explanations (selected model) --------------------------------------------------
    explainer = ModelExplainer(bundle)
    perm = global_permutation_importance(best["pipeline"], test_df, y_te, features)
    native = native_importance(best["pipeline"], features)
    exp_figs, mean_abs_shap = make_global_figures(explainer, test_df[features], perm, native,
                                                  cands[best_key].display_name)
    figures += exp_figs
    bundle["global_importance"] = {
        "permutation": perm.to_dict(orient="records"),
        "native": native.round(5).to_dict() if native is not None else {},
        "mean_abs_shap": mean_abs_shap,
        "explanation_method": explainer.method,
    }

    # Interpretable rules from the tuned decision tree (for the report/dashboard)
    from sklearn.tree import export_text
    tree_pipe = tuned["tree"]["pipeline"]
    tree_rules = export_text(tree_pipe.named_steps["model"], feature_names=features, max_depth=4)

    joblib.dump(bundle, settings.MODEL_BUNDLE_FILE)
    joblib.dump({k: {"pipeline": r["pipeline"], "threshold": r["threshold"],
                     "display_name": cands[k].display_name} for k, r in tuned.items()},
                settings.ALL_MODELS_FILE)
    joblib.dump({"test_curves": curves, "threshold_sweep": sweeps[best_key],
                 "all_sweeps": sweeps}, settings.CURVES_FILE)

    results = {
        "generated_at": now.isoformat(timespec="seconds"),
        "model_version": version,
        "selected_model": best_key,
        "selected_model_name": cands[best_key].display_name,
        "selection_rule": (
            f"Highest mean auction-grouped CV PR-AUC; models within {settings.SELECTION_TOLERANCE} "
            "of the best are treated as tied and the simplest tied model is chosen."
        ),
        "top_cv_models": top_cv[["model", "cv_pr_auc_mean", "cv_pr_auc_std", "complexity"]].to_dict(orient="records"),
        "threshold": best["threshold"],
        "threshold_rule": (
            f"Maximise F{settings.THRESHOLD_BETA:g} on out-of-fold training predictions subject to "
            f"precision >= {settings.MIN_PRECISION_FOR_THRESHOLD}."
        ),
        "split": split_info,
        "features_used": features,
        "feature_decision": feature_decision,
        "feature_drop_log": drop_log,
        "mutual_information_train": mi.round(4).to_dict(),
        "correlated_pairs_train": corr_pairs.to_dict(orient="records"),
        "feature_set_study": fs_study.to_dict(orient="records"),
        "validation_study": val_study.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "classification_reports": reports,
        "bidder_generalisation": bidder_generalisation,
        "global_importance": bundle["global_importance"],
        "decision_tree_rules": tree_rules,
        "figures": figures,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    settings.EVALUATION_RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    from src.evaluation.report import write_model_report
    write_model_report(results)
    logger.info("Training complete. Model %s saved to %s", version, settings.MODEL_BUNDLE_FILE)
    return results
