"""Generate ``reports/model_comparison_report.md`` from measured results."""

from __future__ import annotations

import pandas as pd

from config import settings


def _table(df: pd.DataFrame, digits: int = 4) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_model_report(r: dict) -> None:
    """Write the Markdown model-comparison report (all numbers from ``r``)."""
    comp = pd.DataFrame(r["comparison"])
    cv_cols = comp[["model", "cv_pr_auc_mean", "cv_pr_auc_std", "oof_recall", "oof_precision", "best_params"]]
    test_cols = comp[["model", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc",
                      "pr_auc", "brier", "tp", "fp", "fn", "tn"]]
    at05 = comp[["model", "precision@0.5", "recall@0.5", "f1@0.5"]]
    val = pd.DataFrame(r["validation_study"])
    val_p = val.pivot(index="model", columns="cv_strategy", values="pr_auc_mean").reset_index()
    val_f1 = val.pivot(index="model", columns="cv_strategy", values="f1@0.5").reset_index()
    fs = pd.DataFrame(r["feature_set_study"])[["feature_set", "n_features", "model", "pr_auc_mean", "pr_auc_std", "f1@0.5"]]
    perm = pd.DataFrame(r["global_importance"]["permutation"])
    split = r["split"]
    ov = split["overlap"]
    bg = r["bidder_generalisation"]
    bg_rows = []
    for k, v in bg.items():
        bg_rows.append({"subset": k, "rows": v["rows"], "shill": v["shill"],
                        "precision": v.get("precision", float("nan")),
                        "recall": v.get("recall", float("nan")), "f1": v.get("f1", float("nan"))})
    sel = comp[comp["selected"]].iloc[0]
    rule = comp[comp["key"] == "rule_so"].iloc[0]

    text = f"""# Model Comparison Report

*Generated automatically by `src/training/train.py` on {r['generated_at']}. All values are
measured; nothing is hand-edited.* Model version: `{r['model_version']}`.

## 1. Experimental design

* **Hold-out split:** one fold of a shuffled `StratifiedGroupKFold` grouped by **Auction_ID**
  (seed {settings.RANDOM_SEED}). Train: {split['train_rows']} records / {split['train_auctions']} auctions
  (shill share {split['train_shill_share']:.2%}); test: {split['test_rows']} records / {split['test_auctions']}
  auctions (shill share {split['test_shill_share']:.2%}).
* **Identifier overlap train/test:** Record_ID {ov['Record_ID']['shared_ids']}, Auction_ID
  {ov['Auction_ID']['shared_ids']}, masked Bidder_ID {ov['Bidder_ID']['shared_ids']} shared IDs
  ({ov['Bidder_ID']['test_rows_with_shared_id']} of {split['test_rows']} test rows carry a masked bidder ID
  also present in training - see section 6).
* **Model selection / tuning:** 5-fold `StratifiedGroupKFold` by Auction_ID on the training split only,
  optimising average precision (PR-AUC).
* **Threshold:** {r['threshold_rule']}
* **Selection rule:** {r['selection_rule']}
* The test set was used only after the model had been selected.

## 2. Feature set

Features used ({len(r['features_used'])}): `{', '.join(r['features_used'])}`.

{r['feature_decision']}

{_table(fs)}

The ablation without `Successive_Outbidding` shows how much the models rely on this single indicator.

## 3. Validation strategy study (training data, untuned models)

CV PR-AUC by splitting strategy:

{_table(val_p)}

F1 at threshold 0.5:

{_table(val_f1)}

Row-based CV mixes records of the same auction across folds; auction-grouped CV measures
performance on unseen auctions, which is the intended use case (scoring new, completed
auctions). Bidder-grouped CV relies on masked bidder IDs that are not unique keys, so it is
reported for completeness only.

## 4. Cross-validation results (training split, tuned)

{_table(cv_cols)}

## 5. Held-out test results (unseen auctions)

At each model's optimised threshold:

{_table(test_cols)}

At the default threshold 0.5:

{_table(at05)}

### Selected model: **{r['selected_model_name']}**

* Test PR-AUC {sel['pr_auc']:.4f}, ROC-AUC {sel['roc_auc']:.4f}, precision {sel['precision']:.4f},
  recall {sel['recall']:.4f}, F1 {sel['f1']:.4f} at threshold {sel['threshold']:.2f}.
* Confusion matrix: TP {int(sel['tp'])}, FP {int(sel['fp'])}, FN {int(sel['fn'])}, TN {int(sel['tn'])}.
* For comparison, the one-feature rule `Successive_Outbidding > 0` reaches precision
  {rule['precision']:.4f} / recall {rule['recall']:.4f} / F1 {rule['f1']:.4f} on the same test set.

Classification report (selected model, test set):

```
{r['classification_reports'][r['selected_model_name']]}
```

## 6. Generalisation to unseen bidder IDs (selected model, test set)

{_table(pd.DataFrame(bg_rows))}

## 7. Global feature importance (selected model)

Permutation importance on the test set (decrease in average precision):

{_table(perm)}

Explanation method for individual predictions: **{r['global_importance']['explanation_method']}**.

## 8. Decision-tree rules (tuned Decision Tree, first levels)

```
{r['decision_tree_rules']}
```

## 9. Figures

""" + "\n".join(f"![{p}](../{p.replace(chr(92), '/')})" for p in r["figures"]) + "\n"
    settings.MODEL_REPORT_MD.write_text(text, encoding="utf-8")
