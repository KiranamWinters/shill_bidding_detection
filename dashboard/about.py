"""About page: problem background, data source, methodology and limitations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from dashboard.common import get_results, page_header


def render() -> None:
    page_header("About the Project", "Machine Learning-Based Detection and Analysis of Shill Bidding Activities in Online Auctions")
    r = get_results()

    st.markdown("""
### What is shill bidding?
Shill bidding is a fraudulent practice in which a seller - or an accomplice acting for the seller - places
artificial bids on the seller's own auction. The fake competition pushes the price up and persuades genuine
buyers to bid more than they otherwise would:

1. A seller lists an item.
2. A genuine buyer bids.
3. A shill bidder, colluding with the seller, outbids them with small increments.
4. The genuine buyer raises their bid and ends up paying an inflated price.

### Why it is fraud
The shill bidder has no intention of buying. Their bids misrepresent demand, deceive other participants and
transfer money from buyers to the seller through manipulation. Auction platforms prohibit the practice and it
can breach consumer-protection and fraud law in many jurisdictions.

### Behavioural signals used by the model
""")
    st.dataframe(pd.DataFrame([{"feature": f, "description": settings.FEATURE_DESCRIPTIONS[f]}
                               for f in settings.BEHAVIOURAL_FEATURES]), hide_index=True, width="stretch")

    st.markdown(f"""
### Dataset
* **Shill Bidding Dataset**, UCI Machine Learning Repository, ID {settings.UCI_DATASET_ID} -
  [{settings.UCI_DATASET_URL}]({settings.UCI_DATASET_URL}), DOI [{settings.UCI_DOI}]({settings.UCI_DOI}).
* Built by the dataset creators from scraped eBay auctions of a popular product; 6,321 labelled bidder-auction
  records. Licence: **{settings.DATASET_LICENSE}**.
* Citation: {settings.DATASET_CITATION}
* Retrieved directly from UCI (the `ucimlrepo` client reports this dataset as not importable, so the
  official UCI ZIP download is used). No third-party copies are used.

### Methodology
1. **Validation and cleaning** - schema checks, dtype enforcement, duplicate and missing-value checks.
2. **EDA** - distributions, class balance, correlation, statistical tests, group structure, leakage review.
3. **Leakage control** - identifiers are never predictors; preprocessing lives in scikit-learn pipelines fitted
   on training folds only; the hold-out test set and all CV folds are **grouped by Auction ID**.
4. **Models** - majority and single-rule baselines, Logistic Regression, Decision Tree, SVM (RBF), Random Forest,
   Extra Trees, Gradient Boosting and HistGradientBoosting; class imbalance handled with class weights.
5. **Tuning and selection** - grid search with auction-grouped 5-fold CV optimising PR-AUC; the simplest model
   within a small tolerance of the best is chosen; the test set is used only after selection.
6. **Threshold** - chosen on out-of-fold training predictions (F2 with a precision floor), not on the test set.
7. **Explainability** - permutation importance, model-native importance, SHAP / occlusion for single predictions.
8. **Backlog** - flagged cases are stored in SQLite with review status, notes and an audit trail.
""")
    if r:
        st.markdown(f"Selected model: **{r['selected_model_name']}** (threshold {r['threshold']:.2f}).")

    st.markdown("""
### Limitations
* **Retrospective features.** Several indicators (last bidding, auction bids, winning ratio) are only known after
  an auction ends, so this is a post-auction investigation aid, not real-time prevention.
* **Label provenance.** The labels come with the dataset; UCI does not document the labelling procedure on the
  dataset page. If labels were derived from the same behavioural indicators, near-perfect scores partly reflect
  the model reproducing that procedure rather than discovering independent fraud evidence.
* **One dominant feature.** `Successive_Outbidding` alone separates most cases; without it performance drops
  substantially (see the feature-set ablation), so the model depends heavily on one behaviour.
* **Masked bidder IDs** are not unique, limiting bidder-level analysis and bidder-grouped validation.
* **Single product / platform / period.** The data come from one product category on eBay; behaviour on other
  platforms or products may differ (dataset shift).
* **Predictions are not proof.** A flag means behaviour resembles labelled shill cases and needs human review.
""")
