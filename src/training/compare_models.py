"""Candidate models, baselines and comparative validation studies.

Model choice rationale (small, tabular, moderately imbalanced, mostly
[0, 1]-normalised behavioural features):

* **Logistic Regression** - interpretable linear baseline; coefficients show
  the direction of each behaviour's association with the label.
* **Decision Tree** - human-readable rules, captures thresholds/interactions.
* **Random Forest / Extra Trees** - variance-reduced tree ensembles, robust on
  small tabular data, provide feature importances.
* **SVM (RBF)** - non-linear margin classifier; requires feature scaling.
* **Gradient Boosting / HistGradientBoosting** - boosting ensembles that are
  usually the strongest learners for tabular data.

Class imbalance (~1 : 8.4) is handled with ``class_weight`` (a tunable
hyperparameter) instead of resampling, so no synthetic records are created and
no resampling step can leak across folds. Gradient Boosting has no
``class_weight`` and is left unweighted; HistGradientBoosting supports it.

Baselines: a majority/prior ``DummyClassifier`` and a domain rule that flags a
record when ``Successive_Outbidding > 0`` (the single most informative
feature in EDA).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import settings
from src.preprocessing.transformer import build_preprocessor
from src.training.splitting import get_cv
from src.utils.logging_config import get_logger

logger = get_logger("training.compare")
SEED = settings.RANDOM_SEED


class FeatureRuleClassifier(ClassifierMixin, BaseEstimator):
    """Domain-rule baseline: flag when ``feature > threshold``.

    ``predict_proba`` returns the feature value clipped to [0, 1] (all features
    except Auction_Duration are already on that scale) so that ranking metrics
    such as ROC-AUC can be computed for the rule.
    """

    def __init__(self, feature: str = "Successive_Outbidding", threshold: float = 0.0):
        self.feature = feature
        self.threshold = threshold

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.array(list(X.columns), dtype=object)
        return self

    def predict_proba(self, X):
        p = np.clip(np.asarray(X[self.feature], dtype=float), 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (np.asarray(X[self.feature], dtype=float) > self.threshold).astype(int)


@dataclass
class CandidateSpec:
    """Definition of one candidate model."""

    key: str
    display_name: str
    estimator: BaseEstimator
    scale: bool
    param_grid: dict = field(default_factory=dict)
    complexity: int = 1  # lower = simpler / more interpretable
    is_baseline: bool = False


def get_candidates() -> dict[str, CandidateSpec]:
    """Return all baselines and candidate models with their tuning grids."""
    cands = [
        CandidateSpec("dummy", "Baseline: majority class", DummyClassifier(strategy="prior"),
                      scale=False, complexity=0, is_baseline=True),
        CandidateSpec("rule_so", "Baseline: Successive_Outbidding > 0 rule", FeatureRuleClassifier(),
                      scale=False, complexity=0, is_baseline=True),
        CandidateSpec(
            "logreg", "Logistic Regression",
            LogisticRegression(max_iter=5000, random_state=SEED),
            scale=True, complexity=1,
            param_grid={"model__C": [0.01, 0.1, 1, 10, 100],
                        "model__class_weight": [None, "balanced"]},
        ),
        CandidateSpec(
            "tree", "Decision Tree",
            DecisionTreeClassifier(random_state=SEED),
            scale=False, complexity=2,
            param_grid={"model__max_depth": [3, 4, 5, 6, 8, None],
                        "model__min_samples_leaf": [1, 5, 10, 20],
                        "model__class_weight": [None, "balanced"]},
        ),
        CandidateSpec(
            "svm", "Support Vector Machine (RBF)",
            # Platt-scaled probabilities via CalibratedClassifierCV (the replacement for the
            # deprecated SVC(probability=True)); calibration folds are internal to each fit.
            CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED), method="sigmoid",
                                   ensemble=False, cv=5),
            scale=True, complexity=4,
            param_grid={"model__estimator__C": [0.1, 1, 10, 100],
                        "model__estimator__gamma": ["scale", 0.1, 1.0],
                        "model__estimator__class_weight": [None, "balanced"]},
        ),
        CandidateSpec(
            "rf", "Random Forest",
            RandomForestClassifier(n_estimators=300, random_state=SEED),
            scale=False, complexity=3,
            param_grid={"model__max_depth": [None, 8],
                        "model__min_samples_leaf": [1, 5],
                        "model__max_features": ["sqrt", 0.5],
                        "model__class_weight": [None, "balanced_subsample"]},
        ),
        CandidateSpec(
            "extra_trees", "Extra Trees",
            ExtraTreesClassifier(n_estimators=300, random_state=SEED),
            scale=False, complexity=3,
            param_grid={"model__max_depth": [None, 8],
                        "model__min_samples_leaf": [1, 5],
                        "model__max_features": ["sqrt", 0.5],
                        "model__class_weight": [None, "balanced_subsample"]},
        ),
        CandidateSpec(
            "gb", "Gradient Boosting",
            GradientBoostingClassifier(random_state=SEED),
            scale=False, complexity=4,
            param_grid={"model__n_estimators": [100, 300],
                        "model__learning_rate": [0.05, 0.1],
                        "model__max_depth": [2, 3],
                        "model__subsample": [1.0, 0.8]},
        ),
        CandidateSpec(
            "hgb", "HistGradientBoosting",
            HistGradientBoostingClassifier(random_state=SEED),
            scale=False, complexity=4,
            param_grid={"model__learning_rate": [0.05, 0.1],
                        "model__max_leaf_nodes": [15, 31],
                        "model__l2_regularization": [0.0, 1.0],
                        "model__class_weight": [None, "balanced"]},
        ),
    ]
    return {c.key: c for c in cands}


def build_pipeline(spec: CandidateSpec, features: list[str] | None = None) -> Pipeline:
    """Wrap a candidate estimator with its preprocessing in one pipeline."""
    return Pipeline([
        ("prep", build_preprocessor(features, scale=spec.scale)),
        ("model", clone(spec.estimator)),
    ])


def _cv_scores(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, groups, cv) -> dict:
    """Out-of-fold probabilities -> pooled metrics (fold-wise mean for PR-AUC)."""
    proba = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    fold_ap = []
    for _, val_idx in cv.split(X, y, groups):
        if y.iloc[val_idx].nunique() > 1:
            fold_ap.append(average_precision_score(y.iloc[val_idx], proba[val_idx]))
    return {
        "pr_auc_mean": float(np.mean(fold_ap)),
        "pr_auc_std": float(np.std(fold_ap)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "f1@0.5": float(f1_score(y, (proba >= 0.5).astype(int))),
    }


def validation_strategy_study(train_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Compare row-based vs auction-grouped vs bidder-grouped CV (training data only).

    Untuned default models are used, so the study isolates the effect of the
    splitting strategy on the estimated performance.
    """
    X, y = train_df, train_df[settings.TARGET]
    cands = get_candidates()
    rows = []
    for key in keys:
        for strategy in ("row", "auction", "bidder"):
            cv, group_col = get_cv(strategy)
            groups = train_df[group_col] if group_col else None
            scores = _cv_scores(build_pipeline(cands[key]), X, y, groups, cv)
            rows.append({"model": cands[key].display_name, "cv_strategy": strategy, **scores})
            logger.info("Validation study %-28s %-8s PR-AUC=%.4f", key, strategy, scores["pr_auc_mean"])
    return pd.DataFrame(rows)


def feature_set_study(train_df: pd.DataFrame, feature_sets: dict[str, list[str]],
                      keys: list[str]) -> pd.DataFrame:
    """Compare feature subsets with auction-grouped CV (training data only)."""
    cv, group_col = get_cv("auction")
    y = train_df[settings.TARGET]
    cands = get_candidates()
    rows = []
    for set_name, feats in feature_sets.items():
        for key in keys:
            scores = _cv_scores(build_pipeline(cands[key], feats), train_df, y, train_df[group_col], cv)
            rows.append({"feature_set": set_name, "n_features": len(feats),
                         "model": cands[key].display_name, **scores})
            logger.info("Feature study %-22s %-12s PR-AUC=%.4f", set_name, key, scores["pr_auc_mean"])
    return pd.DataFrame(rows)


def select_best_model(cv_table: pd.DataFrame, tolerance: float = settings.SELECTION_TOLERANCE) -> str:
    """Select the best non-baseline model from cross-validation results only.

    Rule: models whose mean CV PR-AUC is within ``tolerance`` of the best are
    treated as tied; among them the lowest-complexity model is chosen, then the
    one with lower fold-to-fold standard deviation, then higher PR-AUC.
    """
    cands = cv_table[~cv_table["is_baseline"]].copy()
    best = cands["cv_pr_auc_mean"].max()
    tied = cands[cands["cv_pr_auc_mean"] >= best - tolerance]
    tied = tied.sort_values(["complexity", "cv_pr_auc_std", "cv_pr_auc_mean"],
                            ascending=[True, True, False])
    return str(tied.iloc[0]["key"])
