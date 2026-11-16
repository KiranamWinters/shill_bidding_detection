"""Hyperparameter tuning with auction-grouped cross-validation.

Tuning uses the training partition only. The optimisation metric is average
precision (area under the precision-recall curve) for the shill class, which
is informative under class imbalance and independent of the decision
threshold (the threshold is optimised separately afterwards).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_predict

from config import settings
from src.training.compare_models import CandidateSpec, build_pipeline
from src.training.splitting import get_cv
from src.utils.logging_config import get_logger

logger = get_logger("training.tune")


def tune_candidate(spec: CandidateSpec, train_df: pd.DataFrame, features: list[str]) -> dict:
    """Grid-search one candidate and compute its out-of-fold training predictions.

    Returns a dict with the fitted best pipeline, best parameters, CV PR-AUC
    (mean/std across folds) and out-of-fold probabilities for threshold tuning.
    """
    cv, group_col = get_cv("auction")
    y = train_df[settings.TARGET]
    groups = train_df[group_col]
    pipe = build_pipeline(spec, features)
    start = time.perf_counter()

    if spec.param_grid:
        search = GridSearchCV(
            pipe, spec.param_grid, scoring="average_precision", cv=cv,
            n_jobs=-1, refit=True, error_score="raise",
        )
        search.fit(train_df, y, groups=groups)
        best_pipe = search.best_estimator_
        best_params = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
        i = search.best_index_
        cv_mean = float(search.cv_results_["mean_test_score"][i])
        cv_std = float(search.cv_results_["std_test_score"][i])
        n_configs = len(search.cv_results_["params"])
    else:
        best_pipe = clone(pipe).fit(train_df, y)
        best_params = {}
        fold_scores = []
        from sklearn.metrics import average_precision_score
        for tr, va in cv.split(train_df, y, groups):
            fitted = clone(pipe).fit(train_df.iloc[tr], y.iloc[tr])
            fold_scores.append(average_precision_score(y.iloc[va], fitted.predict_proba(train_df.iloc[va])[:, 1]))
        cv_mean, cv_std, n_configs = float(np.mean(fold_scores)), float(np.std(fold_scores)), 1

    # Out-of-fold predictions of the selected configuration (same folds) - used
    # for threshold optimisation without touching the test set.
    oof = cross_val_predict(clone(best_pipe), train_df, y, cv=cv, groups=groups,
                            method="predict_proba", n_jobs=-1)[:, 1]
    elapsed = time.perf_counter() - start
    logger.info("Tuned %-12s configs=%-3d CV PR-AUC=%.4f +/- %.4f (%.1fs) params=%s",
                spec.key, n_configs, cv_mean, cv_std, elapsed, best_params)
    return {
        "pipeline": best_pipe,
        "best_params": best_params,
        "cv_pr_auc_mean": cv_mean,
        "cv_pr_auc_std": cv_std,
        "n_configs": n_configs,
        "oof_proba": oof,
        "tuning_seconds": elapsed,
    }
