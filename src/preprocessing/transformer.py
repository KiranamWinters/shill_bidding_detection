"""Scikit-learn preprocessing components.

Every learned transformation lives inside a :class:`~sklearn.pipeline.Pipeline`
so that it is fitted on training folds only, preventing leakage into
validation and test data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import settings

FORBIDDEN_PREDICTORS = set(settings.IDENTIFIER_COLUMNS) | {settings.TARGET}


class BehaviouralFeatureSelector(BaseEstimator, TransformerMixin):
    """Select an explicit list of behavioural features by name.

    Identifiers (Record/Auction/Bidder ID) and the target are rejected even if
    they are requested, so they can never leak into a model as predictors.
    Columns are always emitted in the configured order.
    """

    def __init__(self, features: Sequence[str] | None = None):
        self.features = features

    def _resolved(self) -> list[str]:
        feats = list(self.features) if self.features is not None else list(
            settings.BEHAVIOURAL_FEATURES
        )
        leaked = FORBIDDEN_PREDICTORS.intersection(feats)
        if leaked:
            raise ValueError(f"Identifiers/target cannot be used as predictors: {sorted(leaked)}")
        return feats

    def fit(self, X: pd.DataFrame, y=None):
        feats = self._resolved()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BehaviouralFeatureSelector expects a pandas DataFrame.")
        missing = [f for f in feats if f not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required features: {missing}")
        self.feature_names_out_ = np.array(feats, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        feats = list(self.feature_names_out_)
        missing = [f for f in feats if f not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required features: {missing}")
        return X[feats].apply(pd.to_numeric, errors="coerce").astype(float)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def build_preprocessor(features: Sequence[str] | None = None, scale: bool = False) -> Pipeline:
    """Build the preprocessing pipeline.

    Args:
        features: behavioural features to use (defaults to all nine).
        scale: standardise features (needed for distance/gradient based
            models such as SVM and Logistic Regression; unnecessary for trees).
    """
    steps = [
        ("select", BehaviouralFeatureSelector(features)),
        # Defensive: the official data has no missing values, but uploaded
        # records might. The median is learned from training data only.
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")
