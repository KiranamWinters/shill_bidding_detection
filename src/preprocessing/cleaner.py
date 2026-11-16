"""Deterministic cleaning of the raw UCI dataset.

Cleaning is limited to operations that do not learn from the data (type
enforcement, whitespace stripping, exact-duplicate removal), so it can safely be
applied to the whole dataset before splitting without leaking information.
"""

from __future__ import annotations

import pandas as pd

from config import settings
from src.data.validator import validate_dataset
from src.utils.logging_config import get_logger

logger = get_logger("preprocessing.cleaner")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataset.

    Steps:
        * keep the documented columns in a fixed order;
        * enforce dtypes (integer identifiers, string bidder ID, float features);
        * drop exact duplicate rows and rows with a missing target;
        * validate the result.
    """
    validate_dataset(df, strict=True)
    out = df[settings.EXPECTED_COLUMNS].copy()

    out[settings.RECORD_ID] = out[settings.RECORD_ID].astype(int)
    out[settings.AUCTION_ID] = out[settings.AUCTION_ID].astype(int)
    out[settings.BIDDER_ID] = out[settings.BIDDER_ID].astype(str).str.strip()
    for col in settings.BEHAVIOURAL_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    n_before = len(out)
    out = out.dropna(subset=[settings.TARGET])
    out[settings.TARGET] = out[settings.TARGET].astype(int)
    out = out.drop_duplicates()
    n_dropped = n_before - len(out)
    if n_dropped:
        logger.info("Removed %d duplicate/unlabelled rows", n_dropped)

    n_missing = int(out[settings.BEHAVIOURAL_FEATURES].isna().sum().sum())
    if n_missing:
        # No missing values exist in the official release; imputation is done
        # inside the model pipeline (fitted on training data only) if any appear.
        logger.warning("%d missing feature values remain after cleaning", n_missing)

    validate_dataset(out, strict=True)
    return out.reset_index(drop=True)


def split_identifiers_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Split a cleaned frame into identifiers, behavioural features and target."""
    ids = df[settings.IDENTIFIER_COLUMNS].copy()
    X = df[settings.BEHAVIOURAL_FEATURES].copy()
    y = df[settings.TARGET].astype(int).copy()
    return ids, X, y
