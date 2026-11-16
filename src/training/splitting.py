"""Leakage-aware data splitting.

Records of the same auction share auction-level features (number of bids,
starting price, duration) and the shill bidders of one auction are labelled
together. A row-level random split therefore lets the model "see" the test
auctions during training. The primary hold-out and cross-validation splits
are grouped by Auction ID so that test auctions are completely unseen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from config import settings


def grouped_holdout_split(
    df: pd.DataFrame,
    group_col: str = settings.PRIMARY_GROUP,
    n_splits: int = settings.TEST_SIZE_FOLDS,
    seed: int = settings.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) with no shared groups and preserved class ratio.

    One fold of a shuffled :class:`StratifiedGroupKFold` is used as the test
    set (~1/n_splits of the data).
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df[settings.TARGET], groups=df[group_col]))
    return train_idx, test_idx


def get_cv(strategy: str = "auction", seed: int = settings.RANDOM_SEED, n_splits: int = settings.CV_FOLDS):
    """Return a CV splitter and the name of the grouping column (or ``None``).

    Strategies:
        * ``"row"``      - stratified K-fold over records (ignores groups);
        * ``"auction"``  - stratified group K-fold by Auction ID (primary);
        * ``"bidder"``   - stratified group K-fold by (masked) Bidder ID.
    """
    if strategy == "row":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), None
    if strategy == "auction":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), settings.AUCTION_ID
    if strategy == "bidder":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), settings.BIDDER_ID
    raise ValueError(f"Unknown CV strategy: {strategy}")


def group_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Count identifiers that appear in both partitions."""
    out = {}
    for col in settings.IDENTIFIER_COLUMNS:
        shared = set(train[col]) & set(test[col])
        out[col] = {
            "shared_ids": len(shared),
            "test_ids": int(test[col].nunique()),
            "test_rows_with_shared_id": int(test[col].isin(shared).sum()),
        }
    return out
