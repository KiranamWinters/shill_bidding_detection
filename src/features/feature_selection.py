"""Feature analysis and selection utilities.

These functions are used on the *training* partition only. Selection decisions
are then evaluated with grouped cross-validation in
:mod:`src.training.compare_models`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from config import settings


def find_correlated_pairs(
    X: pd.DataFrame, threshold: float = settings.CORRELATION_THRESHOLD, method: str = "spearman"
) -> pd.DataFrame:
    """Return feature pairs whose absolute correlation exceeds ``threshold``."""
    corr = X.corr(method=method).abs()
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if corr.loc[a, b] >= threshold:
                rows.append({"feature_a": a, "feature_b": b, f"abs_{method}": round(corr.loc[a, b], 4)})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", f"abs_{method}"])


def mutual_information_ranking(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Mutual information between each feature and the target (descending)."""
    discrete = np.array([X[c].nunique() <= 10 for c in X.columns])
    # Discrete features are passed as integer codes so they are treated as labels.
    values = X.copy()
    for col, is_discrete in zip(X.columns, discrete):
        if is_discrete:
            values[col] = pd.factorize(values[col], sort=True)[0]
    mi = mutual_info_classif(
        values.values, y.values, discrete_features=discrete, random_state=settings.RANDOM_SEED
    )
    return pd.Series(mi, index=X.columns, name="mutual_information").sort_values(ascending=False)


def propose_reduced_feature_set(
    X: pd.DataFrame, y: pd.Series, threshold: float = settings.CORRELATION_THRESHOLD
) -> tuple[list[str], list[dict]]:
    """Drop the less informative member of every highly correlated pair.

    Returns the reduced feature list and a log of what was dropped and why.
    """
    pairs = find_correlated_pairs(X, threshold)
    mi = mutual_information_ranking(X, y)
    dropped, decisions = set(), []
    for _, row in pairs.iterrows():
        a, b = row["feature_a"], row["feature_b"]
        if a in dropped or b in dropped:
            continue
        loser = a if mi[a] < mi[b] else b
        keeper = b if loser == a else a
        dropped.add(loser)
        decisions.append(
            {
                "dropped": loser,
                "kept": keeper,
                "correlation": float(row.iloc[2]),
                "mi_dropped": round(float(mi[loser]), 4),
                "mi_kept": round(float(mi[keeper]), 4),
            }
        )
    reduced = [f for f in X.columns if f not in dropped]
    return reduced, decisions


def leakage_assessment() -> pd.DataFrame:
    """Qualitative assessment of temporal availability / leakage per feature.

    The features are computed retrospectively from complete auctions, so the
    table records when each feature becomes available.
    """
    notes = {
        "Bidder_Tendency": ("history of the bidder across sellers", "Available from bidder history; aggregated over the bidder's auctions."),
        "Bidding_Ratio": ("auction in progress / ended", "Ratio of the bidder's bids to all bids in the auction; final value only known when the auction ends."),
        "Successive_Outbidding": ("auction in progress", "Observable as bids arrive; final value known only at the end."),
        "Last_Bidding": ("auction ended", "Time of the bidder's last bid relative to auction end - requires the auction to be finished."),
        "Auction_Bids": ("auction ended", "Total bids in the auction relative to concurrent auctions - requires the auction to be finished."),
        "Starting_Price_Average": ("auction start", "Known when the auction is listed."),
        "Early_Bidding": ("early in auction", "Time of the bidder's first bid; available once the bidder has bid."),
        "Winning_Ratio": ("auction outcomes", "Proportion of the bidder's auctions won - depends on auction outcomes, i.e. post-auction information."),
        "Auction_Duration": ("auction start", "Duration set by the seller at listing time."),
    }
    rows = [
        {
            "feature": f,
            "measured_at": settings.FEATURE_LEVEL[f],
            "available_at": notes[f][0],
            "comment": notes[f][1],
        }
        for f in settings.BEHAVIOURAL_FEATURES
    ]
    return pd.DataFrame(rows)
