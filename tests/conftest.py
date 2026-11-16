"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small dataset with the official schema (40 auctions x 5 bidders)."""
    rng = np.random.default_rng(0)
    rows = []
    rid = 1
    for auction in range(40):
        for b in range(5):
            shill = int(b == 0 and auction % 2 == 0)
            row = {
                settings.RECORD_ID: rid,
                settings.AUCTION_ID: 1000 + auction,
                settings.BIDDER_ID: f"b***{b}",
                settings.TARGET: shill,
            }
            for f in settings.BEHAVIOURAL_FEATURES:
                row[f] = float(rng.uniform(0.5, 1.0) if shill else rng.uniform(0, 0.5))
            row["Successive_Outbidding"] = 1.0 if shill else 0.0
            row["Auction_Duration"] = 7
            rows.append(row)
            rid += 1
    return pd.DataFrame(rows)[settings.EXPECTED_COLUMNS]


@pytest.fixture
def real_df() -> pd.DataFrame:
    """The cleaned official dataset (skips if the pipeline has not been run)."""
    if not settings.CLEAN_DATA_FILE.exists():
        pytest.skip("Run `python run_pipeline.py` to create the processed dataset.")
    return pd.read_csv(settings.CLEAN_DATA_FILE, dtype={settings.BIDDER_ID: str})


@pytest.fixture
def bundle() -> dict:
    from src.prediction.predict import ModelNotTrainedError, load_bundle

    try:
        return load_bundle()
    except ModelNotTrainedError:
        pytest.skip("Run `python run_pipeline.py` to train the model.")
