"""Dataset retrieval, schema and cleaning tests."""

from __future__ import annotations

import pandas as pd
import pytest

from config import settings
from src.data.validator import SchemaValidationError, validate_dataset, validate_prediction_input
from src.preprocessing.cleaner import clean_dataset, split_identifiers_features_target


def test_raw_download_matches_uci_schema():
    if not settings.RAW_DATA_FILE.exists():
        pytest.skip("Raw UCI file not downloaded yet.")
    raw = pd.read_csv(settings.RAW_DATA_FILE)
    assert list(raw.columns) == settings.EXPECTED_COLUMNS
    assert len(raw) == 6321  # number of instances stated by UCI
    report = validate_dataset(raw)
    assert report.is_valid
    assert report.target_values == [0, 1]


def test_retrieval_metadata_points_to_uci():
    if not settings.RAW_METADATA_FILE.exists():
        pytest.skip("No retrieval metadata yet.")
    from src.data.loader import load_metadata

    meta = load_metadata()
    assert "archive.ics.uci.edu" in meta["source"] or "ucimlrepo" in meta["source"]
    assert "kaggle" not in str(meta).lower()


def test_validator_rejects_missing_column(synthetic_df):
    with pytest.raises(SchemaValidationError):
        validate_dataset(synthetic_df.drop(columns=["Bidding_Ratio"]))


def test_validator_rejects_bad_target(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, settings.TARGET] = 2
    with pytest.raises(SchemaValidationError):
        validate_dataset(bad)


def test_cleaner_removes_duplicates_and_keeps_schema(synthetic_df):
    dup = pd.concat([synthetic_df, synthetic_df.head(3)], ignore_index=True)
    clean = clean_dataset(dup)
    assert len(clean) == len(synthetic_df)
    assert list(clean.columns) == settings.EXPECTED_COLUMNS
    assert clean[settings.BIDDER_ID].dtype == object or pd.api.types.is_string_dtype(clean[settings.BIDDER_ID])


def test_split_identifiers_features_target(synthetic_df):
    ids, X, y = split_identifiers_features_target(clean_dataset(synthetic_df))
    assert list(ids.columns) == settings.IDENTIFIER_COLUMNS
    assert list(X.columns) == settings.BEHAVIOURAL_FEATURES
    assert not set(X.columns) & (set(settings.IDENTIFIER_COLUMNS) | {settings.TARGET})
    assert set(y.unique()) <= {0, 1}


def test_prediction_input_validation(synthetic_df):
    assert validate_prediction_input(synthetic_df) == []
    problems = validate_prediction_input(synthetic_df.drop(columns=["Winning_Ratio"]))
    assert problems and "Winning_Ratio" in problems[0]
    bad = synthetic_df.copy()
    bad["Bidding_Ratio"] = bad["Bidding_Ratio"].astype(object)
    bad.loc[0, "Bidding_Ratio"] = "abc"
    assert validate_prediction_input(bad)
