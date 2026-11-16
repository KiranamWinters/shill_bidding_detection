"""Preprocessing pipeline, leakage prevention and grouped splitting tests."""

from __future__ import annotations

import numpy as np
import pytest

from config import settings
from src.preprocessing.transformer import BehaviouralFeatureSelector, build_preprocessor
from src.training.compare_models import build_pipeline, get_candidates
from src.training.splitting import get_cv, group_overlap, grouped_holdout_split


def test_selector_rejects_identifiers_and_target(synthetic_df):
    for forbidden in settings.IDENTIFIER_COLUMNS + [settings.TARGET]:
        with pytest.raises(ValueError):
            BehaviouralFeatureSelector(settings.BEHAVIOURAL_FEATURES + [forbidden]).fit(synthetic_df)


def test_preprocessor_outputs_only_behavioural_features(synthetic_df):
    out = build_preprocessor(scale=True).fit_transform(synthetic_df)
    assert list(out.columns) == settings.BEHAVIOURAL_FEATURES
    assert np.allclose(out.mean().values, 0, atol=1e-8)


def test_scaler_fitted_on_training_rows_only(synthetic_df):
    train, test = synthetic_df.iloc[:100], synthetic_df.iloc[100:].copy()
    test["Bidding_Ratio"] += 100  # a shift in the test data must not affect the fitted scaler
    prep = build_preprocessor(scale=True).fit(train)
    assert np.isclose(prep.named_steps["scale"].mean_[1], train["Bidding_Ratio"].mean())


def test_every_candidate_pipeline_ignores_identifiers(synthetic_df):
    y = synthetic_df[settings.TARGET]
    shuffled = synthetic_df.copy()
    shuffled[settings.RECORD_ID] = shuffled[settings.RECORD_ID].sample(frac=1, random_state=1).values
    for key in ("logreg", "tree", "rf"):
        pipe = build_pipeline(get_candidates()[key]).fit(synthetic_df, y)
        assert list(pipe[:-1].get_feature_names_out()) == settings.BEHAVIOURAL_FEATURES
        # Changing identifiers must not change predictions.
        assert np.allclose(pipe.predict_proba(synthetic_df), pipe.predict_proba(shuffled))


def test_grouped_holdout_has_no_auction_overlap(synthetic_df):
    tr, te = grouped_holdout_split(synthetic_df)
    train, test = synthetic_df.iloc[tr], synthetic_df.iloc[te]
    ov = group_overlap(train, test)
    assert ov[settings.AUCTION_ID]["shared_ids"] == 0
    assert ov[settings.RECORD_ID]["shared_ids"] == 0
    assert len(tr) + len(te) == len(synthetic_df)


def test_real_split_is_stratified_and_grouped(real_df):
    tr, te = grouped_holdout_split(real_df)
    train, test = real_df.iloc[tr], real_df.iloc[te]
    assert not set(train[settings.AUCTION_ID]) & set(test[settings.AUCTION_ID])
    assert abs(train[settings.TARGET].mean() - test[settings.TARGET].mean()) < 0.02
    assert 0.15 < len(test) / len(real_df) < 0.25


def test_saved_split_matches_reproducible_split(real_df):
    import pandas as pd

    if not settings.SPLIT_FILE.exists():
        pytest.skip("Split file not created yet.")
    saved = pd.read_csv(settings.SPLIT_FILE)
    _, te = grouped_holdout_split(real_df)
    assert set(saved.loc[saved.partition == "test", settings.RECORD_ID]) == set(real_df.iloc[te][settings.RECORD_ID])


@pytest.mark.parametrize("strategy", ["auction", "bidder"])
def test_grouped_cv_folds_do_not_share_groups(real_df, strategy):
    cv, col = get_cv(strategy)
    for tr, va in cv.split(real_df, real_df[settings.TARGET], real_df[col]):
        assert not set(real_df.iloc[tr][col]) & set(real_df.iloc[va][col])
