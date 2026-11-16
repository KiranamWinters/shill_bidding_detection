"""Trained model, prediction, probability and explanation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.evaluation.evaluate import compute_metrics, optimise_threshold
from src.explainability.explain import ModelExplainer
from src.prediction.predict import predict_records


def test_metrics_on_known_example():
    y = [0, 0, 1, 1]
    p = [0.1, 0.6, 0.4, 0.9]
    m = compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["roc_auc"] == 0.75


def test_threshold_optimiser_respects_precision_floor():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(900), np.ones(100)].astype(int)
    p = np.clip(np.r_[rng.normal(0.2, 0.15, 900), rng.normal(0.7, 0.15, 100)], 0, 1)
    thr, sweep = optimise_threshold(y, p, beta=2, min_precision=0.8)
    row = sweep[np.isclose(sweep.threshold, thr)].iloc[0]
    assert row["precision"] >= 0.8
    assert 0.05 <= thr <= 0.95


def test_bundle_contents(bundle):
    assert bundle["features"]
    assert not set(bundle["features"]) & (set(settings.IDENTIFIER_COLUMNS) | {settings.TARGET})
    assert 0 < bundle["threshold"] < 1
    assert bundle["model_version"].startswith(bundle["model_key"])
    assert settings.MODEL_BUNDLE_FILE.exists()


def test_predictions_are_valid_probabilities(bundle, real_df):
    preds = predict_records(real_df.head(300), bundle)
    assert preds["fraud_probability"].between(0, 1).all()
    assert ((preds["fraud_probability"] >= bundle["threshold"]).astype(int) == preds["predicted_class"]).all()
    assert list(preds[settings.RECORD_ID]) == list(real_df.head(300)[settings.RECORD_ID])


def test_probability_matches_pipeline(bundle, real_df):
    sample = real_df.sample(50, random_state=0)
    preds = predict_records(sample, bundle)
    direct = bundle["pipeline"].predict_proba(sample[bundle["features"]])[:, 1]
    assert np.allclose(preds["fraud_probability"].values, direct, atol=1e-6)


def test_target_label_is_not_used_for_prediction(bundle, real_df):
    sample = real_df.head(200).copy()
    flipped = sample.copy()
    flipped[settings.TARGET] = 1 - flipped[settings.TARGET]
    no_label = sample.drop(columns=[settings.TARGET])
    a = predict_records(sample, bundle)["fraud_probability"]
    b = predict_records(flipped, bundle)["fraud_probability"]
    c = predict_records(no_label, bundle)["fraud_probability"]
    assert np.allclose(a, b) and np.allclose(a, c)


def test_prediction_rejects_missing_features(bundle, real_df):
    with pytest.raises(ValueError):
        predict_records(real_df.head(5).drop(columns=["Bidding_Ratio"]), bundle)


def test_model_detects_held_out_shill_records(bundle, real_df):
    split = pd.read_csv(settings.SPLIT_FILE)
    test_ids = set(split.loc[split.partition == "test", settings.RECORD_ID])
    test = real_df[real_df[settings.RECORD_ID].isin(test_ids)]
    preds = predict_records(test, bundle)
    m = compute_metrics(test[settings.TARGET], preds["fraud_probability"], bundle["threshold"])
    # Must reproduce the stored evaluation exactly (no hidden refit on test data).
    assert np.isclose(m["recall"], bundle["test_metrics"]["recall"])
    assert np.isclose(m["precision"], bundle["test_metrics"]["precision"])


def test_explanation_covers_all_features(bundle, real_df):
    explainer = ModelExplainer(bundle)
    record = real_df[real_df["Successive_Outbidding"] == 1].head(1)
    exp = explainer.explain_record(record)
    assert set(exp["contributions"]["feature"]) == set(bundle["features"])
    assert 0 <= exp["probability"] <= 1
    assert exp["narrative"] and "not proof" in exp["narrative"][-1]


def test_occlusion_fallback(bundle, real_df):
    explainer = ModelExplainer(bundle)
    values = explainer.occlusion(real_df.head(3)[bundle["features"]])
    assert values.shape == (3, len(bundle["features"]))
