"""SQLite backlog tests (each test uses its own temporary database)."""

from __future__ import annotations

import pandas as pd
import pytest

from config import settings
from src.backlog.manager import BacklogManager


@pytest.fixture
def manager(tmp_path):
    return BacklogManager(tmp_path / "test_backlog.db")


@pytest.fixture
def predictions():
    return pd.DataFrame({
        settings.RECORD_ID: [1, 2, 3],
        settings.AUCTION_ID: [732, 732, 900],
        settings.BIDDER_ID: ["a***z", "b***y", "a***z"],
        "fraud_probability": [0.95, 0.10, 0.70],
        "predicted_class": [1, 0, 1],
        "prediction": [settings.CLASS_NAMES[1], settings.CLASS_NAMES[0], settings.CLASS_NAMES[1]],
        "threshold": [0.5, 0.5, 0.5],
        "model_version": ["test-v1"] * 3,
    })


def test_only_flagged_records_are_added(manager, predictions):
    ids = manager.add_flagged(predictions)
    assert len(ids) == 2
    data = manager.all_entries()
    assert set(data["record_id"]) == {"1", "3"}
    assert (data["review_status"] == "Pending Review").all()
    assert (data["predicted_class"] == 1).all()


def test_duplicates_skipped_for_same_model_version(manager, predictions):
    manager.add_flagged(predictions)
    assert manager.add_flagged(predictions) == []
    assert len(manager.all_entries()) == 2


def test_search_and_filters(manager, predictions):
    manager.add_flagged(predictions)
    assert len(manager.query(search="a***z")) == 2
    assert len(manager.query(search="900")) == 1
    assert len(manager.query(min_probability=0.9)) == 1
    first = int(manager.all_entries()["backlog_id"].iloc[0])
    manager.update_status(first, "Under Investigation")
    assert len(manager.query(statuses=["Under Investigation"])) == 1
    assert len(manager.query(statuses=["Pending Review"])) == 1


def test_status_update_notes_and_history(manager, predictions):
    bid = manager.add_flagged(predictions)[0]
    manager.update_status(bid, "Confirmed Suspicious", note="Bid pattern verified", changed_by="alice")
    manager.add_note(bid, "Escalated to trust & safety", author="bob")
    case = manager.get_case(bid)
    assert case["review_status"] == "Confirmed Suspicious"
    assert "Bid pattern verified" in case["investigation_notes"]
    assert "Escalated" in case["investigation_notes"]
    assert list(case["history"]["new_status"]) == ["Pending Review", "Confirmed Suspicious", "Confirmed Suspicious"]
    # The model prediction is never overwritten by the review outcome.
    assert case["predicted_class"] == 1


def test_invalid_status_and_missing_case(manager, predictions):
    bid = manager.add_flagged(predictions)[0]
    with pytest.raises(ValueError):
        manager.update_status(bid, "Guilty")
    with pytest.raises(KeyError):
        manager.update_status(9999, "Closed")
    with pytest.raises(ValueError):
        manager.add_note(bid, "   ")


def test_feature_values_and_top_features_stored(manager, predictions):
    feats = pd.DataFrame({"Successive_Outbidding": [1.0, 0.0, 0.5]}, index=predictions.index)
    bid = manager.add_flagged(predictions, features=feats, explanations={0: ["Successive_Outbidding"]})[0]
    case = manager.get_case(bid)
    assert case["feature_values"] == {"Successive_Outbidding": 1.0}
    assert case["top_features"] == ["Successive_Outbidding"]


def test_export_csv(manager, predictions, tmp_path):
    manager.add_flagged(predictions)
    out = tmp_path / "export.csv"
    manager.export_csv(out)
    exported = pd.read_csv(out)
    assert len(exported) == 2 and "review_status" in exported.columns


def test_statistics_and_prediction_log(manager, predictions):
    manager.log_predictions(predictions)
    bid = manager.add_flagged(predictions)[0]
    manager.update_status(bid, "False Positive")
    s = manager.statistics()
    assert s["records_analysed"] == 3
    assert s["predicted_shill"] == 2 and s["predicted_normal"] == 1
    assert s["pending"] == 1
    assert s["confirmation_rate"] == 0.0


def test_persistence_across_instances(tmp_path, predictions):
    path = tmp_path / "persist.db"
    BacklogManager(path).add_flagged(predictions)
    assert len(BacklogManager(path).all_entries()) == 2
