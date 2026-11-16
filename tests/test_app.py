"""Smoke tests: every dashboard page renders without raising an exception."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from config import settings

PAGES = ["overview", "dataset", "performance", "detection", "backlog", "about"]


@pytest.fixture(autouse=True)
def isolated_backlog(tmp_path, monkeypatch):
    """Point the dashboard at a temporary backlog database."""
    monkeypatch.setattr(settings, "BACKLOG_DB_FILE", tmp_path / "app_test.db")


def _page_script(page: str, db_path: str) -> str:
    return f"""
import sys
sys.path.insert(0, {str(settings.PROJECT_ROOT)!r})
from pathlib import Path
from config import settings
settings.BACKLOG_DB_FILE = Path({db_path!r})
from dashboard import {page}
{page}.render()
"""


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, tmp_path):
    if not settings.MODEL_BUNDLE_FILE.exists():
        pytest.skip("Run the pipeline first.")
    at = AppTest.from_string(_page_script(page, str(tmp_path / "page.db")), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any("No trained model" in w.value for w in at.warning), "artefacts were not loaded"


def test_main_app_starts():
    at = AppTest.from_file(str(settings.PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_detection_flow_saves_to_backlog(tmp_path):
    """Analyse the most suspicious held-out record and save it to a temporary backlog."""
    if not settings.MODEL_BUNDLE_FILE.exists():
        pytest.skip("Run the pipeline first.")
    import pandas as pd

    from src.backlog.manager import BacklogManager
    from src.prediction.predict import load_bundle, predict_records

    df = pd.read_csv(settings.CLEAN_DATA_FILE, dtype={settings.BIDDER_ID: str})
    split = pd.read_csv(settings.SPLIT_FILE)
    test = df[df[settings.RECORD_ID].isin(split.loc[split.partition == "test", settings.RECORD_ID])]
    preds = predict_records(test, load_bundle())
    target_id = int(preds.sort_values("fraud_probability").iloc[-1][settings.RECORD_ID])

    db = tmp_path / "flow.db"
    at = AppTest.from_string(_page_script("detection", str(db)), default_timeout=180)
    at.session_state["record_pick"] = target_id
    at.run()
    [b for b in at.button if b.label == "Analyse record"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    save = [b for b in at.button if b.label == "Save to fraud backlog"]
    assert save, "a flagged record must offer 'Save to fraud backlog'"
    save[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    entries = BacklogManager(db).all_entries()
    assert list(entries["record_id"]) == [str(target_id)]
    assert entries["review_status"].iloc[0] == "Pending Review"
    stats = BacklogManager(db).statistics()
    assert stats["records_analysed"] == 1
