"""SQLite persistence layer for predictions and the suspicious-activity backlog.

Tables
------
``prediction_log``   every record scored by the model (flagged or not) - used
                     for "records analysed" statistics.
``backlog``          one row per potentially fraudulent record awaiting or
                     having undergone human review.
``status_history``   audit trail of every review-status change and note.

Model predictions and investigation outcomes are stored separately:
``predicted_class`` never changes, while ``review_status`` records what an
investigator concluded.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_log (
    log_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id          TEXT    NOT NULL,
    record_id         TEXT,
    auction_id        TEXT,
    bidder_id         TEXT,
    fraud_probability REAL    NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    predicted_class   INTEGER NOT NULL CHECK (predicted_class IN (0, 1)),
    threshold         REAL    NOT NULL,
    model_version     TEXT    NOT NULL,
    source            TEXT,
    predicted_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS backlog (
    backlog_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id            TEXT,
    auction_id           TEXT,
    bidder_id            TEXT,
    fraud_probability    REAL    NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    predicted_class      INTEGER NOT NULL CHECK (predicted_class IN (0, 1)),
    prediction_label     TEXT    NOT NULL,
    threshold            REAL    NOT NULL,
    prediction_timestamp TEXT    NOT NULL,
    model_version        TEXT    NOT NULL,
    review_status        TEXT    NOT NULL DEFAULT 'Pending Review',
    investigation_notes  TEXT    NOT NULL DEFAULT '',
    top_features         TEXT,
    feature_values       TEXT,
    source               TEXT,
    updated_at           TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS status_history (
    history_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    backlog_id  INTEGER NOT NULL REFERENCES backlog(backlog_id) ON DELETE CASCADE,
    old_status  TEXT,
    new_status  TEXT,
    note        TEXT,
    changed_by  TEXT,
    changed_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_backlog_bidder  ON backlog(bidder_id);
CREATE INDEX IF NOT EXISTS idx_backlog_auction ON backlog(auction_id);
CREATE INDEX IF NOT EXISTS idx_backlog_status  ON backlog(review_status);
CREATE INDEX IF NOT EXISTS idx_log_batch       ON prediction_log(batch_id);
"""


class BacklogDatabase:
    """Thin wrapper around a SQLite file holding the backlog schema."""

    def __init__(self, db_path: Path | str = settings.BACKLOG_DB_FILE):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialise()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success and rolls back on error."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialise(self) -> None:
        """Create tables and indexes if they do not exist (idempotent)."""
        with self.connect() as conn:
            conn.executescript(SCHEMA)
