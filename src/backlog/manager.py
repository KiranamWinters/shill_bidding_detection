"""Business logic for the suspicious-activity backlog.

A backlog entry is created only for records the model flags as potential shill
bidding. Every entry starts as ``Pending Review``; only an investigator can move
it to ``Confirmed Suspicious`` or ``False Positive``.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import settings
from src.backlog.database import BacklogDatabase
from src.utils.logging_config import get_logger

logger = get_logger("backlog")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _as_text(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


class BacklogManager:
    """CRUD, search, filtering, export and statistics for the backlog."""

    def __init__(self, db_path: Path | str = settings.BACKLOG_DB_FILE):
        self.db = BacklogDatabase(db_path)

    # ----------------------------------------------------------- prediction log
    def log_predictions(self, predictions: pd.DataFrame, source: str = "dashboard") -> str:
        """Store every scored record in ``prediction_log``; returns the batch ID."""
        batch_id = uuid.uuid4().hex[:12]
        ts = _now()
        rows = [
            (batch_id, _as_text(r.get(settings.RECORD_ID)), _as_text(r.get(settings.AUCTION_ID)),
             _as_text(r.get(settings.BIDDER_ID)), float(r["fraud_probability"]),
             int(r["predicted_class"]), float(r["threshold"]), str(r["model_version"]), source, ts)
            for r in predictions.to_dict(orient="records")
        ]
        with self.db.connect() as conn:
            conn.executemany(
                """INSERT INTO prediction_log (batch_id, record_id, auction_id, bidder_id,
                   fraud_probability, predicted_class, threshold, model_version, source, predicted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        logger.info("Logged %d predictions (batch %s)", len(rows), batch_id)
        return batch_id

    # ----------------------------------------------------------------- create
    def add_flagged(
        self,
        predictions: pd.DataFrame,
        features: pd.DataFrame | None = None,
        explanations: dict | None = None,
        source: str = "dashboard",
        skip_duplicates: bool = True,
    ) -> list[int]:
        """Create backlog entries for flagged predictions (``predicted_class == 1``).

        Args:
            predictions: output of :func:`src.prediction.predict.predict_records`.
            features: behavioural feature values (same index) stored for case review.
            explanations: optional ``{index: [top feature names]}``.
            skip_duplicates: do not re-add a record already in the backlog for the
                same model version.

        Returns:
            IDs of the created backlog entries.
        """
        flagged = predictions[predictions["predicted_class"] == 1]
        created = []
        ts = _now()
        with self.db.connect() as conn:
            for idx, r in flagged.iterrows():
                record_id = _as_text(r.get(settings.RECORD_ID))
                if skip_duplicates and record_id is not None:
                    exists = conn.execute(
                        "SELECT 1 FROM backlog WHERE record_id = ? AND model_version = ?",
                        (record_id, str(r["model_version"])),
                    ).fetchone()
                    if exists:
                        continue
                feat_json = (
                    json.dumps({k: float(v) for k, v in features.loc[idx].items()})
                    if features is not None and idx in features.index else None
                )
                top_json = json.dumps(explanations.get(idx)) if explanations and idx in explanations else None
                cur = conn.execute(
                    """INSERT INTO backlog (record_id, auction_id, bidder_id, fraud_probability,
                       predicted_class, prediction_label, threshold, prediction_timestamp,
                       model_version, review_status, investigation_notes, top_features,
                       feature_values, source, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?)""",
                    (record_id, _as_text(r.get(settings.AUCTION_ID)), _as_text(r.get(settings.BIDDER_ID)),
                     float(r["fraud_probability"]), int(r["predicted_class"]), str(r["prediction"]),
                     float(r["threshold"]), ts, str(r["model_version"]),
                     settings.DEFAULT_REVIEW_STATUS, top_json, feat_json, source, ts),
                )
                backlog_id = cur.lastrowid
                conn.execute(
                    """INSERT INTO status_history (backlog_id, old_status, new_status, note, changed_by, changed_at)
                       VALUES (?, NULL, ?, 'Created from model prediction', 'system', ?)""",
                    (backlog_id, settings.DEFAULT_REVIEW_STATUS, ts),
                )
                created.append(backlog_id)
        logger.info("Added %d backlog entries (%d flagged)", len(created), len(flagged))
        return created

    # ------------------------------------------------------------------- read
    def query(
        self,
        search: str | None = None,
        statuses: list[str] | None = None,
        min_probability: float = 0.0,
        max_probability: float = 1.0,
    ) -> pd.DataFrame:
        """Filter the backlog.

        Args:
            search: substring matched against bidder ID, auction ID or record ID.
            statuses: keep only these review statuses.
            min_probability / max_probability: fraud-probability range.
        """
        sql = "SELECT * FROM backlog WHERE fraud_probability BETWEEN ? AND ?"
        params: list = [min_probability, max_probability]
        if search:
            sql += " AND (bidder_id LIKE ? OR auction_id LIKE ? OR record_id LIKE ?)"
            like = f"%{search.strip()}%"
            params += [like, like, like]
        if statuses:
            sql += f" AND review_status IN ({','.join('?' * len(statuses))})"
            params += list(statuses)
        sql += " ORDER BY fraud_probability DESC, backlog_id DESC"
        with self.db.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def all_entries(self) -> pd.DataFrame:
        return self.query()

    def get_case(self, backlog_id: int) -> dict | None:
        """Return one entry with its decoded features and status history."""
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM backlog WHERE backlog_id = ?", (backlog_id,)).fetchone()
            if row is None:
                return None
            history = pd.read_sql_query(
                "SELECT * FROM status_history WHERE backlog_id = ? ORDER BY history_id",
                conn, params=[backlog_id],
            )
        case = dict(row)
        case["feature_values"] = json.loads(case["feature_values"]) if case["feature_values"] else {}
        case["top_features"] = json.loads(case["top_features"]) if case["top_features"] else []
        case["history"] = history
        return case

    # ----------------------------------------------------------------- update
    def update_status(self, backlog_id: int, new_status: str, note: str = "", changed_by: str = "investigator") -> None:
        """Change the review status and record the change in the audit trail."""
        if new_status not in settings.REVIEW_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Allowed: {settings.REVIEW_STATUSES}")
        ts = _now()
        with self.db.connect() as conn:
            row = conn.execute("SELECT review_status FROM backlog WHERE backlog_id = ?", (backlog_id,)).fetchone()
            if row is None:
                raise KeyError(f"Backlog entry {backlog_id} not found")
            conn.execute("UPDATE backlog SET review_status = ?, updated_at = ? WHERE backlog_id = ?",
                         (new_status, ts, backlog_id))
            conn.execute(
                """INSERT INTO status_history (backlog_id, old_status, new_status, note, changed_by, changed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (backlog_id, row["review_status"], new_status, note or None, changed_by, ts),
            )
        if note:
            self.add_note(backlog_id, note, changed_by, _log_history=False)

    def add_note(self, backlog_id: int, note: str, author: str = "investigator", _log_history: bool = True) -> None:
        """Append a timestamped investigation note."""
        note = (note or "").strip()
        if not note:
            raise ValueError("Note must not be empty")
        ts = _now()
        entry = f"[{ts}] {author}: {note}"
        with self.db.connect() as conn:
            row = conn.execute("SELECT investigation_notes, review_status FROM backlog WHERE backlog_id = ?",
                               (backlog_id,)).fetchone()
            if row is None:
                raise KeyError(f"Backlog entry {backlog_id} not found")
            notes = f"{row['investigation_notes']}\n{entry}".strip()
            conn.execute("UPDATE backlog SET investigation_notes = ?, updated_at = ? WHERE backlog_id = ?",
                         (notes, ts, backlog_id))
            if _log_history:
                conn.execute(
                    """INSERT INTO status_history (backlog_id, old_status, new_status, note, changed_by, changed_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (backlog_id, row["review_status"], row["review_status"], note, author, ts),
                )

    def delete_entry(self, backlog_id: int) -> None:
        """Remove an entry (used by tests / data maintenance)."""
        with self.db.connect() as conn:
            conn.execute("DELETE FROM backlog WHERE backlog_id = ?", (backlog_id,))

    # ----------------------------------------------------------------- export
    def export_csv(self, path: Path | str | None = None, frame: pd.DataFrame | None = None) -> str:
        """Export the backlog (or a filtered frame) to CSV; returns CSV text."""
        data = self.all_entries() if frame is None else frame
        text = data.to_csv(index=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    # ------------------------------------------------------------- statistics
    def statistics(self) -> dict:
        """Aggregate statistics over the prediction log and the backlog."""
        with self.db.connect() as conn:
            log = pd.read_sql_query("SELECT * FROM prediction_log", conn)
            backlog = pd.read_sql_query("SELECT * FROM backlog", conn)
        status_counts = backlog["review_status"].value_counts().reindex(settings.REVIEW_STATUSES, fill_value=0)
        reviewed = status_counts[["Confirmed Suspicious", "False Positive"]].sum()
        stats = {
            "records_analysed": int(len(log)),
            "predicted_normal": int((log["predicted_class"] == 0).sum()) if len(log) else 0,
            "predicted_shill": int((log["predicted_class"] == 1).sum()) if len(log) else 0,
            "detection_rate": float(log["predicted_class"].mean()) if len(log) else 0.0,
            "prediction_batches": int(log["batch_id"].nunique()) if len(log) else 0,
            "backlog_total": int(len(backlog)),
            "status_counts": {k: int(v) for k, v in status_counts.items()},
            "pending": int(status_counts["Pending Review"]),
            "open_cases": int(status_counts[["Pending Review", "Under Investigation"]].sum()),
            "confirmation_rate": float(status_counts["Confirmed Suspicious"] / reviewed) if reviewed else None,
            "mean_probability": float(backlog["fraud_probability"].mean()) if len(backlog) else None,
        }
        if len(backlog):
            backlog["date"] = pd.to_datetime(backlog["prediction_timestamp"]).dt.date
            stats["daily_flags"] = backlog.groupby("date").size().rename("flags").reset_index()
            stats["top_bidders"] = (backlog.groupby("bidder_id").size().sort_values(ascending=False)
                                    .head(10).rename("flags").reset_index())
            stats["top_auctions"] = (backlog.groupby("auction_id").size().sort_values(ascending=False)
                                     .head(10).rename("flags").reset_index())
        return stats
