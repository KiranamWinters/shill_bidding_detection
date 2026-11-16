"""Load the trained model bundle and score new bidding records.

Only the nine behavioural features are passed to the model. Identifier
columns are carried through to the output for tracking, and a ``Class``
column, if present in the input, is ignored for prediction.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import settings
from src.data.validator import validate_prediction_input
from src.utils.logging_config import get_logger

logger = get_logger("prediction")


class ModelNotTrainedError(FileNotFoundError):
    """Raised when no trained model bundle exists yet."""


def load_bundle(path: Path = settings.MODEL_BUNDLE_FILE) -> dict:
    """Load the saved model bundle (pipeline, threshold, metadata)."""
    if not Path(path).exists():
        raise ModelNotTrainedError(
            f"No trained model at {path}. Run `python run_pipeline.py` first."
        )
    return joblib.load(path)


def predict_records(df: pd.DataFrame, bundle: dict, threshold: float | None = None) -> pd.DataFrame:
    """Score records and return identifiers, probability and predicted class.

    Args:
        df: frame containing at least the behavioural feature columns.
        bundle: model bundle from :func:`load_bundle`.
        threshold: decision threshold; defaults to the bundle's optimised value.

    Raises:
        ValueError: if required columns are missing or non-numeric.
    """
    problems = validate_prediction_input(df)
    if problems:
        raise ValueError("; ".join(problems))
    threshold = float(bundle["threshold"] if threshold is None else threshold)
    features = list(bundle["features"])

    # The target (if supplied) is deliberately excluded from the model input.
    X = df[features].apply(pd.to_numeric, errors="coerce").astype(float)
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame(index=df.index)
    for col in settings.IDENTIFIER_COLUMNS:
        out[col] = df[col].values if col in df.columns else None
    out["fraud_probability"] = np.round(proba, 6)
    out["predicted_class"] = pred
    out["prediction"] = np.where(pred == 1, settings.CLASS_NAMES[1], settings.CLASS_NAMES[0])
    out["threshold"] = threshold
    out["model_version"] = bundle["model_version"]
    logger.info("Scored %d records: %d flagged (threshold %.2f)", len(out), int(pred.sum()), threshold)
    return out
