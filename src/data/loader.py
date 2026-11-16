"""Retrieve the Shill Bidding Dataset from the official UCI repository.

Retrieval order:

1. Local cache in ``data/raw`` (a previously downloaded official copy).
2. ``ucimlrepo.fetch_ucirepo(id=562)`` - the recommended UCI API client.
3. UCI's official static ZIP download (``archive.ics.uci.edu/static/public/562``).

UCI's metadata for this dataset declares ``target_col: null``, so the
``ucimlrepo`` client may return every column (including ``Class``) under
``features`` and no ``targets``. The loader therefore reassembles a single
DataFrame and identifies the target by name rather than trusting the split.
No third-party mirrors (e.g. Kaggle) are ever used.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone

import pandas as pd

from config import settings
from src.utils.logging_config import get_logger

logger = get_logger("data.loader")


class DatasetRetrievalError(RuntimeError):
    """Raised when the dataset cannot be retrieved from any official source."""


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map minor naming variants (spaces, case) onto the official column names."""
    lookup = {c.lower().replace(" ", "_"): c for c in settings.EXPECTED_COLUMNS}
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in lookup:
            rename[col] = lookup[key]
    return df.rename(columns=rename)


def _fetch_with_ucimlrepo() -> tuple[pd.DataFrame, dict]:
    """Fetch the dataset with the official ``ucimlrepo`` client."""
    from ucimlrepo import fetch_ucirepo  # imported lazily: optional at runtime

    dataset = fetch_ucirepo(id=settings.UCI_DATASET_ID)
    features = dataset.data.features
    targets = dataset.data.targets
    ids = getattr(dataset.data, "ids", None)

    parts = [p for p in (ids, features, targets) if p is not None]
    df = pd.concat(parts, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    metadata = {
        "source": "ucimlrepo.fetch_ucirepo(id=562)",
        "ucimlrepo_target_returned": targets is not None,
        "metadata": _jsonable(dict(dataset.metadata)) if dataset.metadata else {},
        "variables": dataset.variables.to_dict(orient="records")
        if getattr(dataset, "variables", None) is not None
        else [],
    }
    return df, metadata


def _fetch_official_zip() -> tuple[pd.DataFrame, dict]:
    """Download UCI's official ZIP archive and read the CSV inside it."""
    with urllib.request.urlopen(settings.UCI_ZIP_URL, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise DatasetRetrievalError("Official UCI archive contains no CSV file.")
        with archive.open(csv_names[0]) as handle:
            df = pd.read_csv(handle)
    metadata = {"source": settings.UCI_ZIP_URL, "archive_member": csv_names[0]}
    try:
        with urllib.request.urlopen(settings.UCI_METADATA_API, timeout=30) as response:
            metadata["metadata"] = json.loads(response.read().decode("utf-8")).get("data", {})
    except Exception as exc:  # metadata is informative only
        logger.warning("Could not download UCI metadata: %s", exc)
    return df, metadata


def _jsonable(obj):
    """Best-effort conversion of nested metadata into JSON-serialisable types."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return str(obj)


def download_dataset(force: bool = False) -> pd.DataFrame:
    """Download the dataset from UCI and cache it in ``data/raw``.

    Args:
        force: re-download even if a cached copy exists.

    Returns:
        The raw dataset with official column names.
    """
    settings.ensure_directories()
    if settings.RAW_DATA_FILE.exists() and not force:
        logger.info("Using cached UCI dataset at %s", settings.RAW_DATA_FILE)
        return pd.read_csv(settings.RAW_DATA_FILE)

    errors = []
    for fetcher in (_fetch_with_ucimlrepo, _fetch_official_zip):
        try:
            logger.info("Retrieving dataset via %s", fetcher.__name__)
            df, metadata = fetcher()
            df = _normalise_columns(df)
            missing = set(settings.EXPECTED_COLUMNS) - set(df.columns)
            if missing:
                raise DatasetRetrievalError(f"Missing columns after retrieval: {sorted(missing)}")
            df = df[settings.EXPECTED_COLUMNS]
            df.to_csv(settings.RAW_DATA_FILE, index=False)
            metadata["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["n_rows"] = int(len(df))
            settings.RAW_METADATA_FILE.write_text(
                json.dumps(_jsonable(metadata), indent=2), encoding="utf-8"
            )
            logger.info("Saved %d rows from %s", len(df), metadata["source"])
            return df
        except Exception as exc:
            logger.warning("%s failed: %s", fetcher.__name__, exc)
            errors.append(f"{fetcher.__name__}: {exc}")

    raise DatasetRetrievalError("Could not retrieve UCI dataset 562. " + " | ".join(errors))


def load_raw_dataset() -> pd.DataFrame:
    """Return the raw dataset, downloading it from UCI on first use."""
    return download_dataset(force=False)


def load_metadata() -> dict:
    """Return the cached retrieval metadata (empty dict if unavailable)."""
    if settings.RAW_METADATA_FILE.exists():
        return json.loads(settings.RAW_METADATA_FILE.read_text(encoding="utf-8"))
    return {}
