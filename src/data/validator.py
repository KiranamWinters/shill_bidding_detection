"""Schema and integrity validation for the Shill Bidding Dataset."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import settings


class SchemaValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


@dataclass
class ValidationReport:
    """Outcome of a dataset validation run."""

    n_rows: int
    n_columns: int
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    non_numeric_features: list[str] = field(default_factory=list)
    target_values: list = field(default_factory=list)
    out_of_range: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not (self.missing_columns or self.non_numeric_features) and set(
            self.target_values
        ) <= {0, 1}


# Features documented / observed as normalised to [0, 1].
UNIT_INTERVAL_FEATURES = [
    "Bidder_Tendency",
    "Bidding_Ratio",
    "Successive_Outbidding",
    "Last_Bidding",
    "Auction_Bids",
    "Starting_Price_Average",
    "Early_Bidding",
    "Winning_Ratio",
]


def validate_dataset(df: pd.DataFrame, strict: bool = True) -> ValidationReport:
    """Validate column presence, dtypes, target encoding and value ranges.

    Args:
        df: dataset to validate (raw or cleaned).
        strict: raise :class:`SchemaValidationError` on fatal problems.
    """
    report = ValidationReport(n_rows=len(df), n_columns=df.shape[1])
    report.missing_columns = [c for c in settings.EXPECTED_COLUMNS if c not in df.columns]
    report.unexpected_columns = [c for c in df.columns if c not in settings.EXPECTED_COLUMNS]

    for col in settings.BEHAVIOURAL_FEATURES:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            report.non_numeric_features.append(col)

    if settings.TARGET in df.columns:
        report.target_values = sorted(df[settings.TARGET].dropna().unique().tolist())

    for col in UNIT_INTERVAL_FEATURES:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            n_bad = int(((df[col] < 0) | (df[col] > 1)).sum())
            if n_bad:
                report.out_of_range[col] = n_bad
    if report.out_of_range:
        report.warnings.append(f"Values outside [0, 1]: {report.out_of_range}")

    if strict and not report.is_valid:
        raise SchemaValidationError(
            f"Invalid dataset: missing={report.missing_columns}, "
            f"non_numeric={report.non_numeric_features}, target={report.target_values}"
        )
    return report


def validate_prediction_input(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with a frame submitted for prediction."""
    problems = []
    missing = [c for c in settings.BEHAVIOURAL_FEATURES if c not in df.columns]
    if missing:
        problems.append(f"Missing required behavioural columns: {missing}")
        return problems
    for col in settings.BEHAVIOURAL_FEATURES:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            problems.append(f"Column '{col}' contains missing or non-numeric values.")
    return problems
