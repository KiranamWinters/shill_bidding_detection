"""Run the full offline pipeline: download -> validate -> clean -> EDA -> train/evaluate.

Usage:
    python run_pipeline.py              # use cached UCI download if present
    python run_pipeline.py --refresh    # re-download from UCI
    python run_pipeline.py --skip-eda   # skip EDA figures/report
"""

from __future__ import annotations

import argparse
import sys
import time

from config import settings
from src.analysis.eda import run_eda
from src.data.loader import download_dataset
from src.data.validator import validate_dataset
from src.preprocessing.cleaner import clean_dataset
from src.training.train import run_training
from src.utils.logging_config import get_logger

logger = get_logger("pipeline")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download the dataset from UCI")
    parser.add_argument("--skip-eda", action="store_true", help="skip exploratory analysis")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    settings.ensure_directories()
    raw = download_dataset(force=args.refresh)
    report = validate_dataset(raw)
    logger.info("Validation passed: %d rows, %d columns; warnings=%s",
                report.n_rows, report.n_columns, report.warnings)
    clean = clean_dataset(raw)
    clean.to_csv(settings.CLEAN_DATA_FILE, index=False)

    if not args.skip_eda:
        run_eda(clean)
    results = run_training(clean)

    sel = next(r for r in results["comparison"] if r["selected"])
    logger.info(
        "Selected %s | test PR-AUC=%.4f ROC-AUC=%.4f precision=%.4f recall=%.4f F1=%.4f (threshold %.2f)",
        results["selected_model_name"], sel["pr_auc"], sel["roc_auc"], sel["precision"],
        sel["recall"], sel["f1"], sel["threshold"],
    )
    logger.info("Pipeline finished in %.1f s", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
