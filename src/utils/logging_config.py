"""Project-wide logging configuration."""

from __future__ import annotations

import logging
import sys

from config import settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to the console and to ``logs/project.log``."""
    global _CONFIGURED
    if not _CONFIGURED:
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        root = logging.getLogger("shill")
        root.setLevel(logging.INFO)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        file_handler = logging.FileHandler(settings.LOGS_DIR / "project.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(console)
        root.addHandler(file_handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"shill.{name}")
