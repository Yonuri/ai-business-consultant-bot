"""Настройка логирования."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_NOISY = (
    "httpx",
    "httpcore",
    "chromadb",
    "aiogram.event",
    "google_genai",
    "google_genai.models",
    "urllib3",
)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        datefmt=_DATEFMT,
    )
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
