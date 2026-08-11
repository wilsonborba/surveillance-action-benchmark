from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


_LOGGING_READY = False


def configure_logging(level: str = "INFO") -> None:
    global _LOGGING_READY
    if _LOGGING_READY:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    _LOGGING_READY = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not _LOGGING_READY:
        configure_logging()
    return logging.getLogger(name or "sab")
