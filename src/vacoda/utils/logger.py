"""
vacoda.utils.logger
------------------
Utility functions for logging configuration and usage within the vacoda package.
"""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime


_PROJECT_ROOT = Path(__file__).parents[3]


def rel_path(path) -> str:
    """Return path relative to project root for concise log messages."""
    try:
        return str(Path(path).relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger for the given module name.

    Usage
    -----
    from vacoda.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.debug("detailed info for debugging")
    logger.info("general progress message")
    logger.warning("something unexpected but not fatal")
    logger.error("something went wrong")
    """
    return logging.getLogger(name)

def setup_logging(
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """
    Configure root logging for the vacoda package with console and optional file output.

    Call this once at the entry point.
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)

        if not log_path.is_absolute():
            log_path = _PROJECT_ROOT / log_path

        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    for noisy in ("watchdog", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

def parse_log_level(level: str | int | None, default: int = logging.INFO) -> int:
    """
    Convert a YAML/user log level into a logging level integer.

    Examples
    --------
    DEBUG -> logging.DEBUG
    INFO  -> logging.INFO
    10    -> 10
    None  -> default
    """
    if level is None:
        return default

    if isinstance(level, int):
        return level

    return getattr(logging, str(level).strip().upper(), default)


def make_timestamped_log_path(
    log_file: str | Path | None,
    *,
    timestamp: str | None = None,
) -> Path | None:
    """
    Add timestamp before a log file suffix.

    Example
    -------
    LOG_PIPELINE.log
    -> LOG_PIPELINE_20260511_124215.log
    """
    if log_file is None:
        return None

    path = Path(log_file)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")