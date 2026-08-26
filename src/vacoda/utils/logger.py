"""
vacoda.utils.logger
------------------
Utility functions for logging configuration and usage within the vacoda package.
"""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from vacoda.utils.display import get_symbols

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


def log_checkpoint_reuse(logger, description, reason, cached_output_path=None):
    logger.info(
        "%s REUSE — %s:%s %s",
        get_symbols("recycle"),
        description,
        f" ({reason})" if reason is not None else "",
        cached_output_path,
    )

def log_checkpoint_rebuild(logger, description=None, reason=None):
    logger.info(
        "%s REBUILD — %s: %s",
        get_symbols("rebuild"),
        description,
        reason,
    )

def log_checkpoint_skip(logger, description, output_path):
    logger.info(
        "%s SKIP — %s local checkpoint matches: %s",
        get_symbols("skip"),
        description,
        output_path,
    )

def log_semantic_check(logger, name, reason):
    logger.info(
        "%s CHECK — %s requires semantic validation: %s",
        get_symbols("warning"),
        name,
        reason,
    )

def log_audit_info(logger, description):
    logger.info(
        "%s AUDIT — %s",
        get_symbols("check_mark"),
        description,
    )