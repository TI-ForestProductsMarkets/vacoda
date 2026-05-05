"""
scripts/run_comtrade_download.py

Run the Comtrade download pipeline with timestamped logging.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from vacoda.comtrade.pipeline import run_comtrade_download_pipeline
from vacoda.io.paths import get_project_root, resolve_path
from vacoda.utils.logger import get_logger, setup_logging


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML root to be a mapping: {path}")

    return data


def _timestamped_log_path(log_file: str | None) -> Path | None:
    """
    Convert:

        output/reports/LOG_API_PIPELINE.log

    into:

        output/reports/LOG_API_PIPELINE_20260505_094215.log
    """
    if not log_file:
        return None

    path = resolve_path(log_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def _get_log_level(log_level: str | int | None) -> int:
    if log_level is None:
        return logging.INFO

    if isinstance(log_level, int):
        return log_level

    return getattr(logging, str(log_level).upper(), logging.INFO)


def main() -> None:
    project_root = get_project_root()
    config_path = project_root / "configs" / "comtrade.yaml"

    cfg = _load_yaml(config_path)

    log_level_raw = cfg.get("log_level", "INFO")
    log_file_raw = cfg.get("log_file", "output/reports/LOG_API_PIPELINE.log")

    log_level = _get_log_level(log_level_raw)
    log_path = _timestamped_log_path(log_file_raw)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level=log_level,
        log_file=str(log_path) if log_path is not None else None,
    )

    logger = get_logger("SCRIPT.COMTRADE_DOWNLOAD")

    logger.info("=" * 72)
    logger.info("Starting Comtrade download script")
    logger.info("=" * 72)
    logger.info("[Config] Using config file: %s", config_path)
    logger.info("[Config] Logger level set to: %s (%s)", log_level_raw, log_level)
    logger.info("[Config] Log file: %s", log_path)

    run_comtrade_download_pipeline(config_path=config_path)

    logger.info("=" * 72)
    logger.info("Finished Comtrade download script")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()