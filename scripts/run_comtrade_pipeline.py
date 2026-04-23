"""
Script entrypoint for Comtrade API workflows.

How to run:
    python scripts/run_comtrade_api.py 
"""

from __future__ import annotations

from pathlib import Path
import yaml

from vacoda.comtrade.comtrade_pipeline import run_comtrade_api_pipeline
from vacoda.utils.logger import setup_logging


def main() -> None:
    # Load logging config from comtrade.yaml if present
    with open("configs/comtrade.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    import datetime
    log_level = cfg.get("log_level", "INFO").upper()
    log_file = cfg.get("log_file", "output/reports/download.log")
    # If log_file is set, add timestamp before file extension
    if log_file:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(log_file)
        if "." in log_file:
            base, ext = log_file.rsplit(".", 1)
            log_file = f"{base}_{ts}.{ext}"
        else:
            log_file = f"{log_file}_{ts}"
    import logging
    level = getattr(logging, log_level, logging.INFO)
    setup_logging(level=level, log_file=Path(log_file))
    run_comtrade_api_pipeline(
        paths_config_path=Path("configs/paths.yaml"),
        download_config_path=Path("configs/comtrade.yaml"),
    )


if __name__ == "__main__":
    main()
