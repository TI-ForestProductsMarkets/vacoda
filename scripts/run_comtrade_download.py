"""
scripts/run_comtrade_download.py
"""

from vacoda.comtrade.pipeline import run_comtrade_download_pipeline
import logging
from vacoda.io.loaders import load_config
from vacoda.utils.logger import setup_logging
import vacoda.io.loaders as loaders_mod
from vacoda.utils.logger import get_logger


## Read log_level and log_file from config
cfg = load_config()
log_level = cfg.get("log_level", "INFO")
log_file = cfg.get("log_file", None)
if isinstance(log_level, str):
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
else:
    log_level_value = log_level
setup_logging(level=log_level_value, log_file=log_file)

## Logger for this script
logger = get_logger("pipeline_api")
logger.info(f"[Config] Using config file: {loaders_mod.CONFIG_PATH}")
logger.info(f"[Config] Logger level set to: {log_level} ({log_level_value})")

def main() -> None:
    run_comtrade_download_pipeline()


if __name__ == "__main__":
    main()