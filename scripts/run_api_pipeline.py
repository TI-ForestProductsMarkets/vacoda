"""
Script: run_api_pipeline.py
Description: Independent pipeline for processing API data only. Legacy (local+metadata) pipeline remains unchanged.
"""


import logging
from vacoda.io.loaders import load_config
from vacoda.utils.logger import setup_logging

# Read log_level and log_file from config
cfg = load_config()
log_level = cfg.get("log_level", "INFO")
log_file = cfg.get("log_file", None)
if isinstance(log_level, str):
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
else:
    log_level_value = log_level
setup_logging(level=log_level_value, log_file=log_file)

import vacoda.io.loaders as loaders_mod
from vacoda.utils.logger import get_logger
logger = get_logger("pipeline_api")
logger.info(f"[Config] Using config file: {loaders_mod.CONFIG_PATH}")
logger.info(f"[Config] Logger level set to: {log_level} ({log_level_value})")
from vacoda.pipeline.build_processed_api_dataset import build_processed_api_dataset
from vacoda.io.paths import get_rawdata_file_path, get_processed_data_file_path

def main():
    api_csv_path = get_rawdata_file_path(source="api")
    output_path = get_processed_data_file_path()
    df = build_processed_api_dataset(file_path=api_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Processed API data saved to {output_path}")

if __name__ == "__main__":
    main()
