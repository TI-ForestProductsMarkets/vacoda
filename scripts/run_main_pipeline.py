"""
This script is responsible for building the base analytic dataset from the raw data bundle and saving it as a parquet file.
"""

import logging
from pathlib import Path
from vacoda.utils.logger import setup_logging, get_logger, rel_path
from vacoda.pipeline.build_processed_dataset import build_processed_dataset
from vacoda.io.paths import (
    get_processed_data_file_path,
    get_processed_data_dir,
    get_raw_data_dir,
    load_paths_config,
    get_processed_file_path_from_raw
)
from vacoda.io.writer import write_parquet
from vacoda.io.loaders import get_raw_csv_file_list

setup_logging(level=logging.DEBUG)
logger = get_logger("pipeline")


def _normalize_pipeline_mode(mode: str | None) -> str:
    """Normalize pipeline mode to canonical names while supporting legacy aliases."""
    aliases = {
        "single": "per_file",
        "all": "merged",
        "per_file": "per_file",
        "merged": "merged",
    }
    mode_norm = (mode or "per_file").strip().lower()
    if mode_norm not in aliases:
        valid = ", ".join(sorted(set(aliases.values()) | {"single", "all"}))
        raise ValueError(f"Unknown pipeline_mode: {mode!r}. Valid values: {valid}")
    return aliases[mode_norm]

def main() -> None:
    """
    Main function to build the base analytic dataset from the raw data bundle and save it as a parquet file.
    Handles both 'single' and 'all' pipeline modes.
    """
    config = load_paths_config()
    pipeline_cfg = config.get("pipeline", {})
    mode = _normalize_pipeline_mode(pipeline_cfg.get("mode", "per_file"))
    source = pipeline_cfg.get("source", "api")
    logger.info(f"Pipeline mode resolved to: {mode}")
    processed_dir = get_processed_data_dir()
    csv_files = get_raw_csv_file_list(source=source)

    if mode == "per_file":
        if not csv_files:
            logger.error("No CSV files found for processing.")
            return
        for csv_path in csv_files:
            logger.info(f"Processing file: {rel_path(csv_path)}")
            # Only pass file_path if you want to override config; otherwise, use None
            df_base = build_processed_dataset(file_path=None)
            output_path = get_processed_file_path_from_raw(csv_path, processed_dir)
            written = write_parquet(df_base, output_path)
            if written:
                logger.info(f"Parquet saved to: {rel_path(output_path)} | {len(df_base):,} rows, {df_base.shape[1]} columns")
    elif mode == "merged":
        if not csv_files:
            logger.error("No CSV files found for processing.")
            return
        df_base = build_processed_dataset()
        processed_files = config["processed_data"].get("files", ["processed.parquet"])
        if not processed_files:
            logger.error("No output file specified in processed_data.files in config!")
            return
        output_path = processed_dir / processed_files[0]
        written = write_parquet(df_base, output_path)
        if written:
            logger.info(f"Parquet saved to: {rel_path(output_path)} | {len(df_base):,} rows, {df_base.shape[1]} columns")
    else:
        logger.error(f"Unknown pipeline_mode: {mode}")

if __name__ == "__main__":
    main()
