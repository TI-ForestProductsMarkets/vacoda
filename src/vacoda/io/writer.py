"""
This module provides functions for writing data to various file formats.
"""
from pathlib import Path
import pandas as pd

from vacoda.utils.logger import get_logger, rel_path

logger = get_logger("writer")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """
    Writes a DataFrame to a Parquet file. Creates parent directories if needed.
    Prompts before overwriting an existing file.
    """
    path = Path(path)

    if df.empty:
        raise ValueError("DataFrame is empty. Nothing to write.")

    if path.exists():
        prompt = f"File already exists at: {rel_path(path)}\nOverwrite? [y/N]: "
    else:
        prompt = f"Write output parquet to: {rel_path(path)}\nProceed? [y/N]: "

    answer = input(prompt).strip().lower()
    if answer != "y":
        logger.info("Skipping write.")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing DataFrame to parquet at: {rel_path(path)}")
    df.to_parquet(path, index=False)
    logger.info(f"File written successfully at: {rel_path(path)}")
    return True
