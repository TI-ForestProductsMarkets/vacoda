"""
vacoda.io.loaders

Data loading utilities for the vacoda pipeline.

This module loads files. It does not parse YAML directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacoda.config.paths import get_data_dir, get_log_dir, get_metadata_paths
from vacoda.config.pipeline import (
    get_block_files,
    get_block_stage,
    get_block_subdir,
    get_input_block,
    get_output_block,
    get_pipeline_mode,
    get_logging_file,
)
from vacoda.utils.logger import get_logger, rel_path


logger = get_logger("IO.LOADERS")


def resolve_pipeline_data_dir(block: dict) -> Path:
    """
    Resolve a pipeline input/output block to a physical data directory.

    Example:
        {"stage": "raw", "subdir": "api/single"}
        -> /storage/vacoda/data/raw/api/single
    """
    stage = get_block_stage(block)
    subdir = get_block_subdir(block)

    base_dir = get_data_dir(stage)
    return base_dir / subdir if subdir else base_dir


def resolve_pipeline_files(
    block: dict,
    default: str | list[str] | None = None,
) -> list[Path]:
    """
    Resolve files or glob patterns from a pipeline block.
    """
    base_dir = resolve_pipeline_data_dir(block)
    patterns = get_block_files(block, default=default)

    files: list[Path] = []

    for pattern in patterns:
        matches = sorted(base_dir.glob(pattern))

        if matches:
            files.extend(matches)
        else:
            files.append(base_dir / pattern)

    unique: dict[str, Path] = {}
    for path in files:
        unique[str(path)] = path

    return list(unique.values())


def get_input_files(existing_only: bool = True) -> list[Path]:
    """
    Return resolved input files from pipeline.yaml. 
    By default, only existing files are returned.
    """
    block = get_input_block()
    files = resolve_pipeline_files(block, default=["*.csv"])

    if existing_only:
        missing = [path for path in files if not path.exists()]
        if missing:
            msg = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"Configured input file(s) not found:\n{msg}")

    return files


def get_output_files(existing_only: bool = False) -> list[Path]:
    """Return resolved output files from pipeline.yaml."""
    block = get_output_block()
    files = resolve_pipeline_files(block, default=[])

    if existing_only:
        files = [path for path in files if path.exists()]

    return files


def get_pipeline_log_file() -> Path:
    """Return full pipeline log file path."""
    return get_log_dir() / get_logging_file()


def read_csv_with_diagnostics(
    path: Path | str,
    *,
    usecols=None,
    dtype: dict | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Read a CSV file with optional pandas read_csv arguments and basic diagnostics.

    Useful for both full pipeline loading and lightweight analyses that only need
    a subset of columns.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        **read_csv_kwargs,
    )

    logger.debug(
        f"[CSV Load] {rel_path(path)}: "
        f"{df.shape[0]:,} rows × {df.shape[1]} cols"
    )

    if len(df.columns) <= 1:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            preview = f.read(300)

        logger.error(
            f"[CSV Loader] Failed to parse {path.name}. "
            f"Only one column found. Check delimiter/file format."
        )
        logger.error(f"[CSV Loader] File preview:\n{preview}")

        raise ValueError(
            f"CSV parsing failed for {path}. Only one column found. "
            f"Check delimiter or file format."
        )

    return df

def _read_csv_with_diagnostics(path: Path) -> pd.DataFrame:
    """Read CSV and log basic diagnostics."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    logger.info(f"[Data Load] {rel_path(path)}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.debug(f"[Data Load] First columns: {list(df.columns)[:8]}")

    if len(df.columns) <= 1:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            preview = f.read(300)

        logger.error(
            f"[CSV Loader] Failed to parse {path.name}. "
            f"Only one column found. Check delimiter/file format."
        )
        logger.error(f"[CSV Loader] File preview:\n{preview}")

        raise ValueError(
            f"CSV parsing failed for {path}. Only one column found. "
            f"Check delimiter or file format."
        )

    return df


def get_raw_csv_file_list() -> list[Path]:
    """Return configured input CSV files."""
    return get_input_files(existing_only=True)


def load_data_from_config(file_path: Path | str | None = None) -> pd.DataFrame:
    """
    Load input CSV data according to pipeline.yaml.

    Modes:
    - per_file: load first configured CSV
    - merged: load all configured CSVs and concatenate
    """
    if file_path is not None:
        path = Path(file_path)
        logger.info(f"[Config] Loading explicit file: {rel_path(path)}")
        return _read_csv_with_diagnostics(path)

    mode = get_pipeline_mode()
    files = get_input_files(existing_only=True)

    if not files:
        raise FileNotFoundError("No input files found from pipeline.yaml configuration.")

    if mode == "per_file":
        path = files[0]
        logger.info(f"[Config] Loading single file for mode=per_file: {rel_path(path)}")
        return _read_csv_with_diagnostics(path)

    if mode == "merged":
        logger.info(f"[Config] Loading {len(files)} file(s) for mode=merged")

        frames = []
        for path in files:
            logger.debug(f"[Config] Loading: {rel_path(path)}")
            frames.append(_read_csv_with_diagnostics(path))

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            f"[Config] Combined shape: {combined.shape[0]:,} rows × {combined.shape[1]} cols"
        )
        return combined

    raise ValueError(f"Unknown pipeline mode: {mode}")


def load_metadata_from_config() -> dict[str, pd.DataFrame]:
    """Load metadata parquet files specified in paths.yaml."""
    from vacoda.preprocessing.metadata import standardize_metadata_dtypes

    metadata_paths = get_metadata_paths()
    metadata: dict[str, pd.DataFrame] = {}

    for key, path in metadata_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found for '{key}': {path}")

        logger.info(f"Loading metadata '{key}' from: {rel_path(path)}")
        df = pd.read_parquet(path)
        metadata[key] = standardize_metadata_dtypes(df)

    return metadata


def get_processed_parquet_file_list() -> list[Path]:
    """Return existing configured output parquet files."""
    return get_output_files(existing_only=True)


def load_processed_trade_data(file_path: Path | str | None = None) -> pd.DataFrame:
    """
    Load processed trade data parquet.

    Priority:
    1. explicit file_path
    2. first configured existing output parquet
    """
    if file_path is not None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Processed parquet file not found: {path}")

        logger.info(f"Loading processed trade data from: {rel_path(path)}")
        return pd.read_parquet(path)

    files = get_processed_parquet_file_list()

    if not files:
        configured = get_output_files(existing_only=False)
        configured_msg = "\n".join(str(path) for path in configured)
        raise FileNotFoundError(
            "No configured processed parquet file exists yet. "
            "Configured output path(s):\n"
            f"{configured_msg}"
        )

    path = files[0]
    logger.info(f"Loading processed trade data from configured file: {rel_path(path)}")
    return pd.read_parquet(path)