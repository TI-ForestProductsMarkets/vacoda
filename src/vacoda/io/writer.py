"""
vacoda.io.writer

Utilities for writing DataFrames and other tabular outputs to disk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacoda.utils.logger import get_logger, rel_path


logger = get_logger("IO.WRITER")


def _prepare_output_path(
    path: str | Path,
    *,
    overwrite: bool = False,
    create_parent: bool = True,
) -> Path:
    """
    Validate and prepare an output path before writing.

    Parameters
    ----------
    path:
        Target output path.
    overwrite:
        If False, raise FileExistsError when path already exists.
    create_parent:
        If True, create parent directories.

    Returns
    -------
    Path
        Normalized Path object.
    """
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {rel_path(path)}. "
            "Pass overwrite=True to replace it."
        )

    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = False,
    create_parent: bool = True,
    allow_empty: bool = False,
) -> Path:
    """
    Write a DataFrame to a Parquet file.

    Parameters
    ----------
    df:
        DataFrame to write.
    path:
        Output parquet path.
    overwrite:
        If True, overwrite existing file.
    create_parent:
        If True, create parent directories.
    allow_empty:
        If False, raise ValueError for empty DataFrames.

    Returns
    -------
    Path
        Path to the written file.
    """
    if df.empty and not allow_empty:
        raise ValueError("DataFrame is empty. Nothing to write.")

    path = _prepare_output_path(
        path,
        overwrite=overwrite,
        create_parent=create_parent,
    )

    logger.info(
        f"Writing parquet: {rel_path(path)} "
        f"({df.shape[0]:,} rows × {df.shape[1]:,} cols)"
    )

    df.to_parquet(path, index=False)

    logger.info(f"Parquet written successfully: {rel_path(path)}")
    return path


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = False,
    create_parent: bool = True,
    allow_empty: bool = False,
    index: bool = False,
) -> Path:
    """
    Write a DataFrame to a CSV file.

    Parameters
    ----------
    df:
        DataFrame to write.
    path:
        Output CSV path.
    overwrite:
        If True, overwrite existing file.
    create_parent:
        If True, create parent directories.
    allow_empty:
        If False, raise ValueError for empty DataFrames.
    index:
        If True, write row indices.

    Returns
    -------
    Path
        Path to the written file.
    """
    if df.empty and not allow_empty:
        raise ValueError("DataFrame is empty. Nothing to write.")

    path = _prepare_output_path(
        path,
        overwrite=overwrite,
        create_parent=create_parent,
    )

    logger.info(
        f"Writing CSV: {rel_path(path)} "
        f"({df.shape[0]:,} rows × {df.shape[1]:,} cols)"
    )

    df.to_csv(path, index=index)

    logger.info(f"CSV written successfully: {rel_path(path)}")
    return path

def write_parquet_interactive(
    df: pd.DataFrame,
    path: str | Path,
    *,
    allow_empty: bool = False,
) -> Path | None:
    """
    Interactively ask before writing parquet.

    Useful for notebooks/manual exploration.
    Avoid this in automated pipelines.
    """
    path = Path(path)

    if path.exists():
        prompt = f"File already exists at: {rel_path(path)}\nOverwrite? [y/N]: "
    else:
        prompt = f"Write output parquet to: {rel_path(path)}\nProceed? [y/N]: "

    answer = input(prompt).strip().lower()

    if answer != "y":
        logger.info("Skipping write.")
        return None

    return write_parquet(
        df,
        path,
        overwrite=True,
        allow_empty=allow_empty,
    )