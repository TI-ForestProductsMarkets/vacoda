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
    overwrite: bool = True,
    create_parent: bool = True,
    allow_empty: bool = True,
) -> Path:
    """
    Write a DataFrame to a Parquet file.

    Empty DataFrames are allowed only when allow_empty=True.
    A completely empty schema, meaning 0 rows and 0 columns, is always rejected.
    """
    if df.shape[1] == 0:
        raise ValueError(
            "DataFrame has no columns. Refusing to write schema-less parquet."
        )

    if df.empty and not allow_empty:
        raise ValueError("DataFrame is empty. Nothing to write.")

    path = _prepare_output_path(
        path,
        overwrite=overwrite,
        create_parent=create_parent,
    )

    if df.empty:
        logger.warning(
            "Writing empty parquet with schema only: %s "
            "(%s rows × %s cols)",
            rel_path(path),
            f"{df.shape[0]:,}",
            f"{df.shape[1]:,}",
        )
    else:
        logger.info(
            "Writing parquet: %s (%s rows × %s cols)",
            rel_path(path),
            f"{df.shape[0]:,}",
            f"{df.shape[1]:,}",
        )

    df.to_parquet(path, index=False)

    logger.info("Parquet written successfully: %s", rel_path(path))
    return path


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = True,
    create_parent: bool = True,
    allow_empty: bool = False,
    index: bool = False,
) -> Path:
    """
    Write a DataFrame to a CSV file.

    Empty DataFrames are allowed only when allow_empty=True.
    A completely empty schema, meaning 0 rows and 0 columns, is always rejected.
    """
    if df.shape[1] == 0:
        raise ValueError(
            "DataFrame has no columns. Refusing to write schema-less CSV."
        )

    if df.empty and not allow_empty:
        raise ValueError("DataFrame is empty. Nothing to write.")

    path = _prepare_output_path(
        path,
        overwrite=overwrite,
        create_parent=create_parent,
    )

    if df.empty:
        logger.warning(
            "Writing empty CSV with headers only: %s "
            "(%s rows × %s cols)",
            rel_path(path),
            f"{df.shape[0]:,}",
            f"{df.shape[1]:,}",
        )
    else:
        logger.info(
            "Writing CSV: %s (%s rows × %s cols)",
            rel_path(path),
            f"{df.shape[0]:,}",
            f"{df.shape[1]:,}",
        )

    df.to_csv(path, index=index)

    logger.info("CSV written successfully: %s", rel_path(path))
    return path

def write_parquet_interactive(
    df: pd.DataFrame,
    path: str | Path,
    *,
    allow_empty: bool = True,
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