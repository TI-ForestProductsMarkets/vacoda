"""
vacoda.io.readers

Generic file-reading utilities.
This module does not parse YAML and does not know about pipeline configs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacoda.utils.logger import get_logger, rel_path


logger = get_logger("IO.READERS")


def read_csv_with_diagnostics(
    path: Path | str,
    *,
    usecols=None,
    dtype: dict | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read a CSV file with basic diagnostics."""
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
        "[CSV Load] %s: %s rows × %s cols",
        rel_path(path),
        f"{df.shape[0]:,}",
        f"{df.shape[1]:,}",
    )

    if len(df.columns) <= 1:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            preview = f.read(300)

        logger.error(
            "[CSV Loader] Failed to parse %s. Only one column found. "
            "Check delimiter/file format.",
            path.name,
        )
        logger.error("[CSV Loader] File preview:\n%s", preview)

        raise ValueError(
            f"CSV parsing failed for {path}. Only one column found. "
            "Check delimiter or file format."
        )

    return df


def read_parquet_with_diagnostics(path: Path | str) -> pd.DataFrame:
    """Read a parquet file with basic diagnostics."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    logger.debug(
        "[Parquet Load] %s: %s rows × %s cols",
        rel_path(path),
        f"{df.shape[0]:,}",
        f"{df.shape[1]:,}",
    )

    return df