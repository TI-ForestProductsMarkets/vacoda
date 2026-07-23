"""
vacoda.hs_sync.concordance

Functions to load, normalize, and validate HS concordance tables
for HS synchronization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vacoda.config.hs_sync import get_concordance_columns
from vacoda.utils.logger import get_logger
from vacoda.config.paths import rel_path

logger = get_logger("HS_SYNC.CONCORDANCE")


STANDARD_CONCORDANCE_COLUMNS = [
    "source_hs",
    "source_cmd",
    "target_hs",
    "target_cmd",
    "weight",
]

REQUIRED_STANDARD_CONCORDANCE_COLUMNS = set(STANDARD_CONCORDANCE_COLUMNS)


def load_hs_concordance(
    path: str | Path,
    *,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load an HS concordance / conversion-weight table.

    Supported file types:
        .parquet
        .csv

    If cfg is provided, column names are normalized using:

        concordance.columns.source_hs
        concordance.columns.source_cmd
        concordance.columns.target_hs
        concordance.columns.target_cmd
        concordance.columns.weight

    Returns
    -------
    pd.DataFrame
        Concordance table with standard columns:
            source_hs
            source_cmd
            target_hs
            target_cmd
            weight
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Concordance file not found: {rel_path(path)}")

    logger.info("Loading HS concordance: %s", rel_path(path))

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported concordance file format {suffix!r}. "
            "Use .parquet or .csv."
        )

    logger.info("Raw concordance shape: %s", df.shape)

    df = normalize_hs_concordance_columns(df, cfg=cfg)
    validate_hs_concordance_schema(df)

    logger.info("Normalized concordance shape: %s", df.shape)

    return df


def normalize_hs_concordance_columns(
    df: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Normalize concordance columns to standard names.

    Standard output columns:
        source_hs
        source_cmd
        target_hs
        target_cmd
        weight
    """

    out = df.copy()

    if cfg is not None:
        cols = get_concordance_columns(cfg)

        rename_map = {
            cols["source_hs"]: "source_hs",
            cols["source_cmd"]: "source_cmd",
            cols["target_hs"]: "target_hs",
            cols["target_cmd"]: "target_cmd",
            cols["weight"]: "weight",
        }

        # Only rename columns that actually exist.
        rename_map = {
            old: new
            for old, new in rename_map.items()
            if old in out.columns and old != new
        }

        out = out.rename(columns=rename_map)

    missing = REQUIRED_STANDARD_CONCORDANCE_COLUMNS - set(out.columns)
    if missing:
        raise ValueError(
            "Concordance table is missing required columns after normalization: "
            f"{sorted(missing)}. Available columns: {list(out.columns)}"
        )

    out = out.loc[:, STANDARD_CONCORDANCE_COLUMNS].copy()

    out["source_hs"] = out["source_hs"].astype("string").str.strip().str.upper()
    out["target_hs"] = out["target_hs"].astype("string").str.strip().str.upper()

    out["source_cmd"] = (
        out["source_cmd"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    out["target_cmd"] = (
        out["target_cmd"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")

    return out


def validate_hs_concordance_schema(df: pd.DataFrame) -> None:
    """
    Validate normalized HS concordance schema.
    """

    missing = REQUIRED_STANDARD_CONCORDANCE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Concordance table is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Concordance table is empty.")

    null_counts = df[list(REQUIRED_STANDARD_CONCORDANCE_COLUMNS)].isna().sum()
    bad_nulls = null_counts[null_counts > 0]

    if not bad_nulls.empty:
        raise ValueError(
            "Concordance table contains missing values in required columns: "
            f"{bad_nulls.to_dict()}"
        )

    if (df["weight"] < 0).any():
        raise ValueError("Concordance table contains negative weights.")

    if not pd.api.types.is_numeric_dtype(df["weight"]):
        raise TypeError("Concordance column 'weight' must be numeric.")

    duplicated = df.duplicated(
        subset=["source_hs", "source_cmd", "target_hs", "target_cmd"],
        keep=False,
    )

    if duplicated.any():
        n = int(duplicated.sum())
        raise ValueError(
            "Concordance table contains duplicate mapping rows for "
            "source_hs/source_cmd/target_hs/target_cmd. "
            f"Duplicate rows: {n}"
        )