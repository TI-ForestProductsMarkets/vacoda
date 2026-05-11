"""
Coverage engines for building year × HS classification counts.
"""

from pathlib import Path
from typing import Iterable
import duckdb
import pandas as pd
import polars as pl
from vacoda.utils.progress import progress_bar

def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def _quote_ident(name: str) -> str:
    """Quote a SQL identifier for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def build_year_hs_counts_duckdb(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    threads: int = 16,
) -> pd.DataFrame:
    """
    Build year × HS-system counts using DuckDB.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    glob_path = str(data_dir / pattern)
    year_id = _quote_ident(year_col)
    hs_id = _quote_ident(hs_col)

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(threads)}")

    query = f"""
        SELECT
            CAST({year_id} AS INTEGER) AS {year_id},
            CAST({hs_id} AS VARCHAR) AS {hs_id},
            COUNT(*) AS n_rows
        FROM read_csv_auto(
            ?,
            union_by_name=true,
            filename=true
        )
        GROUP BY 1, 2
        ORDER BY 1, 2
    """

    try:
        return con.execute(query, [glob_path]).df()
    finally:
        con.close()


def build_year_hs_counts_polars(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
) -> pd.DataFrame:
    """
    Build year × HS-system counts using Polars lazy scanning.
    """
    data_dir = Path(data_dir)
    glob_path = str(data_dir / pattern)

    result = (
        pl.scan_csv(
            glob_path,
            infer_schema_length=1000,
        )
        .select([
            pl.col(year_col).cast(pl.Int32, strict=False),
            pl.col(hs_col).cast(pl.Utf8, strict=False),
        ])
        .group_by([year_col, hs_col])
        .len()
        .rename({"len": "n_rows"})
        .sort([year_col, hs_col])
        .collect()
    )

    return result.to_pandas()


def _find_csv_files(data_dir: str | Path, pattern: str = "*.csv") -> list[Path]:
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = sorted(data_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {data_dir}")

    return files


def build_year_hs_counts_pandas(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    progress_every: int = 1000,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Build year × HS-system counts using pandas.

    This is the transparent/debug backend.
    """
    files = _find_csv_files(data_dir, pattern=pattern)

    parts: list[pd.DataFrame] = []
    errors: list[tuple[str, str, str]] = []

    for path in progress_bar(files, every=progress_every, disable=not show_progress):
        try:
            df = pd.read_csv(
                path,
                usecols=[year_col, hs_col],
                dtype={
                    year_col: "Int64",
                    hs_col: "string",
                },
            )

            if df.empty:
                continue

            grouped = (
                df.groupby([year_col, hs_col], dropna=False)
                .size()
                .reset_index(name="n_rows")
            )

            parts.append(grouped)

        except Exception as exc:
            errors.append((str(path), type(exc).__name__, str(exc)[:300])) 

    if not parts:
        raise ValueError("No non-empty data found while building year-HS counts.")

    year_hs = pd.concat(parts, ignore_index=True)

    result = (
        year_hs.groupby([year_col, hs_col], dropna=False)["n_rows"]
        .sum()
        .reset_index()
        .sort_values([year_col, hs_col])
        .reset_index(drop=True)
    )

    if errors:
        print(f"WARNING: {len(errors)} file(s) failed while building counts.")
        for error in errors[:10]:
            print(error)

    return result


def build_year_hs_counts_pandas_deprecated(
    files: Iterable[str | Path],
    *,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    progress_every: int = 1000) -> pd.DataFrame:
    """
    Build long-format year × HS-classification row counts.

    Output columns:
        RefYear | ClassificationCode | n_rows

    Parameters
    ----------
    files:
        Iterable of CSV file paths.
    year_col:
        Column containing the reference year.
    hs_col:
        Column containing the HS classification code, e.g. H0, H1, H2.
    progress_every:
        Print progress every N files. Set to 0 or None to disable.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per year × HS classification.
    """
    files = [Path(p) for p in files]
    parts: list[pd.DataFrame] = []
    errors: list[tuple[str, str, str]] = []

    for i, path in enumerate(files, 1):
        try:
            df = pd.read_csv(
                path,
                usecols=[year_col, hs_col],
                dtype={
                    year_col: "int16",
                    hs_col: "string",
                },
            )

            if df.empty:
                continue

            grouped = (
                df.groupby([year_col, hs_col], dropna=False)
                .size()
                .reset_index(name="n_rows")
            )

            parts.append(grouped)

        except Exception as exc:
            errors.append((str(path), type(exc).__name__, str(exc)[:300]))

        if progress_every and i % progress_every == 0:
            print(f"processed {i}/{len(files)} files")

    if not parts:
        raise ValueError("No non-empty data found while building year-HS counts.")

    year_hs = pd.concat(parts, ignore_index=True)

    result = (
        year_hs.groupby([year_col, hs_col], dropna=False)["n_rows"]
        .sum()
        .reset_index()
        .sort_values([year_col, hs_col])
        .reset_index(drop=True)
    )

    if errors:
        print(f"WARNING: {len(errors)} file(s) failed while building counts.")
        for error in errors[:10]:
            print(error)

    return result