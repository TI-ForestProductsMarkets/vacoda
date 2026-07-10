"""
Coverage engines for building year × HS classification counts.
"""

from pathlib import Path
import duckdb
import pandas as pd
import polars as pl
from vacoda.utils.progress import progress_bar
from vacoda.io.duckdb_scan import duckdb_scan_relation

def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def _quote_ident(name: str) -> str:
    """Quote a SQL identifier for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def build_year_hs_counts_duckdb(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    input_format: str,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    threads: int = 16,
) -> pd.DataFrame:
    """
    Build year × HS-system counts using DuckDB.
    """
    data_dir = Path(data_dir)
    #input_format = cfg.coverage.input.format

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    glob_path = str(data_dir / pattern)
    year_id = _quote_ident(year_col)
    hs_id = _quote_ident(hs_col)

    scan = duckdb_scan_relation(
        path=glob_path,
        file_format=input_format,
        alias = "src")

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(threads)}")

    query = f"""
        SELECT
            CAST({year_id} AS INTEGER) AS {year_id},
            CAST({hs_id} AS VARCHAR) AS {hs_id},
            COUNT(*) AS n_rows
        FROM {scan}
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

    for path in progress_bar(files, enabled=show_progress):
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


def build_reporter_year_hs_coverage_duckdb(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    threads: int = 16,
    columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build reporter × year × HS classification coverage using DuckDB.

    The `columns` dict maps logical names to source column names.

    Expected logical keys:
        reporter_code
        reporter_iso
        reporter_desc
        year
        hs
        hs_search
        original
        cmd
        flow
        value
        netweight
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    columns = columns or {}

    reporter_code_col = columns.get("reporter_code", "ReporterCode")
    reporter_iso_col = columns.get("reporter_iso", "ReporterISO")
    reporter_desc_col = columns.get("reporter_desc", "ReporterDesc")
    year_col = columns.get("year", "RefYear")
    hs_col = columns.get("hs", "ClassificationCode")
    hs_search_col = columns.get("hs_search", "ClassificationSearchCode")
    original_col = columns.get("original", "IsOriginalClassification")
    cmd_col = columns.get("cmd", "CmdCode")
    flow_col = columns.get("flow", "FlowCode")
    value_col = columns.get("value", "PrimaryValue")
    netweight_col = columns.get("netweight", "NetWgt")

    glob_path = str(data_dir / pattern)

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(threads)}")

    query = f"""
        WITH base AS (
            SELECT
                CAST({_quote_ident(reporter_code_col)} AS INTEGER) AS ReporterCode,
                CAST({_quote_ident(reporter_iso_col)} AS VARCHAR) AS ReporterISO,
                CAST({_quote_ident(reporter_desc_col)} AS VARCHAR) AS ReporterDesc,
                CAST({_quote_ident(year_col)} AS INTEGER) AS RefYear,
                CAST({_quote_ident(hs_col)} AS VARCHAR) AS ClassificationCode,
                CAST({_quote_ident(hs_search_col)} AS VARCHAR) AS ClassificationSearchCode,
                CAST({_quote_ident(original_col)} AS BOOLEAN) AS IsOriginalClassification,
                CAST({_quote_ident(cmd_col)} AS VARCHAR) AS CmdCode,
                CAST({_quote_ident(flow_col)} AS VARCHAR) AS FlowCode,
                CAST({_quote_ident(value_col)} AS DOUBLE) AS PrimaryValue,
                CAST({_quote_ident(netweight_col)} AS DOUBLE) AS NetWgt
            FROM read_csv_auto(
                ?,
                union_by_name=true,
                filename=true
            )
        ),

        grouped AS (
            SELECT
                ReporterCode,
                ReporterISO,
                ReporterDesc,
                RefYear,
                ClassificationCode,
                ClassificationSearchCode,
                IsOriginalClassification,
                COUNT(*) AS n_rows,
                COUNT(DISTINCT CmdCode) AS n_cmd_codes,
                COUNT(DISTINCT FlowCode) AS n_flows,
                SUM(PrimaryValue) AS primary_value_total,
                SUM(NetWgt) AS netwgt_total
            FROM base
            GROUP BY
                ReporterCode,
                ReporterISO,
                ReporterDesc,
                RefYear,
                ClassificationCode,
                ClassificationSearchCode,
                IsOriginalClassification
        )

        SELECT
            *,
            n_rows * 1.0
            / SUM(n_rows) OVER (
                PARTITION BY ReporterCode, RefYear
            ) AS share_rows_in_reporter_year
        FROM grouped
        ORDER BY
            ReporterCode,
            RefYear,
            ClassificationCode
    """

    try:
        return con.execute(query, [glob_path]).df()
    finally:
        con.close()