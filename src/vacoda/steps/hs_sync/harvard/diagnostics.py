"""
vacoda.hs_sync.diagnostics
Functions to diagnose the results of HS synchronization, checking coverage, value preservation, and weight preservation.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from vacoda.config.paths import rel_path
from vacoda.utils.logger import get_logger
from vacoda.io.duckdb_scan import duckdb_scan_relation
from vacoda.utils.display import progress_bar

logger = get_logger("HS_SYNC.COVERAGE")


def build_trade_source_pair_summary(
    *,
    data_dir: Path,
    pattern: str,
    input_format: str,
    engine: str,
    year_col: str,
    hs_col: str,
    cmd_col: str,
    value_col: str,
    min_year: int | None = None,
    threads: int = 16,
    show_progress: bool = True,
) -> pd.DataFrame:
    engine = engine.lower()

    if engine == "duckdb":
        return build_trade_source_pair_summary_duckdb(
            data_dir=data_dir,
            pattern=pattern,
            input_format=input_format,
            year_col=year_col,
            hs_col=hs_col,
            cmd_col=cmd_col,
            value_col=value_col,
            min_year=min_year,
            threads=threads,
        )

    if engine == "pandas":
        return build_trade_source_pair_summary_pandas(
            data_dir=data_dir,
            pattern=pattern,
            input_format=input_format,
            year_col=year_col,
            hs_col=hs_col,
            cmd_col=cmd_col,
            value_col=value_col,
            min_year=min_year,
            show_progress=show_progress,
        )

    raise ValueError(f"Unsupported HS-sync engine: {engine!r}")


    
def build_trade_source_pair_summary_pandas(
    *,
    data_dir: Path,
    pattern: str,
    input_format: str,
    year_col: str,
    hs_col: str,
    cmd_col: str,
    value_col: str,
    min_year: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:

    files = sorted(data_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No input files found in {rel_path(data_dir)} matching {pattern!r}"
        )

    if input_format != "csv":
        raise ValueError("Pandas source-pair summary currently supports only csv input.")

    parts: list[pd.DataFrame] = []

    for path in progress_bar(
        files,
        total=len(files),
        desc="HS-sync source pairs",
        unit="file",
        enabled=show_progress,
    ):
        df = pd.read_csv(
            path,
            usecols=[year_col, hs_col, cmd_col, value_col],
            dtype={
                year_col: "Int64",
                hs_col: "string",
                cmd_col: "string",
                value_col: "float64",
            },
        )

        if min_year is not None:
            df = df.loc[df[year_col] >= min_year]

        if df.empty:
            continue

        part = (
            df.groupby([hs_col, cmd_col], dropna=False, as_index=False)
            .agg(
                n_trade_rows=(cmd_col, "size"),
                first_year=(year_col, "min"),
                last_year=(year_col, "max"),
                trade_value_total=(value_col, "sum"),
            )
            .rename(
                columns={
                    hs_col: "source_hs",
                    cmd_col: "source_cmd",
                }
            )
        )

        parts.append(part)

    if not parts:
        return pd.DataFrame(
            columns=[
                "source_hs",
                "source_cmd",
                "n_trade_rows",
                "first_year",
                "last_year",
                "trade_value_total",
            ]
        )

    out = pd.concat(parts, ignore_index=True)

    out = (
        out.groupby(["source_hs", "source_cmd"], dropna=False, as_index=False)
        .agg(
            n_trade_rows=("n_trade_rows", "sum"),
            first_year=("first_year", "min"),
            last_year=("last_year", "max"),
            trade_value_total=("trade_value_total", "sum"),
        )
    )

    return out

def build_trade_source_pair_summary_duckdb(
    *,
    data_dir: Path,
    pattern: str,
    input_format: str,
    year_col: str,
    hs_col: str,
    cmd_col: str,
    value_col: str,
    min_year: int | None = None,
    threads: int = 16,
) -> pd.DataFrame:
    """
    Build compact source HS/code summary from staged or raw Comtrade trade files.

    pattern:
        File pattern, e.g. "*.parquet" or "*.csv".

    input_format:
        Reader format, e.g. "parquet" or "csv".
    """

    file_glob = str(data_dir / pattern)

    logger.info("Building source-pair summary with DuckDB.")
    logger.info("Input glob: %s", file_glob)
    logger.info("Input format: %s", input_format)
    logger.info("Threads: %s", threads)

    where_clause = ""
    if min_year is not None:
        where_clause = f'WHERE TRY_CAST(src."{year_col}" AS INTEGER) >= {int(min_year)}'

    scan = duckdb_scan_relation(
        path=file_glob,
        file_format=input_format,
        alias="src",
    )

    query = f"""
        SELECT
            CAST(src."{hs_col}" AS VARCHAR) AS source_hs,
            lpad(CAST(src."{cmd_col}" AS VARCHAR), 6, '0') AS source_cmd,
            COUNT(*) AS n_trade_rows,
            MIN(TRY_CAST(src."{year_col}" AS INTEGER)) AS first_year,
            MAX(TRY_CAST(src."{year_col}" AS INTEGER)) AS last_year,
            SUM(TRY_CAST(src."{value_col}" AS DOUBLE)) AS trade_value_total
        FROM {scan}
        {where_clause}
        GROUP BY
            source_hs,
            source_cmd
        ORDER BY
            source_hs,
            source_cmd
    """

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")
        out = con.execute(query).df()

    logger.info("Source-pair summary shape: %s", out.shape)

    return out



def summarize_hs_concordance_coverage(coverage: pd.DataFrame) -> dict:
    """
    Summarize the output of build_hs_concordance_coverage().
    """

    summary = {
        "n_source_pairs": len(coverage),
        "n_source_hs_systems": coverage["source_hs"].nunique(dropna=True),
        "source_hs_systems": sorted(coverage["source_hs"].dropna().unique().tolist()),
        "n_covered_pairs": int(coverage["has_mapping"].sum()),
        "n_missing_pairs": int((~coverage["has_mapping"]).sum()),
        "n_one_to_one": int((coverage["mapping_type"] == "one_to_one").sum()),
        "n_one_to_many": int((coverage["mapping_type"] == "one_to_many").sum()),
        "n_value_splits": int(coverage["produces_value_split"].sum()),
        "mapping_type_counts": coverage["mapping_type"].value_counts().to_dict(),
        "weight_sum_status_counts": coverage["weight_sum_status"].value_counts().to_dict(),
    }

    if "trade_value_total" in coverage.columns:
        total_value = coverage["trade_value_total"].sum()
        missing_value = coverage.loc[
            ~coverage["has_mapping"],
            "trade_value_total",
        ].sum()

        summary["trade_value_total"] = float(total_value)
        summary["trade_value_missing_mapping"] = float(missing_value)

        if total_value != 0:
            summary["share_trade_value_missing_mapping"] = float(
                missing_value / total_value
            )
        else:
            summary["share_trade_value_missing_mapping"] = None

    return summary

def build_mapping_coverage(
    trade_df: pd.DataFrame,
    concordance_df: pd.DataFrame,
    target_hs: str = "H5",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    value_col: str = "PrimaryValue") -> pd.DataFrame:
    """
    Check which source HS/code pairs in the trade data can be mapped
    to the chosen target HS system.
    """

    source_pairs = (
        trade_df
        .groupby([hs_col, cmd_col], dropna=False, as_index=False)
        .agg(
            n_rows=(cmd_col, "size"),
            value_total=(value_col, "sum"),
        )
    )

    conc_target = concordance_df.loc[
        concordance_df["target_hs"] == target_hs
    ].copy()

    map_summary = (
        conc_target
        .groupby(["source_hs", "source_cmd"], dropna=False, as_index=False)
        .agg(
            n_target_codes=("target_cmd", "nunique"),
            weight_sum=("weight", "sum"),
        )
    )

    out = source_pairs.merge(
        map_summary,
        left_on=[hs_col, cmd_col],
        right_on=["source_hs", "source_cmd"],
        how="left",
    )

    out["has_mapping"] = out["n_target_codes"].notna()

    return out.sort_values(
        ["has_mapping", hs_col, "value_total"],
        ascending=[True, True, False],
    )

def build_trade_source_pair_summary_parquet_duckdb(
    *,
    data_dir: Path,
    pattern: str,
    year_col: str,
    hs_col: str,
    cmd_col: str,
    value_col: str,
    min_year: int | None = None,
    threads: int = 16,
) -> pd.DataFrame:
    file_glob = str(data_dir / pattern)

    logger.info("Building source-pair summary from Parquet with DuckDB.")
    logger.info("Input glob: %s", file_glob)
    logger.info("Threads: %s", threads)

    where_clause = ""
    if min_year is not None:
        where_clause = f'WHERE "{year_col}" >= {int(min_year)}'

    query = f"""
        SELECT
            CAST("{hs_col}" AS VARCHAR) AS source_hs,
            CAST("{cmd_col}" AS VARCHAR) AS source_cmd,
            COUNT(*) AS n_trade_rows,
            MIN(CAST("{year_col}" AS INTEGER)) AS first_year,
            MAX(CAST("{year_col}" AS INTEGER)) AS last_year,
            SUM(CAST("{value_col}" AS DOUBLE)) AS trade_value_total
        FROM read_parquet(
            '{file_glob}',
            union_by_name = true
        )
        {where_clause}
        GROUP BY
            source_hs,
            source_cmd
        ORDER BY
            source_hs,
            source_cmd
    """

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")
        out = con.execute(query).df()

    logger.info("Source-pair summary shape: %s", out.shape)

    return out
    
def summarize_hs_sync_coverage():
    pass
    
def check_value_preservation():
    pass

def check_weight_preservation():
    pass

def compare_before_after_totals():
    pass