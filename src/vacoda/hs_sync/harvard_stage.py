"""
vacoda.hs_sync.harvard_stage

Stage vacoda raw Comtrade CSV files into the folder layout expected by the
Harvard/Growth Lab conversion-weights generator.

Input:
    /storage/vacoda/data/raw/api/single/*.csv

Output:
    /storage/vacoda/data/external/harvard_comtrade/as_reported/raw_parquet/
    ├── H0/1992/*.parquet
    ├── H1/1996/*.parquet
    └── ...
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from vacoda.utils.logger import get_logger, rel_path

logger = get_logger("HS_SYNC.HARVARD_STAGE")

VALID_HS_SYSTEMS = {"H0", "H1", "H2", "H3", "H4", "H5", "H6"}

def audit_csv_to_harvard_parquet_staging(
    *,
    input_dir: Path,
    input_pattern: str,
    output_root: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    value_col: str = "PrimaryValue",
    threads: int = 16,
    value_abs_tolerance: float = 1.0,
    value_rel_tolerance: float = 1e-10,) -> pd.DataFrame:
    """
    Compare raw CSV input against staged Harvard-compatible Parquet output.

    The audit compares row counts and value totals by HS system and year.

    Output columns include:
        hs
        year
        n_rows_csv
        n_rows_parquet
        row_diff
        value_total_csv
        value_total_parquet
        value_diff
        status

    A status of 'ok' means row counts match and value totals are close.
    """

    input_glob = str(input_dir / input_pattern)

    parquet_glob = str(
        output_root
        / "as_reported"
        / "raw_parquet"
        / "*"
        / "*"
        / "*.parquet"
    )

    logger.info("Auditing CSV → Parquet staging.")
    logger.info("CSV input glob: %s", input_glob)
    logger.info("Parquet glob: %s", parquet_glob)

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        csv_summary = con.execute(
            f"""
            SELECT
                CAST("{hs_col}" AS VARCHAR) AS hs,
                CAST("{year_col}" AS INTEGER) AS year,
                COUNT(*) AS n_rows_csv,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total_csv
            FROM read_csv_auto(
                '{input_glob}',
                union_by_name = true,
                filename = true
            )
            WHERE "{hs_col}" IS NOT NULL
              AND "{year_col}" IS NOT NULL
            GROUP BY hs, year
            """
        ).df()

        parquet_summary = con.execute(
            f"""
            SELECT
                CAST("{hs_col}" AS VARCHAR) AS hs,
                CAST("{year_col}" AS INTEGER) AS year,
                COUNT(*) AS n_rows_parquet,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total_parquet
            FROM read_parquet(
                '{parquet_glob}',
                union_by_name = true
            )
            WHERE "{hs_col}" IS NOT NULL
              AND "{year_col}" IS NOT NULL
            GROUP BY hs, year
            """
        ).df()

    audit = csv_summary.merge(
        parquet_summary,
        on=["hs", "year"],
        how="outer",
    )

    audit["n_rows_csv"] = audit["n_rows_csv"].fillna(0).astype("int64")
    audit["n_rows_parquet"] = audit["n_rows_parquet"].fillna(0).astype("int64")

    audit["value_total_csv"] = audit["value_total_csv"].fillna(0.0)
    audit["value_total_parquet"] = audit["value_total_parquet"].fillna(0.0)

    audit["row_diff"] = audit["n_rows_parquet"] - audit["n_rows_csv"]
    audit["value_diff"] = audit["value_total_parquet"] - audit["value_total_csv"]
    audit["abs_value_diff"] = audit["value_diff"].abs()

    audit["status"] = "ok"

    audit.loc[audit["row_diff"] != 0, "status"] = "row_mismatch"

    audit.loc[
        (audit["row_diff"] == 0)
        & (audit["abs_value_diff"] > value_tolerance),
        "status",
    ] = "value_mismatch"

    audit = audit.sort_values(["status", "hs", "year"]).reset_index(drop=True)

    logger.info("Audit rows: %s", audit.shape[0])
    logger.info("Audit status counts: %s", audit["status"].value_counts().to_dict())

    return audit


def stage_comtrade_for_harvard_weights_partitioned(
    *,
    input_dir: Path,
    pattern: str,
    output_root: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    threads: int = 16,
    overwrite: bool = False) -> Path:
    """
    Write partitioned parquet output by HS and year.

    Output will be:
        raw_parquet/hs=H5/year=2019/*.parquet

    This is faster, but the folder names are not exactly Harvard-style.
    """

    input_glob = str(input_dir / pattern)
    output_base = output_root / "as_reported" / "raw_parquet_partitioned"

    if output_base.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {rel_path(output_base)}"
        )

    output_base.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        con.execute(
            f"""
            COPY (
                SELECT
                    *,
                    CAST("{hs_col}" AS VARCHAR) AS hs_partition,
                    CAST("{year_col}" AS INTEGER) AS year_partition
                FROM read_csv_auto(
                    '{input_glob}',
                    union_by_name = true,
                    filename = true
                )
                WHERE "{hs_col}" IS NOT NULL
                  AND "{year_col}" IS NOT NULL
            )
            TO '{output_base}'
            (
                FORMAT PARQUET,
                PARTITION_BY (hs_partition, year_partition),
                OVERWRITE_OR_IGNORE TRUE
            )
            """
        )

    return output_base

def stage_comtrade_for_harvard_weights(
    *,
    input_dir: Path,
    pattern: str,
    output_root: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    threads: int = 16,
    overwrite: bool = False) -> dict[str, int]:
    """
    Stage raw Comtrade CSV data into Harvard-compatible parquet folders.

    Output layout:
        output_root/as_reported/raw_parquet/{HS}/{YEAR}/COMTRADE-FINALCLASSIC-CA-{HS}-{YEAR}.parquet

    Example:
        H5 / 2019 →
        output_root/as_reported/raw_parquet/H5/2019/COMTRADE-FINALCLASSIC-CA-H5-2019.parquet

    Parameters
    ----------
    input_dir:
        Directory containing raw Comtrade CSV files.

    pattern:
        Glob pattern, e.g. "*.csv".

    output_root:
        Base output directory, e.g.
        /storage/vacoda/data/external/harvard_comtrade

    year_col:
        Year column in Comtrade data.

    hs_col:
        HS classification column in Comtrade data.

    threads:
        DuckDB thread count.

    overwrite:
        If False, existing parquet files are skipped.

    Returns
    -------
    dict[str, int]
        Small summary of staged files.
    """

    input_glob = str(input_dir / pattern)
    output_base = output_root / "as_reported" / "raw_parquet"
    output_base.mkdir(parents=True, exist_ok=True)

    logger.info("Input glob: %s", input_glob)
    logger.info("Output base: %s", rel_path(output_base))
    logger.info("Threads: %s", threads)

    written = 0
    skipped = 0

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        logger.info("Scanning available HS/year combinations.")

        combos = con.execute(
            f"""
            SELECT DISTINCT
                CAST("{hs_col}" AS VARCHAR) AS hs,
                CAST("{year_col}" AS INTEGER) AS year
            FROM read_csv_auto(
                '{input_glob}',
                union_by_name = true,
                filename = true
            )
            WHERE "{hs_col}" IS NOT NULL
              AND "{year_col}" IS NOT NULL
            ORDER BY hs, year
            """
        ).df()

        combos["hs"] = combos["hs"].astype("string").str.strip().str.upper()
        combos = combos.loc[combos["hs"].isin(VALID_HS_SYSTEMS)].copy()

        logger.info("Found %s valid HS/year combinations.", f"{len(combos):,}")

        for i, row in enumerate(combos.itertuples(index=False), start=1):
            hs = str(row.hs)
            year = int(row.year)

            out_dir = output_base / hs / str(year)
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / f"COMTRADE-FINALCLASSIC-CA-{hs}-{year}.parquet"

            if out_file.exists() and not overwrite:
                skipped += 1
                logger.info(
                    "[%s/%s] Skipping existing: %s",
                    f"{i:,}",
                    f"{len(combos):,}",
                    rel_path(out_file),
                )
                continue

            logger.info(
                "[%s/%s] Writing %s %s → %s",
                f"{i:,}",
                f"{len(combos):,}",
                hs,
                year,
                rel_path(out_file),
            )

            con.execute(
                f"""
                COPY (
                    SELECT *
                    FROM read_csv_auto(
                        '{input_glob}',
                        union_by_name = true,
                        filename = true
                    )
                    WHERE CAST("{hs_col}" AS VARCHAR) = '{hs}'
                      AND CAST("{year_col}" AS INTEGER) = {year}
                )
                TO '{out_file}'
                (FORMAT PARQUET)
                """
            )

            written += 1

    logger.info("Staging complete. Written=%s, skipped=%s", written, skipped)

    return {
        "written": written,
        "skipped": skipped,
        "total_combinations": int(len(combos)),
    }


def audit_harvard_staging_output(
    *,
    output_root: Path) -> pd.DataFrame:
    """
    Audit staged Harvard-compatible parquet layout.

    Returns one row per HS/year folder with file counts and total size.
    """

    output_base = output_root / "as_reported" / "raw_parquet"

    rows: list[dict] = []

    for hs_dir in sorted(output_base.glob("H*")):
        if not hs_dir.is_dir():
            continue

        for year_dir in sorted(hs_dir.glob("*")):
            if not year_dir.is_dir():
                continue

            files = sorted(year_dir.glob("*.parquet"))

            rows.append(
                {
                    "hs": hs_dir.name,
                    "year": int(year_dir.name),
                    "n_files": len(files),
                    "size_bytes": sum(p.stat().st_size for p in files),
                }
            )

    return pd.DataFrame(rows).sort_values(["hs", "year"]).reset_index(drop=True)


def summarize_staging_audit(audit: pd.DataFrame) -> dict[str, object]:
    """
    Summarize CSV → Parquet staging audit.
    """

    total_rows_csv = int(audit["n_rows_csv"].sum())
    total_rows_parquet = int(audit["n_rows_parquet"].sum())

    total_value_csv = float(audit["value_total_csv"].sum())
    total_value_parquet = float(audit["value_total_parquet"].sum())

    summary = {
        "n_groups": int(len(audit)),
        "status_counts": audit["status"].value_counts().to_dict(),
        "total_rows_csv": total_rows_csv,
        "total_rows_parquet": total_rows_parquet,
        "total_row_diff": int(total_rows_parquet - total_rows_csv),
        "total_value_csv": total_value_csv,
        "total_value_parquet": total_value_parquet,
        "total_value_diff": float(total_value_parquet - total_value_csv),
    }

    return summary


def assert_no_staging_data_loss(
    audit: pd.DataFrame,
    *,
    value_tolerance: float = 0.01) -> None:
    """
    Raise an error if CSV → Parquet staging lost rows or changed values.
    """

    total_row_diff = int(audit["row_diff"].sum())
    total_value_diff = float(audit["value_diff"].sum())

    bad = audit.loc[audit["status"] != "ok"].copy()

    if total_row_diff != 0:
        raise ValueError(
            "CSV → Parquet staging row-count mismatch. "
            f"Total row difference: {total_row_diff:,}"
        )

    if abs(total_value_diff) > value_tolerance:
        raise ValueError(
            "CSV → Parquet staging value-total mismatch. "
            f"Total value difference: {total_value_diff:,.6f}"
        )

    if not bad.empty:
        raise ValueError(
            "CSV → Parquet staging has group-level mismatches. "
            f"Bad HS/year groups: {len(bad):,}"
        )

def audit_global_csv_to_parquet_totals(
    *,
    input_dir: Path,
    input_pattern: str,
    output_root: Path,
    value_col: str = "PrimaryValue",
    threads: int = 16) -> pd.DataFrame:

    input_glob = str(input_dir / input_pattern)
    parquet_glob = str(output_root / "as_reported" / "raw_parquet" / "*" / "*" / "*.parquet")

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        csv = con.execute(
            f"""
            SELECT
                'csv' AS source,
                COUNT(*) AS n_rows,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total
            FROM read_csv_auto(
                '{input_glob}',
                union_by_name = true,
                filename = true
            )
            """
        ).df()

        parquet = con.execute(
            f"""
            SELECT
                'parquet' AS source,
                COUNT(*) AS n_rows,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total
            FROM read_parquet(
                '{parquet_glob}',
                union_by_name = true
            )
            """
        ).df()

    return pd.concat([csv, parquet], ignore_index=True)

def stage_comtrade_for_harvard_weights_fast(
    *,
    input_dir: Path,
    pattern: str,
    output_dir: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    threads: int = 16,
    overwrite: bool = False) -> Path:
    """
    Fast one-pass staging from vacoda raw CSVs to partitioned Parquet.

    This writes:
        raw_parquet_partitioned/hs_partition=H0/year_partition=1988/*.parquet

    It is much faster than scanning all CSVs once per HS/year pair.
    """

    input_glob = str(input_dir / pattern)
    output_base = output_dir

    if output_base.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {rel_path(output_base)}"
        )

    output_base.mkdir(parents=True, exist_ok=True)

    logger.info("Fast staging with DuckDB partitioned Parquet.")
    logger.info("Input glob: %s", input_glob)
    logger.info("Output base: %s", rel_path(output_base))
    logger.info("Threads: %s", threads)

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        con.execute(
            f"""
            COPY (
                SELECT
                    *,
                    CAST("{hs_col}" AS VARCHAR) AS hs_partition,
                    CAST("{year_col}" AS INTEGER) AS year_partition
                FROM read_csv_auto(
                    '{input_glob}',
                    union_by_name = true,
                    filename = true
                )
                WHERE "{hs_col}" IS NOT NULL
                  AND "{year_col}" IS NOT NULL
            )
            TO '{output_base}'
            (
                FORMAT PARQUET,
                PARTITION_BY (hs_partition, year_partition),
                OVERWRITE_OR_IGNORE TRUE
            )
            """
        )

    logger.info("Fast staging complete: %s", rel_path(output_base))

    return output_base


def audit_harvard_staging_partitioned_output(
    *,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Audit fast DuckDB-partitioned Harvard staging output.

    Expected layout:
        output_root/as_reported/raw_parquet_partitioned/
        ├── hs_partition=H0/
        │   ├── year_partition=1988/
        │   └── ...
        └── ...

    Returns one row per HS/year partition with file counts and total size.
    """

    output_base = output_dir

    rows: list[dict] = []

    if not output_base.exists():
        return pd.DataFrame(
            columns=["hs", "year", "n_files", "size_bytes"]
        )

    for hs_dir in sorted(output_base.glob("hs_partition=*")):
        if not hs_dir.is_dir():
            continue

        hs = hs_dir.name.split("=", 1)[1]

        for year_dir in sorted(hs_dir.glob("year_partition=*")):
            if not year_dir.is_dir():
                continue

            year_text = year_dir.name.split("=", 1)[1]

            try:
                year = int(year_text)
            except ValueError:
                year = year_text

            files = sorted(year_dir.glob("*.parquet"))

            rows.append(
                {
                    "hs": hs,
                    "year": year,
                    "n_files": len(files),
                    "size_bytes": sum(p.stat().st_size for p in files),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["hs", "year", "n_files", "size_bytes"]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["hs", "year"])
        .reset_index(drop=True)
    )

def audit_csv_to_partitioned_parquet_staging(
    *,
    input_dir: Path,
    input_pattern: str,
    output_dir: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    value_col: str = "PrimaryValue",
    threads: int = 16,
    value_abs_tolerance: float = 1.0,
    value_rel_tolerance: float = 1e-10,) -> pd.DataFrame:
    """
    Compare raw CSV input against fast DuckDB-partitioned Parquet output.

    This is the actual no-data-loss audit.

    Compares by HS system and year:
        - row counts
        - total PrimaryValue
        - number of distinct CmdCode values

    Value preservation is checked using both:
        - absolute tolerance, useful for small totals
        - relative tolerance, useful for very large totals
    """

    input_glob = str(input_dir / input_pattern)

    parquet_glob = str(
        output_dir
        / "hs_partition=*"
        / "year_partition=*"
        / "*.parquet"
    )

    logger.info("Running CSV → partitioned Parquet preservation audit.")
    logger.info("CSV input glob: %s", input_glob)
    logger.info("Parquet glob: %s", parquet_glob)
    logger.info(
        "Value tolerances: abs=%s, rel=%s",
        value_abs_tolerance,
        value_rel_tolerance,
    )

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        csv_summary = con.execute(
            f"""
            SELECT
                CAST("{hs_col}" AS VARCHAR) AS hs,
                CAST("{year_col}" AS INTEGER) AS year,
                COUNT(*) AS n_rows_csv,
                COUNT(DISTINCT CAST("{cmd_col}" AS VARCHAR)) AS n_cmd_codes_csv,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total_csv
            FROM read_csv_auto(
                '{input_glob}',
                union_by_name = true,
                filename = true
            )
            WHERE "{hs_col}" IS NOT NULL
              AND "{year_col}" IS NOT NULL
            GROUP BY hs, year
            """
        ).df()

        parquet_summary = con.execute(
            f"""
            SELECT
                CAST("{hs_col}" AS VARCHAR) AS hs,
                CAST("{year_col}" AS INTEGER) AS year,
                COUNT(*) AS n_rows_parquet,
                COUNT(DISTINCT CAST("{cmd_col}" AS VARCHAR)) AS n_cmd_codes_parquet,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total_parquet
            FROM read_parquet(
                '{parquet_glob}',
                union_by_name = true
            )
            WHERE "{hs_col}" IS NOT NULL
              AND "{year_col}" IS NOT NULL
            GROUP BY hs, year
            """
        ).df()

    audit = csv_summary.merge(
        parquet_summary,
        on=["hs", "year"],
        how="outer",
    )

    count_cols = [
        "n_rows_csv",
        "n_rows_parquet",
        "n_cmd_codes_csv",
        "n_cmd_codes_parquet",
    ]

    for col in count_cols:
        audit[col] = audit[col].fillna(0).astype("int64")

    audit["value_total_csv"] = audit["value_total_csv"].fillna(0.0)
    audit["value_total_parquet"] = audit["value_total_parquet"].fillna(0.0)

    audit["row_diff"] = audit["n_rows_parquet"] - audit["n_rows_csv"]
    audit["cmd_code_diff"] = audit["n_cmd_codes_parquet"] - audit["n_cmd_codes_csv"]
    audit["value_diff"] = audit["value_total_parquet"] - audit["value_total_csv"]
    audit["abs_value_diff"] = audit["value_diff"].abs()

    # Relative value difference.
    # For very large totals, tiny absolute floating differences are expected.
    denom = audit["value_total_csv"].abs()

    audit["relative_value_diff"] = 0.0
    nonzero = denom > 0

    audit.loc[nonzero, "relative_value_diff"] = (
        audit.loc[nonzero, "abs_value_diff"] / denom.loc[nonzero]
    )

    audit["value_close"] = (
        (audit["abs_value_diff"] <= value_abs_tolerance)
        | (audit["relative_value_diff"] <= value_rel_tolerance)
    )

    audit["status"] = "ok"

    audit.loc[audit["row_diff"] != 0, "status"] = "row_mismatch"

    audit.loc[
        (audit["status"] == "ok") & (audit["cmd_code_diff"] != 0),
        "status",
    ] = "cmd_code_mismatch"

    audit.loc[
        (audit["status"] == "ok") & ~audit["value_close"],
        "status",
    ] = "value_mismatch"

    audit = audit.sort_values(["status", "hs", "year"]).reset_index(drop=True)

    logger.info("Preservation audit shape: %s", audit.shape)
    logger.info("Status counts: %s", audit["status"].value_counts().to_dict())

    return audit