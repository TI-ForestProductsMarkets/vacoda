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
    cmd_col: str = "CmdCode",
    valid_cmd_regex: str | None = None,
    years_by_hs: dict[str, list[int]] | None = None,
    threads: int = 16,
    overwrite: bool = False,
) -> Path:
    """
    Fast one-pass staging from vacoda raw/bulk Comtrade files to
    partitioned Parquet.

    This writes:
        raw_parquet_partitioned/
        ├── hs_partition=H4/
        │   └── year_partition=2016/*.parquet
        └── hs_partition=H5/
            └── year_partition=2017/*.parquet

    Optional filters:
        valid_cmd_regex:
            Keep only commodity codes matching this regex.
            Example: r"^[0-9]{6}$" keeps only 6-digit product codes.

        years_by_hs:
            Keep only selected HS/year pairs.
            Example: {"H4": [2016], "H5": [2017]}.
    """

    input_glob = str(input_dir / pattern)
    output_base = output_dir

    if output_base.exists():
        if overwrite:
            logger.info(
                "Output directory already exists and overwrite is enabled: %s",
                rel_path(output_base),
            )
        else:
            logger.info(
                "Output directory already exists; reusing it without deleting existing files: %s",
                rel_path(output_base),
            )

    output_base.mkdir(parents=True, exist_ok=True)

    where_clauses = [
        f'"{hs_col}" IS NOT NULL',
        f'"{year_col}" IS NOT NULL',
    ]

    if valid_cmd_regex:
        where_clauses.append(
            f"regexp_matches(CAST(\"{cmd_col}\" AS VARCHAR), '{valid_cmd_regex}')"
        )

    if years_by_hs:
        allowed_pairs: list[str] = []

        for hs_system, years in years_by_hs.items():
            hs = str(hs_system).strip().upper()

            for year in years:
                allowed_pairs.append(
                    "("
                    f"CAST(\"{hs_col}\" AS VARCHAR) = '{hs}' "
                    "AND "
                    f"CAST(\"{year_col}\" AS INTEGER) = {int(year)}"
                    ")"
                )

        if allowed_pairs:
            where_clauses.append("(" + " OR ".join(allowed_pairs) + ")")

    where_sql = "\n                  AND ".join(where_clauses)

    logger.info("Fast staging with DuckDB partitioned Parquet.")
    logger.info("Input glob: %s", input_glob)
    logger.info("Output base: %s", rel_path(output_base))
    logger.info("Threads: %s", threads)

    if valid_cmd_regex:
        logger.info("Commodity-code filter: %s", valid_cmd_regex)

    if years_by_hs:
        logger.info("HS/year filter: %s", years_by_hs)

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
                WHERE {where_sql}
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

def stage_comtrade_for_harvard_weights_fast_deprecated(
    *,
    input_dir: Path,
    pattern: str,
    output_dir: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    threads: int = 16,
    overwrite: bool = False,
    cmd_col: str = "CmdCode",
    valid_cmd_regex: str | None = None,
    years_by_hs: dict[str, list[int]] | None = None,) -> Path:
    """
    Fast one-pass staging from vacoda raw CSVs to partitioned Parquet.

    This writes:
        raw_parquet_partitioned/hs_partition=H0/year_partition=1988/*.parquet

    It is much faster than scanning all CSVs once per HS/year pair.
    """

    input_glob = str(input_dir / pattern)
    output_base = output_dir

    if output_base.exists():
        if overwrite:
            logger.info(
                "Output directory already exists and overwrite is enabled: %s",
                rel_path(output_base),
            )
        else:
            logger.info(
                "Output directory already exists; reusing it without deleting existing files: %s",
                rel_path(output_base),
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
    output_dir: Path) -> pd.DataFrame:
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
    valid_cmd_regex: str | None = None,
    years_by_hs: dict[str, list[int]] | None = None,
    threads: int = 16,
    value_abs_tolerance: float = 1.0,
    value_rel_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Compare raw CSV/GZ input against fast DuckDB-partitioned Parquet output.

    This is the actual no-data-loss audit.

    Compares by HS system and year:
        - row counts
        - total PrimaryValue
        - number of distinct CmdCode values

    If staging was filtered by valid_cmd_regex and years_by_hs, the CSV side
    of the audit must use the same filters.
    """

    input_glob = str(input_dir / input_pattern)

    parquet_glob = str(
        output_dir
        / "hs_partition=*"
        / "year_partition=*"
        / "*.parquet"
    )

    # Build the same CSV-side filter used during staging.
    where_clauses = [
        f'"{hs_col}" IS NOT NULL',
        f'"{year_col}" IS NOT NULL',
    ]

    if valid_cmd_regex:
        where_clauses.append(
            f"regexp_matches(CAST(\"{cmd_col}\" AS VARCHAR), '{valid_cmd_regex}')"
        )

    if years_by_hs:
        allowed_pairs: list[str] = []

        for hs_system, years in years_by_hs.items():
            hs = str(hs_system).strip().upper()

            for year in years:
                allowed_pairs.append(
                    "("
                    f"CAST(\"{hs_col}\" AS VARCHAR) = '{hs}' "
                    "AND "
                    f"CAST(\"{year_col}\" AS INTEGER) = {int(year)}"
                    ")"
                )

        if allowed_pairs:
            where_clauses.append("(" + " OR ".join(allowed_pairs) + ")")

    where_sql = "\n              AND ".join(where_clauses)

    logger.info("Running CSV → partitioned Parquet preservation audit.")
    logger.info("CSV input glob: %s", input_glob)
    logger.info("Parquet glob: %s", parquet_glob)
    logger.info("CSV audit filter: %s", where_sql.replace("\n", " "))
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
            WHERE {where_sql}
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

def audit_csv_to_partitioned_parquet_staging_deprecated(
    *,
    input_dir: Path,
    input_pattern: str,
    output_dir: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    value_col: str = "PrimaryValue",
    valid_cmd_regex: str | None = None,
    years_by_hs: dict[str, list[int]] | None = None,
    threads: int = 16,
    value_abs_tolerance: float = 1.0,
    value_rel_tolerance: float = 1e-10,
) -> pd.DataFrame:
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
            WHERE {where_sql}
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


def export_harvard_generator_compatibility_layout(
    *,
    staged_partitioned_dir: Path,
    output_root: Path,
    years_by_hs: dict[str, list[int]] | None = None,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    reporter_code_col: str = "reporterCode",
    partner_code_col: str = "partnerCode",
    flow_col: str = "flowCode",
    value_col: str = "PrimaryValue",
    valid_cmd_regex: str | None = None,
    qty_col: str | None = "NetWgt",
    published_date: str = "2026-01-01",
    threads: int = 16,
    overwrite: bool = False,
) -> dict[str, int]:
    """
    Export fast partitioned vacoda staging into the layout expected by the
    Harvard/Growth Lab conversion-weight generator.

    Input:
        output_root/as_reported/raw_parquet_partitioned/
        ├── hs_partition=H4/year_partition=2016/*.parquet
        └── hs_partition=H5/year_partition=2017/*.parquet

    Output:
        output_root/as_reported/raw_parquet/H4/2016/
        └── COMTRADE-FINALCLASSIC-CA{reporter}{year}H4[{date}].parquet

        output_root/as_reported/aggregated_by_year_not_converted/parquet/H4/
        └── H4_2016.parquet

    Notes
    -----
    Harvard's MatrixBuilder uses raw_parquet mostly to identify reporter files
    and timely classification switchers. The trade values are read from
    aggregated_by_year_not_converted/parquet/{H}/{H}_{year}.parquet.
    """

    as_reported_dir = output_root / "as_reported"

    raw_parquet_dir = as_reported_dir / "raw_parquet"
    aggregated_dir = (
        as_reported_dir
        / "aggregated_by_year_not_converted"
        / "parquet"
    )

    raw_parquet_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting Harvard generator compatibility layout.")
    logger.info("Input partitioned dir: %s", rel_path(staged_partitioned_dir))
    logger.info("Raw parquet output: %s", rel_path(raw_parquet_dir))
    logger.info("Aggregated output: %s", rel_path(aggregated_dir))

    if years_by_hs is None:
        years_by_hs = {}

        for hs_dir in staged_partitioned_dir.glob("hs_partition=*"):
            if not hs_dir.is_dir():
                continue

            hs = hs_dir.name.split("=", 1)[1]
            years: list[int] = []

            for year_dir in hs_dir.glob("year_partition=*"):
                if not year_dir.is_dir():
                    continue
                year_text = year_dir.name.split("=", 1)[1]
                years.append(int(year_text))

            years_by_hs[hs] = sorted(years)

    written_raw_files = 0
    written_aggregated_files = 0
    skipped_raw_files = 0
    skipped_aggregated_files = 0

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        for hs, years in sorted(years_by_hs.items()):
            hs = str(hs).strip().upper()

            for year in sorted(years):
                input_glob = str(
                    staged_partitioned_dir
                    / f"hs_partition={hs}"
                    / f"year_partition={int(year)}"
                    / "*.parquet"
                )

                # ---------------------------------------------------------
                # 1. Aggregated file used by Harvard MatrixBuilder
                # ---------------------------------------------------------

                agg_out_dir = aggregated_dir / hs
                agg_out_dir.mkdir(parents=True, exist_ok=True)

                agg_out_file = agg_out_dir / f"{hs}_{int(year)}.parquet"

                if agg_out_file.exists() and not overwrite:
                    skipped_aggregated_files += 1
                    logger.info(
                        "Skipping existing aggregated file: %s",
                        rel_path(agg_out_file),
                    )
                else:
                    qty_expr = (
                        f'SUM(CAST("{qty_col}" AS DOUBLE)) AS qty'
                        if qty_col
                        else "CAST(NULL AS DOUBLE) AS qty"
                    )

                    logger.info(
                        "Writing aggregated Harvard file: %s",
                        rel_path(agg_out_file),
                    )

                    con.execute(
                        f"""
                        COPY (
                            SELECT
                                LPAD(CAST("{reporter_code_col}" AS VARCHAR), 3, '0') AS reporterISO3,
                                LPAD(CAST("{partner_code_col}" AS VARCHAR), 3, '0') AS partnerISO3,
                                CAST("{cmd_col}" AS VARCHAR) AS cmdCode,
                                CAST("{flow_col}" AS VARCHAR) AS flowCode,
                                LENGTH(CAST("{cmd_col}" AS VARCHAR)) AS digitLevel,
                                {qty_expr},
                                CAST(NULL AS DOUBLE) AS CIFValue,
                                CAST(NULL AS DOUBLE) AS FOBValue,
                                SUM(CAST("{value_col}" AS DOUBLE)) AS primaryValue
                            FROM read_parquet(
                                '{input_glob}',
                                union_by_name = true
                            )
                            WHERE "{hs_col}" IS NOT NULL
                              AND "{year_col}" IS NOT NULL
                            GROUP BY
                                reporterISO3,
                                partnerISO3,
                                cmdCode,
                                flowCode,
                                digitLevel
                        )
                        TO '{agg_out_file}'
                        (FORMAT PARQUET)
                        """
                    )

                    written_aggregated_files += 1

                # ---------------------------------------------------------
                # 2. Raw parquet marker/files used for reporter switching
                # ---------------------------------------------------------

                reporters = con.execute(
                    f"""
                    SELECT DISTINCT
                        LPAD(CAST("{reporter_code_col}" AS VARCHAR), 3, '0') AS reporter_code
                    FROM read_parquet(
                        '{input_glob}',
                        union_by_name = true
                    )
                    WHERE "{reporter_code_col}" IS NOT NULL
                    ORDER BY reporter_code
                    """
                ).fetchall()

                raw_out_dir = raw_parquet_dir / hs / str(int(year))
                raw_out_dir.mkdir(parents=True, exist_ok=True)

                for (reporter_code,) in reporters:
                    out_file = (
                        raw_out_dir
                        / f"COMTRADE-FINALCLASSIC-CA{reporter_code}{int(year)}{hs}[{published_date}].parquet"
                    )

                    if out_file.exists() and not overwrite:
                        skipped_raw_files += 1
                        continue

                    # Keep these as small marker files. Harvard parses filenames
                    # for reporter/year/classification; trade values come from
                    # aggregated_by_year_not_converted.
                    con.execute(
                        f"""
                        COPY (
                            SELECT
                                '{reporter_code}' AS ReporterCode,
                                {int(year)} AS RefYear,
                                '{hs}' AS ClassificationCode
                            LIMIT 1
                        )
                        TO '{out_file}'
                        (FORMAT PARQUET)
                        """
                    )

                    written_raw_files += 1

    summary = {
        "written_raw_files": written_raw_files,
        "skipped_raw_files": skipped_raw_files,
        "written_aggregated_files": written_aggregated_files,
        "skipped_aggregated_files": skipped_aggregated_files,
    }

    logger.info("Harvard compatibility export summary: %s", summary)

    return summary