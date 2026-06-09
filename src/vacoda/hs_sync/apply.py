"""
vacoda.hs_sync.apply

Apply HS-sync mapping weights to observed Comtrade trade rows.

This module consumes the complete mapping-weight table produced by the
allocation stage. It does not decide allocation weights itself.

Core invariant:
    Sum of hs_synced value should equal sum of original value,
    up to floating point tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from vacoda.utils.logger import get_logger
from vacoda.config.hs_sync import HsSyncConfig

logger = get_logger("HS_SYNC.APPLY")


@dataclass(frozen=True)
class HsSyncApplyResult:
    output_path: Path
    audit_path: Path
    audit: pd.DataFrame


def apply_hs_sync_with_weights(
    *,
    input_glob: str,
    weights_path: Path,
    output_path: Path,
    audit_path: Path,
    min_year: int,
    max_year: int,
    year_col: str,
    hs_col: str,
    cmd_col: str,
    value_col: str,
    output_format: str = "parquet",
    threads: int = 16,
    overwrite: bool = True,
) -> HsSyncApplyResult:
    """
    Apply complete HS-sync mapping weights to trade data.

    Parameters
    ----------
    input_glob:
        Glob of raw/staged Comtrade CSV files.

    weights_path:
        Complete mapping-weight table CSV.

    output_path:
        Final synced output path.

    audit_path:
        CSV audit path.

    min_year, max_year:
        Year range to process.

    year_col, hs_col, cmd_col, value_col:
        Input column names.

    output_format:
        Currently supports parquet or csv.

    threads:
        DuckDB threads.

    overwrite:
        Whether to overwrite existing outputs.
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    if audit_path.exists() and not overwrite:
        raise FileExistsError(f"Audit already exists: {audit_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)}")

    try:
        logger.info("Applying HS-sync mapping weights.")
        logger.info("  Input glob: %s", input_glob)
        logger.info("  Weights: %s", weights_path)
        logger.info("  Output: %s", output_path)
        logger.info("  Years: %s–%s", min_year, max_year)
        logger.info("  Threads: %s", threads)

        _register_weights(
            con=con,
            weights_path=weights_path,
        )

        _validate_weight_table(con)

        audit_before = _build_pre_apply_audit(
            con=con,
            input_glob=input_glob,
            min_year=min_year,
            max_year=max_year,
            year_col=year_col,
            hs_col=hs_col,
            cmd_col=cmd_col,
            value_col=value_col,
        )

        _write_synced_output(
            con=con,
            input_glob=input_glob,
            output_path=output_path,
            min_year=min_year,
            max_year=max_year,
            year_col=year_col,
            hs_col=hs_col,
            cmd_col=cmd_col,
            value_col=value_col,
            output_format=output_format,
        )

        audit_after = _build_post_apply_audit(
            con=con,
            output_path=output_path,
            output_format=output_format,
        )

        audit = _combine_apply_audit(
            audit_before=audit_before,
            audit_after=audit_after,
        )

        audit.to_csv(audit_path, index=False)

        failed = audit[audit["status"] == "fail"]
        if not failed.empty:
            raise ValueError(
                "HS-sync apply audit failed:\n"
                + failed.to_string(index=False)
            )

        logger.info("HS-sync apply completed.")
        logger.info("  Output: %s", output_path)
        logger.info("  Audit: %s", audit_path)

        return HsSyncApplyResult(
            output_path=output_path,
            audit_path=audit_path,
            audit=audit,
        )

    finally:
        con.close()


def _sql_literal(value: str | Path) -> str:
    """
    Safely quote a Python string/path as a SQL string literal.

    DuckDB does not allow prepared parameters in every statement type,
    for example CREATE VIEW AS SELECT read_csv_auto(?).
    """
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def _register_weights(
    *,
    con: duckdb.DuckDBPyConnection,
    weights_path: Path,
) -> None:
    """
    Register complete mapping-weight table as DuckDB temp view.
    """

    if not weights_path.exists():
        raise FileNotFoundError(f"Mapping weights file not found: {weights_path}")

    weights_sql = _sql_literal(weights_path)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW hs_weights_raw AS
        SELECT *
        FROM read_csv_auto(
            {weights_sql},
            HEADER = TRUE,
            ALL_VARCHAR = TRUE
        )
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW hs_weights AS
        SELECT
            source_hs,
            lpad(source_cmd, 6, '0') AS source_cmd,
            target_hs,
            lpad(target_cmd, 6, '0') AS target_cmd,
            conversion_need,
            hs_sync_method,
            relationship_original,
            actual_mapping_category,
            CAST(n_target_codes AS INTEGER) AS n_target_codes,
            CAST(allocation_weight AS DOUBLE) AS allocation_weight,
            allocation_strategy,
            allocation_is_assumption
        FROM hs_weights_raw
        """
    )

def _register_weights_obsolete(
    *,
    con: duckdb.DuckDBPyConnection,
    weights_path: Path,
) -> None:
    """
    Register complete mapping-weight table as DuckDB temp view.
    """

    if not weights_path.exists():
        raise FileNotFoundError(f"Mapping weights file not found: {weights_path}")

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW hs_weights_raw AS
        SELECT *
        FROM read_csv_auto(?,
            HEADER = TRUE,
            ALL_VARCHAR = TRUE
        )
        """,
        [str(weights_path)],
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW hs_weights AS
        SELECT
            source_hs,
            lpad(source_cmd, 6, '0') AS source_cmd,
            target_hs,
            lpad(target_cmd, 6, '0') AS target_cmd,
            conversion_need,
            hs_sync_method,
            relationship_original,
            actual_mapping_category,
            CAST(n_target_codes AS INTEGER) AS n_target_codes,
            CAST(allocation_weight AS DOUBLE) AS allocation_weight,
            allocation_strategy,
            allocation_is_assumption
        FROM hs_weights_raw
        """
    )


def _validate_weight_table(con: duckdb.DuckDBPyConnection) -> None:
    """
    Fail if complete mapping weights are not valid.
    """

    bad = con.execute(
        """
        WITH sums AS (
            SELECT
                source_hs,
                source_cmd,
                SUM(allocation_weight) AS weight_sum,
                COUNT(*) AS n_rows
            FROM hs_weights
            GROUP BY source_hs, source_cmd
        )
        SELECT *
        FROM sums
        WHERE abs(weight_sum - 1.0) > 1e-9
           OR n_rows < 1
        LIMIT 20
        """
    ).fetchdf()

    if not bad.empty:
        raise ValueError(
            "Invalid complete mapping weights. Some source weights do not sum "
            "to 1.0:\n"
            + bad.to_string(index=False)
        )


def _build_pre_apply_audit(
    *,
    con: duckdb.DuckDBPyConnection,
    input_glob: str,
    min_year: int,
    max_year: int,
    year_col: str,
    hs_col: str,
    cmd_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Audit input rows and whether all source_hs/source_cmd pairs find weights.
    """

    input_glob_sql = _sql_literal(input_glob)

    query = f"""
    WITH src AS (
        SELECT
            CAST("{year_col}" AS INTEGER) AS ref_year,
            CAST("{hs_col}" AS VARCHAR) AS source_hs,
            lpad(CAST("{cmd_col}" AS VARCHAR), 6, '0') AS source_cmd,
            TRY_CAST("{value_col}" AS DOUBLE) AS value_original
        FROM read_csv_auto(
            {input_glob_sql},
            HEADER = TRUE,
            ALL_VARCHAR = TRUE,
            union_by_name = TRUE
        )
        WHERE CAST("{year_col}" AS INTEGER) BETWEEN ? AND ?
    ),
    source_pairs AS (
        SELECT DISTINCT source_hs, source_cmd
        FROM src
    ),
    missing_pairs AS (
        SELECT s.*
        FROM source_pairs s
        LEFT JOIN (
            SELECT DISTINCT source_hs, source_cmd
            FROM hs_weights
        ) w
        USING (source_hs, source_cmd)
        WHERE w.source_cmd IS NULL
    )
    SELECT
        COUNT(*) AS input_rows,
        SUM(value_original) AS input_value,
        (SELECT COUNT(*) FROM source_pairs) AS input_source_pairs,
        (SELECT COUNT(*) FROM missing_pairs) AS missing_source_pairs
    FROM src
    """

    row = con.execute(query, [min_year, max_year]).fetchdf().iloc[0]

    checks = []

    checks.append(
        {
            "check": "input_rows_defined",
            "expected": "> 0",
            "actual": row["input_rows"],
            "status": "pass" if row["input_rows"] > 0 else "fail",
            "detail": "",
        }
    )

    checks.append(
        {
            "check": "all_source_pairs_have_weights",
            "expected": 0,
            "actual": row["missing_source_pairs"],
            "status": "pass" if row["missing_source_pairs"] == 0 else "fail",
            "detail": f"input_source_pairs={row['input_source_pairs']}",
        }
    )

    checks.append(
        {
            "check": "input_value_total",
            "expected": "recorded",
            "actual": row["input_value"],
            "status": "info",
            "detail": "",
        }
    )

    return pd.DataFrame(checks)

def _write_synced_output(
    *,
    con: duckdb.DuckDBPyConnection,
    input_glob: str,
    output_path: Path,
    min_year: int,
    max_year: int,
    year_col: str,
    hs_col: str,
    cmd_col: str,
    value_col: str,
    output_format: str,
) -> None:
    """
    Write HS-synced rows.
    """

    input_glob_sql = _sql_literal(input_glob)
    output_path_sql = _sql_literal(output_path)

    synced_query = f"""
    SELECT
        src.*,

        src."{hs_col}" AS source_hs_original,
        lpad(CAST(src."{cmd_col}" AS VARCHAR), 6, '0') AS source_cmd_original,

        w.target_hs AS target_hs,
        w.target_cmd AS target_cmd,

        w.allocation_weight,
        w.conversion_need,
        w.hs_sync_method,
        w.allocation_strategy,
        w.allocation_is_assumption,

        TRY_CAST(src."{value_col}" AS DOUBLE) AS primary_value_original,
        TRY_CAST(src."{value_col}" AS DOUBLE) * w.allocation_weight
            AS primary_value_hs_synced

    FROM read_csv_auto(
        {input_glob_sql},
        HEADER = TRUE,
        ALL_VARCHAR = TRUE,
        union_by_name = TRUE
    ) src
    INNER JOIN hs_weights w
      ON CAST(src."{hs_col}" AS VARCHAR) = w.source_hs
     AND lpad(CAST(src."{cmd_col}" AS VARCHAR), 6, '0') = w.source_cmd
    WHERE CAST(src."{year_col}" AS INTEGER)
          BETWEEN {int(min_year)} AND {int(max_year)}
    """

    if output_format == "parquet":
        con.execute(
            f"""
            COPY (
                {synced_query}
            )
            TO {output_path_sql}
            (FORMAT PARQUET)
            """
        )

    elif output_format == "csv":
        con.execute(
            f"""
            COPY (
                {synced_query}
            )
            TO {output_path_sql}
            (HEADER, DELIMITER ',')
            """
        )

    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def _build_post_apply_audit(
    *,
    con: duckdb.DuckDBPyConnection,
    output_path: Path,
    output_format: str,
) -> pd.DataFrame:
    """
    Audit written HS-synced output.
    """

    output_path_sql = _sql_literal(output_path)

    if output_format == "parquet":
        scan = f"read_parquet({output_path_sql})"
    elif output_format == "csv":
        scan = f"read_csv_auto({output_path_sql}, HEADER = TRUE, ALL_VARCHAR = TRUE)"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    query = f"""
    SELECT
        COUNT(*) AS output_rows,
        SUM(CAST(primary_value_original AS DOUBLE)) AS expanded_original_value,
        SUM(CAST(primary_value_hs_synced AS DOUBLE)) AS synced_value,
        COUNT(DISTINCT source_hs_original || ':' || source_cmd_original)
            AS output_source_pairs,
        COUNT(DISTINCT target_hs || ':' || target_cmd)
            AS output_target_pairs
    FROM {scan}
    """

    row = con.execute(query).fetchdf().iloc[0]

    checks = []

    checks.append(
        {
            "check": "output_rows_defined",
            "expected": "> 0",
            "actual": row["output_rows"],
            "status": "pass" if row["output_rows"] > 0 else "fail",
            "detail": "",
        }
    )

    checks.append(
        {
            "check": "synced_value_total",
            "expected": "recorded",
            "actual": row["synced_value"],
            "status": "info",
            "detail": (
                f"expanded_original_value={row['expanded_original_value']}; "
                f"output_source_pairs={row['output_source_pairs']}; "
                f"output_target_pairs={row['output_target_pairs']}"
            ),
        }
    )

    return pd.DataFrame(checks)


def _combine_apply_audit(
    *,
    audit_before: pd.DataFrame,
    audit_after: pd.DataFrame,
) -> pd.DataFrame:
    audit = pd.concat([audit_before, audit_after], ignore_index=True)

    # Compare input_value_total and synced_value_total.
    input_value_rows = audit[audit["check"] == "input_value_total"]
    synced_value_rows = audit[audit["check"] == "synced_value_total"]

    if not input_value_rows.empty and not synced_value_rows.empty:
        input_value = float(input_value_rows["actual"].iloc[0])
        synced_value = float(synced_value_rows["actual"].iloc[0])

        diff = input_value - synced_value
        rel_diff = abs(diff) / abs(input_value) if input_value else 0.0

        audit = pd.concat(
            [
                audit,
                pd.DataFrame(
                    [
                        {
                            "check": "value_preserved_after_hs_sync",
                            "expected": input_value,
                            "actual": synced_value,
                            "status": "pass" if rel_diff <= 1e-10 else "fail",
                            "detail": f"diff={diff}; rel_diff={rel_diff}",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return audit

def run_hs_sync_apply(
    *,
    cfg: HsSyncConfig,
    run_id: str,
    logger: Logger,
) -> dict[str, object]:
    """
    Apply complete HS-sync mapping weights to raw Comtrade rows.

    This is the config-driven workflow backend for the HS-sync apply step.
    It expects the complete mapping-weight table to already exist.
    """

    target_hs = cfg.coverage.target_hs
    target_hs_lower = target_hs.lower()

    input_dir = cfg.coverage_input_dir()
    input_pattern = cfg.coverage.input.pattern

    if input_pattern is None:
        raise ValueError("coverage.input.pattern is required for HS-sync apply.")

    input_glob = str(input_dir / input_pattern)

    weights_dir = cfg.weights_output_dir()

    weights_path = (
        weights_dir
        / f"target_{target_hs_lower}_complete_mapping_weights.csv"
    )

    output_dir = (
        cfg.coverage_output_dir().parent
        / f"target_{target_hs_lower}_applied"
    )

    output_path = (
        output_dir
        / f"comtrade_target_{target_hs_lower}_hs_synced.parquet"
    )

    audit_path = (
        output_dir
        / f"comtrade_target_{target_hs_lower}_hs_sync_audit.csv"
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Complete mapping-weight table does not exist: {weights_path}"
        )

    logger.info("Starting HS-sync apply step.")
    logger.info("  Run ID: %s", run_id)
    logger.info("  Target HS: %s", target_hs)
    logger.info("  Input glob: %s", input_glob)
    logger.info("  Mapping weights: %s", weights_path)
    logger.info("  Output path: %s", output_path)
    logger.info("  Audit path: %s", audit_path)

    result = apply_hs_sync_with_weights(
        input_glob=input_glob,
        weights_path=weights_path,
        output_path=output_path,
        audit_path=audit_path,
        min_year=cfg.coverage.min_year,
        max_year=cfg.coverage.max_year,
        year_col=cfg.columns.year,
        hs_col=cfg.columns.hs,
        cmd_col=cfg.columns.cmd,
        value_col=cfg.columns.value,
        output_format="parquet",
        threads=cfg.engine.threads,
        overwrite=cfg.coverage.write.overwrite,
    )

    failed = result.audit[result.audit["status"] == "fail"]
    n_failed = len(failed)

    logger.info("HS-sync apply step completed.")
    logger.info("  Output file: %s", result.output_path)
    logger.info("  Audit file: %s", result.audit_path)
    logger.info("  Failed audit rows: %s", n_failed)

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": result.output_path,
        "audit_path": result.audit_path,
        "n_audit_rows": len(result.audit),
        "n_failed_audit_rows": n_failed,
    }