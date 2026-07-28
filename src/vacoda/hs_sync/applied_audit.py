from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.io.duckdb_scan import duckdb_scan_relation

def _sql_literal(value: str | Path) -> str:
    """Safely quote a Python string/path as a SQL string literal."""
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def file_tree_size(path: Path, pattern: str = "*") -> int:
    if path.is_file():
        return path.stat().st_size

    return sum(
        p.stat().st_size
        for p in path.glob(pattern)
        if p.is_file()
    )


def _audit_row(
    *,
    check: str,
    expected: Any,
    actual: Any,
    status: str,
    detail: str = "",
) -> dict[str, object]:
    return {
        "check": check,
        "expected": expected,
        "actual": actual,
        "status": status,
        "detail": detail,
    }


def _status_pass(condition: bool) -> str:
    return "pass" if condition else "fail"


def _to_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def _to_float(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _create_audit_views(
    *,
    con: duckdb.DuckDBPyConnection,
    raw_input_glob: Path | str,
    raw_format: str,
    parquet_sql: str,
    weights_sql: str,
    year_col: str,
    hs_col: str,
    cmd_col: str,
    value_col: str,
    min_year: int,
    max_year: int,
) -> None:
    """
    Register normalized raw, applied, and weight views for integrity checks.
    """

    raw_scan = duckdb_scan_relation(
        path=raw_input_glob,
        file_format=raw_format,
        alias = "raw",
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW raw_rows AS
        SELECT
            TRY_CAST(raw."{year_col}" AS INTEGER) AS ref_year,
            CAST(raw."{hs_col}" AS VARCHAR) AS source_hs,
            lpad(CAST(raw."{cmd_col}" AS VARCHAR), 6, '0') AS source_cmd,
            TRY_CAST(raw."{value_col}" AS DOUBLE) AS value_original
        FROM {raw_scan}
        WHERE TRY_CAST(raw."{year_col}" AS INTEGER)
            BETWEEN {int(min_year)} AND {int(max_year)}
    """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW applied_rows AS
        SELECT *
        FROM read_parquet({parquet_sql})
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW weights_raw AS
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
        CREATE OR REPLACE TEMP VIEW weights AS
        SELECT
            CAST(source_hs AS VARCHAR) AS source_hs,
            lpad(CAST(source_cmd AS VARCHAR), 6, '0') AS source_cmd,
            CAST(target_hs AS VARCHAR) AS target_hs,
            lpad(CAST(target_cmd AS VARCHAR), 6, '0') AS target_cmd,
            TRY_CAST(allocation_weight AS DOUBLE) AS allocation_weight,
            CAST(conversion_need AS VARCHAR) AS conversion_need,
            CAST(hs_sync_method AS VARCHAR) AS hs_sync_method,
            CAST(allocation_strategy AS VARCHAR) AS allocation_strategy
        FROM weights_raw
        """
    )


def _required_column_audit(
    *,
    con: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, object], set[str]]:
    required_columns = {
        "hs_sync_input_row_id",
        "source_hs_original",
        "source_cmd_original",
        "target_hs",
        "target_cmd",
        "allocation_weight",
        "conversion_need",
        "hs_sync_method",
        "allocation_strategy",
        "allocation_is_assumption",
        "primary_value_original",
        "primary_value_hs_synced",
    }

    schema = con.execute("DESCRIBE SELECT * FROM applied_rows").fetchdf()
    actual_columns = set(schema["column_name"].astype(str))
    missing = sorted(required_columns - actual_columns)

    return (
        _audit_row(
            check="required_hs_sync_columns_present",
            expected="all required HS-sync columns",
            actual=0 if not missing else len(missing),
            status=_status_pass(not missing),
            detail="missing_columns=" + ",".join(missing) if missing else "",
        ),
        actual_columns,
    )


def _build_summary_tables(
    *,
    con: duckdb.DuckDBPyConnection,
    target_hs: str,
    year_col: str,
) -> dict[str, pd.Series]:
    raw_summary = con.execute(
        """
        SELECT
            COUNT(*) AS raw_rows,
            COUNT(DISTINCT source_hs || ':' || source_cmd) AS raw_source_pairs,
            SUM(value_original) AS raw_value_total,
            MIN(ref_year) AS raw_min_year,
            MAX(ref_year) AS raw_max_year,
            COUNT(*) FILTER (WHERE value_original IS NULL) AS raw_null_value_rows
        FROM raw_rows
        """
    ).fetchdf().iloc[0]

    applied_summary = con.execute(
        f"""
        SELECT
            COUNT(*) AS applied_rows,
            COUNT(DISTINCT source_hs_original || ':' || source_cmd_original)
                AS applied_source_pairs,
            COUNT(DISTINCT target_hs || ':' || target_cmd)
                AS applied_target_pairs,
            SUM(TRY_CAST(primary_value_hs_synced AS DOUBLE)) AS applied_value_total,
            SUM(TRY_CAST(primary_value_original AS DOUBLE)) AS expanded_original_value,
            MIN(TRY_CAST("{year_col}" AS INTEGER)) AS applied_min_year,
            MAX(TRY_CAST("{year_col}" AS INTEGER)) AS applied_max_year,
            SUM(CASE WHEN target_hs = '{target_hs}' THEN 1 ELSE 0 END)
                AS rows_with_target_hs,
            COUNT(*) - SUM(CASE WHEN target_hs = '{target_hs}' THEN 1 ELSE 0 END)
                AS rows_not_target_hs
        FROM applied_rows
        """
    ).fetchdf().iloc[0]

    weight_summary = con.execute(
        """
        SELECT
            COUNT(*) AS weight_rows,
            COUNT(DISTINCT source_hs || ':' || source_cmd) AS weight_source_pairs,
            COUNT(*) FILTER (
                WHERE allocation_weight IS NULL
                   OR allocation_weight <= 0
                   OR allocation_weight > 1
            ) AS invalid_weight_rows
        FROM weights
        """
    ).fetchdf().iloc[0]

    return {
        "raw": raw_summary,
        "applied": applied_summary,
        "weights": weight_summary,
    }


def _build_pair_coverage_audit(
    *,
    con: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    result = con.execute(
        """
        WITH raw_pairs AS (
            SELECT DISTINCT source_hs, source_cmd
            FROM raw_rows
        ),
        output_pairs AS (
            SELECT DISTINCT
                source_hs_original AS source_hs,
                source_cmd_original AS source_cmd
            FROM applied_rows
        )
        SELECT
            (
                SELECT COUNT(*)
                FROM raw_pairs r
                LEFT JOIN output_pairs o
                USING (source_hs, source_cmd)
                WHERE o.source_cmd IS NULL
            ) AS missing_source_pairs,
            (
                SELECT COUNT(*)
                FROM output_pairs o
                LEFT JOIN raw_pairs r
                USING (source_hs, source_cmd)
                WHERE r.source_cmd IS NULL
            ) AS extra_output_source_pairs
        """
    ).fetchdf().iloc[0]

    return {
        "missing_source_pairs": _to_int(result["missing_source_pairs"]),
        "extra_output_source_pairs": _to_int(result["extra_output_source_pairs"]),
    }


def _build_null_audit(
    *,
    con: duckdb.DuckDBPyConnection,
) -> pd.Series:
    return con.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE hs_sync_input_row_id IS NULL) AS null_hs_sync_input_row_id,
            COUNT(*) FILTER (WHERE source_hs_original IS NULL) AS null_source_hs_original,
            COUNT(*) FILTER (WHERE source_cmd_original IS NULL) AS null_source_cmd_original,
            COUNT(*) FILTER (WHERE target_hs IS NULL) AS null_target_hs,
            COUNT(*) FILTER (WHERE target_cmd IS NULL) AS null_target_cmd,
            COUNT(*) FILTER (WHERE allocation_weight IS NULL) AS null_allocation_weight,
            COUNT(*) FILTER (WHERE conversion_need IS NULL) AS null_conversion_need,
            COUNT(*) FILTER (WHERE hs_sync_method IS NULL) AS null_hs_sync_method,
            COUNT(*) FILTER (WHERE allocation_strategy IS NULL) AS null_allocation_strategy,
            COUNT(*) FILTER (WHERE primary_value_original IS NULL) AS null_primary_value_original,
            COUNT(*) FILTER (WHERE primary_value_hs_synced IS NULL) AS null_primary_value_hs_synced
        FROM applied_rows
        """
    ).fetchdf().iloc[0]


def _build_content_audit(
    *,
    con: duckdb.DuckDBPyConnection,
) -> pd.Series:
    return con.execute(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE TRY_CAST(allocation_weight AS DOUBLE) IS NULL
                   OR TRY_CAST(allocation_weight AS DOUBLE) <= 0
                   OR TRY_CAST(allocation_weight AS DOUBLE) > 1
            ) AS invalid_allocation_weight_rows,

            COUNT(*) FILTER (
                WHERE length(CAST(source_cmd_original AS VARCHAR)) != 6
                   OR regexp_matches(CAST(source_cmd_original AS VARCHAR), '^[0-9]{6}$') = false
            ) AS invalid_source_cmd_rows,

            COUNT(*) FILTER (
                WHERE length(CAST(target_cmd AS VARCHAR)) != 6
                   OR regexp_matches(CAST(target_cmd AS VARCHAR), '^[0-9]{6}$') = false
            ) AS invalid_target_cmd_rows,

            COUNT(*) FILTER (
                WHERE abs(
                    TRY_CAST(primary_value_hs_synced AS DOUBLE)
                    - TRY_CAST(primary_value_original AS DOUBLE)
                      * TRY_CAST(allocation_weight AS DOUBLE)
                ) > 1e-7
            ) AS invalid_row_value_formula_rows
        FROM applied_rows
        """
    ).fetchdf().iloc[0]


def _build_weight_reference_audit(
    *,
    con: duckdb.DuckDBPyConnection,
) -> pd.Series:
    """
    Check that applied rows correspond to the complete mapping-weight table.
    """

    return con.execute(
        """
        WITH applied_mapping_rows AS (
            SELECT DISTINCT
                source_hs_original AS source_hs,
                source_cmd_original AS source_cmd,
                target_hs,
                target_cmd,
                TRY_CAST(allocation_weight AS DOUBLE) AS allocation_weight
            FROM applied_rows
        ),
        missing_weight_rows AS (
            SELECT a.*
            FROM applied_mapping_rows a
            LEFT JOIN weights w
              ON a.source_hs = w.source_hs
             AND a.source_cmd = w.source_cmd
             AND a.target_hs = w.target_hs
             AND a.target_cmd = w.target_cmd
            WHERE w.source_cmd IS NULL
        ),
        mismatched_weight_rows AS (
            SELECT a.*
            FROM applied_mapping_rows a
            INNER JOIN weights w
              ON a.source_hs = w.source_hs
             AND a.source_cmd = w.source_cmd
             AND a.target_hs = w.target_hs
             AND a.target_cmd = w.target_cmd
            WHERE abs(a.allocation_weight - w.allocation_weight) > 1e-12
        ),
        unused_weight_rows AS (
            SELECT w.*
            FROM weights w
            LEFT JOIN applied_mapping_rows a
              ON a.source_hs = w.source_hs
             AND a.source_cmd = w.source_cmd
             AND a.target_hs = w.target_hs
             AND a.target_cmd = w.target_cmd
            WHERE a.source_cmd IS NULL
        )
        SELECT
            (SELECT COUNT(*) FROM applied_mapping_rows) AS applied_distinct_mapping_rows,
            (SELECT COUNT(*) FROM missing_weight_rows) AS applied_mapping_rows_missing_from_weights,
            (SELECT COUNT(*) FROM mismatched_weight_rows) AS applied_mapping_rows_with_wrong_weight,
            (SELECT COUNT(*) FROM unused_weight_rows) AS unused_weight_rows
        """
    ).fetchdf().iloc[0]


def _build_expected_expansion_audit(
    *,
    con: duckdb.DuckDBPyConnection,
) -> pd.Series:
    """
    Compare actual applied row count to expected expansion from raw rows and weights.

    This is stronger than comparing raw rows to applied rows, because split
    mappings intentionally expand rows.
    """

    return con.execute(
        """
        WITH raw_pair_counts AS (
            SELECT
                source_hs,
                source_cmd,
                COUNT(*) AS raw_rows_for_pair
            FROM raw_rows
            GROUP BY source_hs, source_cmd
        ),
        weight_pair_counts AS (
            SELECT
                source_hs,
                source_cmd,
                COUNT(*) AS target_rows_for_pair
            FROM weights
            GROUP BY source_hs, source_cmd
        ),
        expected AS (
            SELECT
                r.source_hs,
                r.source_cmd,
                r.raw_rows_for_pair,
                COALESCE(w.target_rows_for_pair, 0) AS target_rows_for_pair,
                r.raw_rows_for_pair * COALESCE(w.target_rows_for_pair, 0)
                    AS expected_applied_rows_for_pair
            FROM raw_pair_counts r
            LEFT JOIN weight_pair_counts w
            USING (source_hs, source_cmd)
        ),
        actual AS (
            SELECT
                source_hs_original AS source_hs,
                source_cmd_original AS source_cmd,
                COUNT(*) AS actual_applied_rows_for_pair
            FROM applied_rows
            GROUP BY source_hs_original, source_cmd_original
        ),
        mismatches AS (
            SELECT
                e.source_hs,
                e.source_cmd,
                e.expected_applied_rows_for_pair,
                COALESCE(a.actual_applied_rows_for_pair, 0)
                    AS actual_applied_rows_for_pair
            FROM expected e
            LEFT JOIN actual a
            USING (source_hs, source_cmd)
            WHERE e.expected_applied_rows_for_pair
                  != COALESCE(a.actual_applied_rows_for_pair, 0)
        )
        SELECT
            SUM(expected_applied_rows_for_pair) AS expected_applied_rows,
            (SELECT COUNT(*) FROM applied_rows) AS actual_applied_rows,
            COUNT(*) FILTER (WHERE target_rows_for_pair = 0)
                AS raw_source_pairs_without_weights,
            (SELECT COUNT(*) FROM mismatches) AS source_pairs_with_row_count_mismatch
        FROM expected
        """
    ).fetchdf().iloc[0]


def _build_distribution_info(
    *,
    con: duckdb.DuckDBPyConnection,
) -> list[dict[str, object]]:
    """
    Add compact informational distribution rows to the audit.
    """

    rows: list[dict[str, object]] = []

    target_distribution = con.execute(
        """
        SELECT
            target_hs,
            COUNT(*) AS n_rows
        FROM applied_rows
        GROUP BY target_hs
        ORDER BY n_rows DESC
        """
    ).fetchdf()

    rows.append(
        _audit_row(
            check="target_hs_distribution",
            expected="recorded",
            actual=len(target_distribution),
            status="info",
            detail=target_distribution.to_dict(orient="records"),
        )
    )

    method_distribution = con.execute(
        """
        SELECT
            conversion_need,
            hs_sync_method,
            allocation_strategy,
            COUNT(*) AS n_rows,
            SUM(TRY_CAST(primary_value_hs_synced AS DOUBLE)) AS value_synced
        FROM applied_rows
        GROUP BY 1, 2, 3
        ORDER BY n_rows DESC
        """
    ).fetchdf()

    rows.append(
        _audit_row(
            check="method_distribution",
            expected="recorded",
            actual=len(method_distribution),
            status="info",
            detail=method_distribution.to_dict(orient="records"),
        )
    )

    return rows


def run_hs_sync_applied_output_audit(
    *,
    cfg: HsSyncConfig,
    run_id: str,
    logger: Logger,
) -> dict[str, object]:
    """
    Audit HS-sync applied output against raw Comtrade input.

    This verifies that the generated target-HS parquet preserves raw input
    coverage after deterministic mapping and equal-split allocation.

    Important:
    - CSV and parquet file sizes are informational only.
    - Row counts are not expected to match because allocation expands rows.
    - Value totals should match after allocation.
    - Expected applied rows are checked against the complete mapping-weight table.
    """

    target_hs = cfg.apply.target_hs
    output_format = cfg.apply.write.format.strip().lower()
    raw_format = cfg.coverage.input.format

    input_dir = cfg.apply_trade_data_dir()
    input_pattern = cfg.coverage.input.pattern

    if input_pattern is None:
        raise ValueError("coverage.input.pattern is required for applied-output audit.")

    input_glob = input_dir / input_pattern

    applied_path = cfg.apply_output_path()
    weights_path = cfg.apply_weights_path()
    audit_path = cfg.apply_integrity_audit_path()

    year_col = cfg.columns.year
    hs_col = cfg.columns.hs
    cmd_col = cfg.columns.cmd
    value_col = cfg.columns.value
    threads = cfg.engine.threads

    if output_format != "parquet":
        raise ValueError(
            "applied_output_audit currently supports only apply.write.format='parquet'. "
            f"Got {output_format!r}."
        )

    if not applied_path.exists():
        raise FileNotFoundError(f"Applied output does not exist: {applied_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Complete mapping weights do not exist: {weights_path}")

    raw_size = file_tree_size(input_dir, input_pattern)
    parquet_size = applied_path.stat().st_size

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)}")

    try:
        parquet_sql = _sql_literal(applied_path)
        weights_sql = _sql_literal(weights_path)

        logger.info("Auditing HS-sync applied output.")
        logger.info("  Run ID: %s", run_id)
        logger.info("  Raw input glob: %s", input_glob)
        logger.info("  Complete mapping weights: %s", weights_path)
        logger.info("  Applied output: %s", applied_path)
        logger.info("  Applied output format: %s", output_format)
        logger.info("  Audit output: %s", audit_path)

        _create_audit_views(
                con=con,
                raw_input_glob=input_glob,
                raw_format=raw_format,
                parquet_sql=parquet_sql,
                weights_sql=weights_sql,
                year_col=year_col,
                hs_col=hs_col,
                cmd_col=cmd_col,
                value_col=value_col,
                min_year=cfg.coverage.min_year,
                max_year=cfg.coverage.max_year,
            )

        row_level_summary = con.execute(
    f"""
    WITH row_level AS (
        SELECT
            hs_sync_input_row_id,
            COUNT(*) AS output_rows_for_input,
            COUNT(DISTINCT TRY_CAST(primary_value_original AS DOUBLE))
                AS n_original_values,
            MAX(TRY_CAST(primary_value_original AS DOUBLE)) AS original_value,
            SUM(TRY_CAST(primary_value_hs_synced AS DOUBLE)) AS synced_value_sum,
            SUM(TRY_CAST(allocation_weight AS DOUBLE)) AS weight_sum
        FROM read_parquet({parquet_sql})
        GROUP BY hs_sync_input_row_id
    )
    SELECT
        COUNT(*) AS distinct_input_row_ids,

        COUNT(*) FILTER (
            WHERE n_original_values != 1
        ) AS input_rows_with_inconsistent_original_value,

        COUNT(*) FILTER (
            WHERE abs(weight_sum - 1.0) > 1e-9
        ) AS input_rows_with_bad_weight_sum,

        COUNT(*) FILTER (
            WHERE abs(original_value - synced_value_sum)
                  > greatest(1e-7, abs(original_value) * 1e-10)
        ) AS input_rows_with_bad_value_sum,

        MIN(output_rows_for_input) AS min_output_rows_per_input,
        MAX(output_rows_for_input) AS max_output_rows_per_input,
        AVG(output_rows_for_input) AS avg_output_rows_per_input
    FROM row_level
    """
    ).fetchdf().iloc[0]

        required_columns_row, _ = _required_column_audit(con=con)
        summaries = _build_summary_tables(
            con=con,
            target_hs=target_hs,
            year_col=year_col,
        )
        pair_coverage = _build_pair_coverage_audit(con=con)
        nulls = _build_null_audit(con=con)
        content = _build_content_audit(con=con)
        weight_ref = _build_weight_reference_audit(con=con)
        expansion = _build_expected_expansion_audit(con=con)

        raw_summary = summaries["raw"]
        applied_summary = summaries["applied"]
        weight_summary = summaries["weights"]

        raw_value = _to_float(raw_summary["raw_value_total"])
        applied_value = _to_float(applied_summary["applied_value_total"])
        value_diff = raw_value - applied_value
        value_rel_diff = abs(value_diff) / abs(raw_value) if raw_value else 0.0

        expected_applied_rows = _to_int(expansion["expected_applied_rows"])
        actual_applied_rows = _to_int(expansion["actual_applied_rows"])

        audit_rows = [
            _audit_row(
                check="raw_file_size",
                expected="recorded",
                actual=format_bytes(raw_size),
                status="info",
                detail=str(input_glob),
            ),
            _audit_row(
                check="applied_parquet_size",
                expected="recorded",
                actual=format_bytes(parquet_size),
                status="info",
                detail=str(applied_path),
            ),
            required_columns_row,
            _audit_row(
                check="raw_rows_defined",
                expected="> 0",
                actual=raw_summary["raw_rows"],
                status=_status_pass(_to_int(raw_summary["raw_rows"]) > 0),
            ),
            _audit_row(
                check="applied_rows_defined",
                expected="> 0",
                actual=applied_summary["applied_rows"],
                status=_status_pass(_to_int(applied_summary["applied_rows"]) > 0),
                detail="output rows may exceed raw rows because split mappings expand rows",
            ),
            _audit_row(
                check="raw_primary_value_parseable",
                expected=0,
                actual=raw_summary["raw_null_value_rows"],
                status=_status_pass(_to_int(raw_summary["raw_null_value_rows"]) == 0),
                detail="Rows with non-parseable raw PrimaryValue inside configured year range.",
            ),
            _audit_row(
                check="all_raw_source_pairs_present_in_output",
                expected=0,
                actual=pair_coverage["missing_source_pairs"],
                status=_status_pass(pair_coverage["missing_source_pairs"] == 0),
                detail=(
                    f"raw_source_pairs={raw_summary['raw_source_pairs']}; "
                    f"applied_source_pairs={applied_summary['applied_source_pairs']}"
                ),
            ),
            _audit_row(
                check="no_extra_output_source_pairs",
                expected=0,
                actual=pair_coverage["extra_output_source_pairs"],
                status=_status_pass(pair_coverage["extra_output_source_pairs"] == 0),
                detail="Applied output should not contain source pairs absent from raw input.",
            ),
            _audit_row(
                check="all_output_rows_target_hs",
                expected=0,
                actual=applied_summary["rows_not_target_hs"],
                status=_status_pass(_to_int(applied_summary["rows_not_target_hs"]) == 0),
                detail=f"target_hs={target_hs}",
            ),
            _audit_row(
                check="value_preserved_after_hs_sync",
                expected=raw_value,
                actual=applied_value,
                status=_status_pass(value_rel_diff <= 1e-10),
                detail=f"diff={value_diff}; rel_diff={value_rel_diff}",
            ),
            _audit_row(
                check="expected_applied_row_count",
                expected=expected_applied_rows,
                actual=actual_applied_rows,
                status=_status_pass(expected_applied_rows == actual_applied_rows),
                detail=(
                    "Expected rows = sum over raw source pairs of "
                    "raw_rows_for_pair * number_of_weight_targets_for_pair."
                ),
            ),
            _audit_row(
                check="no_raw_source_pairs_without_weights",
                expected=0,
                actual=expansion["raw_source_pairs_without_weights"],
                status=_status_pass(
                    _to_int(expansion["raw_source_pairs_without_weights"]) == 0
                ),
            ),
            _audit_row(
                check="no_source_pair_row_count_mismatch",
                expected=0,
                actual=expansion["source_pairs_with_row_count_mismatch"],
                status=_status_pass(
                    _to_int(expansion["source_pairs_with_row_count_mismatch"]) == 0
                ),
            ),
            _audit_row(
                check="allocation_weights_valid_range",
                expected=0,
                actual=content["invalid_allocation_weight_rows"],
                status=_status_pass(_to_int(content["invalid_allocation_weight_rows"]) == 0),
                detail="allocation_weight must be > 0 and <= 1.",
            ),
            _audit_row(
                check="source_cmd_original_valid_6_digit",
                expected=0,
                actual=content["invalid_source_cmd_rows"],
                status=_status_pass(_to_int(content["invalid_source_cmd_rows"]) == 0),
            ),
            _audit_row(
                check="target_cmd_valid_6_digit",
                expected=0,
                actual=content["invalid_target_cmd_rows"],
                status=_status_pass(_to_int(content["invalid_target_cmd_rows"]) == 0),
            ),
            _audit_row(
                check="row_value_formula_valid",
                expected=0,
                actual=content["invalid_row_value_formula_rows"],
                status=_status_pass(
                    _to_int(content["invalid_row_value_formula_rows"]) == 0
                ),
                detail="primary_value_hs_synced must equal primary_value_original * allocation_weight.",
            ),
            _audit_row(
                check="applied_mapping_rows_exist_in_weights",
                expected=0,
                actual=weight_ref["applied_mapping_rows_missing_from_weights"],
                status=_status_pass(
                    _to_int(weight_ref["applied_mapping_rows_missing_from_weights"]) == 0
                ),
                detail=(
                    f"applied_distinct_mapping_rows="
                    f"{weight_ref['applied_distinct_mapping_rows']}"
                ),
            ),
            _audit_row(
                check="applied_allocation_weights_match_reference",
                expected=0,
                actual=weight_ref["applied_mapping_rows_with_wrong_weight"],
                status=_status_pass(
                    _to_int(weight_ref["applied_mapping_rows_with_wrong_weight"]) == 0
                ),
            ),
            _audit_row(
                check="complete_weight_rows_used_by_current_raw_scope",
                expected=0,
                actual=weight_ref["unused_weight_rows"],
                status="info",
                detail=(
                    "Information only. Nonzero can be acceptable if the complete "
                    "weights table covers a broader source-code universe than current raw input."
                ),
            ),
            _audit_row(
                check="weight_table_rows",
                expected="recorded",
                actual=weight_summary["weight_rows"],
                status="info",
                detail=(
                    f"weight_source_pairs={weight_summary['weight_source_pairs']}; "
                    f"invalid_weight_rows={weight_summary['invalid_weight_rows']}"
                ),
            ),
            _audit_row(
                check="expanded_original_value_recorded",
                expected="recorded",
                actual=applied_summary["expanded_original_value"],
                status="info",
                detail="This is larger than raw value because split rows repeat original value.",
            ),
            _audit_row(
                check="raw_year_range",
                expected=f"{cfg.coverage.min_year}-{cfg.coverage.max_year}",
                actual=(
                    f"{int(raw_summary['raw_min_year'])}-"
                    f"{int(raw_summary['raw_max_year'])}"
                ),
                status="pass",
            ),
            _audit_row(
                check="applied_year_range",
                expected=f"{cfg.coverage.min_year}-{cfg.coverage.max_year}",
                actual=(
                    f"{int(applied_summary['applied_min_year'])}-"
                    f"{int(applied_summary['applied_max_year'])}"
                ),
                status=_status_pass(
                    int(applied_summary["applied_min_year"]) == int(cfg.coverage.min_year)
                    and int(applied_summary["applied_max_year"]) == int(cfg.coverage.max_year)
                ),
                detail="Ensures final applied output is fully scoped to configured years.",
            ),
            _audit_row(
                check="target_pair_count",
                expected="recorded",
                actual=applied_summary["applied_target_pairs"],
                status="info",
            ),
            _audit_row(
                check="hs_sync_input_row_id_present_for_every_raw_row",
                expected=raw_summary["raw_rows"],
                actual=row_level_summary["distinct_input_row_ids"],
                status=(
                    "pass"
                    if int(row_level_summary["distinct_input_row_ids"]) == int(raw_summary["raw_rows"])
                    else "fail"
                ),
                detail=(
                    "Each raw input row in scope should appear as one "
                    "hs_sync_input_row_id after HS-sync expansion."
                ),
            ),
            _audit_row(
                check="row_level_original_value_consistent",
                expected=0,
                actual=row_level_summary["input_rows_with_inconsistent_original_value"],
                status=(
                    "pass"
                    if int(row_level_summary["input_rows_with_inconsistent_original_value"]) == 0
                    else "fail"
                ),
                detail=(
                    "All expanded rows from one input row should carry the same "
                    "primary_value_original."
                ),
            ),
            _audit_row(
                check="row_level_allocation_weights_sum_to_one",
                expected=0,
                actual=row_level_summary["input_rows_with_bad_weight_sum"],
                status=(
                    "pass"
                    if int(row_level_summary["input_rows_with_bad_weight_sum"]) == 0
                    else "fail"
                ),
                detail=(
                    "For each input row, allocation weights across expanded target rows "
                    "should sum to 1."
                ),
            ),
            _audit_row(
                check="row_level_value_preserved_after_expansion",
                expected=0,
                actual=row_level_summary["input_rows_with_bad_value_sum"],
                status=(
                    "pass"
                    if int(row_level_summary["input_rows_with_bad_value_sum"]) == 0
                    else "fail"
                ),
                detail=(
                    "For each input row, sum(primary_value_hs_synced) should equal "
                    "primary_value_original."
                ),
            ),
            _audit_row(
                check="row_level_expansion_distribution",
                expected="recorded",
                actual=(
                    f"min={row_level_summary['min_output_rows_per_input']}; "
                    f"max={row_level_summary['max_output_rows_per_input']}; "
                    f"avg={row_level_summary['avg_output_rows_per_input']}"
                ),
                status="info",
                detail="Number of output rows created per original input row.",
            ),
        ]

        for column_name, null_count in nulls.items():
            audit_rows.append(
                _audit_row(
                    check=f"no_null_{column_name.removeprefix('null_')}",
                    expected=0,
                    actual=null_count,
                    status=_status_pass(_to_int(null_count) == 0),
                )
            )

        audit_rows.extend(_build_distribution_info(con=con))


        ## Make final audit DataFrame and write to CSV
        audit = pd.DataFrame(audit_rows)

        failed = audit[audit["status"] == "fail"]
        ready_for_cif_fob = failed.empty

        audit_rows.append(
            _audit_row(
                    check="ready_for_cif_fob",
                    expected=True,
                    actual=ready_for_cif_fob,
                    status="pass" if ready_for_cif_fob else "fail",
                    detail="HS-sync output passed integrity audit and is ready for CIF-FOB correction.",
                )
            )  

        audit = pd.DataFrame(audit_rows)
        failed = audit.loc[
                audit["status"].astype(str).str.strip().str.lower() == "fail"
            ]

        audit_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        audit.to_csv(
            audit_path,
            index=False,
        )

        logger.info("HS-sync applied-output integrity audit complete.")
        logger.info("  Input file size: %s", format_bytes(raw_size))
        logger.info("  Parquet file size: %s", format_bytes(parquet_size))
        logger.info("  Input rows: %s", f"{_to_int(raw_summary['raw_rows']):,}")
        logger.info("  Applied rows: %s", f"{_to_int(applied_summary['applied_rows']):,}")
        logger.info("  Expected applied rows: %s", f"{expected_applied_rows:,}")
        logger.info("  Input value: %.6f", raw_value)
        logger.info("  Applied synced value: %.6f", applied_value)
        logger.info("  Relative value diff: %.3e", value_rel_diff)
        logger.info("  Failed checks: %s", len(failed))
        logger.info("  Audit file: %s", audit_path)

        if not failed.empty:
            raise ValueError(
                "HS-sync applied-output integrity audit failed:\n"
                + failed.to_string(index=False)
            )

        return {
            "status": "completed",
            "run_id": run_id,
            
            "audit_path": audit_path,
            "applied_path": applied_path,
            "weights_path": weights_path,

            "ready_for_cif_fob": ready_for_cif_fob,
            "audit_passed": ready_for_cif_fob,
            "n_failed_audit_rows": len(failed),

            "raw_file_size": raw_size,
            "applied_parquet_size": parquet_size,
            "raw_rows": _to_int(raw_summary["raw_rows"]),
            "applied_rows": _to_int(applied_summary["applied_rows"]),
            "expected_applied_rows": expected_applied_rows,
            "raw_value_total": raw_value,
            "applied_value_total": applied_value,
            "value_diff": value_diff,
            "value_rel_diff": value_rel_diff,
        }

    finally:
        con.close()
