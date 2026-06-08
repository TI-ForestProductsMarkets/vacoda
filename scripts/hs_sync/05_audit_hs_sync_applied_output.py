"""
Audit HS-sync applied output against raw Comtrade input.

This script checks that the generated HS-synced parquet preserves the raw
input coverage after deterministic mapping and equal-split allocation.

Important:
- CSV and parquet file sizes are not expected to match.
- Row counts are not expected to match because equal-split allocation expands
  some rows.
- Value totals should match after allocation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.paths import resolve_log_path
from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)


logger = get_logger("SCRIPT.HS_SYNC_APPLIED_AUDIT")


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


def sql_literal(value: str | Path) -> str:
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def main() -> None:
    cfg = HsSyncConfig.from_yaml()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_level = parse_log_level(cfg.logging.level)
    log_path = resolve_log_path(
        cfg.logging.resolve_log_file(),
        timestamp=cfg.logging.timestamp,
    )

    setup_logging(level=log_level, log_file=log_path)

    target_hs = cfg.coverage.target_hs
    target_hs_lower = target_hs.lower()

    input_dir = cfg.coverage_input_dir()
    input_pattern = cfg.coverage.input.pattern
    input_glob = input_dir / input_pattern

    output_dir = Path(
        f"/storage/vacoda/data/interim/hs_sync/target_{target_hs_lower}_applied"
    )

    applied_path = output_dir / f"comtrade_target_{target_hs_lower}_hs_synced.parquet"

    audit_path = output_dir / f"comtrade_target_{target_hs_lower}_hs_sync_integrity_audit.csv"

    year_col = cfg.columns.year
    hs_col = cfg.columns.hs
    cmd_col = cfg.columns.cmd
    value_col = cfg.columns.value
    threads = cfg.engine.threads

    if not applied_path.exists():
        raise FileNotFoundError(f"Applied parquet does not exist: {applied_path}")
    else:
        logger.info("Found applied parquet file: %s", applied_path)

    raw_size = file_tree_size(input_dir, input_pattern)
    parquet_size = applied_path.stat().st_size

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)}")

    try:
        raw_sql = sql_literal(input_glob)
        parquet_sql = sql_literal(applied_path)

        logger.info("Auditing HS-sync applied output.")
        logger.info("  Raw input glob: %s", input_glob)
        logger.info("  Applied parquet: %s", applied_path)
        logger.info("  Audit output: %s", audit_path)

        raw_summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS raw_rows,
                COUNT(DISTINCT CAST("{hs_col}" AS VARCHAR) || ':' ||
                               lpad(CAST("{cmd_col}" AS VARCHAR), 6, '0'))
                    AS raw_source_pairs,
                SUM(TRY_CAST("{value_col}" AS DOUBLE)) AS raw_value_total,
                MIN(CAST("{year_col}" AS INTEGER)) AS raw_min_year,
                MAX(CAST("{year_col}" AS INTEGER)) AS raw_max_year
            FROM read_csv_auto(
                {raw_sql},
                HEADER = TRUE,
                ALL_VARCHAR = TRUE,
                union_by_name = TRUE
            )
            WHERE CAST("{year_col}" AS INTEGER)
                  BETWEEN {int(cfg.coverage.min_year)} AND {int(cfg.coverage.max_year)}
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
                SUM(CAST(primary_value_hs_synced AS DOUBLE)) AS applied_value_total,
                SUM(CAST(primary_value_original AS DOUBLE)) AS expanded_original_value,
                SUM(CASE WHEN target_hs = '{target_hs}' THEN 1 ELSE 0 END)
                    AS rows_with_target_hs,
                COUNT(*) - SUM(CASE WHEN target_hs = '{target_hs}' THEN 1 ELSE 0 END)
                    AS rows_not_target_hs
            FROM read_parquet({parquet_sql})
            """
        ).fetchdf().iloc[0]

        missing_source_pairs = con.execute(
            f"""
            WITH raw_pairs AS (
                SELECT DISTINCT
                    CAST("{hs_col}" AS VARCHAR) AS source_hs,
                    lpad(CAST("{cmd_col}" AS VARCHAR), 6, '0') AS source_cmd
                FROM read_csv_auto(
                    {raw_sql},
                    HEADER = TRUE,
                    ALL_VARCHAR = TRUE,
                    union_by_name = TRUE
                )
                WHERE CAST("{year_col}" AS INTEGER)
                      BETWEEN {int(cfg.coverage.min_year)} AND {int(cfg.coverage.max_year)}
            ),
            output_pairs AS (
                SELECT DISTINCT
                    source_hs_original AS source_hs,
                    source_cmd_original AS source_cmd
                FROM read_parquet({parquet_sql})
            )
            SELECT COUNT(*) AS missing_source_pairs
            FROM raw_pairs r
            LEFT JOIN output_pairs o
            USING (source_hs, source_cmd)
            WHERE o.source_cmd IS NULL
            """
        ).fetchdf().iloc[0]["missing_source_pairs"]

        raw_value = float(raw_summary["raw_value_total"])
        applied_value = float(applied_summary["applied_value_total"])
        value_diff = raw_value - applied_value
        value_rel_diff = abs(value_diff) / abs(raw_value) if raw_value else 0.0

        audit_rows = [
            {
                "check": "raw_file_size",
                "expected": "recorded",
                "actual": format_bytes(raw_size),
                "status": "info",
                "detail": str(input_glob),
            },
            {
                "check": "applied_parquet_size",
                "expected": "recorded",
                "actual": format_bytes(parquet_size),
                "status": "info",
                "detail": str(applied_path),
            },
            {
                "check": "raw_rows_defined",
                "expected": "> 0",
                "actual": raw_summary["raw_rows"],
                "status": "pass" if raw_summary["raw_rows"] > 0 else "fail",
                "detail": "",
            },
            {
                "check": "applied_rows_defined",
                "expected": "> 0",
                "actual": applied_summary["applied_rows"],
                "status": "pass" if applied_summary["applied_rows"] > 0 else "fail",
                "detail": "output rows may exceed raw rows because equal-split expands rows",
            },
            {
                "check": "all_raw_source_pairs_present_in_output",
                "expected": 0,
                "actual": missing_source_pairs,
                "status": "pass" if int(missing_source_pairs) == 0 else "fail",
                "detail": (
                    f"raw_source_pairs={raw_summary['raw_source_pairs']}; "
                    f"applied_source_pairs={applied_summary['applied_source_pairs']}"
                ),
            },
            {
                "check": "all_output_rows_target_hs",
                "expected": 0,
                "actual": applied_summary["rows_not_target_hs"],
                "status": "pass" if int(applied_summary["rows_not_target_hs"]) == 0 else "fail",
                "detail": f"target_hs={target_hs}",
            },
            {
                "check": "value_preserved_after_hs_sync",
                "expected": raw_value,
                "actual": applied_value,
                "status": "pass" if value_rel_diff <= 1e-10 else "fail",
                "detail": f"diff={value_diff}; rel_diff={value_rel_diff}",
            },
            {
                "check": "expanded_original_value_recorded",
                "expected": "recorded",
                "actual": applied_summary["expanded_original_value"],
                "status": "info",
                "detail": "This is larger than raw value because split rows repeat original value.",
            },
            {
                "check": "raw_year_range",
                "expected": f"{cfg.coverage.min_year}-{cfg.coverage.max_year}",
                "actual": f"{int(raw_summary['raw_min_year'])}-{int(raw_summary['raw_max_year'])}",
                "status": "pass",
                "detail": "",
            },
            {
                "check": "target_pair_count",
                "expected": "recorded",
                "actual": applied_summary["applied_target_pairs"],
                "status": "info",
                "detail": "",
            },
        ]

        audit = pd.DataFrame(audit_rows)
        audit.to_csv(audit_path, index=False)

        failed = audit[audit["status"] == "fail"]

        logger.info("HS-sync applied-output integrity audit complete.")
        logger.info("  Raw file size: %s", format_bytes(raw_size))
        logger.info("  Parquet file size: %s", format_bytes(parquet_size))
        logger.info("  Raw rows: %s", f"{int(raw_summary['raw_rows']):,}")
        logger.info("  Applied rows: %s", f"{int(applied_summary['applied_rows']):,}")
        logger.info("  Raw value: %.6f", raw_value)
        logger.info("  Applied synced value: %.6f", applied_value)
        logger.info("  Relative value diff: %.3e", value_rel_diff)
        logger.info("  Failed checks: %s", len(failed))
        logger.info("  Audit file: %s", audit_path)
        logger.info("  Log file: %s", log_path)

        if not failed.empty:
            raise ValueError(
                "HS-sync applied-output integrity audit failed:\n"
                + failed.to_string(index=False)
            )

    finally:
        con.close()


if __name__ == "__main__":
    main()