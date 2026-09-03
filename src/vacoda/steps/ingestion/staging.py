"""
src.vacoda.ingestion.staging.py

This module contains the logic for building the Comtrade ingestion layer from raw CSV files.
The ingestion layer is the boundary between raw data and staged data, and it performs the following tasks:
- reads many raw CSV files
- validates required raw columns
- writes one typed standardized parquet
- writes a small audit manifest
"""
from __future__ import annotations

from logging import Logger
from pathlib import Path

import duckdb
import pandas as pd
from vacoda.config.ingestion import IngestionConfig
from vacoda.utils.sql import sql_literal

COMTRADE_QUANTITY_UNIT_LABELS = {
    2: "m²",
    4: "m",
    5: "u",
    6: "2u",
    7: "l",
    8: "kg",
    9: "1000u",
    10: "U (jeu/pack)",
    11: "12u",
    12: "m³",
}


def _duckdb_type(dtype: str) -> str:
    dtype = dtype.strip().lower()

    mapping = {
        "string": "VARCHAR",
        "int8": "TINYINT",
        "int16": "SMALLINT",
        "int32": "INTEGER",
        "int64": "BIGINT",
        "float64": "DOUBLE",
        "boolean": "BOOLEAN",
        "bool": "BOOLEAN",
    }

    if dtype not in mapping:
        raise ValueError(f"Unsupported staging dtype: {dtype!r}")

    return mapping[dtype]

def _quantity_unit_expression(
    *,
    raw_code_col: str,
    raw_abbr_col: str,
    out_col: str,
) -> str:
    cases = "\n".join(
        f'WHEN TRY_CAST("{raw_code_col}" AS INTEGER) = {code} '
        f"THEN '{label}'"
        for code, label in COMTRADE_QUANTITY_UNIT_LABELS.items()
    )

    return f"""
        CASE
            {cases}
            ELSE CAST("{raw_abbr_col}" AS VARCHAR)
        END AS {out_col}
    """

def _log_staging_summary(
    *,
    logger: Logger,
    status: str,
    ready_for_hs_sync: bool,
    output_path: Path,
    audit_path: Path,
    row: pd.Series,
    failed_count: int,
    n_input_files: int,
) -> None:
    logger.info("=" * 72)
    logger.info("COMTRADE STAGING COMPLETED")
    logger.info("=" * 72)
    logger.info("Status: %s", status)
    logger.info("Ready for HS-sync: %s", "YES" if ready_for_hs_sync else "NO")
    logger.info("Input files: %s", f"{n_input_files:,}")
    logger.info("Rows: %s", f"{int(row['n_rows']):,}")
    logger.info("Year range: %s-%s", int(row["min_year"]), int(row["max_year"]))
    logger.info("Source HS-code pairs: %s", f"{int(row['source_pairs']):,}")
    logger.info("Primary value total: %.2f", float(row["primary_value_total"]))
    logger.info("Failed audit checks: %s", failed_count)
    logger.info("Staged parquet:")
    logger.info("  %s", output_path)
    logger.info("Audit:")
    logger.info("  %s", audit_path)
    logger.info("=" * 72)

def _cast_expression(*, raw_col: str, out_col: str, dtype: str) -> str:
    duck_type = _duckdb_type(dtype)

    # HS commodity codes must stay string and must be 6-digit padded.
    if out_col == "source_cmd":
        return (
            f"CASE "
            f"WHEN regexp_matches(CAST(\"{raw_col}\" AS VARCHAR), '^[0-9]+$') "
            f"THEN lpad(CAST(\"{raw_col}\" AS VARCHAR), 6, '0') "
            f"ELSE CAST(\"{raw_col}\" AS VARCHAR) "
            f"END AS {out_col}"
        )

    if dtype.strip().lower() == "string":
        return f"CAST(\"{raw_col}\" AS VARCHAR) AS {out_col}"

    return f"TRY_CAST(\"{raw_col}\" AS {duck_type}) AS {out_col}"


def _unit_consistency_condition(
    *,
    code_col: str,
    unit_col: str,
) -> str:
    conditions = [
        (
            f"({code_col} = {code} "
            f"AND {unit_col} <> '{label}')"
        )
        for code, label
        in COMTRADE_QUANTITY_UNIT_LABELS.items()
    ]

    return " OR ".join(conditions)

def build_comtrade_ingestion_layer(
    *,
    cfg: IngestionConfig,
    run_id: str,
    force: bool = False,
    logger: Logger,
) -> dict[str, object]:
    """
    Build staged Comtrade parquet from raw API CSV files.

    This is the raw -> staging boundary:
    - reads many raw CSV files
    - validates required raw columns
    - writes one typed standardized parquet
    - writes a small audit manifest
    """

    if cfg.ingestion.mode != "multi_file":
        raise ValueError(
            f"Expected ingestion.mode='multi_file' for staging, got {cfg.ingestion.mode!r}."
        )

    if cfg.input.format != "csv":
        raise ValueError(f"Expected input.format='csv', got {cfg.input.format!r}.")

    if cfg.output.format != "parquet":
        raise ValueError(f"Expected output.format='parquet', got {cfg.output.format!r}.")

    input_glob = cfg.input_glob()
    output_path = cfg.output_path()
    audit_path = cfg.audit_path()

    if output_path.exists():
        if output_path.is_dir():
            raise IsADirectoryError(
                f"Expected output file but found directory: {output_path}"
            )

        if not force and not cfg.write.overwrite:
            logger.warning("=" * 72)
            logger.warning("COMTRADE STAGING SKIPPED")
            logger.warning("=" * 72)
            logger.warning(
                "Reason: output already exists and "
                "write.overwrite=false and force=false."
            )
            logger.warning("Existing staged parquet:")
            logger.warning("  %s", output_path)
            logger.warning("Audit file:")
            logger.warning("  %s", audit_path)
            logger.warning(
                "Ready for HS-sync: YES, using existing staged parquet."
            )
            logger.warning("=" * 72)

            return {
                "status": "skipped_existing",
                "ready_for_next_step": True,
                "ready_for_hs_sync": True,
                "staged_parquet": output_path,
                "audit_file": audit_path,
                "output_path": output_path,
                "audit_path": audit_path,
                "n_failed_audit_rows": 0,
            }

        logger.info(
            "Output will be overwritten: %s%s",
            output_path,
            " (force=True)" if force else "",
        )
        output_path.unlink()
    else:
        logger.info("Output will be created: %s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    mapping = cfg.schema.standardized.field_mapping
    dtypes = cfg.schema.standardized.dtypes

    select_exprs: list[str] = []
    for raw_col, out_col in mapping.items():
        if out_col not in dtypes:
            raise KeyError(
                f"Missing dtype for standardized field {out_col!r} "
                f"mapped from raw column {raw_col!r}."
            )

        if out_col == "quantity_unit":
            select_exprs.append(
                _quantity_unit_expression(
                    raw_code_col="QtyUnitCode",
                    raw_abbr_col="QtyUnitAbbr",
                    out_col=out_col,
                )
            )
            continue

        if out_col == "alt_quantity_unit":
            select_exprs.append(
                _quantity_unit_expression(
                    raw_code_col="AltQtyUnitCode",
                    raw_abbr_col="AltQtyUnitAbbr",
                    out_col=out_col,
                )
            )
            continue

        select_exprs.append(
            _cast_expression(
                raw_col=raw_col,
                out_col=out_col,
                dtype=dtypes[out_col],
            )
        )

    select_sql = ",\n            ".join(select_exprs)

    input_sql = sql_literal(input_glob)
    output_sql = sql_literal(output_path)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")

    input_files = sorted(input_glob.parent.glob(input_glob.name))
    n_input_files = len(input_files)

    if n_input_files == 0:
        raise FileNotFoundError(f"No input files found for glob: {input_glob}")

    logger.info("  Input files: %s", f"{n_input_files:,}")

    try:
        logger.info("Building Comtrade ingestion layer.")
        logger.info("  Engine: %s", cfg.engine.name)
        logger.info("  Threads: %s", cfg.engine.threads)
        logger.info("  Input glob: %s", input_glob)
        logger.info("  Output path: %s", output_path)
        logger.info("  Audit path: %s", audit_path)

        # Register raw view.
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW raw_input AS
            SELECT *
            FROM read_csv_auto(
                {input_sql},
                HEADER = TRUE,
                ALL_VARCHAR = TRUE,
                union_by_name = TRUE
            )
            """
        )

        # Validate raw required fields.
        schema_df = con.execute("DESCRIBE SELECT * FROM raw_input").fetchdf()
        raw_columns = set(schema_df["column_name"].astype(str))

        missing_raw = [
            col
            for col in cfg.schema.raw.required_fields
            if col not in raw_columns
        ]

        if missing_raw:
            raise ValueError(
                "Raw input is missing required columns: "
                + ", ".join(missing_raw)
            )

        # Write staged parquet.
        con.execute(
            f"""
            COPY (
                SELECT
                    {select_sql}
                FROM raw_input
            )
            TO {output_sql}
            (FORMAT PARQUET)
            """
        )

        quantity_condition = _unit_consistency_condition(
            code_col="quantity_unit_code",
            unit_col="quantity_unit",
        )

        alt_quantity_condition = _unit_consistency_condition(
            code_col="alt_quantity_unit_code",
            unit_col="alt_quantity_unit",
        )

        unit_audit = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE {quantity_condition}
                ) AS bad_quantity_units,

                COUNT(*) FILTER (
                    WHERE {alt_quantity_condition}
                ) AS bad_alt_quantity_units

            FROM read_parquet({output_sql})
            """
        ).fetchone()

        (
            bad_quantity_units,
            bad_alt_quantity_units,
        ) = unit_audit

        n_unit_inconsistencies = sum(unit_audit)

        row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,
                MIN(year) AS min_year,
                MAX(year) AS max_year,
                COUNT(DISTINCT source_hs || ':' || source_cmd) AS source_pairs,
                SUM(primary_value) AS primary_value_total,
                COUNT(*) FILTER (WHERE primary_value IS NULL) AS null_primary_value_rows,
                COUNT(*) FILTER (WHERE source_hs IS NULL) AS null_source_hs_rows,
                COUNT(*) FILTER (WHERE source_cmd IS NULL) AS null_source_cmd_rows
            FROM read_parquet({output_sql})
            """
        ).fetchdf().iloc[0]

        ready_for_hs_sync = (
                int(row["n_rows"]) > 0
                and int(row["null_primary_value_rows"]) == 0
                and int(row["null_source_hs_rows"]) == 0
                and int(row["null_source_cmd_rows"]) == 0
            )

        audit_rows = [
            {
                "check": "raw_required_columns_present",
                "expected": 0,
                "actual": len(missing_raw),
                "status": "pass" if not missing_raw else "fail",
                "detail": ",".join(missing_raw),
            },
            {
                "check": "staged_rows_defined",
                "expected": "> 0",
                "actual": int(row["n_rows"]),
                "status": "pass" if int(row["n_rows"]) > 0 else "fail",
                "detail": "",
            },
            {
                "check": "staged_year_range",
                "expected": "recorded",
                "actual": f"{int(row['min_year'])}-{int(row['max_year'])}",
                "status": "info",
                "detail": "",
            },
            {
                "check": "staged_source_pair_count",
                "expected": "recorded",
                "actual": int(row["source_pairs"]),
                "status": "info",
                "detail": "",
            },
            {
                "check": "staged_primary_value_total",
                "expected": "recorded",
                "actual": float(row["primary_value_total"]),
                "status": "info",
                "detail": "",
            },
            {
                "check": "no_null_primary_value",
                "expected": 0,
                "actual": int(row["null_primary_value_rows"]),
                "status": "pass" if int(row["null_primary_value_rows"]) == 0 else "fail",
                "detail": "",
            },
            {
                "check": "no_null_source_hs",
                "expected": 0,
                "actual": int(row["null_source_hs_rows"]),
                "status": "pass" if int(row["null_source_hs_rows"]) == 0 else "fail",
                "detail": "",
            },
            {
                "check": "no_null_source_cmd",
                "expected": 0,
                "actual": int(row["null_source_cmd_rows"]),
                "status": "pass" if int(row["null_source_cmd_rows"]) == 0 else "fail",
                "detail": "",
            },
            {
                "check": "quantity_unit_code_label_consistency",
                "expected": 0,
                "actual": int(n_unit_inconsistencies),
                "status": (
                    "pass"
                    if n_unit_inconsistencies == 0
                    else "fail"
                ),
                "detail": (
                    f"quantity_units={bad_quantity_units}; "
                    f"alt_quantity_units={bad_alt_quantity_units}"
                ),
            },
        ]

        audit = pd.DataFrame(audit_rows)
        failed = audit[audit["status"] == "fail"]

        ready_for_hs_sync = ready_for_hs_sync and failed.empty

        ready_row = pd.DataFrame(
        [
            {
            "check": "ready_for_hs_sync",
            "expected": True,
            "actual": ready_for_hs_sync,
            "status": "pass" if ready_for_hs_sync else "fail",
            "detail": "Staged parquet satisfies the minimum input contract for HS-sync.",
            }
        ]
        )

        audit = pd.concat([audit, ready_row], ignore_index=True)

        audit.insert(0, "run_id", run_id)
        audit.insert(1, "dataset_name", cfg.ingestion.name)
        audit.insert(2, "schema_name", cfg.schema.name)
        audit.insert(3, "schema_version", cfg.schema.version)
        audit.to_csv(audit_path, index=False)

        failed = audit[audit["status"] == "fail"]

        if not cfg.write.allow_empty and int(row["n_rows"]) == 0:
            raise ValueError("Staging output is empty and write.allow_empty=false.")

        if not failed.empty:
            raise ValueError(
                "Comtrade staging audit failed:\n"
                + failed.to_string(index=False)
            )


        logger.info("Comtrade ingestion built successfully.")
        logger.info("  Rows: %s", f"{int(row['n_rows']):,}")
        logger.info("  Year range: %s-%s", int(row["min_year"]), int(row["max_year"]))
        logger.info("  Source pairs: %s", f"{int(row['source_pairs']):,}")
        logger.info("  Output: %s", output_path)
        logger.info("  Audit: %s", audit_path)

        _log_staging_summary(
                logger=logger,
                status="PASSED",
                ready_for_hs_sync=ready_for_hs_sync,
                output_path=output_path,
                audit_path=audit_path,
                row=row,
                failed_count=len(failed),
                n_input_files=n_input_files,
            )

        return {
                "status": "completed",
                "ready_for_next_step": ready_for_hs_sync,
                "ready_for_hs_sync": ready_for_hs_sync,
                "staged_parquet": output_path,
                "audit_file": audit_path,
                "output_path": output_path,
                "audit_path": audit_path,
                "n_rows": int(row["n_rows"]),
                "min_year": int(row["min_year"]),
                "max_year": int(row["max_year"]),
                "source_pairs": int(row["source_pairs"]),
                "primary_value_total": float(row["primary_value_total"]),
                "n_failed_audit_rows": len(failed),
                "n_input_files": n_input_files,
            }

    finally:
        con.close()