"""
vacoda.steps.validation.structural

Audit structural integrity of the canonical VACODA Validation input.

This module performs structural and technical validation only. It verifies
that the prepared Validation dataset satisfies the contract required by
downstream model development and scientific validation.

Scientific plausibility, anomaly detection, quantity correction, imputation,
and release decisions do not belong here.
"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any
import pandas as pd
import duckdb

from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal


# ======================================================================
# Helpers
# ======================================================================


def _safe_identifier(name: str) -> str:
    """Return a safely quoted SQL identifier."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"Unsafe Validation column identifier: {name!r}"
        )

    return f'"{name}"'



# ======================================================================
# Public step
# ======================================================================


def audit_structural_integrity(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Audit structural integrity of the prepared Validation input.

    The audit verifies:

    - required Validation columns;
    - canonical-key completeness;
    - canonical-key uniqueness;
    - configured year scope;
    - target HS-code structure;
    - finite and positive trade values when present;
    - finite and positive synchronized quantities when present;
    - consistency between synchronized quantity, method, and status.

    The audit artifact is always written before an audit failure is raised.
    """

    start = perf_counter()

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Prepared Validation input not found: {input_path}"
        )

    output_path = cfg.product_path("structural_audit")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "%s REBUILD — Validation structural audit",
        get_symbols("rebuilt"),
    )
    logger.info("Auditing Validation structural integrity.")
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)

    structural_cfg = cfg.method.structural

    required_fields = tuple(
        structural_cfg.required_fields
    )
    canonical_key = tuple(
        structural_cfg.canonical_key
    )

    require_positive_value = bool(
        structural_cfg.require_positive_value
    )

    require_positive_quantity = bool(
        structural_cfg.require_nonnegative_quantity
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"PRAGMA threads={int(cfg.engine.threads)}"
        )

        input_sql = sql_literal(input_path)

        # --------------------------------------------------------------
        # Schema
        # --------------------------------------------------------------

        source_columns = {
            str(row[0])
            for row in con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet({input_sql})
                """
            ).fetchall()
        }

        missing_required_fields = sorted(
            set(required_fields) - source_columns
        )

        required_fields_present = (
            len(missing_required_fields) == 0
        )

        if not required_fields_present:
            raise RuntimeError(
                "Prepared Validation input is missing required "
                "column(s): "
                + ", ".join(missing_required_fields)
            )

        missing_key_fields = sorted(
            set(canonical_key) - source_columns
        )

        if missing_key_fields:
            raise RuntimeError(
                "Validation canonical key references missing "
                "column(s): "
                + ", ".join(missing_key_fields)
            )

        logger.info(
            "  Required columns: %s",
            len(required_fields),
        )
        logger.info(
            "  Canonical key: %s",
            " × ".join(canonical_key),
        )

        # --------------------------------------------------------------
        # Canonical-key SQL
        # --------------------------------------------------------------

        key_sql = ", ".join(
            _safe_identifier(column)
            for column in canonical_key
        )

        key_null_condition = " OR ".join(
            f"{_safe_identifier(column)} IS NULL"
            for column in canonical_key
        )

        # --------------------------------------------------------------
        # Main structural statistics
        # --------------------------------------------------------------

        stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                MIN(year) AS min_year,
                MAX(year) AS max_year,

                COUNT(*) FILTER (
                    WHERE {key_null_condition}
                ) AS n_null_key_rows,

                COUNT(*) FILTER (
                    WHERE year < ?
                       OR year > ?
                       OR year IS NULL
                ) AS n_invalid_year_rows,

                COUNT(*) FILTER (
                    WHERE target_cmd IS NULL
                    OR NOT regexp_matches(
                            CAST(target_cmd AS VARCHAR),
                            '^[0-9]{{6}}$'
                    )
                ) AS n_invalid_hs_rows,

                COUNT(*) FILTER (
                    WHERE trade_value_reconciled IS NOT NULL
                      AND (
                            NOT isfinite(
                                CAST(
                                    trade_value_reconciled
                                    AS DOUBLE
                                )
                            )
                            OR trade_value_reconciled <= 0
                      )
                ) AS n_invalid_nominal_value_rows,

                COUNT(*) FILTER (
                    WHERE trade_value_reconciled_constant_usd
                          IS NOT NULL
                      AND (
                            NOT isfinite(
                                CAST(
                                    trade_value_reconciled_constant_usd
                                    AS DOUBLE
                                )
                            )
                            OR trade_value_reconciled_constant_usd <= 0
                      )
                ) AS n_invalid_constant_value_rows,

                COUNT(*) FILTER (
                    WHERE quantity_simple_mean_kg IS NOT NULL
                    AND (
                        NOT isfinite(quantity_simple_mean_kg)
                        OR quantity_simple_mean_kg <= 0
                    )
                ) AS n_invalid_simple_mean_rows,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_kg IS NOT NULL
                    AND (
                        NOT isfinite(
                            CAST(trade_synced_quantity_kg AS DOUBLE)
                        )
                        OR trade_synced_quantity_kg <= 0
                    )
                ) AS n_invalid_quantity_rows,

                COUNT(*) FILTER (
                    WHERE quantity_reliability_weighted_kg IS NOT NULL
                    AND (
                        NOT isfinite(quantity_reliability_weighted_kg)
                        OR quantity_reliability_weighted_kg <= 0
                    )
                ) AS n_invalid_quantity_reliability_weighted_rows,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_kg IS NULL
                ) AS n_missing_quantity_rows,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_kg IS NOT NULL
                ) AS n_available_quantity_rows

            FROM read_parquet({input_sql})
            """,
            [
                int(cfg.scope.min_year),
                int(cfg.scope.max_year),
            ],
        ).fetchone()

        (
            n_rows,
            min_year,
            max_year,
            n_null_key_rows,
            n_invalid_year_rows,
            n_invalid_hs_rows,
            n_invalid_nominal_value_rows,
            n_invalid_constant_value_rows,
            n_invalid_simple_mean_rows,
            n_invalid_quantity_rows,
            n_invalid_quantity_reliability_weighted_rows,
            n_missing_quantity_rows,
            n_available_quantity_rows,
        ) = stats

        n_rows = int(n_rows)
        n_null_key_rows = int(n_null_key_rows)
        n_invalid_year_rows = int(n_invalid_year_rows)
        n_invalid_hs_rows = int(n_invalid_hs_rows)
        n_invalid_nominal_value_rows = int(
            n_invalid_nominal_value_rows
        )
        n_invalid_constant_value_rows = int(
            n_invalid_constant_value_rows
        )
        n_invalid_simple_mean_rows = int(
            n_invalid_simple_mean_rows
        )
        n_invalid_quantity_rows = int(
            n_invalid_quantity_rows
        )
        n_invalid_quantity_reliability_weighted_rows = int(
            n_invalid_quantity_reliability_weighted_rows
        )
        n_missing_quantity_rows = int(
            n_missing_quantity_rows
        )
        n_available_quantity_rows = int(
            n_available_quantity_rows
        )

        quantity_candidates_numeric_valid = (
            n_invalid_simple_mean_rows == 0
            and n_invalid_quantity_rows == 0
            and n_invalid_quantity_reliability_weighted_rows == 0
        )
        # --------------------------------------------------------------
        # Duplicate canonical keys
        # --------------------------------------------------------------

        duplicate_stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_duplicate_keys,
                COALESCE(
                    SUM(n_rows - 1),
                    0
                ) AS n_duplicate_rows

            FROM (
                SELECT
                    {key_sql},
                    COUNT(*) AS n_rows

                FROM read_parquet({input_sql})

                GROUP BY
                    {key_sql}

                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()

        n_duplicate_keys = int(
            duplicate_stats[0]
        )
        n_duplicate_rows = int(
            duplicate_stats[1]
        )

        # --------------------------------------------------------------
        # Quantity method/status contract
        # --------------------------------------------------------------

        quantity_contract = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_status
                          NOT IN (
                              'bilateral_synced',
                              'unilateral_retained',
                              'missing'
                          )
                       OR trade_synced_quantity_status IS NULL
                ) AS n_invalid_quantity_status,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_status
                          = 'bilateral_synced'
                      AND (
                            trade_synced_quantity_kg IS NULL
                            OR trade_synced_quantity_method
                               NOT IN (
                                   'unit_value_guided_export',
                                   'unit_value_guided_import',
                                   'unit_value_guided_mean',
                                   'bilateral_mean_no_unit_value_reference'
                               )
                      )
                ) AS n_invalid_bilateral_quantity_contract,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_status
                          = 'unilateral_retained'
                      AND (
                            trade_synced_quantity_kg IS NULL
                            OR trade_synced_quantity_method
                               NOT IN (
                                   'export_quantity_retained',
                                   'import_quantity_retained'
                               )
                      )
                ) AS n_invalid_unilateral_quantity_contract,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_status
                          = 'missing'
                      AND (
                            trade_synced_quantity_kg IS NOT NULL
                            OR trade_synced_quantity_method
                               <> 'no_usable_quantity'
                      )
                ) AS n_invalid_missing_quantity_contract

            FROM read_parquet({input_sql})
            """
        ).fetchone()

        (
            n_invalid_quantity_status,
            n_invalid_bilateral_quantity_contract,
            n_invalid_unilateral_quantity_contract,
            n_invalid_missing_quantity_contract,
        ) = map(int, quantity_contract)

        # --------------------------------------------------------------
        # Individual audit decisions
        # --------------------------------------------------------------

        has_rows = n_rows > 0

        key_complete = (
            n_null_key_rows == 0
        )

        canonical_key_unique = (
            n_duplicate_keys == 0
            and n_duplicate_rows == 0
        )

        year_scope_valid = (
            n_invalid_year_rows == 0
            and min_year is not None
            and max_year is not None
            and int(min_year) >= int(cfg.scope.min_year)
            and int(max_year) <= int(cfg.scope.max_year)
        )

        target_hs_valid = (
            n_invalid_hs_rows == 0
        )

        nominal_value_valid = (
            n_invalid_nominal_value_rows == 0
            if require_positive_value
            else True
        )

        constant_value_valid = (
            n_invalid_constant_value_rows == 0
            if require_positive_value
            else True
        )

        quantity_numeric_valid = (
            quantity_candidates_numeric_valid
            if require_positive_quantity
            else True
        )

        quantity_status_valid = (
            n_invalid_quantity_status == 0
        )

        bilateral_quantity_contract_valid = (
            n_invalid_bilateral_quantity_contract == 0
        )

        unilateral_quantity_contract_valid = (
            n_invalid_unilateral_quantity_contract == 0
        )

        missing_quantity_contract_valid = (
            n_invalid_missing_quantity_contract == 0
        )

        audit_passed = bool(
            required_fields_present
            and has_rows
            and key_complete
            and canonical_key_unique
            and year_scope_valid
            and target_hs_valid
            and nominal_value_valid
            and constant_value_valid
            and quantity_numeric_valid
            and quantity_status_valid
            and bilateral_quantity_contract_valid
            and unilateral_quantity_contract_valid
            and missing_quantity_contract_valid
        )

        ready_for_diagnostics = audit_passed
        # --------------------------------------------------------------
        # Write audit artifact
        # --------------------------------------------------------------

        elapsed = perf_counter() - start

        audit_row = {
            "run_id": run_id,
            "audit_passed": audit_passed,

            "n_rows": n_rows,
            "min_year": (
                int(min_year)
                if min_year is not None
                else None
            ),
            "max_year": (
                int(max_year)
                if max_year is not None
                else None
            ),

            "required_fields_present": (
                required_fields_present
            ),
            "has_rows": has_rows,

            "key_complete": key_complete,
            "n_null_key_rows": n_null_key_rows,

            "canonical_key_unique": (
                canonical_key_unique
            ),
            "n_duplicate_keys": n_duplicate_keys,
            "n_duplicate_rows": n_duplicate_rows,

            "year_scope_valid": year_scope_valid,
            "n_invalid_year_rows": n_invalid_year_rows,

            "target_hs_valid": target_hs_valid,
            "n_invalid_hs_rows": n_invalid_hs_rows,

            "nominal_value_valid": nominal_value_valid,
            "n_invalid_nominal_value_rows": (
                n_invalid_nominal_value_rows
            ),

            "constant_value_valid": constant_value_valid,
            "n_invalid_constant_value_rows": (
                n_invalid_constant_value_rows
            ),

            "quantity_numeric_valid": (
                quantity_numeric_valid
            ),

            "n_available_quantity_rows": (
                n_available_quantity_rows
            ),
            "n_missing_quantity_rows": (
                n_missing_quantity_rows
            ),

            "quantity_status_valid": (
                quantity_status_valid
            ),
            "n_invalid_quantity_status": (
                n_invalid_quantity_status
            ),

            "bilateral_quantity_contract_valid": (
                bilateral_quantity_contract_valid
            ),
            "n_invalid_bilateral_quantity_contract": (
                n_invalid_bilateral_quantity_contract
            ),

            "unilateral_quantity_contract_valid": (
                unilateral_quantity_contract_valid
            ),
            "n_invalid_unilateral_quantity_contract": (
                n_invalid_unilateral_quantity_contract
            ),

            "missing_quantity_contract_valid": (
                missing_quantity_contract_valid
            ),
            "n_invalid_missing_quantity_contract": (
                n_invalid_missing_quantity_contract
            ),
            "quantity_candidates_numeric_valid": (
                quantity_candidates_numeric_valid
            ),

            "n_invalid_simple_mean_rows": (
                n_invalid_simple_mean_rows
            ),

            "n_invalid_quantity_rows": (
                n_invalid_quantity_rows
            ),

            "n_invalid_quantity_reliability_weighted_rows": (
                n_invalid_quantity_reliability_weighted_rows
            ),

            "duration_seconds": elapsed,
        }

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        if temporary_path.exists():
            temporary_path.unlink()

        temporary_sql = sql_literal(
            temporary_path
        )

        audit_frame = pd.DataFrame(
            [audit_row]
        )

        con.register(
            "_validation_structural_audit",
            audit_frame,
        )

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM _validation_structural_audit
            )
            TO {temporary_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        temporary_path.replace(
            output_path
        )

    finally:
        con.close()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    logger.info(
        "  Rows: %s",
        f"{n_rows:,}",
    )
    logger.info(
        "  Canonical key duplicates: %s",
        f"{n_duplicate_rows:,}",
    )
    logger.info(
        "  Invalid key rows: %s",
        f"{n_null_key_rows:,}",
    )
    logger.info(
        "  Invalid year rows: %s",
        f"{n_invalid_year_rows:,}",
    )
    logger.info(
        "  Invalid HS rows: %s",
        f"{n_invalid_hs_rows:,}",
    )
    logger.info(
        "  Invalid nominal values: %s",
        f"{n_invalid_nominal_value_rows:,}",
    )
    logger.info(
        "  Invalid constant values: %s",
        f"{n_invalid_constant_value_rows:,}",
    )
    logger.info(
        "  Missing synchronized quantities: %s",
        f"{n_missing_quantity_rows:,}",
    )
    logger.info(
        "  Invalid simple-mean quantities: %s",
        f"{n_invalid_simple_mean_rows:,}",
    )

    logger.info(
        "  Invalid quantities: %s",
        f"{n_invalid_quantity_rows:,}",
    )

    logger.info(
        "  Invalid reliability-weighted quantities: %s",
        f"{n_invalid_quantity_reliability_weighted_rows:,}",
    )
    logger.info(
        "  Invalid quantity contracts: %s",
        f"{(
            n_invalid_quantity_status
            + n_invalid_bilateral_quantity_contract
            + n_invalid_unilateral_quantity_contract
            + n_invalid_missing_quantity_contract
        ):,}",
    )
    logger.info(
        "  Audit artifact: %s",
        output_path,
    )

    # ------------------------------------------------------------------
    # Audit owns its own pass/fail decision
    # ------------------------------------------------------------------

    if not audit_passed:
        logger.error(
            "%s AUDIT — Validation structural integrity failed.",
            get_symbols("warning"),
        )

        raise RuntimeError(
            "Validation structural-integrity audit failed. "
            f"See audit artifact: {output_path}"
        )

    logger.info(
        "%s AUDIT — Validation structural integrity passed.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,

        "output_path": output_path,
        "structural_audit_path": output_path,

        "audit_passed": True,
        "ready_for_diagnostics": ready_for_diagnostics,

        "n_rows": n_rows,
        "n_missing_quantity_rows": (
            n_missing_quantity_rows
        ),

        "duration_seconds": elapsed,
        "reused": False,
    }