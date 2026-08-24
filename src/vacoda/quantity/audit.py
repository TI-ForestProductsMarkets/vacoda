"""
vacoda.quantity.audit
---------------------

Final integrity audit for the Quantity module.

This audit is intentionally different from the broader Quantity diagnostics
and decision report. Diagnostics describe the data and support methodological
decisions; this audit verifies that the final synchronized Quantity parquet
satisfies the structural and row-level invariants required before downstream
CIF/FOB processing.

The audit writes a compact CSV with the standard VACODA audit schema:

    check, expected, actual, status, detail

where ``status`` is one of ``pass``, ``fail``, or ``info``.

The downstream readiness flag ``ready_for_cif_fob`` is true only when no
audit row has status ``fail``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from vacoda.config.paths import get_manifests_dir
from vacoda.config.quantity import QuantityConfig
from vacoda.utils.sql import sql_literal


# ======================================================================
# Methodology contracts
# ======================================================================

ALLOWED_SYNC_STATUSES = {
    "reported_three_exact",
    "reported_pair_exact_quantity_alt",
    "reported_pair_exact_quantity_net_weight",
    "reported_pair_exact_alt_net_weight",
    "reported_quantity_precedence",
    "reported_alt_quantity_precedence",
    "reported_net_weight_precedence",
    "estimated_quantity_fallback",
    "estimated_alt_quantity_fallback",
    "estimated_net_weight_fallback",
    "no_kg_evidence",
}


EXPECTED_SYNC_SOURCE_BY_STATUS = {
    "reported_three_exact": "quantity_alt_quantity_net_weight",
    "reported_pair_exact_quantity_alt": "quantity_alt_quantity",
    "reported_pair_exact_quantity_net_weight": "quantity_net_weight",
    "reported_pair_exact_alt_net_weight": "alt_quantity_net_weight",
    "reported_quantity_precedence": "quantity",
    "reported_alt_quantity_precedence": "alt_quantity",
    "reported_net_weight_precedence": "net_weight",
    "estimated_quantity_fallback": "quantity",
    "estimated_alt_quantity_fallback": "alt_quantity",
    "estimated_net_weight_fallback": "net_weight",
    "no_kg_evidence": "unavailable",
}


REQUIRED_OUTPUT_COLUMNS = {
    # Main physical outputs.
    "synchronized_quantity_kg",
    "supplementary_quantity",
    "supplementary_unit",

    # kg provenance / decision fields.
    "synchronized_quantity_source",
    "quantity_sync_status",
    "is_synchronized_quantity_estimated",
    "quantity_sync_resolved",

    # Supplementary provenance.
    "supplementary_quantity_source",
    "is_supplementary_quantity_estimated",

    # Original quantity evidence retained in output.
    "quantity",
    "quantity_unit",
    "quantity_unit_code",
    "is_quantity_estimated",
    "alt_quantity",
    "alt_quantity_unit",
    "alt_quantity_unit_code",
    "is_alt_quantity_estimated",
    "net_weight_kg",
    "is_net_weight_estimated",
    "gross_weight_kg",
    "is_gross_weight_estimated",

    # Canonical kg candidates used by synchronization.
    "quantity_kg_candidate",
    "quantity_kg_status",
    "alt_quantity_kg_candidate",
    "alt_quantity_kg_status",
    "net_weight_kg_candidate",
    "net_weight_kg_status",

    # Scope / commodity identifiers.
    "target_hs",
    "target_cmd",
}


# ======================================================================
# Small helpers
# ======================================================================

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
    if value is None or pd.isna(value):
        return 0
    return int(value)


def quantity_audit_path(cfg: QuantityConfig) -> Path:
    """
    Return the final Quantity integrity-audit CSV path.

    This follows the same VACODA convention as the HS-sync final audit:
    machine-readable integrity audits live under output/manifests rather
    than inside the interim-data directory.
    """
    target_hs = str(cfg.scope.target_hs).lower()

    return (
        get_manifests_dir()
        / "quantity"
        / f"target_{target_hs}"
        / f"target_{target_hs}_quantity_integrity_audit.csv"
    )


def _parquet_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> set[str]:
    rows = con.execute(
        """
        DESCRIBE
        SELECT *
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchall()

    return {str(row[0]) for row in rows}


# ======================================================================
# Main audit
# ======================================================================

def run_quantity_output_audit(
    *,
    cfg: QuantityConfig,
    run_id: str,
    logger,
    synchronized_quantity_path: Path | None = None,
    input_path: Path | None = None,
) -> dict[str, Any]:
    """
    Audit the final synchronized Quantity output.

    Parameters
    ----------
    cfg:
        Parsed Quantity configuration.
    run_id:
        Master-pipeline/workflow run identifier.
    logger:
        Configured workflow logger.
    synchronized_quantity_path:
        Final Quantity parquet. If omitted, resolves from
        ``cfg.product_path("synchronized_quantity")``.
    input_path:
        HS-synchronized Quantity source input. If omitted, resolves from
        ``cfg.input_path()``.

    Returns
    -------
    dict
        Audit metadata including ``audit_path``, ``audit_passed``,
        ``ready_for_cif_fob``, row counts, and selected output counts.

    Raises
    ------
    FileNotFoundError
        If input or output artifacts are missing.
    ValueError
        If any final integrity check fails. The audit CSV is written before
        the exception is raised.
    """

    input_path = Path(
        input_path
        if input_path is not None
        else cfg.input_path()
    )

    synchronized_quantity_path = Path(
        synchronized_quantity_path
        if synchronized_quantity_path is not None
        else cfg.product_path("synchronized_quantity")
    )

    audit_path = quantity_audit_path(cfg)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Quantity audit input not found: {input_path}"
        )

    if not synchronized_quantity_path.is_file():
        raise FileNotFoundError(
            "Synchronized Quantity output not found: "
            f"{synchronized_quantity_path}"
        )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_sql = sql_literal(input_path)
    output_sql = sql_literal(synchronized_quantity_path)

    logger.info("Auditing final Quantity output.")
    logger.info("  Run ID: %s", run_id)
    logger.info("  Quantity input: %s", input_path)
    logger.info(
        "  Synchronized Quantity output: %s",
        synchronized_quantity_path,
    )
    logger.info("  Audit output: %s", audit_path)

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # ==============================================================
        # Schema
        # ==============================================================

        output_columns = _parquet_columns(
            con,
            synchronized_quantity_path,
        )

        missing_columns = sorted(
            REQUIRED_OUTPUT_COLUMNS - output_columns
        )

        # ==============================================================
        # Population accounting
        # ==============================================================

        input_rows = _to_int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({input_sql})
                """
            ).fetchone()[0]
        )

        output_rows = _to_int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({output_sql})
                """
            ).fetchone()[0]
        )

        audit_rows: list[dict[str, object]] = [
            _audit_row(
                check="quantity_input_exists",
                expected=True,
                actual=input_path.is_file(),
                status=_status_pass(input_path.is_file()),
                detail=str(input_path),
            ),
            _audit_row(
                check="synchronized_quantity_output_exists",
                expected=True,
                actual=synchronized_quantity_path.is_file(),
                status=_status_pass(
                    synchronized_quantity_path.is_file()
                ),
                detail=str(synchronized_quantity_path),
            ),
            _audit_row(
                check="quantity_input_rows_defined",
                expected="> 0",
                actual=input_rows,
                status=_status_pass(input_rows > 0),
            ),
            _audit_row(
                check="synchronized_quantity_rows_defined",
                expected="> 0",
                actual=output_rows,
                status=_status_pass(output_rows > 0),
            ),
            _audit_row(
                check="input_output_row_count_preserved",
                expected=input_rows,
                actual=output_rows,
                status=_status_pass(
                    input_rows == output_rows
                ),
                detail=(
                    "Quantity synchronization is row-preserving; "
                    "it adds canonical physical-quantity fields "
                    "without expanding or dropping trade rows."
                ),
            ),
            _audit_row(
                check="required_quantity_output_columns_present",
                expected=0,
                actual=len(missing_columns),
                status=_status_pass(
                    not missing_columns
                ),
                detail=(
                    "missing_columns="
                    + ",".join(missing_columns)
                    if missing_columns
                    else ""
                ),
            ),
        ]

        # We cannot safely execute the row-level SQL below if required
        # columns are missing. Write the structural failure first.
        if missing_columns:
            audit = pd.DataFrame(audit_rows)

            audit_rows.append(
                _audit_row(
                    check="ready_for_cif_fob",
                    expected=True,
                    actual=False,
                    status="fail",
                    detail=(
                        "Required Quantity output columns are missing."
                    ),
                )
            )

            audit = pd.DataFrame(audit_rows)
            audit.to_csv(
                audit_path,
                index=False,
            )

            raise ValueError(
                "Quantity output integrity audit failed because "
                "required columns are missing:\n"
                + "\n".join(missing_columns)
            )

        # ==============================================================
        # One-pass final-output integrity summary
        # ==============================================================

        summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                -- ------------------------------------------------------
                -- Main kg output
                -- ------------------------------------------------------

                COUNT(*) FILTER (
                    WHERE synchronized_quantity_kg > 0
                ) AS n_synchronized_kg,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status = 'no_kg_evidence'
                ) AS n_no_kg_evidence,

                COUNT(*) FILTER (
                    WHERE synchronized_quantity_kg IS NOT NULL
                      AND synchronized_quantity_kg <= 0
                ) AS n_nonpositive_synchronized_kg,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status = 'no_kg_evidence'
                      AND synchronized_quantity_kg IS NOT NULL
                ) AS n_no_kg_status_with_value,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status <> 'no_kg_evidence'
                      AND synchronized_quantity_kg IS NULL
                ) AS n_resolved_status_without_value,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status = 'no_kg_evidence'
                      AND quantity_sync_resolved IS DISTINCT FROM FALSE
                ) AS n_no_kg_with_wrong_resolved_flag,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status <> 'no_kg_evidence'
                      AND quantity_sync_resolved IS DISTINCT FROM TRUE
                ) AS n_resolved_with_wrong_resolved_flag,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE 'reported_%'
                      AND is_synchronized_quantity_estimated
                          IS DISTINCT FROM FALSE
                ) AS n_reported_with_wrong_estimated_flag,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE 'estimated_%'
                      AND is_synchronized_quantity_estimated
                          IS DISTINCT FROM TRUE
                ) AS n_estimated_with_wrong_estimated_flag,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status = 'no_kg_evidence'
                      AND is_synchronized_quantity_estimated
                          IS NOT NULL
                ) AS n_no_kg_with_estimated_flag,

                COUNT(*) FILTER (
                    WHERE synchronized_quantity_source =
                          'gross_weight_proxy'
                ) AS n_gross_weight_proxy,

                -- ------------------------------------------------------
                -- Supplementary output
                -- ------------------------------------------------------

                COUNT(*) FILTER (
                    WHERE supplementary_quantity > 0
                      AND supplementary_unit IS NOT NULL
                ) AS n_supplementary,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity IS NOT NULL
                      AND supplementary_quantity <= 0
                ) AS n_nonpositive_supplementary,

                COUNT(*) FILTER (
                    WHERE (
                        supplementary_quantity IS NULL
                        AND supplementary_unit IS NOT NULL
                    )
                    OR (
                        supplementary_quantity IS NOT NULL
                        AND supplementary_unit IS NULL
                    )
                ) AS n_supplementary_null_mismatch,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source = 'quantity'
                ) AS n_supplementary_from_quantity,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source = 'alt_quantity'
                ) AS n_supplementary_from_alt_quantity,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source = 'unavailable'
                ) AS n_supplementary_unavailable,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source NOT IN (
                        'quantity',
                        'alt_quantity',
                        'unavailable'
                    )
                    OR supplementary_quantity_source IS NULL
                ) AS n_invalid_supplementary_source,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source = 'unavailable'
                      AND (
                          supplementary_quantity IS NOT NULL
                          OR supplementary_unit IS NOT NULL
                      )
                ) AS n_unavailable_supplementary_with_value,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source <> 'unavailable'
                      AND (
                          supplementary_quantity IS NULL
                          OR supplementary_unit IS NULL
                      )
                ) AS n_available_supplementary_without_value,

                -- Final supplementary value must equal the original
                -- selected source exactly.
                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source = 'quantity'
                      AND (
                          supplementary_quantity IS DISTINCT FROM quantity
                          OR supplementary_unit
                              IS DISTINCT FROM quantity_unit
                      )
                ) AS n_quantity_supplementary_value_mismatch,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source = 'alt_quantity'
                      AND (
                          supplementary_quantity
                              IS DISTINCT FROM alt_quantity
                          OR supplementary_unit
                              IS DISTINCT FROM alt_quantity_unit
                      )
                ) AS n_alt_supplementary_value_mismatch,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source = 'quantity'
                      AND is_supplementary_quantity_estimated
                          IS DISTINCT FROM
                          COALESCE(is_quantity_estimated, FALSE)
                ) AS n_quantity_supplementary_estimated_mismatch,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source = 'alt_quantity'
                      AND is_supplementary_quantity_estimated
                          IS DISTINCT FROM
                          COALESCE(is_alt_quantity_estimated, FALSE)
                ) AS n_alt_supplementary_estimated_mismatch,

                COUNT(*) FILTER (
                    WHERE supplementary_quantity_source = 'unavailable'
                      AND is_supplementary_quantity_estimated IS NOT NULL
                ) AS n_unavailable_supplementary_with_estimated_flag

            FROM read_parquet({output_sql})
            """
        ).fetchone()

        (
            n_rows,
            n_synchronized_kg,
            n_no_kg_evidence,
            n_nonpositive_synchronized_kg,
            n_no_kg_status_with_value,
            n_resolved_status_without_value,
            n_no_kg_with_wrong_resolved_flag,
            n_resolved_with_wrong_resolved_flag,
            n_reported_with_wrong_estimated_flag,
            n_estimated_with_wrong_estimated_flag,
            n_no_kg_with_estimated_flag,
            n_gross_weight_proxy,
            n_supplementary,
            n_nonpositive_supplementary,
            n_supplementary_null_mismatch,
            n_supplementary_from_quantity,
            n_supplementary_from_alt_quantity,
            n_supplementary_unavailable,
            n_invalid_supplementary_source,
            n_unavailable_supplementary_with_value,
            n_available_supplementary_without_value,
            n_quantity_supplementary_value_mismatch,
            n_alt_supplementary_value_mismatch,
            n_quantity_supplementary_estimated_mismatch,
            n_alt_supplementary_estimated_mismatch,
            n_unavailable_supplementary_with_estimated_flag,
        ) = [
            _to_int(value)
            for value in summary
        ]

        # ==============================================================
        # Synchronization-status distribution and allowed vocabulary
        # ==============================================================

        status_distribution = con.execute(
            f"""
            SELECT
                quantity_sync_status,
                synchronized_quantity_source,
                COUNT(*) AS n_rows
            FROM read_parquet({output_sql})
            GROUP BY
                quantity_sync_status,
                synchronized_quantity_source
            ORDER BY n_rows DESC
            """
        ).fetchdf()

        observed_statuses = {
            str(value)
            for value in status_distribution[
                "quantity_sync_status"
            ].dropna()
        }

        unknown_statuses = sorted(
            observed_statuses - ALLOWED_SYNC_STATUSES
        )

        null_status_rows = _to_int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({output_sql})
                WHERE quantity_sync_status IS NULL
                """
            ).fetchone()[0]
        )

        # Validate exact status -> source mapping.
        expected_mapping_df = pd.DataFrame(
            [
                {
                    "quantity_sync_status": status,
                    "expected_source": source,
                }
                for status, source
                in EXPECTED_SYNC_SOURCE_BY_STATUS.items()
            ]
        )

        con.register(
            "expected_quantity_status_source",
            expected_mapping_df,
        )

        status_source_mismatch = _to_int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({output_sql}) q
                LEFT JOIN expected_quantity_status_source e
                  ON q.quantity_sync_status
                     = e.quantity_sync_status
                WHERE e.quantity_sync_status IS NOT NULL
                  AND q.synchronized_quantity_source
                      IS DISTINCT FROM e.expected_source
                """
            ).fetchone()[0]
        )

        # ==============================================================
        # Validate synchronized kg value against the rule-selected source
        # ==============================================================

        kg_value_mismatch = _to_int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({output_sql})
                WHERE
                    CASE quantity_sync_status
                        WHEN 'reported_three_exact'
                            THEN synchronized_quantity_kg
                                 IS DISTINCT FROM
                                 quantity_kg_candidate

                        WHEN 'reported_pair_exact_quantity_alt'
                            THEN synchronized_quantity_kg
                                 IS DISTINCT FROM
                                 quantity_kg_candidate

                        WHEN 'reported_pair_exact_quantity_net_weight'
                            THEN synchronized_quantity_kg
                                 IS DISTINCT FROM
                                 quantity_kg_candidate

                        WHEN 'reported_pair_exact_alt_net_weight'
                            THEN synchronized_quantity_kg
                                 IS DISTINCT FROM
                                 alt_quantity_kg_candidate

                        WHEN 'reported_quantity_precedence'
                            THEN synchronized_quantity_kg
                                 IS DISTINCT FROM
                                 quantity_kg_candidate

                        WHEN 'reported_alt_quantity_precedence'
                            THEN synchronized_quantity_kg
                                 IS DISTINCT FROM
                                 alt_quantity_kg_candidate

                        WHEN 'reported_net_weight_precedence'
                            THEN synchronized_quantity_kg
                                 IS DISTINCT FROM
                                 net_weight_kg_candidate

                        WHEN 'estimated_quantity_fallback'
                            THEN synchronized_quantity_kg
                                 IS DISTINCT FROM
                                 quantity_kg_candidate

                        WHEN 'estimated_alt_quantity_fallback'
                            THEN synchronized_quantity_kg
                                 IS DISTINCT FROM
                                 alt_quantity_kg_candidate

                        WHEN 'estimated_net_weight_fallback'
                            THEN synchronized_quantity_kg
                                 IS DISTINCT FROM
                                 net_weight_kg_candidate

                        WHEN 'no_kg_evidence'
                            THEN synchronized_quantity_kg IS NOT NULL

                        ELSE TRUE
                    END
                """
            ).fetchone()[0]
        )

        # ==============================================================
        # Methodology partition
        # ==============================================================

        methodology_counts = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE quantity_sync_status =
                          'reported_three_exact'
                ) AS n_reported_three_exact,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE
                          'reported_pair_exact_%'
                ) AS n_reported_pair_exact,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE
                          'reported_%_precedence'
                ) AS n_reported_precedence,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE
                          'estimated_%_fallback'
                ) AS n_estimated_fallback,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status =
                          'no_kg_evidence'
                ) AS n_no_kg

            FROM read_parquet({output_sql})
            """
        ).fetchone()

        (
            n_reported_three_exact,
            n_reported_pair_exact,
            n_reported_precedence,
            n_estimated_fallback,
            n_no_kg_partition,
        ) = [
            _to_int(value)
            for value in methodology_counts
        ]

        methodology_partition = (
            n_reported_three_exact
            + n_reported_pair_exact
            + n_reported_precedence
            + n_estimated_fallback
            + n_no_kg_partition
        )

        supplementary_source_partition = (
            n_supplementary_from_quantity
            + n_supplementary_from_alt_quantity
            + n_supplementary_unavailable
        )

        # ==============================================================
        # PASS / FAIL audit rows
        # ==============================================================

        audit_rows.extend(
            [
                _audit_row(
                    check="quantity_methodology_status_vocabulary_valid",
                    expected=0,
                    actual=len(unknown_statuses) + null_status_rows,
                    status=_status_pass(
                        not unknown_statuses
                        and null_status_rows == 0
                    ),
                    detail=(
                        f"unknown_statuses={unknown_statuses}; "
                        f"null_status_rows={null_status_rows}"
                    ),
                ),
                _audit_row(
                    check="quantity_methodology_partition_complete",
                    expected=n_rows,
                    actual=methodology_partition,
                    status=_status_pass(
                        methodology_partition == n_rows
                    ),
                    detail=(
                        "reported_three_exact + "
                        "reported_pair_exact + "
                        "reported_precedence + "
                        "estimated_fallback + "
                        "no_kg_evidence"
                    ),
                ),
                _audit_row(
                    check="quantity_status_source_mapping_valid",
                    expected=0,
                    actual=status_source_mismatch,
                    status=_status_pass(
                        status_source_mismatch == 0
                    ),
                ),
                _audit_row(
                    check="synchronized_kg_matches_applied_rule",
                    expected=0,
                    actual=kg_value_mismatch,
                    status=_status_pass(
                        kg_value_mismatch == 0
                    ),
                    detail=(
                        "Final synchronized_quantity_kg must equal "
                        "the candidate selected by quantity_sync_status."
                    ),
                ),
                _audit_row(
                    check="synchronized_kg_positive_when_present",
                    expected=0,
                    actual=n_nonpositive_synchronized_kg,
                    status=_status_pass(
                        n_nonpositive_synchronized_kg == 0
                    ),
                ),
                _audit_row(
                    check="no_kg_evidence_has_no_synchronized_value",
                    expected=0,
                    actual=n_no_kg_status_with_value,
                    status=_status_pass(
                        n_no_kg_status_with_value == 0
                    ),
                ),
                _audit_row(
                    check="resolved_kg_status_has_synchronized_value",
                    expected=0,
                    actual=n_resolved_status_without_value,
                    status=_status_pass(
                        n_resolved_status_without_value == 0
                    ),
                ),
                _audit_row(
                    check="quantity_sync_resolved_flag_consistent",
                    expected=0,
                    actual=(
                        n_no_kg_with_wrong_resolved_flag
                        + n_resolved_with_wrong_resolved_flag
                    ),
                    status=_status_pass(
                        n_no_kg_with_wrong_resolved_flag == 0
                        and n_resolved_with_wrong_resolved_flag == 0
                    ),
                    detail=(
                        "no_kg_evidence must resolve False; "
                        "all selected kg rules must resolve True."
                    ),
                ),
                _audit_row(
                    check="kg_estimated_provenance_flag_consistent",
                    expected=0,
                    actual=(
                        n_reported_with_wrong_estimated_flag
                        + n_estimated_with_wrong_estimated_flag
                        + n_no_kg_with_estimated_flag
                    ),
                    status=_status_pass(
                        n_reported_with_wrong_estimated_flag == 0
                        and n_estimated_with_wrong_estimated_flag == 0
                        and n_no_kg_with_estimated_flag == 0
                    ),
                ),
                _audit_row(
                    check="gross_weight_not_used_as_quantity_proxy",
                    expected=0,
                    actual=n_gross_weight_proxy,
                    status=_status_pass(
                        n_gross_weight_proxy == 0
                    ),
                    detail=(
                        "Gross weight is diagnostic-only under the "
                        "adopted Quantity methodology."
                    ),
                ),
                _audit_row(
                    check="supplementary_quantity_positive_when_present",
                    expected=0,
                    actual=n_nonpositive_supplementary,
                    status=_status_pass(
                        n_nonpositive_supplementary == 0
                    ),
                ),
                _audit_row(
                    check="supplementary_quantity_unit_null_consistency",
                    expected=0,
                    actual=n_supplementary_null_mismatch,
                    status=_status_pass(
                        n_supplementary_null_mismatch == 0
                    ),
                ),
                _audit_row(
                    check="supplementary_source_vocabulary_valid",
                    expected=0,
                    actual=n_invalid_supplementary_source,
                    status=_status_pass(
                        n_invalid_supplementary_source == 0
                    ),
                ),
                _audit_row(
                    check="supplementary_source_partition_complete",
                    expected=n_rows,
                    actual=supplementary_source_partition,
                    status=_status_pass(
                        supplementary_source_partition == n_rows
                    ),
                ),
                _audit_row(
                    check="supplementary_unavailable_has_no_value",
                    expected=0,
                    actual=n_unavailable_supplementary_with_value,
                    status=_status_pass(
                        n_unavailable_supplementary_with_value == 0
                    ),
                ),
                _audit_row(
                    check="supplementary_selected_source_has_value",
                    expected=0,
                    actual=n_available_supplementary_without_value,
                    status=_status_pass(
                        n_available_supplementary_without_value == 0
                    ),
                ),
                _audit_row(
                    check="supplementary_quantity_source_value_preserved",
                    expected=0,
                    actual=(
                        n_quantity_supplementary_value_mismatch
                        + n_alt_supplementary_value_mismatch
                    ),
                    status=_status_pass(
                        n_quantity_supplementary_value_mismatch == 0
                        and n_alt_supplementary_value_mismatch == 0
                    ),
                    detail=(
                        "Selected supplementary value and unit must "
                        "exactly match the original chosen source."
                    ),
                ),
                _audit_row(
                    check="supplementary_estimated_provenance_preserved",
                    expected=0,
                    actual=(
                        n_quantity_supplementary_estimated_mismatch
                        + n_alt_supplementary_estimated_mismatch
                        + n_unavailable_supplementary_with_estimated_flag
                    ),
                    status=_status_pass(
                        n_quantity_supplementary_estimated_mismatch == 0
                        and n_alt_supplementary_estimated_mismatch == 0
                        and n_unavailable_supplementary_with_estimated_flag
                            == 0
                    ),
                ),
            ]
        )

        # ==============================================================
        # Informational audit rows
        # ==============================================================

        audit_rows.extend(
            [
                _audit_row(
                    check="quantity_output_rows",
                    expected="recorded",
                    actual=n_rows,
                    status="info",
                ),
                _audit_row(
                    check="rows_with_synchronized_quantity_kg",
                    expected="recorded",
                    actual=n_synchronized_kg,
                    status="info",
                    detail=(
                        f"share={100.0 * n_synchronized_kg / n_rows:.4f}%"
                        if n_rows
                        else "share=0%"
                    ),
                ),
                _audit_row(
                    check="rows_with_no_kg_evidence",
                    expected="recorded",
                    actual=n_no_kg_evidence,
                    status="info",
                    detail=(
                        f"share={100.0 * n_no_kg_evidence / n_rows:.4f}%"
                        if n_rows
                        else "share=0%"
                    ),
                ),
                _audit_row(
                    check="rows_reported_three_exact",
                    expected="recorded",
                    actual=n_reported_three_exact,
                    status="info",
                ),
                _audit_row(
                    check="rows_reported_pair_exact",
                    expected="recorded",
                    actual=n_reported_pair_exact,
                    status="info",
                ),
                _audit_row(
                    check="rows_reported_precedence",
                    expected="recorded",
                    actual=n_reported_precedence,
                    status="info",
                ),
                _audit_row(
                    check="rows_estimated_fallback",
                    expected="recorded",
                    actual=n_estimated_fallback,
                    status="info",
                ),
                _audit_row(
                    check="rows_with_supplementary_quantity",
                    expected="recorded",
                    actual=n_supplementary,
                    status="info",
                    detail=(
                        f"share={100.0 * n_supplementary / n_rows:.4f}%"
                        if n_rows
                        else "share=0%"
                    ),
                ),
                _audit_row(
                    check="supplementary_selected_from_quantity",
                    expected="recorded",
                    actual=n_supplementary_from_quantity,
                    status="info",
                ),
                _audit_row(
                    check="supplementary_selected_from_alt_quantity",
                    expected="recorded",
                    actual=n_supplementary_from_alt_quantity,
                    status="info",
                ),
                _audit_row(
                    check="quantity_sync_status_distribution",
                    expected="recorded",
                    actual=len(status_distribution),
                    status="info",
                    detail=status_distribution.to_dict(
                        orient="records"
                    ),
                ),
            ]
        )

        # ==============================================================
        # Final readiness decision
        # ==============================================================

        provisional_audit = pd.DataFrame(audit_rows)

        failed = provisional_audit.loc[
            provisional_audit["status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "fail"
        ]

        ready_for_cif_fob = failed.empty

        audit_rows.append(
            _audit_row(
                check="ready_for_cif_fob",
                expected=True,
                actual=ready_for_cif_fob,
                status=(
                    "pass"
                    if ready_for_cif_fob
                    else "fail"
                ),
                detail=(
                    "Quantity output passed final integrity audit "
                    "and is ready for CIF/FOB."
                    if ready_for_cif_fob
                    else
                    "One or more Quantity final-output integrity "
                    "checks failed."
                ),
            )
        )

        audit = pd.DataFrame(audit_rows)

        failed = audit.loc[
            audit["status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "fail"
        ]

        audit.to_csv(
            audit_path,
            index=False,
        )

        logger.info("Quantity final-output integrity audit complete.")
        logger.info("  Input rows: %s", f"{input_rows:,}")
        logger.info("  Output rows: %s", f"{output_rows:,}")
        logger.info(
            "  Synchronized kg rows: %s",
            f"{n_synchronized_kg:,}",
        )
        logger.info(
            "  Supplementary quantity rows: %s",
            f"{n_supplementary:,}",
        )
        logger.info(
            "  Failed checks: %s",
            f"{len(failed):,}",
        )
        logger.info(
            "  Ready for CIF/FOB: %s",
            ready_for_cif_fob,
        )
        logger.info("  Audit file: %s", audit_path)

        if not failed.empty:
            raise ValueError(
                "Quantity final-output integrity audit failed:\n"
                + failed.to_string(index=False)
            )

        return {
            "status": "completed",
            "run_id": run_id,

            "audit_path": audit_path,
            "output_path": audit_path,

            "synchronized_quantity_path":
                synchronized_quantity_path,
            "input_path": input_path,

            "ready_for_cif_fob": ready_for_cif_fob,
            "audit_passed": ready_for_cif_fob,
            "n_failed_audit_rows": int(len(failed)),

            "input_rows": int(input_rows),
            "output_rows": int(output_rows),

            "n_synchronized_quantity_kg": int(
                n_synchronized_kg
            ),
            "n_no_kg_evidence": int(
                n_no_kg_evidence
            ),
            "n_supplementary_quantity": int(
                n_supplementary
            ),
            "n_supplementary_from_quantity": int(
                n_supplementary_from_quantity
            ),
            "n_supplementary_from_alt_quantity": int(
                n_supplementary_from_alt_quantity
            ),
        }

    finally:
        con.close()
