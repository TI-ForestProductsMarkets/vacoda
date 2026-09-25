"""
vacoda.steps.validation.quantity

Select the bilateral quantity methodology used by Validation.

This step owns the methodological choice among the three fixed quantity
candidates retained from Trade Sync. It also writes the row-level evidence
needed by downstream diagnostics so candidate formulas are not recomputed in
multiple modules.
"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import pandas as pd

from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal


# ======================================================================
# Helpers
# ======================================================================


def _safe_identifier(name: str) -> str:
    """Return a safely quoted SQL identifier."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
        raise ValueError(
            f"Unsafe Validation quantity-selection column identifier: {name!r}"
        )

    return f'"{name}"'


def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    required_columns: set[str],
) -> None:
    """Require the columns needed for quantity selection."""

    input_sql = sql_literal(input_path)

    available = {
        str(row[0])
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet({input_sql})
            """
        ).fetchall()
    }

    missing = sorted(required_columns - available)

    if missing:
        raise RuntimeError(
            "Validation quantity-selection input is missing required column(s): "
            + ", ".join(missing)
        )


# ======================================================================
# Public step
# ======================================================================


def select_quantity_candidate(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    structural_audit_path: Path,
    run_id: str,
    output_path: Path | None = None,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Select one bilateral quantity candidate and write selection evidence.

    Decision rule
    -------------
    Candidates are compared on the same directly observed bilateral
    population using the HS x year quantity-conditioned UV residual:

        | log(V / Q_candidate)
          - [alpha_(p,t) + beta_(p,t) log(Q_candidate)] |

    where alpha and beta are estimated from direct bilateral observations for
    the same HS product and year, using the symmetric bilateral quantity
    sqrt(Q_export * Q_import). Only HS x year groups with at least 30 direct
    bilateral observations are used.

    The selected candidate is the one with the smallest median conditioned
    residual. P95 and then P90 conditioned residual are deterministic
    tie-breakers. If the three criteria cannot resolve a tie, the step fails
    rather than choosing arbitrarily.

    The selected methodology is applied only when both reporter quantities are
    usable. Unilateral rows retain the available reporter quantity; rows with
    neither usable quantity remain missing.
    """

    start = perf_counter()

    input_path = Path(input_path)
    structural_audit_path = Path(
        structural_audit_path
    )

    output_path = (Path(output_path) if output_path is not None else cfg.product_path("quantity_selection"))

    selection_evidence_path = (output_path.parent / f"{output_path.stem}_evidence.parquet")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Validation quantity-selection input not found: {input_path}"
        )

    if not structural_audit_path.is_file():
        raise FileNotFoundError(
            "Validation structural-audit artifact not found: "
            f"{structural_audit_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        not force
        and output_path.is_file()
        and output_path.stat().st_size > 0
        and selection_evidence_path.is_file()
        and selection_evidence_path.stat().st_size > 0
    ):
        logger.info(
            "%s REUSE — Validation quantity selection: %s",
            get_symbols("reuse"),
            output_path,
        )
        logger.info(
            "  Selection evidence: %s",
            selection_evidence_path,
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "output_path": output_path,
            "quantity_selected_path": output_path,
            "selection_evidence_path": selection_evidence_path,
            "reused": True,
        }

    logger.info(
        "%s REBUILD — Validation quantity selection",
        get_symbols("rebuilt"),
    )
    logger.info("  Input: %s", input_path)
    logger.info("  Structural audit: %s", structural_audit_path)
    logger.info("  Output: %s", output_path)
    logger.info("  Selection evidence: %s", selection_evidence_path)

    diagnostics_cfg = cfg.diagnostics
    columns_cfg = diagnostics_cfg.columns

    year_col = str(columns_cfg.year)
    product_col = str(columns_cfg.product)
    nominal_value_col = str(columns_cfg.nominal_value)
    export_quantity_col = str(columns_cfg.export_quantity)
    import_quantity_col = str(columns_cfg.import_quantity)
    export_estimated_col = str(columns_cfg.export_estimated)
    import_estimated_col = str(columns_cfg.import_estimated)
    uv_reference_col = str(columns_cfg.unit_value_reference)

    simple_mean_col = "quantity_simple_mean_kg"
    value_weighted_col = "quantity_value_weighted_kg"
    quantity_reliability_weighted_col = (
        "quantity_reliability_weighted_kg"
    )

    required_columns = {
        year_col,
        product_col,
        "exporter_iso",
        "importer_iso",
        nominal_value_col,
        export_quantity_col,
        import_quantity_col,
        export_estimated_col,
        import_estimated_col,
        uv_reference_col,
        simple_mean_col,
        value_weighted_col,
        quantity_reliability_weighted_col,
    }

    con = duckdb.connect()

    try:
        con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")

        input_sql = sql_literal(input_path)
        structural_audit_sql = sql_literal(structural_audit_path)

        audit_passed_row = con.execute(
            f"""
            SELECT audit_passed
            FROM read_parquet({structural_audit_sql})
            LIMIT 1
            """
        ).fetchone()

        if audit_passed_row is None or not bool(audit_passed_row[0]):
            raise RuntimeError(
                "Validation quantity selection requires a passing structural audit."
            )

        _require_columns(
            con=con,
            input_path=input_path,
            required_columns=required_columns,
        )

        year_sql = _safe_identifier(year_col)
        product_sql = _safe_identifier(product_col)
        nominal_value_sql = _safe_identifier(nominal_value_col)
        export_quantity_sql = _safe_identifier(export_quantity_col)
        import_quantity_sql = _safe_identifier(import_quantity_col)
        export_estimated_sql = _safe_identifier(export_estimated_col)
        import_estimated_sql = _safe_identifier(import_estimated_col)
        uv_reference_sql = _safe_identifier(uv_reference_col)
        simple_mean_sql = _safe_identifier(simple_mean_col)
        value_weighted_sql = _safe_identifier(value_weighted_col)
        quantity_reliability_weighted_sql = _safe_identifier(
            quantity_reliability_weighted_col
        )

        # --------------------------------------------------------------
        # Selection base
        # --------------------------------------------------------------

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW validation_quantity_selection_base AS
            SELECT
                *,

                (
                    {export_quantity_sql} IS NOT NULL
                    AND {export_quantity_sql} > 0
                    AND isfinite(CAST({export_quantity_sql} AS DOUBLE))
                ) AS has_export_quantity_selection,

                (
                    {import_quantity_sql} IS NOT NULL
                    AND {import_quantity_sql} > 0
                    AND isfinite(CAST({import_quantity_sql} AS DOUBLE))
                ) AS has_import_quantity_selection,

                (
                    {export_quantity_sql} IS NOT NULL
                    AND {import_quantity_sql} IS NOT NULL
                    AND {export_quantity_sql} > 0
                    AND {import_quantity_sql} > 0
                    AND isfinite(CAST({export_quantity_sql} AS DOUBLE))
                    AND isfinite(CAST({import_quantity_sql} AS DOUBLE))
                    AND COALESCE({export_estimated_sql}, FALSE) = FALSE
                    AND COALESCE({import_estimated_sql}, FALSE) = FALSE
                ) AS both_quantities_observed_selection,

                CASE
                    WHEN {export_quantity_sql} > 0
                     AND {import_quantity_sql} > 0
                     AND {nominal_value_sql} > 0
                     AND isfinite(CAST({export_quantity_sql} AS DOUBLE))
                     AND isfinite(CAST({import_quantity_sql} AS DOUBLE))
                     AND isfinite(CAST({nominal_value_sql} AS DOUBLE))
                    THEN LN(
                        SQRT(
                            CAST({export_quantity_sql} AS DOUBLE)
                            * CAST({import_quantity_sql} AS DOUBLE)
                        )
                    )
                    ELSE NULL
                END AS selection_log_bilateral_quantity,

                CASE
                    WHEN {export_quantity_sql} > 0
                     AND {import_quantity_sql} > 0
                     AND {nominal_value_sql} > 0
                     AND isfinite(CAST({export_quantity_sql} AS DOUBLE))
                     AND isfinite(CAST({import_quantity_sql} AS DOUBLE))
                     AND isfinite(CAST({nominal_value_sql} AS DOUBLE))
                    THEN LN(
                        CAST({nominal_value_sql} AS DOUBLE)
                        /
                        SQRT(
                            CAST({export_quantity_sql} AS DOUBLE)
                            * CAST({import_quantity_sql} AS DOUBLE)
                        )
                    )
                    ELSE NULL
                END AS selection_log_bilateral_unit_value

            FROM read_parquet({input_sql})
            """
        )

        # --------------------------------------------------------------
        # HS x year quantity-conditioned UV relationship
        # --------------------------------------------------------------

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW validation_quantity_selection_models AS
            SELECT
                {year_sql} AS year,
                {product_sql} AS target_cmd,
                COUNT(*) AS n_model_rows,
                REGR_INTERCEPT(
                    selection_log_bilateral_unit_value,
                    selection_log_bilateral_quantity
                ) AS log_uv_intercept,
                REGR_SLOPE(
                    selection_log_bilateral_unit_value,
                    selection_log_bilateral_quantity
                ) AS log_uv_slope,
                REGR_R2(
                    selection_log_bilateral_unit_value,
                    selection_log_bilateral_quantity
                ) AS log_uv_r2
            FROM validation_quantity_selection_base
            WHERE both_quantities_observed_selection
              AND selection_log_bilateral_quantity IS NOT NULL
              AND selection_log_bilateral_unit_value IS NOT NULL
            GROUP BY {year_sql}, {product_sql}
            HAVING COUNT(*) >= 30
               AND REGR_INTERCEPT(
                    selection_log_bilateral_unit_value,
                    selection_log_bilateral_quantity
               ) IS NOT NULL
               AND REGR_SLOPE(
                    selection_log_bilateral_unit_value,
                    selection_log_bilateral_quantity
               ) IS NOT NULL
            """
        )

        # --------------------------------------------------------------
        # Row-level evidence, stored WIDE so Diagnostics can reuse it
        # without recomputing candidate definitions or UV residuals.
        # --------------------------------------------------------------

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW validation_quantity_selection_evidence AS
            SELECT
                b.*,
                m.n_model_rows AS quantity_conditioned_model_n_rows,
                m.log_uv_intercept AS quantity_conditioned_log_uv_intercept,
                m.log_uv_slope AS quantity_conditioned_log_uv_slope,
                m.log_uv_r2 AS quantity_conditioned_log_uv_r2,

                CASE
                    WHEN {simple_mean_sql} > 0
                     AND {nominal_value_sql} > 0
                     AND {uv_reference_sql} > 0
                    THEN ABS(
                        LN(CAST({nominal_value_sql} AS DOUBLE) / CAST({simple_mean_sql} AS DOUBLE))
                        - LN(CAST({uv_reference_sql} AS DOUBLE))
                    )
                END AS candidate_simple_mean_uv_log_distance,

                CASE
                    WHEN {value_weighted_sql} > 0
                     AND {nominal_value_sql} > 0
                     AND {uv_reference_sql} > 0
                    THEN ABS(
                        LN(CAST({nominal_value_sql} AS DOUBLE) / CAST({value_weighted_sql} AS DOUBLE))
                        - LN(CAST({uv_reference_sql} AS DOUBLE))
                    )
                END AS candidate_value_weighted_uv_log_distance,

                CASE
                    WHEN {quantity_reliability_weighted_sql} > 0
                     AND {nominal_value_sql} > 0
                     AND {uv_reference_sql} > 0
                    THEN ABS(
                        LN(CAST({nominal_value_sql} AS DOUBLE) / CAST({quantity_reliability_weighted_sql} AS DOUBLE))
                        - LN(CAST({uv_reference_sql} AS DOUBLE))
                    )
                END AS candidate_quantity_reliability_weighted_uv_log_distance,

                CASE WHEN {simple_mean_sql} > 0 AND {export_quantity_sql} > 0
                    THEN ABS(LN(CAST({simple_mean_sql} AS DOUBLE)) - LN(CAST({export_quantity_sql} AS DOUBLE)))
                END AS candidate_simple_mean_export_log_gap,

                CASE WHEN {simple_mean_sql} > 0 AND {import_quantity_sql} > 0
                    THEN ABS(LN(CAST({simple_mean_sql} AS DOUBLE)) - LN(CAST({import_quantity_sql} AS DOUBLE)))
                END AS candidate_simple_mean_import_log_gap,

                CASE WHEN {value_weighted_sql} > 0 AND {export_quantity_sql} > 0
                    THEN ABS(LN(CAST({value_weighted_sql} AS DOUBLE)) - LN(CAST({export_quantity_sql} AS DOUBLE)))
                END AS candidate_value_weighted_export_log_gap,

                CASE WHEN {value_weighted_sql} > 0 AND {import_quantity_sql} > 0
                    THEN ABS(LN(CAST({value_weighted_sql} AS DOUBLE)) - LN(CAST({import_quantity_sql} AS DOUBLE)))
                END AS candidate_value_weighted_import_log_gap,

                CASE WHEN {quantity_reliability_weighted_sql} > 0 AND {export_quantity_sql} > 0
                    THEN ABS(LN(CAST({quantity_reliability_weighted_sql} AS DOUBLE)) - LN(CAST({export_quantity_sql} AS DOUBLE)))
                END AS candidate_quantity_reliability_weighted_export_log_gap,

                CASE WHEN {quantity_reliability_weighted_sql} > 0 AND {import_quantity_sql} > 0
                    THEN ABS(LN(CAST({quantity_reliability_weighted_sql} AS DOUBLE)) - LN(CAST({import_quantity_sql} AS DOUBLE)))
                END AS candidate_quantity_reliability_weighted_import_log_gap,

                CASE
                    WHEN {simple_mean_sql} > 0
                     AND {nominal_value_sql} > 0
                     AND m.log_uv_intercept IS NOT NULL
                     AND m.log_uv_slope IS NOT NULL
                    THEN ABS(
                        LN(CAST({nominal_value_sql} AS DOUBLE) / CAST({simple_mean_sql} AS DOUBLE))
                        - (
                            m.log_uv_intercept
                            + m.log_uv_slope * LN(CAST({simple_mean_sql} AS DOUBLE))
                        )
                    )
                END AS candidate_simple_mean_conditioned_uv_abs_residual,

                CASE
                    WHEN {value_weighted_sql} > 0
                     AND {nominal_value_sql} > 0
                     AND m.log_uv_intercept IS NOT NULL
                     AND m.log_uv_slope IS NOT NULL
                    THEN ABS(
                        LN(CAST({nominal_value_sql} AS DOUBLE) / CAST({value_weighted_sql} AS DOUBLE))
                        - (
                            m.log_uv_intercept
                            + m.log_uv_slope * LN(CAST({value_weighted_sql} AS DOUBLE))
                        )
                    )
                END AS candidate_value_weighted_conditioned_uv_abs_residual,

                CASE
                    WHEN {quantity_reliability_weighted_sql} > 0
                     AND {nominal_value_sql} > 0
                     AND m.log_uv_intercept IS NOT NULL
                     AND m.log_uv_slope IS NOT NULL
                    THEN ABS(
                        LN(CAST({nominal_value_sql} AS DOUBLE) / CAST({quantity_reliability_weighted_sql} AS DOUBLE))
                        - (
                            m.log_uv_intercept
                            + m.log_uv_slope * LN(CAST({quantity_reliability_weighted_sql} AS DOUBLE))
                        )
                    )
                END AS candidate_quantity_reliability_weighted_conditioned_uv_abs_residual

            FROM validation_quantity_selection_base AS b
            LEFT JOIN validation_quantity_selection_models AS m
              ON m.year = b.{year_sql}
             AND m.target_cmd = b.{product_sql}
            """
        )

        # --------------------------------------------------------------
        # Candidate-level decision audit
        # --------------------------------------------------------------

        selection_summary = con.execute(
            f"""
            WITH long AS (
                SELECT
                    'simple_mean' AS candidate,
                    candidate_simple_mean_conditioned_uv_abs_residual AS conditioned_residual,
                    candidate_simple_mean_uv_log_distance AS uv_log_distance
                FROM validation_quantity_selection_evidence
                WHERE both_quantities_observed_selection

                UNION ALL

                SELECT
                    'value_weighted' AS candidate,
                    candidate_value_weighted_conditioned_uv_abs_residual AS conditioned_residual,
                    candidate_value_weighted_uv_log_distance AS uv_log_distance
                FROM validation_quantity_selection_evidence
                WHERE both_quantities_observed_selection

                UNION ALL

                SELECT
                    'quantity_reliability_weighted' AS candidate,
                    candidate_quantity_reliability_weighted_conditioned_uv_abs_residual AS conditioned_residual,
                    candidate_quantity_reliability_weighted_uv_log_distance AS uv_log_distance
                FROM validation_quantity_selection_evidence
                WHERE both_quantities_observed_selection
            )
            SELECT
                candidate,
                COUNT(*) FILTER (WHERE conditioned_residual IS NOT NULL)
                    AS n_conditioned_rows,
                MEDIAN(conditioned_residual)
                    FILTER (WHERE conditioned_residual IS NOT NULL)
                    AS median_conditioned_uv_abs_residual,
                QUANTILE_CONT(conditioned_residual, 0.90)
                    FILTER (WHERE conditioned_residual IS NOT NULL)
                    AS p90_conditioned_uv_abs_residual,
                QUANTILE_CONT(conditioned_residual, 0.95)
                    FILTER (WHERE conditioned_residual IS NOT NULL)
                    AS p95_conditioned_uv_abs_residual,
                COUNT(*) FILTER (WHERE uv_log_distance IS NOT NULL)
                    AS n_uv_rows,
                MEDIAN(uv_log_distance)
                    FILTER (WHERE uv_log_distance IS NOT NULL)
                    AS median_uv_log_distance,
                QUANTILE_CONT(uv_log_distance, 0.95)
                    FILTER (WHERE uv_log_distance IS NOT NULL)
                    AS p95_uv_log_distance
            FROM long
            GROUP BY candidate
            ORDER BY candidate
            """
        ).fetchdf()

        expected_candidates = {
            "simple_mean",
            "value_weighted",
            "quantity_reliability_weighted",
        }

        if set(selection_summary["candidate"]) != expected_candidates:
            raise RuntimeError(
                "Quantity selection did not produce evidence for all three candidates."
            )

        if selection_summary[
            "median_conditioned_uv_abs_residual"
        ].isna().any():
            raise RuntimeError(
                "Quantity selection requires conditioned UV evidence for all candidates."
            )

        ranked = selection_summary.sort_values(
            [
                "median_conditioned_uv_abs_residual",
                "p95_conditioned_uv_abs_residual",
                "p90_conditioned_uv_abs_residual",
            ],
            ascending=True,
            kind="mergesort",
        ).reset_index(drop=True)

        # Require the decision metrics to distinguish the leading candidate.
        if len(ranked) > 1:
            first = ranked.iloc[0]
            second = ranked.iloc[1]
            decision_columns = [
                "median_conditioned_uv_abs_residual",
                "p95_conditioned_uv_abs_residual",
                "p90_conditioned_uv_abs_residual",
            ]
            if all(
                abs(float(first[col]) - float(second[col])) <= 1e-12
                for col in decision_columns
            ):
                raise RuntimeError(
                    "Quantity-selection criteria are tied; refusing to choose arbitrarily."
                )

        selected_candidate = str(ranked.iloc[0]["candidate"])

        selection_summary = selection_summary.copy()
        selection_summary["run_id"] = run_id
        selection_summary["selected"] = (
            selection_summary["candidate"] == selected_candidate
        )
        selection_summary["selection_rule"] = (
            "minimum median quantity-conditioned UV absolute residual; "
            "tie-break P95 then P90"
        )
        selection_summary["decision_rank"] = selection_summary["candidate"].map(
            {
                str(row.candidate): rank
                for rank, row in enumerate(
                    ranked.itertuples(index=False),
                    start=1,
                )
            }
        )

        # --------------------------------------------------------------
        # Write selected row-level dataset
        # --------------------------------------------------------------

        selected_quantity_sql = {
            "simple_mean": simple_mean_sql,
            "value_weighted": value_weighted_sql,
            "quantity_reliability_weighted": quantity_reliability_weighted_sql,
        }[selected_candidate]

        output_sql = sql_literal(output_path)
        temporary_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_output_sql = sql_literal(temporary_output_path)

        if temporary_output_path.exists():
            temporary_output_path.unlink()

        con.execute(
            f"""
            COPY (
                SELECT
                    *,
                    '{selected_candidate}' AS quantity_selected_candidate,
                    CASE
                        WHEN has_export_quantity_selection
                         AND has_import_quantity_selection
                            THEN CAST({selected_quantity_sql} AS DOUBLE)
                        WHEN has_export_quantity_selection
                            THEN CAST({export_quantity_sql} AS DOUBLE)
                        WHEN has_import_quantity_selection
                            THEN CAST({import_quantity_sql} AS DOUBLE)
                        ELSE NULL
                    END AS quantity_selected_kg,
                    CASE
                        WHEN has_export_quantity_selection
                         AND has_import_quantity_selection
                            THEN '{selected_candidate}'
                        WHEN has_export_quantity_selection
                            THEN 'export_only'
                        WHEN has_import_quantity_selection
                            THEN 'import_only'
                        ELSE 'missing'
                    END AS quantity_selected_source
                FROM validation_quantity_selection_evidence
            )
            TO {temporary_output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        temporary_output_path.replace(output_path)

        # --------------------------------------------------------------
        # Write candidate-selection evidence
        # --------------------------------------------------------------

        temporary_evidence_path = (
            selection_evidence_path.with_suffix(
                selection_evidence_path.suffix + ".tmp"
            )
        )

        if temporary_evidence_path.exists():
            temporary_evidence_path.unlink()

        temporary_evidence_sql = sql_literal(
            temporary_evidence_path
        )

        con.register(
            "_quantity_selection_evidence_summary",
            selection_summary,
        )

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM _quantity_selection_evidence_summary
                ORDER BY decision_rank
            )
            TO {temporary_evidence_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        temporary_evidence_path.replace(
            selection_evidence_path
        )

        n_rows = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet({output_sql})"
            ).fetchone()[0]
        )

    finally:
        con.close()

    elapsed = perf_counter() - start

    logger.info("  Selected candidate: %s", selected_candidate)
    logger.info("  Output rows: %s", f"{n_rows:,}")
    logger.info("  Output: %s", output_path)
    logger.info("  Selection evidence: %s", selection_evidence_path)
    logger.info(
        "%s CHECK — Quantity candidate selected.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "quantity_selected_path": output_path,
        "selection_evidence_path": selection_evidence_path,
        "selected_candidate": selected_candidate,
        "n_rows": n_rows,
        "duration_seconds": elapsed,
        "reused": False,
    }
