"""
vacoda.steps.validation.diagnostics

Build diagnostics for the canonical bilateral Validation workflow.

The diagnostics are deliberately separated into three concepts:

1. source-reporter diagnostics, which describe exporter/importer quantity
   availability and disagreement but do not define model-training eligibility;
2. training-sample diagnostics, which audit the canonical bilateral training
   population defined by HS × year UV support, positive UV MAD, and normalized
   UV distance;
3. model diagnostics, which summarize expanding-window out-of-fold prediction
   performance.

No reporter-side disagreement criterion is used to define the training sample.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import pandas as pd

from vacoda.config.steps.validation import ValidationConfig
from vacoda.steps.validation.training_sample import training_definition_name
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal


PRE_TRAINING_REFERENCE_WINDOWS = (
    ("0_0", 0, "Same year"),
    ("1_0", 1, "Current + 1 previous year"),
    ("2_0", 2, "Current + 2 previous years"),
)

NORMALIZED_UV_CUTOFFS = (
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
)

SOURCE_DISAGREEMENT_BANDS = (
    ("0-5%", 0.00, 0.05),
    ("5-10%", 0.05, 0.10),
    ("10-25%", 0.10, 0.25),
    ("25-50%", 0.25, 0.50),
    ("50-100%", 0.50, 1.00),
)

DIAGNOSTIC_QUANTILES = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
)


# ======================================================================
# Helpers
# ======================================================================


def _share(
    numerator: int,
    denominator: int,
) -> float | None:
    """Return numerator / denominator, or NULL for an empty denominator."""

    if denominator <= 0:
        return None

    return float(numerator) / float(denominator)



def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    required_columns: set[str],
) -> None:
    """Require all columns needed by Validation diagnostics."""

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
            "Validation diagnostics input is missing required column(s): "
            + ", ".join(missing)
        )



def _append_metric(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    criterion: str,
    variable: str | None = None,
    value: float | None = None,
    n_rows: int | None = None,
    total_rows: int | None = None,
    share_of_all: float | None = None,
    share_of_parent: float | None = None,
    year: int | None = None,
    target_cmd: str | None = None,
    quantile: float | None = None,
    candidate: str | None = None,
    model_name: str | None = None,
    uv_support_threshold: float | None = None,
    uv_distance_threshold: float | None = None,
    disagreement_band: str | None = None,
) -> None:
    """Append one normalized diagnostics row."""

    rows.append(
        {
            "scope": scope,
            "criterion": criterion,
            "candidate": candidate,
            "model_name": model_name,
            "disagreement_threshold": None,
            "uv_support_threshold": uv_support_threshold,
            "uv_distance_threshold": uv_distance_threshold,
            "disagreement_band": disagreement_band,
            "uv_distance_band": None,
            "year": year,
            "target_cmd": target_cmd,
            "variable": variable,
            "quantile": quantile,
            "n_rows": n_rows,
            "total_rows": total_rows,
            "share_of_all": share_of_all,
            "share_of_parent": share_of_parent,
            "value": value,
        }
    )


# ======================================================================
# Public step
# ======================================================================


def build_diagnostics(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    structural_audit_path: Path,
    quantity_selection_audit_path: Path,
    model_predictions_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build post-training Validation diagnostics.

    ``input_path`` is the training-sample artifact.  Reporter-side quantities
    are retained only for source-data diagnostics; the model-training population
    is audited exclusively through ``training_eligible`` and the configured
    HS × year unit-value criteria.
    """

    start = perf_counter()

    input_path = Path(input_path)
    structural_audit_path = Path(structural_audit_path)
    quantity_selection_audit_path = Path(quantity_selection_audit_path)
    model_predictions_path = Path(model_predictions_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Validation diagnostics input not found: {input_path}"
        )

    if not structural_audit_path.is_file():
        raise FileNotFoundError(
            "Validation diagnostics structural-audit artifact not found: "
            f"{structural_audit_path}"
        )

    if not quantity_selection_audit_path.is_file():
        raise FileNotFoundError(
            "Validation diagnostics quantity-selection audit artifact not found: "
            f"{quantity_selection_audit_path}"
        )

    if not model_predictions_path.is_file():
        raise FileNotFoundError(
            "Validation model-prediction artifact not found: "
            f"{model_predictions_path}"
        )

    output_path = cfg.product_path("diagnostics")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        not force
        and output_path.is_file()
        and output_path.stat().st_size > 0
    ):
        logger.info(
            "%s REUSE — Validation diagnostics: %s",
            get_symbols("reuse"),
            output_path,
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "output_path": output_path,
            "diagnostics_path": output_path,
            "reused": True,
            "fitted": False,
            "selection_already_applied": True,
        }

    logger.info(
        "%s REBUILD — Validation diagnostics",
        get_symbols("rebuilt"),
    )
    logger.info("  Input: %s", input_path)
    logger.info("  Model predictions: %s", model_predictions_path)
    logger.info("  Output: %s", output_path)

    columns_cfg = cfg.diagnostics.columns
    training_cfg = cfg.model.training_sample

    year_col = str(columns_cfg.year)
    product_col = str(columns_cfg.product)
    nominal_value_col = str(columns_cfg.nominal_value)
    constant_value_col = str(columns_cfg.constant_value)
    export_quantity_col = str(columns_cfg.export_quantity)
    import_quantity_col = str(columns_cfg.import_quantity)
    export_estimated_col = str(columns_cfg.export_estimated)
    import_estimated_col = str(columns_cfg.import_estimated)
    relative_difference_col = str(columns_cfg.quantity_relative_difference)

    selected_quantity_col = "quantity_selected_kg"
    selected_candidate_col = "quantity_selected_candidate"
    selected_source_col = "quantity_selected_source"

    min_reference_observations = int(
        training_cfg.min_hs_year_reference_observations
    )
    max_normalized_uv_distance_mad = float(
        training_cfg.max_normalized_uv_distance_mad
    )
    reference_year_before = int(
        training_cfg.reference_year_window.before
    )
    reference_year_after = int(
        training_cfg.reference_year_window.after
    )

    training_definition = training_definition_name(cfg=cfg)

    required_columns = {
        year_col,
        product_col,
        "exporter_iso",
        "importer_iso",
        nominal_value_col,
        constant_value_col,
        export_quantity_col,
        import_quantity_col,
        export_estimated_col,
        import_estimated_col,
        relative_difference_col,
        selected_quantity_col,
        selected_candidate_col,
        selected_source_col,
        "selected_unit_value",
        "training_eligible",
        "hs_year_uv_reference",
        "hs_year_uv_reference_log",
        "hs_year_uv_reference_n",
        "hs_year_uv_log_mad",
        "selected_uv_log_distance",
        "training_normalized_uv_distance",
        "uv_evidence_available",
        "training_selected",
    }

    con = duckdb.connect()

    try:
        con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")

        # --------------------------------------------------------------
        # Require passing quantity-method audit
        # --------------------------------------------------------------
        audit_passed_row = con.execute(
            f"""
            SELECT audit_passed
            FROM read_parquet({sql_literal(quantity_selection_audit_path)})
            LIMIT 1
            """
        ).fetchone()

        if audit_passed_row is None or not bool(audit_passed_row[0]):
            raise RuntimeError(
                "Validation diagnostics require a passing quantity-selection "
                "audit."
            )

        _require_columns(
            con=con,
            input_path=input_path,
            required_columns=required_columns,
        )

        input_sql = sql_literal(input_path)
        predictions_sql = sql_literal(model_predictions_path)

        # --------------------------------------------------------------
        # Canonical diagnostics view
        # --------------------------------------------------------------
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW validation_diagnostics_base AS
            SELECT
                "{year_col}" AS year,
                exporter_iso,
                importer_iso,
                "{product_col}" AS target_cmd,

                "{selected_quantity_col}" AS quantity_selected_kg,
                "{selected_candidate_col}" AS quantity_selected_candidate,
                "{selected_source_col}" AS quantity_selected_source,

                "{nominal_value_col}" AS trade_value_reconciled,
                "{constant_value_col}" AS trade_value_reconciled_constant_usd,

                "{export_quantity_col}" AS export_quantity_kg,
                "{import_quantity_col}" AS import_quantity_kg,
                "{export_estimated_col}" AS export_quantity_estimated,
                "{import_estimated_col}" AS import_quantity_estimated,
                "{relative_difference_col}" AS quantity_relative_difference,

                selected_unit_value,
                training_eligible,
                hs_year_uv_reference,
                hs_year_uv_reference_log,
                hs_year_uv_reference_n,
                hs_year_uv_log_mad,
                selected_uv_log_distance,
                training_normalized_uv_distance,
                uv_evidence_available,
                training_selected,

                (
                    "{export_quantity_col}" IS NOT NULL
                    AND "{export_quantity_col}" > 0
                    AND isfinite(CAST("{export_quantity_col}" AS DOUBLE))
                ) AS has_export_quantity,

                (
                    "{import_quantity_col}" IS NOT NULL
                    AND "{import_quantity_col}" > 0
                    AND isfinite(CAST("{import_quantity_col}" AS DOUBLE))
                ) AS has_import_quantity,

                (
                    "{export_quantity_col}" IS NOT NULL
                    AND "{import_quantity_col}" IS NOT NULL
                    AND "{export_quantity_col}" > 0
                    AND "{import_quantity_col}" > 0
                    AND isfinite(CAST("{export_quantity_col}" AS DOUBLE))
                    AND isfinite(CAST("{import_quantity_col}" AS DOUBLE))
                ) AS has_both_quantities,

                (
                    "{export_quantity_col}" IS NOT NULL
                    AND "{import_quantity_col}" IS NOT NULL
                    AND "{export_quantity_col}" > 0
                    AND "{import_quantity_col}" > 0
                    AND isfinite(CAST("{export_quantity_col}" AS DOUBLE))
                    AND isfinite(CAST("{import_quantity_col}" AS DOUBLE))
                    AND COALESCE("{export_estimated_col}", FALSE) = FALSE
                    AND COALESCE("{import_estimated_col}", FALSE) = FALSE
                ) AS both_quantities_non_estimated

            FROM read_parquet({input_sql})
            """
        )

        # --------------------------------------------------------------
        # Model predictions
        # --------------------------------------------------------------
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW validation_model_predictions AS
            SELECT
                model_name,
                model_type,
                held_out_year,
                target_transformed,
                prediction_transformed,
                target_quantity_kg,
                prediction_quantity_kg,
                prediction_transformed - target_transformed AS log_error,
                ABS(prediction_transformed - target_transformed)
                    AS absolute_log_error,
                POWER(prediction_transformed - target_transformed, 2)
                    AS squared_log_error
            FROM read_parquet({predictions_sql})
            WHERE target_transformed IS NOT NULL
              AND prediction_transformed IS NOT NULL
              AND isfinite(target_transformed)
              AND isfinite(prediction_transformed)
            """
        )

        diagnostics_rows: list[dict[str, Any]] = []

        n_all_rows = int(
            con.execute(
                "SELECT COUNT(*) FROM validation_diagnostics_base"
            ).fetchone()[0]
        )

        # ==================================================================
        # 1. Selected canonical quantity methodology
        # ==================================================================
        selected_candidates = con.execute(
            """
            SELECT DISTINCT quantity_selected_candidate
            FROM validation_diagnostics_base
            WHERE quantity_selected_candidate IS NOT NULL
            ORDER BY quantity_selected_candidate
            """
        ).fetchall()

        if len(selected_candidates) != 1:
            raise RuntimeError(
                "Validation diagnostics require exactly one selected canonical "
                "bilateral quantity methodology."
            )

        selected_candidate = str(selected_candidates[0][0])

        _append_metric(
            diagnostics_rows,
            scope="quantity_methodology",
            criterion="selected_canonical_quantity",
            candidate=selected_candidate,
            n_rows=n_all_rows,
            total_rows=n_all_rows,
            share_of_all=1.0,
        )

        # ==================================================================
        # 2. Source-reporter diagnostics
        # ==================================================================
        # These describe the original reporter evidence only.  They are not
        # training-sample criteria and are not used as truth labels.

        source_counts = con.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE has_export_quantity AND has_import_quantity
                ) AS n_both,
                COUNT(*) FILTER (
                    WHERE has_export_quantity AND NOT has_import_quantity
                ) AS n_export_only,
                COUNT(*) FILTER (
                    WHERE NOT has_export_quantity AND has_import_quantity
                ) AS n_import_only,
                COUNT(*) FILTER (
                    WHERE NOT has_export_quantity AND NOT has_import_quantity
                ) AS n_neither,
                COUNT(*) FILTER (
                    WHERE both_quantities_non_estimated
                ) AS n_both_non_estimated
            FROM validation_diagnostics_base
            """
        ).fetchone()

        (
            n_both_quantities,
            n_export_only,
            n_import_only,
            n_neither,
            n_both_non_estimated,
        ) = map(int, source_counts)

        for criterion, n_rows in (
            ("both", n_both_quantities),
            ("export_only", n_export_only),
            ("import_only", n_import_only),
            ("neither", n_neither),
            ("both_non_estimated", n_both_non_estimated),
        ):
            _append_metric(
                diagnostics_rows,
                scope="source_quantity_availability",
                criterion=criterion,
                n_rows=n_rows,
                total_rows=n_all_rows,
                share_of_all=_share(n_rows, n_all_rows),
            )

        n_disagreement_evaluable = int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM validation_diagnostics_base
                WHERE has_both_quantities
                  AND quantity_relative_difference IS NOT NULL
                  AND isfinite(quantity_relative_difference)
                """
            ).fetchone()[0]
        )

        for quantile in DIAGNOSTIC_QUANTILES:
            value = con.execute(
                f"""
                SELECT QUANTILE_CONT(
                    quantity_relative_difference,
                    {float(quantile):.17g}
                )
                FROM validation_diagnostics_base
                WHERE has_both_quantities
                  AND quantity_relative_difference IS NOT NULL
                  AND isfinite(quantity_relative_difference)
                """
            ).fetchone()[0]

            _append_metric(
                diagnostics_rows,
                scope="source_quantity_disagreement_distribution",
                criterion="both_source_quantities_available",
                variable="quantity_relative_difference",
                quantile=float(quantile),
                value=(float(value) if value is not None else None),
                n_rows=n_disagreement_evaluable,
            )

        for index, (label, lower, upper) in enumerate(
            SOURCE_DISAGREEMENT_BANDS
        ):
            lower_condition = (
                f"quantity_relative_difference >= {lower:.17g}"
                if index == 0
                else f"quantity_relative_difference > {lower:.17g}"
            )

            n_rows = int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM validation_diagnostics_base
                    WHERE has_both_quantities
                      AND quantity_relative_difference IS NOT NULL
                      AND isfinite(quantity_relative_difference)
                      AND {lower_condition}
                      AND quantity_relative_difference <= {upper:.17g}
                    """
                ).fetchone()[0]
            )

            _append_metric(
                diagnostics_rows,
                scope="source_quantity_disagreement_band",
                criterion="reporter_disagreement",
                disagreement_band=label,
                n_rows=n_rows,
                total_rows=n_disagreement_evaluable,
                share_of_parent=_share(n_rows, n_disagreement_evaluable),
            )

        # ==================================================================
        # 3. Training-population audit
        # ==================================================================
        # Canonical sequence:
        #   training_eligible
        #   -> support >= configured minimum
        #   -> positive finite MAD
        #   -> UV distance available
        #   -> normalized UV distance <= configured maximum

        funnel = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_all,

                COUNT(*) FILTER (
                    WHERE training_eligible
                ) AS n_eligible,

                COUNT(*) FILTER (
                    WHERE training_eligible
                      AND hs_year_uv_reference_n IS NOT NULL
                      AND hs_year_uv_reference_n >= {min_reference_observations}
                ) AS n_support,

                COUNT(*) FILTER (
                    WHERE training_eligible
                      AND hs_year_uv_reference_n IS NOT NULL
                      AND hs_year_uv_reference_n >= {min_reference_observations}
                      AND hs_year_uv_log_mad IS NOT NULL
                      AND hs_year_uv_log_mad > 0
                      AND isfinite(hs_year_uv_log_mad)
                ) AS n_positive_mad,

                COUNT(*) FILTER (
                    WHERE training_eligible
                      AND hs_year_uv_reference_n IS NOT NULL
                      AND hs_year_uv_reference_n >= {min_reference_observations}
                      AND hs_year_uv_log_mad IS NOT NULL
                      AND hs_year_uv_log_mad > 0
                      AND isfinite(hs_year_uv_log_mad)
                      AND selected_uv_log_distance IS NOT NULL
                      AND isfinite(selected_uv_log_distance)
                ) AS n_uv_evaluable,

                COUNT(*) FILTER (
                    WHERE training_selected
                ) AS n_selected

            FROM validation_diagnostics_base
            """
        ).fetchone()

        (
            n_training_all,
            n_training_eligible,
            n_after_support,
            n_after_positive_mad,
            n_uv_evaluable,
            n_training_selected,
        ) = map(int, funnel)

        funnel_specs = (
            ("all_rows", n_training_all),
            ("training_eligible", n_training_eligible),
            ("support", n_after_support),
            ("positive_uv_mad", n_after_positive_mad),
            ("uv_distance_available", n_uv_evaluable),
            ("training_selected", n_training_selected),
        )

        previous = n_training_all

        for criterion, n_rows in funnel_specs:
            _append_metric(
                diagnostics_rows,
                scope="training_population_funnel",
                criterion=criterion,
                n_rows=n_rows,
                total_rows=previous,
                share_of_all=_share(n_rows, n_training_all),
                share_of_parent=_share(n_rows, previous),
                uv_support_threshold=(
                    float(min_reference_observations)
                    if criterion in {
                        "support",
                        "positive_uv_mad",
                        "uv_distance_available",
                        "training_selected",
                    }
                    else None
                ),
                uv_distance_threshold=(
                    float(max_normalized_uv_distance_mad)
                    if criterion == "training_selected"
                    else None
                ),
            )
            previous = n_rows

        # Guard against an accidental mismatch between the explicit rule above
        # and the upstream training_selected column.
        n_rule_selected = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM validation_diagnostics_base
                WHERE training_eligible
                  AND hs_year_uv_reference_n IS NOT NULL
                  AND hs_year_uv_reference_n >= {min_reference_observations}
                  AND hs_year_uv_log_mad IS NOT NULL
                  AND hs_year_uv_log_mad > 0
                  AND isfinite(hs_year_uv_log_mad)
                  AND selected_uv_log_distance IS NOT NULL
                  AND isfinite(selected_uv_log_distance)
                  AND (
                        selected_uv_log_distance / hs_year_uv_log_mad
                      ) <= {max_normalized_uv_distance_mad:.17g}
                """
            ).fetchone()[0]
        )

        if n_rule_selected != n_training_selected:
            raise RuntimeError(
                "Diagnostics training rule disagrees with training_selected: "
                f"rule={n_rule_selected:,}, flag={n_training_selected:,}."
            )

        _append_metric(
            diagnostics_rows,
            scope="training_definition",
            criterion=training_definition,
            variable="training_selected",
            n_rows=n_training_selected,
            total_rows=n_training_all,
            share_of_all=_share(n_training_selected, n_training_all),
            uv_support_threshold=float(min_reference_observations),
            uv_distance_threshold=float(max_normalized_uv_distance_mad),
        )

        # ==================================================================
        # 4. Training coverage by year and product
        # ==================================================================
        training_by_year = con.execute(
            """
            SELECT
                year,
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (WHERE training_eligible) AS eligible_rows,
                COUNT(*) FILTER (WHERE training_selected) AS selected_rows,
                COUNT(DISTINCT target_cmd) AS total_products,
                COUNT(DISTINCT target_cmd) FILTER (
                    WHERE training_selected
                ) AS selected_products
            FROM validation_diagnostics_base
            GROUP BY year
            ORDER BY year
            """
        ).fetchdf()

        for row in training_by_year.itertuples(index=False):
            _append_metric(
                diagnostics_rows,
                scope="training_coverage_by_year",
                criterion="final_training_sample",
                year=int(row.year),
                variable="training_selected_rows",
                n_rows=int(row.selected_rows),
                total_rows=int(row.total_rows),
                share_of_all=_share(
                    int(row.selected_rows),
                    int(row.total_rows),
                ),
                share_of_parent=_share(
                    int(row.selected_rows),
                    int(row.eligible_rows),
                ),
                value=float(row.selected_products),
            )

        training_by_product = con.execute(
            """
            SELECT
                target_cmd,
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (WHERE training_eligible) AS eligible_rows,
                COUNT(*) FILTER (WHERE training_selected) AS selected_rows,
                COUNT(DISTINCT year) AS validation_years,
                COUNT(DISTINCT year) FILTER (
                    WHERE training_selected
                ) AS training_years,
                MIN(year) FILTER (
                    WHERE training_selected
                ) AS first_training_year,
                MAX(year) FILTER (
                    WHERE training_selected
                ) AS last_training_year
            FROM validation_diagnostics_base
            GROUP BY target_cmd
            ORDER BY selected_rows ASC, target_cmd ASC
            """
        ).fetchdf()

        for row in training_by_product.itertuples(index=False):
            selected_rows = int(row.selected_rows)
            total_rows = int(row.total_rows)
            eligible_rows = int(row.eligible_rows)
            validation_years = int(row.validation_years)
            training_years = int(row.training_years)

            diagnostics_rows.append(
                {
                    "scope": "training_coverage_by_product",
                    "criterion": (
                        "seen" if selected_rows > 0 else "unseen"
                    ),
                    "candidate": None,
                    "model_name": None,
                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,
                    "disagreement_band": None,
                    "uv_distance_band": None,
                    "year": None,
                    "target_cmd": str(row.target_cmd),
                    "variable": "training_selected_rows",
                    "quantile": None,
                    "n_rows": selected_rows,
                    "total_rows": total_rows,
                    "share_of_all": _share(selected_rows, total_rows),
                    "share_of_parent": _share(selected_rows, eligible_rows),
                    "value": None,
                    "eligible_rows": eligible_rows,
                    "validation_years": validation_years,
                    "training_years": training_years,
                    "first_training_year": (
                        int(row.first_training_year)
                        if pd.notna(row.first_training_year)
                        else None
                    ),
                    "last_training_year": (
                        int(row.last_training_year)
                        if pd.notna(row.last_training_year)
                        else None
                    ),
                    "training_year_coverage": _share(
                        training_years,
                        validation_years,
                    ),
                }
            )

        # Cumulative rejection stages for products absent from final training.
        unseen_products = [
            str(row.target_cmd)
            for row in training_by_product.itertuples(index=False)
            if int(row.selected_rows) == 0
        ]

        if unseen_products:
            escaped = ", ".join(
                "'" + value.replace("'", "''") + "'"
                for value in unseen_products
            )

            rejection = con.execute(
                f"""
                SELECT
                    target_cmd,

                    COUNT(*) FILTER (
                        WHERE training_eligible
                    ) AS eligible_rows,

                    COUNT(*) FILTER (
                        WHERE training_eligible
                          AND hs_year_uv_reference_n IS NOT NULL
                          AND hs_year_uv_reference_n >= {min_reference_observations}
                    ) AS after_support_rows,

                    COUNT(*) FILTER (
                        WHERE training_eligible
                          AND hs_year_uv_reference_n IS NOT NULL
                          AND hs_year_uv_reference_n >= {min_reference_observations}
                          AND hs_year_uv_log_mad IS NOT NULL
                          AND hs_year_uv_log_mad > 0
                          AND isfinite(hs_year_uv_log_mad)
                    ) AS after_positive_mad_rows,

                    COUNT(*) FILTER (
                        WHERE training_eligible
                          AND hs_year_uv_reference_n IS NOT NULL
                          AND hs_year_uv_reference_n >= {min_reference_observations}
                          AND hs_year_uv_log_mad IS NOT NULL
                          AND hs_year_uv_log_mad > 0
                          AND isfinite(hs_year_uv_log_mad)
                          AND selected_uv_log_distance IS NOT NULL
                          AND isfinite(selected_uv_log_distance)
                    ) AS uv_evaluable_rows,

                    COUNT(*) FILTER (
                        WHERE training_selected
                    ) AS selected_rows

                FROM validation_diagnostics_base
                WHERE target_cmd IN ({escaped})
                GROUP BY target_cmd
                ORDER BY target_cmd
                """
            ).fetchdf()

            for row in rejection.itertuples(index=False):
                stages = (
                    ("training_eligible", int(row.eligible_rows)),
                    ("reference_support", int(row.after_support_rows)),
                    ("positive_uv_mad", int(row.after_positive_mad_rows)),
                    ("uv_distance_available", int(row.uv_evaluable_rows)),
                    ("normalized_uv_distance", int(row.selected_rows)),
                )

                previous_rows: int | None = None

                for stage, n_rows in stages:
                    _append_metric(
                        diagnostics_rows,
                        scope="training_rejection_by_product",
                        criterion=stage,
                        target_cmd=str(row.target_cmd),
                        variable="cumulative_training_rows",
                        n_rows=n_rows,
                        total_rows=(
                            previous_rows
                            if previous_rows is not None
                            else n_rows
                        ),
                        share_of_parent=(
                            _share(n_rows, previous_rows)
                            if previous_rows is not None
                            else 1.0
                        ),
                        uv_support_threshold=(
                            float(min_reference_observations)
                            if stage != "training_eligible"
                            else None
                        ),
                        uv_distance_threshold=(
                            float(max_normalized_uv_distance_mad)
                            if stage == "normalized_uv_distance"
                            else None
                        ),
                    )
                    previous_rows = n_rows

        # ==================================================================
        # 5. Reference-window diagnostics
        # ==================================================================
        reference_window_queries: list[str] = []

        for window_name, before, _ in PRE_TRAINING_REFERENCE_WINDOWS:
            reference_window_queries.append(
                f"""
                WITH
                focal_groups AS (
                    SELECT DISTINCT
                        year AS focal_year,
                        target_cmd
                    FROM validation_diagnostics_base
                ),
                eligible_reference_rows AS (
                    SELECT
                        year AS source_year,
                        target_cmd,
                        LN(selected_unit_value) AS log_unit_value
                    FROM validation_diagnostics_base
                    WHERE training_eligible
                      AND selected_unit_value IS NOT NULL
                      AND selected_unit_value > 0
                      AND isfinite(selected_unit_value)
                ),
                window_rows AS (
                    SELECT
                        f.focal_year,
                        f.target_cmd,
                        r.log_unit_value
                    FROM focal_groups AS f
                    INNER JOIN eligible_reference_rows AS r
                        ON r.target_cmd = f.target_cmd
                       AND r.source_year BETWEEN
                           f.focal_year - {before}
                           AND f.focal_year
                ),
                centres AS (
                    SELECT
                        focal_year,
                        target_cmd,
                        COUNT(*) AS reference_n,
                        MEDIAN(log_unit_value) AS median_log_unit_value
                    FROM window_rows
                    GROUP BY focal_year, target_cmd
                ),
                deviations AS (
                    SELECT
                        r.focal_year,
                        r.target_cmd,
                        ABS(
                            r.log_unit_value - c.median_log_unit_value
                        ) AS log_absolute_deviation
                    FROM window_rows AS r
                    INNER JOIN centres AS c
                        USING (focal_year, target_cmd)
                ),
                dispersion AS (
                    SELECT
                        focal_year,
                        target_cmd,
                        MEDIAN(log_absolute_deviation) AS log_mad
                    FROM deviations
                    GROUP BY focal_year, target_cmd
                )
                SELECT
                    '{window_name}' AS reference_window,
                    {before} AS reference_before,
                    f.focal_year AS year,
                    f.target_cmd,
                    COALESCE(c.reference_n, 0) AS reference_n,
                    c.median_log_unit_value,
                    d.log_mad
                FROM focal_groups AS f
                LEFT JOIN centres AS c
                    ON c.focal_year = f.focal_year
                   AND c.target_cmd = f.target_cmd
                LEFT JOIN dispersion AS d
                    ON d.focal_year = f.focal_year
                   AND d.target_cmd = f.target_cmd
                """
            )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE pre_training_reference_windows AS
            """
            + "\nUNION ALL\n".join(
                f"SELECT * FROM ({query})"
                for query in reference_window_queries
            )
        )

        for window_name, before, _ in PRE_TRAINING_REFERENCE_WINDOWS:
            row = con.execute(
                """
                SELECT
                    COUNT(*) AS n_groups,
                    COUNT(*) FILTER (
                        WHERE reference_n > 0
                    ) AS n_with_reference,
                    COUNT(*) FILTER (
                        WHERE reference_n >= ?
                    ) AS n_with_support,
                    MEDIAN(reference_n) AS median_support,
                    MEDIAN(log_mad) FILTER (
                        WHERE log_mad IS NOT NULL
                    ) AS median_log_mad
                FROM pre_training_reference_windows
                WHERE reference_window = ?
                """,
                [min_reference_observations, window_name],
            ).fetchone()

            (
                n_groups,
                n_with_reference,
                n_with_support,
                median_support,
                median_log_mad,
            ) = row

            _append_metric(
                diagnostics_rows,
                scope="reference_window_summary",
                criterion=window_name,
                variable="groups_with_support",
                n_rows=int(n_with_support),
                total_rows=int(n_groups),
                share_of_parent=_share(int(n_with_support), int(n_groups)),
                value=(
                    float(median_support)
                    if median_support is not None
                    else None
                ),
                uv_support_threshold=float(min_reference_observations),
            )

            _append_metric(
                diagnostics_rows,
                scope="reference_window_summary",
                criterion=window_name,
                variable="median_log_mad",
                n_rows=int(n_with_reference),
                total_rows=int(n_groups),
                value=(
                    float(median_log_mad)
                    if median_log_mad is not None
                    else None
                ),
            )

        # ==================================================================
        # 6. Current-reference distributions
        # ==================================================================
        for quantile in DIAGNOSTIC_QUANTILES:
            support_value = con.execute(
                f"""
                SELECT QUANTILE_CONT(
                    hs_year_uv_reference_n,
                    {float(quantile):.17g}
                )
                FROM (
                    SELECT DISTINCT
                        year,
                        target_cmd,
                        hs_year_uv_reference_n
                    FROM validation_diagnostics_base
                    WHERE hs_year_uv_reference_n IS NOT NULL
                )
                """
            ).fetchone()[0]

            _append_metric(
                diagnostics_rows,
                scope="training_reference_support_distribution",
                criterion="hs_year_groups",
                variable="hs_year_uv_reference_n",
                quantile=float(quantile),
                value=(
                    float(support_value)
                    if support_value is not None
                    else None
                ),
            )

            mad_value = con.execute(
                f"""
                SELECT QUANTILE_CONT(
                    hs_year_uv_log_mad,
                    {float(quantile):.17g}
                )
                FROM (
                    SELECT DISTINCT
                        year,
                        target_cmd,
                        hs_year_uv_log_mad
                    FROM validation_diagnostics_base
                    WHERE hs_year_uv_log_mad IS NOT NULL
                      AND isfinite(hs_year_uv_log_mad)
                )
                """
            ).fetchone()[0]

            _append_metric(
                diagnostics_rows,
                scope="training_uv_mad_distribution",
                criterion="hs_year_groups",
                variable="hs_year_uv_log_mad",
                quantile=float(quantile),
                value=(float(mad_value) if mad_value is not None else None),
            )

            normalized_value = con.execute(
                f"""
                SELECT QUANTILE_CONT(
                    training_normalized_uv_distance,
                    {float(quantile):.17g}
                )
                FROM validation_diagnostics_base
                WHERE training_eligible
                  AND hs_year_uv_reference_n IS NOT NULL
                  AND hs_year_uv_reference_n >= {min_reference_observations}
                  AND hs_year_uv_log_mad IS NOT NULL
                  AND hs_year_uv_log_mad > 0
                  AND isfinite(hs_year_uv_log_mad)
                  AND training_normalized_uv_distance IS NOT NULL
                  AND isfinite(training_normalized_uv_distance)
                """
            ).fetchone()[0]

            _append_metric(
                diagnostics_rows,
                scope="training_normalized_uv_distance_distribution",
                criterion="pre_cutoff_population",
                variable="training_normalized_uv_distance",
                quantile=float(quantile),
                value=(
                    float(normalized_value)
                    if normalized_value is not None
                    else None
                ),
                n_rows=n_uv_evaluable,
            )

        # ==================================================================
        # 7. Normalized-UV cutoff sensitivity
        # ==================================================================
        for cutoff in NORMALIZED_UV_CUTOFFS:
            n_rows = int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM validation_diagnostics_base
                    WHERE training_eligible
                      AND hs_year_uv_reference_n IS NOT NULL
                      AND hs_year_uv_reference_n >= {min_reference_observations}
                      AND hs_year_uv_log_mad IS NOT NULL
                      AND hs_year_uv_log_mad > 0
                      AND isfinite(hs_year_uv_log_mad)
                      AND training_normalized_uv_distance IS NOT NULL
                      AND isfinite(training_normalized_uv_distance)
                      AND training_normalized_uv_distance <= {float(cutoff):.17g}
                    """
                ).fetchone()[0]
            )

            _append_metric(
                diagnostics_rows,
                scope="training_normalized_uv_cutoff_sensitivity",
                criterion="canonical_pre_cutoff_population",
                variable="training_normalized_uv_distance",
                n_rows=n_rows,
                total_rows=n_uv_evaluable,
                share_of_all=_share(n_rows, n_training_all),
                share_of_parent=_share(n_rows, n_uv_evaluable),
                uv_support_threshold=float(min_reference_observations),
                uv_distance_threshold=float(cutoff),
                value=float(cutoff),
            )

        # ==================================================================
        # 8. Model evaluation
        # ==================================================================
        overall_models = con.execute(
            """
            SELECT
                model_name,
                COUNT(*) AS n_rows,
                AVG(absolute_log_error) AS mae_log,
                SQRT(AVG(squared_log_error)) AS rmse_log,
                MEDIAN(absolute_log_error) AS median_absolute_log_error
            FROM validation_model_predictions
            GROUP BY model_name
            ORDER BY model_name
            """
        ).fetchdf()

        for row in overall_models.itertuples(index=False):
            for variable, value in (
                ("mae_log", row.mae_log),
                ("rmse_log", row.rmse_log),
                (
                    "median_absolute_log_error",
                    row.median_absolute_log_error,
                ),
            ):
                _append_metric(
                    diagnostics_rows,
                    scope="model_performance_overall",
                    criterion="out_of_fold",
                    model_name=str(row.model_name),
                    variable=variable,
                    value=float(value),
                    n_rows=int(row.n_rows),
                )

        by_year_models = con.execute(
            """
            SELECT
                model_name,
                held_out_year,
                COUNT(*) AS n_rows,
                AVG(absolute_log_error) AS mae_log,
                SQRT(AVG(squared_log_error)) AS rmse_log,
                MEDIAN(absolute_log_error) AS median_absolute_log_error
            FROM validation_model_predictions
            GROUP BY model_name, held_out_year
            ORDER BY model_name, held_out_year
            """
        ).fetchdf()

        for row in by_year_models.itertuples(index=False):
            for variable, value in (
                ("mae_log", row.mae_log),
                ("rmse_log", row.rmse_log),
                (
                    "median_absolute_log_error",
                    row.median_absolute_log_error,
                ),
            ):
                _append_metric(
                    diagnostics_rows,
                    scope="model_performance_by_year",
                    criterion="out_of_fold",
                    model_name=str(row.model_name),
                    year=int(row.held_out_year),
                    variable=variable,
                    value=float(value),
                    n_rows=int(row.n_rows),
                )

        # ==================================================================
        # Diagnostics artifact
        # ==================================================================
        diagnostics = pd.DataFrame(diagnostics_rows)

        if diagnostics.empty:
            raise RuntimeError("Validation diagnostics contain no rows.")

        optional_columns = (
            "candidate",
            "model_name",
            "disagreement_threshold",
            "uv_support_threshold",
            "uv_distance_threshold",
            "disagreement_band",
            "uv_distance_band",
            "year",
            "target_cmd",
            "variable",
            "quantile",
            "n_rows",
            "total_rows",
            "share_of_all",
            "share_of_parent",
            "value",
            "eligible_rows",
            "validation_years",
            "training_years",
            "first_training_year",
            "last_training_year",
            "training_year_coverage",
        )

        for column in optional_columns:
            if column not in diagnostics.columns:
                diagnostics[column] = None

        diagnostics = diagnostics[
            [
                "scope",
                "criterion",
                "candidate",
                "model_name",
                "disagreement_threshold",
                "uv_support_threshold",
                "uv_distance_threshold",
                "disagreement_band",
                "uv_distance_band",
                "year",
                "target_cmd",
                "variable",
                "quantile",
                "n_rows",
                "total_rows",
                "share_of_all",
                "share_of_parent",
                "value",
                "eligible_rows",
                "validation_years",
                "training_years",
                "first_training_year",
                "last_training_year",
                "training_year_coverage",
            ]
        ]

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        if temporary_path.exists():
            temporary_path.unlink()

        con.register("_validation_diagnostics", diagnostics)

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM _validation_diagnostics
            )
            TO {sql_literal(temporary_path)}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        temporary_path.replace(output_path)

    finally:
        con.close()

    elapsed = perf_counter() - start

    logger.info("  All Validation rows: %s", f"{n_all_rows:,}")
    logger.info(
        "  Canonical training-eligible rows: %s",
        f"{n_training_eligible:,}",
    )
    logger.info(
        "  Final selected training rows: %s",
        f"{n_training_selected:,}",
    )
    logger.info("  Diagnostics rows: %s", f"{len(diagnostics):,}")
    logger.info("  Output: %s", output_path)
    logger.info(
        "%s CHECK — Validation diagnostics built.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "diagnostics_path": output_path,
        "n_rows": n_all_rows,
        "n_training_eligible_rows": n_training_eligible,
        "n_training_selected_rows": n_training_selected,
        "n_both_source_quantities": n_both_quantities,
        "fitted": False,
        "selection_already_applied": True,
        "duration_seconds": elapsed,
        "reused": False,
    }
