"""
vacoda.steps.validation.diagnostics

Build the diagnostics used by Validation workflow.

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


def _safe_identifier(
    name: str,
) -> str:
    """Return a safely quoted SQL identifier."""

    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*",
        str(name),
    ):
        raise ValueError(
            f"Unsafe Validation diagnostics column identifier: {name!r}"
        )

    return f'"{name}"'


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
    """Require configured diagnostics-development columns in the input."""

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

    missing = sorted(
        required_columns - available
    )

    if missing:
        raise RuntimeError(
            "Validation diagnostics input is missing required column(s): "
            + ", ".join(missing)
        )


# ======================================================================
# Public step
# ======================================================================


def build_diagnostics(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    structural_audit_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Characterize candidate diagnostics training populations.

    """

    start = perf_counter()

    input_path = Path(input_path)
    structural_audit_path = Path(
        structural_audit_path
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Validation diagnostics input not found: {input_path}"
        )

    if not structural_audit_path.is_file():
        raise FileNotFoundError(
            "Validation diagnostics structural-audit artifact not found: "
            f"{structural_audit_path}"
        )

    output_path = cfg.product_path(
        "diagnostics"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Reuse
    # ------------------------------------------------------------------

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
            "ready_for_training_decision": True,
        }

    logger.info(
        "%s REBUILD — Validation diagnostics",
        get_symbols("rebuilt"),
    )
    logger.info(
        "Characterizing candidate diagnostics training populations."
    )
    logger.info(
        "  Input: %s",
        input_path,
    )
    logger.info(
        "  Structural audit: %s",
        structural_audit_path,
    )
    logger.info(
        "  Output: %s",
        output_path,
    )

    # ==================================================================
    # Configuration
    # ==================================================================

    diagnostics_cfg = cfg.diagnostics
    columns_cfg = diagnostics_cfg.columns
    training_cfg = diagnostics_cfg.training_population

    year_col = str(
        columns_cfg.year
    )
    product_col = str(
        columns_cfg.product
    )

    quantity_col = str(
        columns_cfg.quantity
    )

    nominal_value_col = str(
        columns_cfg.nominal_value
    )

    constant_value_col = str(
        columns_cfg.constant_value
    )

    export_quantity_col = str(
        columns_cfg.export_quantity
    )

    import_quantity_col = str(
        columns_cfg.import_quantity
    )

    export_estimated_col = str(
        columns_cfg.export_estimated
    )

    import_estimated_col = str(
        columns_cfg.import_estimated
    )

    relative_difference_col = str(
        columns_cfg.quantity_relative_difference
    )

    uv_reference_col = str(
        columns_cfg.unit_value_reference
    )

    uv_reference_n_col = str(
        columns_cfg.unit_value_reference_observations
    )

    disagreement_thresholds = tuple(
        training_cfg.disagreement_thresholds
    )

    uv_support_thresholds = tuple(
        training_cfg.unit_value_reference_min_observations
    )

    uv_distance_thresholds = tuple(
        training_cfg.unit_value_log_distance_thresholds
    )

    require_bilateral = bool(
        training_cfg.require_bilateral_quantities
    )

    require_non_estimated = bool(
        training_cfg.require_non_estimated_quantities
    )


    logger.info(
        "  Require bilateral quantities: %s",
        require_bilateral,
    )
    logger.info(
        "  Require non-estimated quantities: %s",
        require_non_estimated,
    )
    logger.info(
        "  Mirror-disagreement thresholds: %s",
        ", ".join(
            f"{value:.0%}"
            for value in disagreement_thresholds
        ),
    )
    logger.info(
        "  Unit-value support thresholds: %s",
        ", ".join(
            f"{value:,}"
            for value in uv_support_thresholds
        ),
    )
    logger.info(
        "  Unit-value log-distance thresholds: %s",
        ", ".join(
            f"{value:g}"
            for value in uv_distance_thresholds
        ),
    )
    logger.info(
        "  Final training selection enabled: %s",
        diagnostics_cfg.selection.enabled,
    )

    # ==================================================================
    # Input contract
    # ==================================================================

    required_columns = {
        year_col,
        product_col,
        quantity_col,
        nominal_value_col,
        constant_value_col,
        export_quantity_col,
        import_quantity_col,
        export_estimated_col,
        import_estimated_col,
        relative_difference_col,
        uv_reference_col,
        uv_reference_n_col,
    }

    con = duckdb.connect()

    try:
        con.execute(
            f"PRAGMA threads={int(cfg.engine.threads)}"
        )

        _require_columns(
            con=con,
            input_path=input_path,
            required_columns=required_columns,
        )

        input_sql = sql_literal(
            input_path
        )

        # --------------------------------------------------------------
        # Safe SQL identifiers
        # --------------------------------------------------------------

        year_sql = _safe_identifier(
            year_col
        )

        product_sql = _safe_identifier(
            product_col
        )

        quantity_sql = _safe_identifier(
            quantity_col
        )

        nominal_value_sql = _safe_identifier(
            nominal_value_col
        )

        constant_value_sql = _safe_identifier(
            constant_value_col
        )

        export_quantity_sql = _safe_identifier(
            export_quantity_col
        )

        import_quantity_sql = _safe_identifier(
            import_quantity_col
        )

        export_estimated_sql = _safe_identifier(
            export_estimated_col
        )

        import_estimated_sql = _safe_identifier(
            import_estimated_col
        )

        relative_difference_sql = _safe_identifier(
            relative_difference_col
        )

        uv_reference_sql = _safe_identifier(
            uv_reference_col
        )

        uv_reference_n_sql = _safe_identifier(
            uv_reference_n_col
        )

        # ==================================================================
        # Canonical diagnostics-development view
        # ==================================================================
        #
        # The unit-value plausibility comparison intentionally uses
        # nominal reconciled trade value.
        #
        # The product-year unit-value reference was derived from nominal
        # within-year trade values. Using constant-price USD against that
        # nominal reference would mix two price bases.
        #
        # Constant-price value is retained here for later predictive model
        # development, where cross-year comparability becomes relevant.
        # ==================================================================

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW
                validation_diagnostics_base
            AS

            SELECT
                {year_sql}
                    AS year,

                {product_sql}
                    AS target_cmd,

                {quantity_sql}
                    AS trade_synced_quantity_kg,

                {nominal_value_sql}
                    AS trade_value_reconciled,

                {constant_value_sql}
                    AS trade_value_reconciled_constant_usd,

                {export_quantity_sql}
                    AS export_quantity_kg,

                {import_quantity_sql}
                    AS import_quantity_kg,

                {export_estimated_sql}
                    AS export_quantity_estimated,

                {import_estimated_sql}
                    AS import_quantity_estimated,

                {relative_difference_sql}
                    AS quantity_relative_difference,

                                    CASE
                    WHEN
                        {export_quantity_sql} IS NOT NULL
                        AND {import_quantity_sql} IS NOT NULL

                        AND {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0

                        AND isfinite(
                            CAST(
                                {export_quantity_sql}
                                AS DOUBLE
                            )
                        )

                        AND isfinite(
                            CAST(
                                {import_quantity_sql}
                                AS DOUBLE
                            )
                        )

                    THEN
                        (
                            CAST(
                                {export_quantity_sql}
                                AS DOUBLE
                            )
                            -
                            CAST(
                                {import_quantity_sql}
                                AS DOUBLE
                            )
                        )
                        /
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )

                    ELSE NULL
                END
                    AS quantity_difference_vs_import,

                CASE
                    WHEN
                        {export_quantity_sql} IS NOT NULL
                        AND {import_quantity_sql} IS NOT NULL

                        AND {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0

                        AND isfinite(
                            CAST(
                                {export_quantity_sql}
                                AS DOUBLE
                            )
                        )

                        AND isfinite(
                            CAST(
                                {import_quantity_sql}
                                AS DOUBLE
                            )
                        )

                    THEN
                        ABS(
                            CAST(
                                {export_quantity_sql}
                                AS DOUBLE
                            )
                            -
                            CAST(
                                {import_quantity_sql}
                                AS DOUBLE
                            )
                        )
                        /
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )

                    ELSE NULL
                END
                    AS quantity_absolute_difference_vs_import,

                CASE
                    WHEN
                        {export_quantity_sql} IS NOT NULL
                        AND {import_quantity_sql} IS NOT NULL
                        AND {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0
                        AND isfinite(
                            CAST(
                                {export_quantity_sql}
                                AS DOUBLE
                            )
                        )
                        AND isfinite(
                            CAST(
                                {import_quantity_sql}
                                AS DOUBLE
                            )
                        )
                    THEN SQRT(
                        CAST(
                            {export_quantity_sql}
                            AS DOUBLE
                        )
                        * CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )
                    )
                END
                    AS bilateral_quantity_kg,

                CASE
                    WHEN
                        {export_quantity_sql} IS NOT NULL
                        AND {import_quantity_sql} IS NOT NULL
                        AND {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0
                        AND {nominal_value_sql} IS NOT NULL
                        AND {nominal_value_sql} > 0
                        AND isfinite(CAST({export_quantity_sql} AS DOUBLE))
                        AND isfinite(CAST({import_quantity_sql} AS DOUBLE))
                        AND isfinite(CAST({nominal_value_sql} AS DOUBLE))
                    THEN
                        CAST({nominal_value_sql} AS DOUBLE)
                        /
                        SQRT(
                            CAST({export_quantity_sql} AS DOUBLE)
                            * CAST({import_quantity_sql} AS DOUBLE)
                        )
                END
                    AS bilateral_unit_value,

                CASE
                    WHEN
                        {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0
                    THEN LN(
                        SQRT(
                            CAST({export_quantity_sql} AS DOUBLE)
                            * CAST({import_quantity_sql} AS DOUBLE)
                        )
                    )
                END
                    AS log_bilateral_quantity,

                CASE
                    WHEN
                        {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0
                        AND {nominal_value_sql} > 0
                    THEN LN(
                        CAST({nominal_value_sql} AS DOUBLE)
                        /
                        SQRT(
                            CAST({export_quantity_sql} AS DOUBLE)
                            * CAST({import_quantity_sql} AS DOUBLE)
                        )
                    )
                END
                    AS log_bilateral_unit_value,

                {uv_reference_sql}
                    AS product_year_median_unit_value,

                {uv_reference_n_sql}
                    AS unit_value_reference_observations,

                (
                    {quantity_sql} IS NOT NULL
                    AND {quantity_sql} > 0
                    AND isfinite(
                        CAST(
                            {quantity_sql}
                            AS DOUBLE
                        )
                    )
                )
                    AS has_synced_quantity,

                (
                    {export_quantity_sql} IS NOT NULL
                    AND {import_quantity_sql} IS NOT NULL

                    AND {export_quantity_sql} > 0
                    AND {import_quantity_sql} > 0

                    AND isfinite(
                        CAST(
                            {export_quantity_sql}
                            AS DOUBLE
                        )
                    )

                    AND isfinite(
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )
                    )
                )
                    AS has_both_quantities,

                (
                    {export_quantity_sql} IS NOT NULL
                    AND {import_quantity_sql} IS NOT NULL

                    AND {export_quantity_sql} > 0
                    AND {import_quantity_sql} > 0

                    AND isfinite(
                        CAST(
                            {export_quantity_sql}
                            AS DOUBLE
                        )
                    )

                    AND isfinite(
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )
                    )

                    AND COALESCE(
                        {export_estimated_sql},
                        FALSE
                    ) = FALSE

                    AND COALESCE(
                        {import_estimated_sql},
                        FALSE
                    ) = FALSE
                )
                    AS both_quantities_observed,

        CASE
            WHEN
                {export_quantity_sql} IS NOT NULL
                AND {export_quantity_sql} > 0

                AND {nominal_value_sql} IS NOT NULL
                AND {nominal_value_sql} > 0

                AND {uv_reference_sql} IS NOT NULL
                AND {uv_reference_sql} > 0

                AND isfinite(
                    CAST(
                        {export_quantity_sql}
                        AS DOUBLE
                    )
                )

                AND isfinite(
                    CAST(
                        {nominal_value_sql}
                        AS DOUBLE
                    )
                )

                AND isfinite(
                    CAST(
                        {uv_reference_sql}
                        AS DOUBLE
                    )
                )

            THEN
                ABS(
                    LN(
                        CAST(
                            {nominal_value_sql}
                            AS DOUBLE
                        )
                        /
                        CAST(
                            {export_quantity_sql}
                            AS DOUBLE
                        )
                    )
                    -
                    LN(
                        CAST(
                            {uv_reference_sql}
                            AS DOUBLE
                        )
                    )
                )

            ELSE NULL
        END
            AS export_unit_value_log_distance,

        CASE
            WHEN
                {import_quantity_sql} IS NOT NULL
                AND {import_quantity_sql} > 0

                AND {nominal_value_sql} IS NOT NULL
                AND {nominal_value_sql} > 0

                AND {uv_reference_sql} IS NOT NULL
                AND {uv_reference_sql} > 0

                AND isfinite(
                    CAST(
                        {import_quantity_sql}
                        AS DOUBLE
                    )
                )

                AND isfinite(
                    CAST(
                        {nominal_value_sql}
                        AS DOUBLE
                    )
                )

                AND isfinite(
                    CAST(
                        {uv_reference_sql}
                        AS DOUBLE
                    )
                )

            THEN
                ABS(
                    LN(
                        CAST(
                            {nominal_value_sql}
                            AS DOUBLE
                        )
                        /
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )
                    )
                    -
                    LN(
                        CAST(
                            {uv_reference_sql}
                            AS DOUBLE
                        )
                    )
                )

            ELSE NULL
        END
            AS import_unit_value_log_distance,
                    
                CASE
                    WHEN
                        {quantity_sql} IS NOT NULL
                        AND {quantity_sql} > 0

                        AND {nominal_value_sql} IS NOT NULL
                        AND {nominal_value_sql} > 0

                        AND {uv_reference_sql} IS NOT NULL
                        AND {uv_reference_sql} > 0

                        AND isfinite(
                            CAST(
                                {quantity_sql}
                                AS DOUBLE
                            )
                        )

                        AND isfinite(
                            CAST(
                                {nominal_value_sql}
                                AS DOUBLE
                            )
                        )

                        AND isfinite(
                            CAST(
                                {uv_reference_sql}
                                AS DOUBLE
                            )
                        )

                    THEN
                        ABS(
                            LN(
                                CAST(
                                    {nominal_value_sql}
                                    AS DOUBLE
                                )
                                /
                                CAST(
                                    {quantity_sql}
                                    AS DOUBLE
                                )
                            )
                            -
                            LN(
                                CAST(
                                    {uv_reference_sql}
                                    AS DOUBLE
                                )
                            )
                        )

                    ELSE NULL
                END
                    AS synced_unit_value_log_distance

            FROM read_parquet({input_sql})
            """
        )

        # ==================================================================
        # 1. Population funnel
        # ==================================================================

        funnel = con.execute(
            """
            SELECT
                COUNT(*)
                    AS n_all_rows,

                COUNT(*) FILTER (
                    WHERE has_synced_quantity
                )
                    AS n_synced_quantity,

                COUNT(*) FILTER (
                    WHERE has_both_quantities
                )
                    AS n_both_quantities,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                )
                    AS n_both_observed,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_relative_difference
                          IS NOT NULL
                )
                    AS n_both_observed_with_disagreement,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                      AND synced_unit_value_log_distance
                          IS NOT NULL
                )
                    AS n_both_observed_with_uv_distance,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_relative_difference
                          IS NOT NULL
                      AND synced_unit_value_log_distance
                          IS NOT NULL
                      AND unit_value_reference_observations
                          IS NOT NULL
                )
                    AS n_complete_training_evidence

            FROM validation_diagnostics_base
            """
        ).fetchone()

        (
            n_all_rows,
            n_synced_quantity,
            n_both_quantities,
            n_both_observed,
            n_both_observed_with_disagreement,
            n_both_observed_with_uv_distance,
            n_complete_training_evidence,
        ) = map(
            int,
            funnel,
        )

        diagnostics_rows: list[
            dict[str, Any]
        ] = []

        funnel_specs = (
            (
                "all_rows",
                n_all_rows,
                n_all_rows,
            ),
            (
                "trade_synced_quantity_available",
                n_synced_quantity,
                n_all_rows,
            ),
            (
                "both_quantities_available",
                n_both_quantities,
                n_synced_quantity,
            ),
            (
                "both_quantities_non_estimated",
                n_both_observed,
                n_both_quantities,
            ),
            (
                "both_observed_with_disagreement",
                n_both_observed_with_disagreement,
                n_both_observed,
            ),
            (
                "both_observed_with_uv_distance",
                n_both_observed_with_uv_distance,
                n_both_observed,
            ),
            (
                "complete_training_evidence",
                n_complete_training_evidence,
                n_both_observed,
            ),
        )

        for (
            criterion,
            n_rows,
            n_parent_rows,
        ) in funnel_specs:

            diagnostics_rows.append(
                {
                    "scope": "population_funnel",
                    "criterion": criterion,

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": None,
                    "quantile": None,

                    "n_rows": int(n_rows),

                    "share_of_all": _share(
                        n_rows,
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        n_rows,
                        n_parent_rows,
                    ),

                    "value": None,
                }
            )

        # ==================================================================
        # 2. Individual threshold sensitivity
        # ==================================================================

        threshold_expressions: list[str] = []
        threshold_metadata: list[
            tuple[str, float | int, str]
        ] = []

        # --------------------------------------------------------------
        # Mirror disagreement
        # --------------------------------------------------------------

        for index, threshold in enumerate(
            disagreement_thresholds
        ):
            alias = (
                f"disagreement_{index}"
            )

            threshold_expressions.append(
                f"""
                COUNT(*) FILTER (
                    WHERE both_quantities_observed

                      AND quantity_relative_difference
                          IS NOT NULL

                      AND quantity_relative_difference
                          <= {float(threshold):.17g}
                ) AS {alias}
                """
            )

            threshold_metadata.append(
                (
                    "quantity_relative_difference",
                    float(threshold),
                    alias,
                )
            )

        # --------------------------------------------------------------
        # Product-year unit-value support
        # --------------------------------------------------------------

        for index, threshold in enumerate(
            uv_support_thresholds
        ):
            alias = (
                f"uv_support_{index}"
            )

            threshold_expressions.append(
                f"""
                COUNT(*) FILTER (
                    WHERE both_quantities_observed

                      AND unit_value_reference_observations
                          IS NOT NULL

                      AND unit_value_reference_observations
                          >= {int(threshold)}
                ) AS {alias}
                """
            )

            threshold_metadata.append(
                (
                    "unit_value_reference_observations",
                    int(threshold),
                    alias,
                )
            )

        # --------------------------------------------------------------
        # Synchronized unit-value distance
        # --------------------------------------------------------------

        for index, threshold in enumerate(
            uv_distance_thresholds
        ):
            alias = (
                f"uv_distance_{index}"
            )

            threshold_expressions.append(
                f"""
                COUNT(*) FILTER (
                    WHERE both_quantities_observed

                      AND synced_unit_value_log_distance
                          IS NOT NULL

                      AND synced_unit_value_log_distance
                          <= {float(threshold):.17g}
                ) AS {alias}
                """
            )

            threshold_metadata.append(
                (
                    "synced_unit_value_log_distance",
                    float(threshold),
                    alias,
                )
            )

        sensitivity_row = con.execute(
            f"""
            SELECT
                {", ".join(threshold_expressions)}
            FROM validation_diagnostics_base
            """
        ).fetchone()

        for (
            metadata,
            value,
        ) in zip(
            threshold_metadata,
            sensitivity_row,
        ):
            (
                variable,
                threshold,
                _,
            ) = metadata

            n_rows = int(value)

            diagnostics_rows.append(
                {
                    "scope": "threshold_sensitivity",
                    "criterion": variable,

                    "disagreement_threshold": (
                        float(threshold)
                        if variable
                        == "quantity_relative_difference"
                        else None
                    ),

                    "uv_support_threshold": (
                        float(threshold)
                        if variable
                        == "unit_value_reference_observations"
                        else None
                    ),

                    "uv_distance_threshold": (
                        float(threshold)
                        if variable
                        == "synced_unit_value_log_distance"
                        else None
                    ),

                    "year": None,
                    "target_cmd": None,

                    "variable": variable,
                    "quantile": None,

                    "n_rows": n_rows,

                    "share_of_all": _share(
                        n_rows,
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        n_rows,
                        n_both_observed,
                    ),

                    "value": None,
                }
            )

        # ==================================================================
        # 3. Full candidate-training threshold grid
        # ==================================================================
        #
        # Every configured combination is evaluated.
        #
        # Nothing here is declared to be the final training population.
        # That decision is made only after diagnostics/reporting review.
        # ==================================================================

        grid_expressions: list[str] = []

        grid_metadata: list[
            tuple[
                float,
                int,
                float,
                str,
            ]
        ] = []

        grid_index = 0

        for disagreement in disagreement_thresholds:

            for support in uv_support_thresholds:

                for uv_distance in uv_distance_thresholds:

                    alias = (
                        f"training_grid_{grid_index}"
                    )

                    grid_expressions.append(
                        f"""
                        COUNT(*) FILTER (
                            WHERE both_quantities_observed

                              AND quantity_relative_difference
                                  IS NOT NULL

                              AND quantity_relative_difference
                                  <= {float(disagreement):.17g}

                              AND unit_value_reference_observations
                                  IS NOT NULL

                              AND unit_value_reference_observations
                                  >= {int(support)}

                              AND synced_unit_value_log_distance
                                  IS NOT NULL

                              AND synced_unit_value_log_distance
                                  <= {float(uv_distance):.17g}
                        ) AS {alias}
                        """
                    )

                    grid_metadata.append(
                        (
                            float(disagreement),
                            int(support),
                            float(uv_distance),
                            alias,
                        )
                    )

                    grid_index += 1

        grid_row = con.execute(
            f"""
            SELECT
                {", ".join(grid_expressions)}
            FROM validation_diagnostics_base
            """
        ).fetchone()

        for (
            metadata,
            value,
        ) in zip(
            grid_metadata,
            grid_row,
        ):
            (
                disagreement,
                support,
                uv_distance,
                _,
            ) = metadata

            n_rows = int(value)

            diagnostics_rows.append(
                {
                    "scope": "threshold_grid",

                    "criterion": (
                        "candidate_training_population"
                    ),

                    "disagreement_threshold": (
                        disagreement
                    ),

                    "uv_support_threshold": float(
                        support
                    ),

                    "uv_distance_threshold": (
                        uv_distance
                    ),

                    "year": None,
                    "target_cmd": None,

                    "variable": None,
                    "quantile": None,

                    "n_rows": n_rows,

                    "share_of_all": _share(
                        n_rows,
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        n_rows,
                        n_both_observed,
                    ),

                    "value": None,
                }
            )

        # ==================================================================
        # 4. Mirror-agreement × bilateral unit-value plausibility matrix
        # ==================================================================
        #
        # Unlike the cumulative threshold grid above, this diagnostic uses
        # mutually exclusive bands.
        #
        # Unit-value plausibility is defined using the WORSE of the exporter
        # and importer distances. Therefore, a row can enter a low-distance
        # band only when both reported quantities independently imply
        # plausible unit values.
        # ==================================================================

        disagreement_edges = tuple(
            float(value)
            for value in disagreement_thresholds
        )

        uv_edges = tuple(
            float(value)
            for value in uv_distance_thresholds
        )

        def _band_case(
            *,
            column: str,
            edges: tuple[float, ...],
            prefix: str,
        ) -> str:

            clauses: list[str] = []

            lower = 0.0

            for index, upper in enumerate(
                edges
            ):
                if index == 0:
                    label = (
                        f"{100 * upper:.0f}% or less"
                        if prefix == "mirror"
                        else f"{upper:g} or less"
                    )

                    clauses.append(
                        f"""
                        WHEN {column} <= {upper:.17g}
                            THEN '{label}'
                        """
                    )

                else:
                    label = (
                        f"{100 * lower:.0f}–{100 * upper:.0f}%"
                        if prefix == "mirror"
                        else f"{lower:g}–{upper:g}"
                    )

                    clauses.append(
                        f"""
                        WHEN {column} <= {upper:.17g}
                            THEN '{label}'
                        """
                    )

                lower = upper

            final_label = (
                f"> {100 * edges[-1]:.0f}%"
                if prefix == "mirror"
                else f"> {edges[-1]:g}"
            )

            return (
                "CASE "
                + " ".join(clauses)
                + f" ELSE '{final_label}' END"
            )

        disagreement_band_sql = _band_case(
            column="quantity_relative_difference",
            edges=disagreement_edges,
            prefix="mirror",
        )

        bilateral_uv_distance_sql = """
            GREATEST(
                export_unit_value_log_distance,
                import_unit_value_log_distance
            )
        """

        uv_band_sql = _band_case(
            column=bilateral_uv_distance_sql,
            edges=uv_edges,
            prefix="uv",
        )

        evidence_matrix = con.execute(
            f"""
            WITH evidence AS (
                SELECT
                    {disagreement_band_sql}
                        AS disagreement_band,

                    {uv_band_sql}
                        AS uv_distance_band

                FROM validation_diagnostics_base

                WHERE both_quantities_observed

                  AND quantity_relative_difference
                      IS NOT NULL

                  AND export_unit_value_log_distance
                      IS NOT NULL

                  AND import_unit_value_log_distance
                      IS NOT NULL
            )

            SELECT
                disagreement_band,
                uv_distance_band,
                COUNT(*) AS n_rows

            FROM evidence

            GROUP BY
                disagreement_band,
                uv_distance_band
            """
        ).fetchdf()

        n_matrix_rows = int(
            evidence_matrix["n_rows"].sum()
        )

        for row in evidence_matrix.itertuples(
            index=False
        ):
            diagnostics_rows.append(
                {
                    "scope": "evidence_matrix",

                    "criterion": (
                        "mirror_agreement_x_bilateral_uv_plausibility"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": (
                        row.disagreement_band
                    ),

                    "uv_distance_band": (
                        row.uv_distance_band
                    ),

                    "year": None,
                    "target_cmd": None,

                    "variable": None,
                    "quantile": None,

                    "n_rows": int(
                        row.n_rows
                    ),

                    "share_of_all": _share(
                        int(row.n_rows),
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        int(row.n_rows),
                        n_matrix_rows,
                    ),

                    "value": None,
                }
            )


        # ==================================================================
        # 5. Side-specific plausibility by mirror-disagreement band
        # ==================================================================
        uv_strict = float(uv_distance_thresholds[0])
        uv_moderate = float(uv_distance_thresholds[1])
        uv_loose = float(uv_distance_thresholds[2])

        side_plausibility = con.execute(
            f"""
            WITH evidence AS (
                SELECT
                    {disagreement_band_sql}
                        AS disagreement_band,

                    export_unit_value_log_distance
                        AS export_distance,

                    import_unit_value_log_distance
                        AS import_distance,

                    GREATEST(
                        export_unit_value_log_distance,
                        import_unit_value_log_distance
                    )
                        AS bilateral_max_distance

                FROM validation_diagnostics_base

                WHERE both_quantities_observed

                  AND quantity_relative_difference
                      IS NOT NULL

                  AND export_unit_value_log_distance
                      IS NOT NULL

                  AND import_unit_value_log_distance
                      IS NOT NULL
            )

            SELECT
                disagreement_band,

                COUNT(*) AS n_rows,

                MEDIAN(
                    export_distance
                ) AS median_export_distance,

                MEDIAN(
                    import_distance
                ) AS median_import_distance,

                MEDIAN(
                    bilateral_max_distance
                ) AS median_bilateral_max_distance,

                AVG(
                    CASE
                        WHEN bilateral_max_distance <= {uv_strict:.17g}
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS both_within_025_share,

                AVG(
                    CASE
                        WHEN bilateral_max_distance <= {uv_moderate:.17g}
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS both_within_050_share,

                AVG(
                    CASE
                        WHEN bilateral_max_distance <= {uv_loose:.17g}
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS both_within_100_share

            FROM evidence

            GROUP BY disagreement_band
            """
        ).fetchdf()

        for row in side_plausibility.itertuples(
            index=False
        ):

            metrics = {
                "median_export_uv_distance": (
                    row.median_export_distance
                ),

                "median_import_uv_distance": (
                    row.median_import_distance
                ),

                "median_bilateral_max_uv_distance": (
                    row.median_bilateral_max_distance
                ),

                "both_within_025_share": (
                    row.both_within_025_share
                ),

                "both_within_050_share": (
                    row.both_within_050_share
                ),

                "both_within_100_share": (
                    row.both_within_100_share
                ),
            }

            for metric, value in metrics.items():

                diagnostics_rows.append(
                    {
                        "scope": "side_plausibility",

                        "criterion": (
                            row.disagreement_band
                        ),

                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,

                        "disagreement_band": (
                            row.disagreement_band
                        ),

                        "uv_distance_band": None,

                        "year": None,
                        "target_cmd": None,

                        "variable": metric,
                        "quantile": None,

                        "n_rows": int(
                            row.n_rows
                        ),

                        "share_of_all": None,
                        "share_of_parent": None,

                        "value": (
                            float(value)
                            if value is not None
                            else None
                        ),
                    }
                )


        # ==================================================================
        # 6a. Exporter quantity relative to importer reference
        # ==================================================================
        #
        # Import quantity is used as the reference denominator.
        #
        # Signed difference:
        #
        #     (Q_export - Q_import) / Q_import
        #
        # Positive  -> exporter quantity is larger.
        # Negative  -> exporter quantity is smaller.
        # Zero      -> exact agreement.
        #
        # The absolute version measures only the magnitude of disagreement.
        # ==================================================================

        importer_reference = con.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS n_comparable_rows,

                AVG(
                    quantity_difference_vs_import
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS mean_signed_difference,

                MEDIAN(
                    quantity_difference_vs_import
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS median_signed_difference,

                MEDIAN(
                    quantity_absolute_difference_vs_import
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_absolute_difference_vs_import IS NOT NULL
                )
                    AS median_absolute_difference,

                AVG(
                    CASE
                        WHEN quantity_difference_vs_import > 0
                            THEN 1.0
                        ELSE 0.0
                    END
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS exporter_higher_share,

                AVG(
                    CASE
                        WHEN quantity_difference_vs_import < 0
                            THEN 1.0
                        ELSE 0.0
                    END
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS importer_higher_share,

                AVG(
                    CASE
                        WHEN quantity_difference_vs_import = 0
                            THEN 1.0
                        ELSE 0.0
                    END
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS exact_agreement_share

            FROM validation_diagnostics_base
            """
        ).fetchone()

        (
            n_import_reference_rows,
            mean_signed_difference,
            median_signed_difference,
            median_absolute_difference,
            exporter_higher_share,
            importer_higher_share,
            exact_agreement_share,
        ) = importer_reference

        importer_reference_metrics = {
            "mean_quantity_difference_vs_import": (
                mean_signed_difference
            ),
            "median_quantity_difference_vs_import": (
                median_signed_difference
            ),
            "median_absolute_quantity_difference_vs_import": (
                median_absolute_difference
            ),
            "exporter_quantity_higher_share": (
                exporter_higher_share
            ),
            "importer_quantity_higher_share": (
                importer_higher_share
            ),
            "exact_quantity_agreement_share": (
                exact_agreement_share
            ),
        }

        for metric, value in importer_reference_metrics.items():

            diagnostics_rows.append(
                {
                    "scope": "importer_reference",

                    "criterion": (
                        "exporter_quantity_relative_to_importer"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": None,
                    "uv_distance_band": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": metric,
                    "quantile": None,

                    "n_rows": int(
                        n_import_reference_rows
                    ),

                    "share_of_all": _share(
                        int(n_import_reference_rows),
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        int(n_import_reference_rows),
                        n_both_observed,
                    ),

                    "value": (
                        float(value)
                        if value is not None
                        else None
                    ),
                }
            )

        # ==================================================================
        # 6b. Exporter / importer quantity-ratio distribution
        # ==================================================================

        quantity_ratio_distribution = con.execute(
            """
            WITH ratios AS (
                SELECT
                    LN(
                        export_quantity_kg
                        / import_quantity_kg
                    ) AS log_ratio

                FROM validation_diagnostics_base

                WHERE both_quantities_observed
                  AND export_quantity_kg > 0
                  AND import_quantity_kg > 0
                  AND isfinite(
                      LN(
                          export_quantity_kg
                          / import_quantity_kg
                      )
                  )
            ),

            bounds AS (
                SELECT
                    QUANTILE_CONT(
                        log_ratio,
                        0.005
                    ) AS lower_bound,

                    QUANTILE_CONT(
                        log_ratio,
                        0.995
                    ) AS upper_bound

                FROM ratios
            ),

            displayed AS (
                SELECT
                    log_ratio

                FROM ratios
                CROSS JOIN bounds

                WHERE log_ratio >= lower_bound
                  AND log_ratio <= upper_bound
            ),

            binned AS (
                SELECT
                    FLOOR(log_ratio / 0.10) * 0.10
                        AS log_ratio_bin,

                    COUNT(*) AS n_rows

                FROM displayed

                GROUP BY log_ratio_bin
            )

            SELECT
                log_ratio_bin,
                n_rows

            FROM binned

            ORDER BY log_ratio_bin
            """
        ).fetchdf()

        n_quantity_ratio_rows = int(
            quantity_ratio_distribution[
                "n_rows"
            ].sum()
        )

        for row in quantity_ratio_distribution.itertuples(
            index=False
        ):

            diagnostics_rows.append(
                {
                    "scope": "quantity_ratio_distribution",

                    "criterion": (
                        "direct_bilateral_quantity_observations"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": None,
                    "uv_distance_band": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": (
                        "export_import_log_quantity_ratio"
                    ),

                    "quantile": None,

                    "n_rows": int(
                        row.n_rows
                    ),

                    "share_of_all": _share(
                        int(row.n_rows),
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        int(row.n_rows),
                        n_quantity_ratio_rows,
                    ),

                    "value": float(
                        row.log_ratio_bin
                    ),
                }
            )


        # ==================================================================
        # Quantity–UV relationship
        # ==================================================================
        #
        # Explore whether unit value systematically varies with trade
        # quantity. This is diagnostic only and does not currently affect
        # candidate-training selection.
        #
        # A symmetric bilateral quantity proxy is used so that neither
        # exporter nor importer quantity is treated as the true quantity.
        # ==================================================================

        quantity_uv_relationship = con.execute(
            """
            SELECT
                year,
                target_cmd,

                COUNT(*) AS n_rows,

                REGR_SLOPE(
                    log_bilateral_unit_value,
                    log_bilateral_quantity
                ) AS log_uv_quantity_slope,

                REGR_R2(
                    log_bilateral_unit_value,
                    log_bilateral_quantity
                ) AS log_uv_quantity_r2,

                CORR(
                    log_bilateral_unit_value,
                    log_bilateral_quantity
                ) AS log_uv_quantity_correlation

            FROM validation_diagnostics_base

            WHERE both_quantities_observed
              AND log_bilateral_quantity IS NOT NULL
              AND log_bilateral_unit_value IS NOT NULL

            GROUP BY
                year,
                target_cmd

            ORDER BY
                year,
                target_cmd
            """
        ).fetchdf()

        for row in quantity_uv_relationship.itertuples(
            index=False
        ):

            metrics = {
                "log_uv_quantity_slope": (
                    row.log_uv_quantity_slope
                ),
                "log_uv_quantity_r2": (
                    row.log_uv_quantity_r2
                ),
                "log_uv_quantity_correlation": (
                    row.log_uv_quantity_correlation
                ),
            }

            for variable, value in metrics.items():

                diagnostics_rows.append(
                    {
                        "scope": "quantity_uv_relationship",

                        "criterion": (
                            "direct_bilateral_quantity_observations"
                        ),

                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,

                        "disagreement_band": None,
                        "uv_distance_band": None,

                        "year": int(
                            row.year
                        ),

                        "target_cmd": str(
                            row.target_cmd
                        ),

                        "variable": variable,
                        "quantile": None,

                        "n_rows": int(
                            row.n_rows
                        ),

                        "share_of_all": None,
                        "share_of_parent": None,

                        "value": (
                            float(value)
                            if value is not None
                            and pd.notna(value)
                            else None
                        ),
                    }
                )

        quantity_uv_summary = con.execute(
            """
            WITH relationship AS (
                SELECT
                    year,
                    target_cmd,

                    COUNT(*) AS n_rows,

                    REGR_SLOPE(
                        log_bilateral_unit_value,
                        log_bilateral_quantity
                    ) AS slope,

                    REGR_R2(
                        log_bilateral_unit_value,
                        log_bilateral_quantity
                    ) AS r2

                FROM validation_diagnostics_base

                WHERE both_quantities_observed
                  AND log_bilateral_quantity IS NOT NULL
                  AND log_bilateral_unit_value IS NOT NULL

                GROUP BY
                    year,
                    target_cmd
            )

            SELECT
                COUNT(*) AS n_product_year_groups,

                MEDIAN(slope)
                    AS median_slope,

                QUANTILE_CONT(slope, 0.25)
                    AS p25_slope,

                QUANTILE_CONT(slope, 0.75)
                    AS p75_slope,

                AVG(
                    CASE
                        WHEN slope < 0 THEN 1.0
                        ELSE 0.0
                    END
                )
                    AS negative_slope_share,

                MEDIAN(r2)
                    AS median_r2

            FROM relationship

            WHERE n_rows >= 30
              AND slope IS NOT NULL
            """
        ).fetchone()

        (
            n_quantity_uv_groups,
            median_quantity_uv_slope,
            p25_quantity_uv_slope,
            p75_quantity_uv_slope,
            negative_quantity_uv_share,
            median_quantity_uv_r2,
        ) = quantity_uv_summary

        summary_metrics = {
            "n_product_year_groups": (
                n_quantity_uv_groups
            ),
            "median_log_uv_quantity_slope": (
                median_quantity_uv_slope
            ),
            "p25_log_uv_quantity_slope": (
                p25_quantity_uv_slope
            ),
            "p75_log_uv_quantity_slope": (
                p75_quantity_uv_slope
            ),
            "negative_slope_share": (
                negative_quantity_uv_share
            ),
            "median_log_uv_quantity_r2": (
                median_quantity_uv_r2
            ),
        }

        for variable, value in summary_metrics.items():

            diagnostics_rows.append(
                {
                    "scope": "quantity_uv_summary",
                    "criterion": (
                        "product_year_groups_with_at_least_30_rows"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": None,
                    "uv_distance_band": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": variable,
                    "quantile": None,

                    "n_rows": int(
                        n_quantity_uv_groups
                    ),

                    "share_of_all": None,
                    "share_of_parent": None,

                    "value": (
                        float(value)
                        if value is not None
                        else None
                    ),
                }
            )

        supported_quantity_uv = quantity_uv_relationship[
            quantity_uv_relationship["n_rows"] >= 30
        ].copy()

        for row in supported_quantity_uv.itertuples(
            index=False):

            diagnostics_rows.append(
                {
                    "scope": "quantity_uv_relationship_groups",
                    "criterion": "hs_year_group_with_min_30_rows",

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": None,
                    "uv_distance_band": None,

                    "year": int(
                        row.year
                    ),

                    "target_cmd": str(
                        row.target_cmd
                    ),

                    "variable": "quantity_uv_slope",
                    "quantile": None,

                    "n_rows": int(
                        row.n_rows
                    ),

                    "share_of_all": None,
                    "share_of_parent": None,

                    "value": (
                        float(
                            row.log_uv_quantity_slope
                        )
                        if pd.notna(
                            row.log_uv_quantity_slope
                        )
                        else None
                    ),
                }
            )

            diagnostics_rows.append(
                {
                    "scope": "quantity_uv_relationship_groups",
                    "criterion": "hs_year_group_with_min_30_rows",

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": None,
                    "uv_distance_band": None,

                    "year": int(
                        row.year
                    ),

                    "target_cmd": str(
                        row.target_cmd
                    ),

                    "variable": "quantity_uv_r_squared",
                    "quantile": None,

                    "n_rows": int(
                        row.n_rows
                    ),

                    "share_of_all": None,
                    "share_of_parent": None,

                    "value": (
                        float(
                            row.log_uv_quantity_r2
                        )
                        if pd.notna(
                            row.log_uv_quantity_r2
                        )
                        else None
                    ),
                }
            )
        # ==================================================================
        # 7. Year coverage
        # ==================================================================

        year_frame = con.execute(
            """
            SELECT
                year,

                COUNT(*)
                    AS n_rows,

                COUNT(*) FILTER (
                    WHERE has_both_quantities
                )
                    AS n_both_quantities,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                )
                    AS n_both_observed,

                MEDIAN(
                    quantity_relative_difference
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_relative_difference
                          IS NOT NULL
                )
                    AS median_quantity_relative_difference,

                MEDIAN(
                    synced_unit_value_log_distance
                ) FILTER (
                    WHERE both_quantities_observed
                      AND synced_unit_value_log_distance
                          IS NOT NULL
                )
                    AS median_synced_uv_log_distance,

                MEDIAN(
                    unit_value_reference_observations
                ) FILTER (
                    WHERE both_quantities_observed
                      AND unit_value_reference_observations
                          IS NOT NULL
                )
                    AS median_uv_reference_observations,

                MEDIAN(
                    quantity_difference_vs_import
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS median_quantity_difference_vs_import,

                MEDIAN(
                    quantity_absolute_difference_vs_import
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_absolute_difference_vs_import IS NOT NULL
                )
                    AS median_absolute_quantity_difference_vs_import,

                AVG(
                    CASE
                        WHEN quantity_difference_vs_import > 0
                            THEN 1.0
                        ELSE 0.0
                    END
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS exporter_higher_share

            FROM validation_diagnostics_base

            GROUP BY year
            ORDER BY year
            """
        ).fetchdf()

        for row in year_frame.itertuples(
            index=False
        ):
            diagnostics_rows.append(
                {
                    "scope": "year_coverage",

                    "criterion": (
                        "both_quantities_non_estimated"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": int(
                        row.year
                    ),

                    "target_cmd": None,

                    "variable": None,
                    "quantile": None,

                    "n_rows": int(
                        row.n_both_observed
                    ),

                    "share_of_all": _share(
                        int(row.n_both_observed),
                        int(row.n_rows),
                    ),

                    "share_of_parent": None,

                    "value": None,

                    "total_rows": int(
                        row.n_rows
                    ),

                    "n_both_quantities": int(
                        row.n_both_quantities
                    ),

                    "median_quantity_relative_difference": (
                        float(
                            row.median_quantity_relative_difference
                        )
                        if pd.notna(
                            row.median_quantity_relative_difference
                        )
                        else None
                    ),

                    "median_synced_uv_log_distance": (
                        float(
                            row.median_synced_uv_log_distance
                        )
                        if pd.notna(
                            row.median_synced_uv_log_distance
                        )
                        else None
                    ),

                    "median_quantity_difference_vs_import": (
                        float(
                            row.median_quantity_difference_vs_import
                        )
                        if pd.notna(
                            row.median_quantity_difference_vs_import
                        )
                        else None
                    ),

                    "median_absolute_quantity_difference_vs_import": (
                        float(
                            row.median_absolute_quantity_difference_vs_import
                        )
                        if pd.notna(
                            row.median_absolute_quantity_difference_vs_import
                        )
                        else None
                    ),

                    "exporter_higher_share": (
                        float(
                            row.exporter_higher_share
                        )
                        if pd.notna(
                            row.exporter_higher_share
                        )
                        else None
                    ),
                    "median_uv_reference_observations": (
                        float(
                            row.median_uv_reference_observations
                        )
                        if pd.notna(
                            row.median_uv_reference_observations
                        )
                        else None
                    ),
                }
            )

        # ==================================================================
        # 8. Product coverage
        # ==================================================================

        product_frame = con.execute(
            """
            SELECT
                target_cmd,

                COUNT(*)
                    AS n_rows,

                COUNT(*) FILTER (
                    WHERE has_both_quantities
                )
                    AS n_both_quantities,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                )
                    AS n_both_observed,

                MEDIAN(
                    quantity_relative_difference
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_relative_difference
                          IS NOT NULL
                )
                    AS median_quantity_relative_difference,

                MEDIAN(
                    synced_unit_value_log_distance
                ) FILTER (
                    WHERE both_quantities_observed
                      AND synced_unit_value_log_distance
                          IS NOT NULL
                )
                    AS median_synced_uv_log_distance,

                MEDIAN(
                    unit_value_reference_observations
                ) FILTER (
                    WHERE both_quantities_observed
                      AND unit_value_reference_observations
                          IS NOT NULL
                )
                    AS median_uv_reference_observations

            FROM validation_diagnostics_base

            GROUP BY target_cmd
            ORDER BY target_cmd
            """
        ).fetchdf()

        for row in product_frame.itertuples(
            index=False
        ):
            diagnostics_rows.append(
                {
                    "scope": "product_coverage",

                    "criterion": (
                        "both_quantities_non_estimated"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": None,

                    "target_cmd": str(
                        row.target_cmd
                    ),

                    "variable": None,
                    "quantile": None,

                    "n_rows": int(
                        row.n_both_observed
                    ),

                    "share_of_all": _share(
                        int(row.n_both_observed),
                        int(row.n_rows),
                    ),

                    "share_of_parent": None,

                    "value": None,

                    "total_rows": int(
                        row.n_rows
                    ),

                    "n_both_quantities": int(
                        row.n_both_quantities
                    ),

                    "median_quantity_relative_difference": (
                        float(
                            row.median_quantity_relative_difference
                        )
                        if pd.notna(
                            row.median_quantity_relative_difference
                        )
                        else None
                    ),

                    "median_synced_uv_log_distance": (
                        float(
                            row.median_synced_uv_log_distance
                        )
                        if pd.notna(
                            row.median_synced_uv_log_distance
                        )
                        else None
                    ),

                    "median_uv_reference_observations": (
                        float(
                            row.median_uv_reference_observations
                        )
                        if pd.notna(
                            row.median_uv_reference_observations
                        )
                        else None
                    ),
                }
            )

        # ==================================================================
        # 9. Distribution evidence
        # ==================================================================

        quantiles = (
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

        distribution_variables = (
            "quantity_relative_difference",

            "quantity_difference_vs_import",
            "quantity_absolute_difference_vs_import",

            "synced_unit_value_log_distance",
            "unit_value_reference_observations",
        )

        quantile_sql = ", ".join(
            f"""
            quantile_cont(
                {{variable}},
                {float(quantile):.17g}
            ) AS q_{index}
            """
            for index, quantile in enumerate(
                quantiles
            )
        )

        for variable in distribution_variables:

            variable_quantile_sql = (
                quantile_sql.replace(
                    "{variable}",
                    variable,
                )
            )

            values = con.execute(
                f"""
                SELECT
                    COUNT(*) AS n_rows,
                    {variable_quantile_sql}

                FROM validation_diagnostics_base

                WHERE both_quantities_observed
                  AND {variable} IS NOT NULL
                """
            ).fetchone()

            n_distribution_rows = int(
                values[0]
            )

            for (
                quantile,
                value,
            ) in zip(
                quantiles,
                values[1:],
            ):
                diagnostics_rows.append(
                    {
                        "scope": "distribution",

                        "criterion": (
                            "both_quantities_non_estimated"
                        ),

                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,

                        "year": None,
                        "target_cmd": None,

                        "variable": variable,

                        "quantile": float(
                            quantile
                        ),

                        "n_rows": (
                            n_distribution_rows
                        ),

                        "share_of_all": None,
                        "share_of_parent": None,

                        "value": (
                            float(value)
                            if value is not None
                            else None
                        ),
                    }
                )

        # ==================================================================
        # Diagnostics artifact
        # ==================================================================

        diagnostics = pd.DataFrame(
            diagnostics_rows
        )

        if diagnostics.empty:
            raise RuntimeError(
                "Validation diagnostics contain no rows."
            )

        # Ensure optional columns exist consistently across scopes.
        optional_columns = (
            "disagreement_band",
            "uv_distance_band",

            "total_rows",
            "n_both_quantities",

            "median_quantity_relative_difference",
            "median_synced_uv_log_distance",
            "median_uv_reference_observations",
            "median_quantity_difference_vs_import",
            "median_absolute_quantity_difference_vs_import",
            "exporter_higher_share",
        )

        for column in optional_columns:

            if column not in diagnostics.columns:
                diagnostics[column] = None

        # Stable column order.
        diagnostics = diagnostics[
            [
                "scope",
                "criterion",

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
                "n_both_quantities",

                "share_of_all",
                "share_of_parent",

                "value",

                "median_quantity_relative_difference",
                "median_synced_uv_log_distance",
                "median_uv_reference_observations",
                "median_quantity_difference_vs_import",
                "median_absolute_quantity_difference_vs_import",
                "exporter_higher_share",
            ]
        ]

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        if temporary_path.exists():
            temporary_path.unlink()

        temporary_sql = sql_literal(
            temporary_path
        )

        con.register(
            "_validation_diagnostics",
            diagnostics,
        )

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM _validation_diagnostics
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

    # ==================================================================
    # Logging
    # ==================================================================

    elapsed = perf_counter() - start

    logger.info(
        "  All Validation rows: %s",
        f"{n_all_rows:,}",
    )

    logger.info(
        "  Trade-Synced quantity available: %s",
        f"{n_synced_quantity:,}",
    )

    logger.info(
        "  Both quantities available: %s",
        f"{n_both_quantities:,}",
    )

    logger.info(
        "  Both quantities non-estimated: %s",
        f"{n_both_observed:,}",
    )

    logger.info(
        "  Complete candidate-training evidence: %s",
        f"{n_complete_training_evidence:,}",
    )

    logger.info(
        "  Threshold combinations evaluated: %s",
        f"{len(grid_metadata):,}",
    )

    logger.info(
        "  Diagnostics rows: %s",
        f"{len(diagnostics):,}",
    )

    logger.info(
        "  Output: %s",
        output_path,
    )

    logger.info(
        "%s CHECK — Quantity training population characterized.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,

        "output_path": output_path,
        "diagnostics_path": output_path,

        "n_rows": n_all_rows,

        "n_synced_quantity": (
            n_synced_quantity
        ),

        "n_both_quantities": (
            n_both_quantities
        ),

        "n_both_observed": (
            n_both_observed
        ),

        "n_both_observed_with_disagreement": (
            n_both_observed_with_disagreement
        ),

        "n_both_observed_with_uv_distance": (
            n_both_observed_with_uv_distance
        ),

        "n_complete_training_evidence": (
            n_complete_training_evidence
        ),

        "n_threshold_combinations": len(
            grid_metadata
        ),

        "fitted": False,

        "ready_for_training_decision": True,

        "duration_seconds": elapsed,
        "reused": False,
    }