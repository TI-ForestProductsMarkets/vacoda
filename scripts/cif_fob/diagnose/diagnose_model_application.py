from pathlib import Path

import duckdb


PATH = Path(
    "/storage/vacoda/data/interim/cif_fob/target_h5/cif_fob_runs/cif_fob_upper_limit_1p457/"
    "comtrade_target_h5_cif_fob_corrected.parquet"
)

con = duckdb.connect()

try:
    # ------------------------------------------------------------
    # 1. Overall model-only versus old direct-priority counterfactual
    # ------------------------------------------------------------
    overall = con.execute(
        """
        WITH base AS (
            SELECT
                *,

                -- Current production value: model only
                CASE
                    WHEN prediction_eligible
                     AND import_value_cif IS NOT NULL
                     AND import_value_cif >= 0
                     AND estimated_cif_fob_factor IS NOT NULL
                     AND estimated_cif_fob_factor > 0
                    THEN import_value_cif / estimated_cif_fob_factor
                END AS model_only_fob,

                -- Reconstruct previous application rule:
                -- direct factor first, otherwise model
                CASE
                    WHEN factor_source = 'direct'
                     AND import_direct_cif_fob_factor IS NOT NULL
                     AND import_direct_cif_fob_factor > 0
                     AND import_value_cif IS NOT NULL
                     AND import_value_cif >= 0
                    THEN import_value_cif / import_direct_cif_fob_factor

                    WHEN prediction_eligible
                     AND estimated_cif_fob_factor IS NOT NULL
                     AND estimated_cif_fob_factor > 0
                     AND import_value_cif IS NOT NULL
                     AND import_value_cif >= 0
                    THEN import_value_cif / estimated_cif_fob_factor
                END AS old_direct_priority_fob

            FROM read_parquet(?)
        ),

        compared AS (
            SELECT
                *,
                model_only_fob - old_direct_priority_fob
                    AS absolute_change,

                ABS(model_only_fob - old_direct_priority_fob)
                    AS absolute_change_abs,

                ABS(model_only_fob - old_direct_priority_fob)
                / NULLIF(ABS(old_direct_priority_fob), 0)
                    AS relative_change
            FROM base
        )

        SELECT
            COUNT(*) AS total_rows,

            COUNT(*) FILTER (
                WHERE model_only_fob IS NOT NULL
            ) AS model_only_rows,

            COUNT(*) FILTER (
                WHERE old_direct_priority_fob IS NOT NULL
            ) AS old_rule_rows,

            COUNT(*) FILTER (
                WHERE factor_source = 'direct'
                  AND import_direct_cif_fob_factor IS NOT NULL
                  AND import_direct_cif_fob_factor > 0
            ) AS rows_with_direct_factor,

            COUNT(*) FILTER (
                WHERE model_only_fob IS NOT NULL
                  AND old_direct_priority_fob IS NOT NULL
                  AND ABS(model_only_fob - old_direct_priority_fob) > 1e-8
            ) AS rows_changed,

            SUM(model_only_fob) AS model_only_total,

            SUM(old_direct_priority_fob) AS old_rule_total,

            SUM(model_only_fob)
                - SUM(old_direct_priority_fob)
                AS total_value_change,

            MEDIAN(relative_change) FILTER (
                WHERE relative_change IS NOT NULL
            ) AS median_relative_change,

            QUANTILE_CONT(relative_change, 0.90) FILTER (
                WHERE relative_change IS NOT NULL
            ) AS p90_relative_change,

            QUANTILE_CONT(relative_change, 0.95) FILTER (
                WHERE relative_change IS NOT NULL
            ) AS p95_relative_change,

            MAX(relative_change) AS maximum_relative_change

        FROM compared
        """,
        [str(PATH)],
    ).df()

    print("\n=== OVERALL IMPACT ===")
    print(overall.to_string(index=False))


    # ------------------------------------------------------------
    # 2. Only rows where the old rule would have used DIRECT
    # ------------------------------------------------------------
    direct_rows = con.execute(
        """
        WITH compared AS (
            SELECT
                year,
                import_value_cif,
                import_direct_cif_fob_factor AS direct_factor,
                estimated_cif_fob_factor AS model_factor,

                import_value_cif
                    / import_direct_cif_fob_factor
                    AS direct_priority_fob,

                import_value_cif
                    / estimated_cif_fob_factor
                    AS model_only_fob

            FROM read_parquet(?)

            WHERE factor_source = 'direct'
              AND import_direct_cif_fob_factor IS NOT NULL
              AND import_direct_cif_fob_factor > 0
              AND estimated_cif_fob_factor IS NOT NULL
              AND estimated_cif_fob_factor > 0
              AND prediction_eligible
              AND import_value_cif IS NOT NULL
              AND import_value_cif > 0
        )

        SELECT
            COUNT(*) AS rows,

            MEDIAN(direct_factor) AS median_direct_factor,
            MEDIAN(model_factor) AS median_model_factor,

            MEDIAN(
                ABS(model_factor - direct_factor)
                / direct_factor
            ) AS median_factor_difference,

            QUANTILE_CONT(
                ABS(model_factor - direct_factor)
                / direct_factor,
                0.90
            ) AS p90_factor_difference,

            QUANTILE_CONT(
                ABS(model_factor - direct_factor)
                / direct_factor,
                0.95
            ) AS p95_factor_difference,

            SUM(direct_priority_fob) AS direct_priority_total,
            SUM(model_only_fob) AS model_only_total,

            SUM(model_only_fob)
                - SUM(direct_priority_fob)
                AS total_change

        FROM compared
        """,
        [str(PATH)],
    ).df()

    print("\n=== DIRECT-FACTOR ROWS ONLY ===")
    print(direct_rows.to_string(index=False))


    # ------------------------------------------------------------
    # 3. Annual effect
    # ------------------------------------------------------------
    annual = con.execute(
        """
        WITH compared AS (
            SELECT
                year,

                CASE
                    WHEN prediction_eligible
                     AND estimated_cif_fob_factor IS NOT NULL
                     AND estimated_cif_fob_factor > 0
                     AND import_value_cif IS NOT NULL
                     AND import_value_cif >= 0
                    THEN import_value_cif / estimated_cif_fob_factor
                END AS model_only_fob,

                CASE
                    WHEN factor_source = 'direct'
                     AND import_direct_cif_fob_factor IS NOT NULL
                     AND import_direct_cif_fob_factor > 0
                     AND import_value_cif IS NOT NULL
                     AND import_value_cif >= 0
                    THEN import_value_cif / import_direct_cif_fob_factor

                    WHEN prediction_eligible
                     AND estimated_cif_fob_factor IS NOT NULL
                     AND estimated_cif_fob_factor > 0
                     AND import_value_cif IS NOT NULL
                     AND import_value_cif >= 0
                    THEN import_value_cif / estimated_cif_fob_factor
                END AS old_fob

            FROM read_parquet(?)
        )

        SELECT
            year,
            COUNT(*) FILTER (
                WHERE model_only_fob IS NOT NULL
            ) AS model_rows,

            COUNT(*) FILTER (
                WHERE old_fob IS NOT NULL
            ) AS old_rows,

            SUM(old_fob) AS old_total,
            SUM(model_only_fob) AS model_total,

            SUM(model_only_fob)
                - SUM(old_fob)
                AS difference,

            100.0 * (
                SUM(model_only_fob)
                - SUM(old_fob)
            )
            / NULLIF(SUM(old_fob), 0)
                AS difference_percent

        FROM compared

        GROUP BY year
        ORDER BY year
        """,
        [str(PATH)],
    ).df()

    print("\n=== ANNUAL IMPACT ===")
    print(annual.to_string(index=False))


    decomposition = con.execute(
    """
        WITH base AS (
            SELECT
                *,

                CASE
                    WHEN prediction_eligible
                    AND import_value_cif IS NOT NULL
                    AND import_value_cif >= 0
                    AND estimated_cif_fob_factor IS NOT NULL
                    AND estimated_cif_fob_factor > 0
                    THEN import_value_cif / estimated_cif_fob_factor
                END AS model_only_fob,

                CASE
                    WHEN factor_source = 'direct'
                    AND import_direct_cif_fob_factor IS NOT NULL
                    AND import_direct_cif_fob_factor > 0
                    AND import_value_cif IS NOT NULL
                    AND import_value_cif >= 0
                    THEN import_value_cif / import_direct_cif_fob_factor

                    WHEN prediction_eligible
                    AND estimated_cif_fob_factor IS NOT NULL
                    AND estimated_cif_fob_factor > 0
                    AND import_value_cif IS NOT NULL
                    AND import_value_cif >= 0
                    THEN import_value_cif / estimated_cif_fob_factor
                END AS old_fob

            FROM read_parquet(?)
        )

        SELECT
            CASE
                WHEN old_fob IS NOT NULL
                AND model_only_fob IS NOT NULL
                    THEN 'both_rules'

                WHEN old_fob IS NOT NULL
                AND model_only_fob IS NULL
                    THEN 'old_only'

                WHEN old_fob IS NULL
                AND model_only_fob IS NOT NULL
                    THEN 'model_only'

                ELSE 'neither'
            END AS category,

            COUNT(*) AS rows,
            SUM(old_fob) AS old_total,
            SUM(model_only_fob) AS model_total,
            SUM(model_only_fob) - SUM(old_fob) AS difference

        FROM base
        GROUP BY 1
        ORDER BY 1
        """,
        [str(PATH)],
    ).df()

    print(decomposition.to_string(index=False))

    old_only_reasons = con.execute(
    """
        WITH base AS (
            SELECT
                *,

                CASE
                    WHEN prediction_eligible
                    AND import_value_cif IS NOT NULL
                    AND import_value_cif >= 0
                    AND estimated_cif_fob_factor IS NOT NULL
                    AND estimated_cif_fob_factor > 0
                    THEN import_value_cif / estimated_cif_fob_factor
                END AS model_only_fob,

                CASE
                    WHEN factor_source = 'direct'
                    AND import_direct_cif_fob_factor IS NOT NULL
                    AND import_direct_cif_fob_factor > 0
                    AND import_value_cif IS NOT NULL
                    AND import_value_cif >= 0
                    THEN import_value_cif / import_direct_cif_fob_factor

                    WHEN prediction_eligible
                    AND estimated_cif_fob_factor IS NOT NULL
                    AND estimated_cif_fob_factor > 0
                    AND import_value_cif IS NOT NULL
                    AND import_value_cif >= 0
                    THEN import_value_cif / estimated_cif_fob_factor
                END AS old_fob

            FROM read_parquet(?)
        )

        SELECT
            prediction_eligible,
            initial_predictors_complete,
            factor_source,
            COUNT(*) AS rows,
            SUM(old_fob) AS old_value

        FROM base

        WHERE old_fob IS NOT NULL
        AND model_only_fob IS NULL

        GROUP BY
            prediction_eligible,
            initial_predictors_complete,
            factor_source

        ORDER BY old_value DESC
        """,
        [str(PATH)],
    ).df()

    print(old_only_reasons.to_string(index=False))

finally:
    con.close()