from __future__ import annotations

from pathlib import Path

import duckdb


INPUT_PATH = Path(
    "/storage/vacoda/data/interim/cif_fob/target_h5/"
    "target_h5_cif_fob_estimated_rates.parquet"
)

HOLDOUT_YEAR = 2025


def print_diagnostics_table(
    dataframe,
    *,
    percentage_columns: tuple[str, ...] = (),
    factor_columns: tuple[str, ...] = (),
) -> None:
    """Print a diagnostic DataFrame with human-readable formatting."""
    display = dataframe.copy()

    for column in percentage_columns:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: (
                    f"{value * 100:.2f}%"
                    if value is not None
                    else ""
                )
            )

    for column in factor_columns:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: (
                    f"{value:.4f}"
                    if value is not None
                    else ""
                )
            )

    print(display.to_string(index=False))


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Estimated-rates parquet not found: {INPUT_PATH}"
        )

    con = duckdb.connect()

    try:
        # --------------------------------------------------------------
        # Build holdout diagnostics only from clean estimation evidence.
        #
        # The model was validated using:
        #   training years < 2025
        #   holdout year = 2025
        # --------------------------------------------------------------

        con.read_parquet(str(INPUT_PATH)).create_view(
            "rates",
            replace=True,
        )
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW training_medians AS
            SELECT
                MEDIAN(observed_cif_fob_factor)
                    AS overall_training_median

            FROM rates

            WHERE clean_estimation_eligible
              AND year < {HOLDOUT_YEAR}
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW product_group_medians AS
            SELECT
                product_group,

                MEDIAN(observed_cif_fob_factor)
                    AS product_group_training_median,

                COUNT(*) AS product_group_training_observations

            FROM rates

            WHERE clean_estimation_eligible
              AND year < {HOLDOUT_YEAR}

            GROUP BY product_group
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW holdout AS
            SELECT
                r.*,

                tm.overall_training_median,

                COALESCE(
                    pg.product_group_training_median,
                    tm.overall_training_median
                ) AS product_group_baseline_factor,

                pg.product_group_training_observations,

                ABS(
                    r.holdout_predicted_factor
                    - r.observed_cif_fob_factor
                ) AS model_absolute_error,

                ABS(
                    tm.overall_training_median
                    - r.observed_cif_fob_factor
                ) AS overall_baseline_absolute_error,

                ABS(
                    COALESCE(
                        pg.product_group_training_median,
                        tm.overall_training_median
                    )
                    - r.observed_cif_fob_factor
                ) AS product_baseline_absolute_error,

                ABS(
                    r.holdout_predicted_factor
                    - r.observed_cif_fob_factor
                )
                / r.observed_cif_fob_factor
                    AS model_absolute_percentage_error,

                ABS(
                    tm.overall_training_median
                    - r.observed_cif_fob_factor
                )
                / r.observed_cif_fob_factor
                    AS overall_baseline_absolute_percentage_error,

                ABS(
                    COALESCE(
                        pg.product_group_training_median,
                        tm.overall_training_median
                    )
                    - r.observed_cif_fob_factor
                )
                / r.observed_cif_fob_factor
                    AS product_baseline_absolute_percentage_error

            FROM rates AS r

            CROSS JOIN training_medians AS tm

            LEFT JOIN product_group_medians AS pg
              ON r.product_group = pg.product_group

            WHERE r.clean_estimation_eligible
              AND r.year = {HOLDOUT_YEAR}
              AND r.holdout_predicted_factor IS NOT NULL
            """
        )

        # --------------------------------------------------------------
        # 1. Holdout target and prediction distributions
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("HOLDOUT DISTRIBUTIONS")
        print("=" * 72)

        distribution = con.execute(
            """
            SELECT
                COUNT(*) AS n_holdout,

                MEDIAN(observed_cif_fob_factor)
                    AS observed_median,

                QUANTILE_CONT(
                    observed_cif_fob_factor,
                    0.05
                ) AS observed_p05,

                QUANTILE_CONT(
                    observed_cif_fob_factor,
                    0.95
                ) AS observed_p95,

                MEDIAN(holdout_predicted_factor)
                    AS predicted_median,

                QUANTILE_CONT(
                    holdout_predicted_factor,
                    0.05
                ) AS predicted_p05,

                QUANTILE_CONT(
                    holdout_predicted_factor,
                    0.95
                ) AS predicted_p95,

                STDDEV_SAMP(observed_cif_fob_factor)
                    AS observed_stddev,

                STDDEV_SAMP(holdout_predicted_factor)
                    AS predicted_stddev

            FROM holdout
            """
        ).df()

        print_diagnostics_table(
            distribution,
            factor_columns=(
                "observed_median",
                "observed_p05",
                "observed_p95",
                "predicted_median",
                "predicted_p05",
                "predicted_p95",
                "observed_stddev",
                "predicted_stddev",
            ),
        )

        # --------------------------------------------------------------
        # 2. Model versus simple baselines
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("MODEL VERSUS BASELINES")
        print("=" * 72)

        comparison = con.execute(
            """
            SELECT
                MEDIAN(
                    model_absolute_percentage_error
                ) AS model_median_ape,

                QUANTILE_CONT(
                    model_absolute_percentage_error,
                    0.90
                ) AS model_p90_ape,

                MEDIAN(
                    overall_baseline_absolute_percentage_error
                ) AS overall_baseline_median_ape,

                QUANTILE_CONT(
                    overall_baseline_absolute_percentage_error,
                    0.90
                ) AS overall_baseline_p90_ape,

                MEDIAN(
                    product_baseline_absolute_percentage_error
                ) AS product_baseline_median_ape,

                QUANTILE_CONT(
                    product_baseline_absolute_percentage_error,
                    0.90
                ) AS product_baseline_p90_ape,

                AVG(
                    CAST(
                    model_absolute_percentage_error
                    < overall_baseline_absolute_percentage_error
                    AS INTEGER
                    )
                ) AS model_win_rate_vs_overall,

                AVG(
                    CAST(
                    model_absolute_percentage_error
                    < product_baseline_absolute_percentage_error
                    AS INTEGER
                    )
                ) AS model_win_rate_vs_product

            FROM holdout
            """
        ).df()

        print_diagnostics_table(
            comparison,
            percentage_columns=(
                "model_median_ape",
                "model_p90_ape",
                "overall_baseline_median_ape",
                "overall_baseline_p90_ape",
                "product_baseline_median_ape",
                "product_baseline_p90_ape",
                "model_win_rate_vs_overall",
                "model_win_rate_vs_product",
            ),
        )

        # --------------------------------------------------------------
        # 3. Performance by evidence source
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("PERFORMANCE BY FACTOR SOURCE")
        print("=" * 72)

        by_source = con.execute(
            """
            SELECT
                factor_source,
                COUNT(*) AS n_rows,

                MEDIAN(observed_cif_fob_factor)
                    AS observed_median,

                MEDIAN(holdout_predicted_factor)
                    AS predicted_median,

                MEDIAN(
                    ABS(
                        holdout_predicted_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    )
                ) AS model_median_ape,

                QUANTILE_CONT(
                    ABS(
                        holdout_predicted_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    ),
                    0.90
                ) AS model_p90_ape,

                MEDIAN(
                    ABS(
                        product_group_baseline_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    )
                ) AS product_baseline_median_ape,

                AVG(
                    CAST(
                        ABS(
                            holdout_predicted_factor
                            - observed_cif_fob_factor
                        )
                        <
                        ABS(
                            product_group_baseline_factor
                            - observed_cif_fob_factor
                        )
                        AS DOUBLE
                    )
                ) AS model_win_rate_vs_product

            FROM holdout

            GROUP BY factor_source

            ORDER BY factor_source
            """
        ).df()

        print_diagnostics_table(
            by_source,
            percentage_columns=(
                "model_median_ape",
                "model_p90_ape",
                "product_baseline_median_ape",
                "model_win_rate_vs_product",
            ),
            factor_columns=(
                "observed_median",
                "predicted_median",
            ),
        )

        # --------------------------------------------------------------
        # Performance by availability of lagged pair history
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("PERFORMANCE BY LAGGED PAIR HISTORY")
        print("=" * 72)

        by_pair_history = con.execute(
            """
            SELECT
                CASE
                    WHEN lagged_pair_median_factor IS NOT NULL
                        THEN 'history_available'
                    ELSE 'history_missing'
                END AS pair_history_status,

                COUNT(*) AS n_rows,

                MEDIAN(observed_cif_fob_factor)
                    AS observed_median,

                MEDIAN(holdout_predicted_factor)
                    AS predicted_median,

                MEDIAN(
                    ABS(
                        holdout_predicted_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    )
                ) AS model_median_ape,

                QUANTILE_CONT(
                    ABS(
                        holdout_predicted_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    ),
                    0.90
                ) AS model_p90_ape,

                MEDIAN(
                    ABS(
                        product_group_baseline_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    )
                ) AS product_baseline_median_ape,

                AVG(
                    CAST(
                        ABS(
                            holdout_predicted_factor
                            - observed_cif_fob_factor
                        )
                        <
                        ABS(
                            product_group_baseline_factor
                            - observed_cif_fob_factor
                        )
                        AS DOUBLE
                    )
                ) AS model_win_rate_vs_product

            FROM holdout

            GROUP BY pair_history_status

            ORDER BY pair_history_status
            """
        ).df()

        print(by_pair_history.to_string(index=False))

        # --------------------------------------------------------------
        # Model versus raw lagged-pair baseline
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("MODEL VERSUS LAGGED PAIR BASELINE")
        print("=" * 72)

        pair_baseline = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,

                MEDIAN(
                    ABS(
                        holdout_predicted_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    )
                ) AS model_median_ape,

                QUANTILE_CONT(
                    ABS(
                        holdout_predicted_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    ),
                    0.90
                ) AS model_p90_ape,

                MEDIAN(
                    ABS(
                        lagged_pair_median_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    )
                ) AS pair_baseline_median_ape,

                QUANTILE_CONT(
                    ABS(
                        lagged_pair_median_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    ),
                    0.90
                ) AS pair_baseline_p90_ape,

                AVG(
                    CAST(
                        ABS(
                            holdout_predicted_factor
                            - observed_cif_fob_factor
                        )
                        <
                        ABS(
                            lagged_pair_median_factor
                            - observed_cif_fob_factor
                        )
                        AS DOUBLE
                    )
                ) AS model_win_rate_vs_pair_baseline

            FROM holdout

            WHERE lagged_pair_median_factor IS NOT NULL
            """
        ).df()

        print(pair_baseline.to_string(index=False))




        # --------------------------------------------------------------
        # 4. Calibration by predicted-factor decile
        #
        # If observed medians remain almost flat across deciles, the
        # model has little ranking power.
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("CALIBRATION BY PREDICTION DECILE")
        print("=" * 72)

        calibration = con.execute(
            """
            WITH ranked AS (
                SELECT
                    *,

                    NTILE(10) OVER (
                        ORDER BY holdout_predicted_factor
                    ) AS prediction_decile

                FROM holdout
            )

            SELECT
                prediction_decile,
                COUNT(*) AS n_rows,

                MIN(holdout_predicted_factor)
                    AS predicted_min,

                MEDIAN(holdout_predicted_factor)
                    AS predicted_median,

                MAX(holdout_predicted_factor)
                    AS predicted_max,

                MEDIAN(observed_cif_fob_factor)
                    AS observed_median,

                MEDIAN(
                    model_absolute_percentage_error
                ) AS median_ape

            FROM ranked

            GROUP BY prediction_decile

            ORDER BY prediction_decile
            """
        ).df()

        print_diagnostics_table(
            calibration,
            percentage_columns=("median_ape",),
            factor_columns=(
                "predicted_min",
                "predicted_median",
                "predicted_max",
                "observed_median",
            ),
        )

        # --------------------------------------------------------------
        # 5. Worst product groups
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("WORST PRODUCT GROUPS")
        print("=" * 72)

        worst_products = con.execute(
            """
            SELECT
                product_group,
                COUNT(*) AS n_rows,

                MEDIAN(observed_cif_fob_factor)
                    AS observed_median,

                MEDIAN(holdout_predicted_factor)
                    AS predicted_median,

                MEDIAN(
                    model_absolute_percentage_error
                ) AS model_median_ape,

                MEDIAN(
                    product_baseline_absolute_percentage_error
                ) AS product_baseline_median_ape,

                AVG(
                    CAST(
                        model_absolute_percentage_error
                        < product_baseline_absolute_percentage_error
                        AS INTEGER
                    )
                ) AS model_win_rate_vs_product

            FROM holdout

            GROUP BY product_group

            HAVING COUNT(*) >= 100

            ORDER BY model_median_ape DESC

            LIMIT 30
            """
        ).df()

        print_diagnostics_table(
            worst_products,
            percentage_columns=(
                "model_median_ape",
                "product_baseline_median_ape",
                "model_win_rate_vs_product",
            ),
            factor_columns=(
                "observed_median",
                "predicted_median",
            ),
        )

    finally:
        con.close()


if __name__ == "__main__":
    main()