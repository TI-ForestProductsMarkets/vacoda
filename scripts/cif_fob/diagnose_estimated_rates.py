from __future__ import annotations

from pathlib import Path

import duckdb


INPUT_PATH = Path(
    "/storage/vacoda/data/interim/cif_fob/target_h5/"
    "target_h5_cif_fob_estimated_rates.parquet"
)

HOLDOUT_YEAR = 2025


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
                    r.estimated_cif_fob_factor
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
                    r.estimated_cif_fob_factor
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
              AND r.estimated_cif_fob_factor IS NOT NULL
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

                MEDIAN(estimated_cif_fob_factor)
                    AS predicted_median,

                QUANTILE_CONT(
                    estimated_cif_fob_factor,
                    0.05
                ) AS predicted_p05,

                QUANTILE_CONT(
                    estimated_cif_fob_factor,
                    0.95
                ) AS predicted_p95,

                STDDEV_SAMP(observed_cif_fob_factor)
                    AS observed_stddev,

                STDDEV_SAMP(estimated_cif_fob_factor)
                    AS predicted_stddev

            FROM holdout
            """
        ).df()

        print(distribution.to_string(index=False))

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

        print(comparison.to_string(index=False))

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

                MEDIAN(estimated_cif_fob_factor)
                    AS predicted_median,

                MEDIAN(
                    CAST(
                        model_absolute_percentage_error
                        AS INTEGER
                    )
                ) AS model_median_ape,

                QUANTILE_CONT(
                    CAST(
                        model_absolute_percentage_error
                        AS INTEGER
                    ),
                    0.90
                ) AS model_p90_ape,

                MEDIAN(
                    CAST(
                        product_baseline_absolute_percentage_error
                        AS INTEGER
                    )
                ) AS product_baseline_median_ape,

                AVG(
                    CAST(
                        model_absolute_percentage_error
                        < product_baseline_absolute_percentage_error
                        AS INTEGER
                    )
                ) AS model_win_rate_vs_product

            FROM holdout

            GROUP BY factor_source

            ORDER BY factor_source
            """
        ).df()

        print(by_source.to_string(index=False))

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
                        ORDER BY estimated_cif_fob_factor
                    ) AS prediction_decile

                FROM holdout
            )

            SELECT
                prediction_decile,
                COUNT(*) AS n_rows,

                MIN(estimated_cif_fob_factor)
                    AS predicted_min,

                MEDIAN(estimated_cif_fob_factor)
                    AS predicted_median,

                MAX(estimated_cif_fob_factor)
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

        print(calibration.to_string(index=False))

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

                MEDIAN(estimated_cif_fob_factor)
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

        print(worst_products.to_string(index=False))

    finally:
        con.close()


if __name__ == "__main__":
    main()