from __future__ import annotations

from pathlib import Path

import duckdb


INPUT_PATH = Path(
    "/storage/vacoda/data/interim/cif_fob/target_h5/"
    "target_h5_cif_fob_estimated_rates.parquet"
)

if INPUT_PATH.parent != Path("/storage/vacoda/data/interim/cif_fob/target_h5/"):
    raise ValueError(
        f"Unexpected input path: {INPUT_PATH}. "
        "Expected parent directory: /storage/vacoda/data/interim/cif_fob/target_h5/"
    )
else:
   print(f"Using input path: {INPUT_PATH}")

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

        print_diagnostics_table(
            by_pair_history,
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

        print_diagnostics_table(
                pair_baseline,
                percentage_columns=(
                    "model_median_ape",
                    "model_p90_ape",
                    "pair_baseline_median_ape",
                    "pair_baseline_p90_ape",
                    "model_win_rate_vs_pair_baseline",
                ),
            )


        # --------------------------------------------------------------
        # Model versus pair baseline by historical evidence strength
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("MODEL VERSUS PAIR BASELINE BY HISTORY STRENGTH")
        print("=" * 72)

        pair_history_strength = con.execute(
            """
            WITH classified AS (
                SELECT
                    *,

                    CASE
                        WHEN lagged_pair_factor_observations = 1
                            THEN '01_observation'

                        WHEN lagged_pair_factor_observations
                             BETWEEN 2 AND 5
                            THEN '02_to_05_observations'

                        WHEN lagged_pair_factor_observations
                             BETWEEN 6 AND 20
                            THEN '06_to_20_observations'

                        WHEN lagged_pair_factor_observations
                             BETWEEN 21 AND 100
                            THEN '21_to_100_observations'

                        WHEN lagged_pair_factor_observations > 100
                            THEN 'over_100_observations'

                        ELSE 'missing'
                    END AS history_strength

                FROM holdout

                WHERE lagged_pair_median_factor IS NOT NULL
            )

            SELECT
                history_strength,
                COUNT(*) AS n_rows,

                MEDIAN(
                    lagged_pair_factor_observations
                ) AS median_history_observations,

                MEDIAN(
                    lagged_pair_factor_years
                ) AS median_history_years,

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
                        holdout_predicted_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(
                        observed_cif_fob_factor,
                        0
                    ),
                    0.90
                ) AS model_p90_ape,

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

            FROM classified

            GROUP BY history_strength

            ORDER BY history_strength
            """
        ).df()

        print_diagnostics_table(
            pair_history_strength,
            percentage_columns=(
                "model_median_ape",
                "model_p90_ape",
                "pair_baseline_median_ape",
                "pair_baseline_p90_ape",
                "model_win_rate_vs_pair_baseline",
            ),
        )


        print("\n" + "=" * 72)
        print("INSPECT TWO-TO-THREE-YEAR PAIR HISTORY GROUP")
        print("=" * 72)

        inspect_short_history = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE observed_cif_fob_factor
                          = lagged_pair_median_factor
                ) AS n_exact_pair_matches,

                AVG(
                    CAST(
                        observed_cif_fob_factor
                        = lagged_pair_median_factor
                        AS DOUBLE
                    )
                ) AS exact_pair_match_rate,

                COUNT(DISTINCT exporter_iso || '|' || importer_iso)
                    AS n_distinct_pairs,

                COUNT(DISTINCT observed_cif_fob_factor)
                    AS n_distinct_observed_factors,

                MEDIAN(observed_cif_fob_factor)
                    AS observed_median,

                MEDIAN(lagged_pair_median_factor)
                    AS pair_baseline_median

            FROM holdout

            WHERE lagged_pair_factor_years BETWEEN 2 AND 3
            """
        ).df()

        print_diagnostics_table(
            inspect_short_history,
            percentage_columns=("exact_pair_match_rate",),
            factor_columns=(
                "observed_median",
                "pair_baseline_median",
            ),
        )

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
        # Signed-error and systematic-bias diagnostics
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("SIGNED ERROR AND BIAS")
        print("=" * 72)

        bias = con.execute(
            """
            SELECT
                AVG(
                    holdout_predicted_factor
                    - observed_cif_fob_factor
                ) AS mean_signed_error,

                MEDIAN(
                    holdout_predicted_factor
                    - observed_cif_fob_factor
                ) AS median_signed_error,

                AVG(
                    (
                        holdout_predicted_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(observed_cif_fob_factor, 0)
                ) AS mean_signed_percentage_error,

                MEDIAN(
                    (
                        holdout_predicted_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(observed_cif_fob_factor, 0)
                ) AS median_signed_percentage_error,

                AVG(
                    CAST(
                        holdout_predicted_factor
                        < observed_cif_fob_factor
                        AS DOUBLE
                    )
                ) AS underprediction_rate,

                AVG(
                    CAST(
                        holdout_predicted_factor
                        > observed_cif_fob_factor
                        AS DOUBLE
                    )
                ) AS overprediction_rate

            FROM holdout
            """
        ).df()

        print_diagnostics_table(
            bias,
            percentage_columns=(
                "mean_signed_percentage_error",
                "median_signed_percentage_error",
                "underprediction_rate",
                "overprediction_rate",
            ),
            factor_columns=(
                "mean_signed_error",
                "median_signed_error",
            ),
        )

        # --------------------------------------------------------------
        # Prediction ranking and association
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("PREDICTION ASSOCIATION")
        print("=" * 72)

        association = con.execute(
            """
            WITH ranked AS (
                SELECT
                    holdout_predicted_factor,
                    observed_cif_fob_factor,

                    RANK() OVER (
                        ORDER BY holdout_predicted_factor
                    ) AS predicted_rank,

                    RANK() OVER (
                        ORDER BY observed_cif_fob_factor
                    ) AS observed_rank

                FROM holdout
            )

            SELECT
                CORR(
                    holdout_predicted_factor,
                    observed_cif_fob_factor
                ) AS pearson_correlation,

                CORR(
                    predicted_rank,
                    observed_rank
                ) AS spearman_rank_correlation

            FROM ranked
            """
        ).df()

        print(association.to_string(index=False))


        # --------------------------------------------------------------
        # Model versus pair baseline by number of historical years
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("MODEL VERSUS PAIR BASELINE BY HISTORY YEARS")
        print("=" * 72)

        pair_history_years = con.execute(
            """
            WITH classified AS (
                SELECT
                    *,

                    CASE
                        WHEN lagged_pair_factor_years = 1
                            THEN '01_one_year'

                        WHEN lagged_pair_factor_years BETWEEN 2 AND 3
                            THEN '02_two_to_three_years'

                        WHEN lagged_pair_factor_years BETWEEN 4 AND 5
                            THEN '03_four_to_five_years'

                        WHEN lagged_pair_factor_years BETWEEN 6 AND 7
                            THEN '04_six_to_seven_years'

                        WHEN lagged_pair_factor_years >= 8
                            THEN '05_eight_or_more_years'

                        ELSE 'missing'
                    END AS history_year_group

                FROM holdout

                WHERE lagged_pair_median_factor IS NOT NULL
            )

            SELECT
                history_year_group,
                COUNT(*) AS n_rows,

                MEDIAN(
                    lagged_pair_factor_observations
                ) AS median_history_observations,

                MEDIAN(
                    lagged_pair_factor_years
                ) AS median_history_years,

                MEDIAN(
                    model_absolute_percentage_error
                ) AS model_median_ape,

                MEDIAN(
                    ABS(
                        lagged_pair_median_factor
                        - observed_cif_fob_factor
                    )
                    / NULLIF(observed_cif_fob_factor, 0)
                ) AS pair_baseline_median_ape,

                AVG(
                    CAST(
                        model_absolute_percentage_error
                        <
                        ABS(
                            lagged_pair_median_factor
                            - observed_cif_fob_factor
                        )
                        / NULLIF(observed_cif_fob_factor, 0)
                        AS DOUBLE
                    )
                ) AS model_win_rate_vs_pair_baseline

            FROM classified

            GROUP BY history_year_group

            ORDER BY history_year_group
            """
        ).df()

        print_diagnostics_table(
            pair_history_years,
            percentage_columns=(
                "model_median_ape",
                "pair_baseline_median_ape",
                "model_win_rate_vs_pair_baseline",
            ),
        )
        # --------------------------------------------------------------
        # Performance by observed CIF/FOB-factor range
        #
        # This reveals whether the model performs well around the dense
        # central range but underpredicts unusually high factors.
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("PERFORMANCE BY OBSERVED FACTOR RANGE")
        print("=" * 72)

        by_observed_factor_range = con.execute(
            """
            WITH classified AS (
                SELECT
                    *,

                    CASE
                        WHEN observed_cif_fob_factor < 1.02
                            THEN '01_below_1.02'

                        WHEN observed_cif_fob_factor < 1.05
                            THEN '02_1.02_to_1.05'

                        WHEN observed_cif_fob_factor < 1.10
                            THEN '03_1.05_to_1.10'

                        WHEN observed_cif_fob_factor < 1.25
                            THEN '04_1.10_to_1.25'

                        WHEN observed_cif_fob_factor < 1.50
                            THEN '05_1.25_to_1.50'

                        ELSE '06_1.50_and_above'
                    END AS observed_factor_range

                FROM holdout
            )

            SELECT
                observed_factor_range,
                COUNT(*) AS n_rows,

                MEDIAN(observed_cif_fob_factor)
                    AS observed_median,

                MEDIAN(holdout_predicted_factor)
                    AS predicted_median,

                MEDIAN(
                    holdout_predicted_factor
                    - observed_cif_fob_factor
                ) AS median_signed_error,

                MEDIAN(
                    model_absolute_percentage_error
                ) AS model_median_ape,

                QUANTILE_CONT(
                    model_absolute_percentage_error,
                    0.90
                ) AS model_p90_ape,

                MEDIAN(
                    product_baseline_absolute_percentage_error
                ) AS product_baseline_median_ape,

                AVG(
                    CAST(
                        model_absolute_percentage_error
                        < product_baseline_absolute_percentage_error
                        AS DOUBLE
                    )
                ) AS model_win_rate_vs_product,

                AVG(
                    CAST(
                        holdout_predicted_factor
                        < observed_cif_fob_factor
                        AS DOUBLE
                    )
                ) AS underprediction_rate

            FROM classified

            GROUP BY observed_factor_range

            ORDER BY observed_factor_range
            """
        ).df()

        print_diagnostics_table(
            by_observed_factor_range,
            percentage_columns=(
                "model_median_ape",
                "model_p90_ape",
                "product_baseline_median_ape",
                "model_win_rate_vs_product",
                "underprediction_rate",
            ),
            factor_columns=(
                "observed_median",
                "predicted_median",
                "median_signed_error",
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