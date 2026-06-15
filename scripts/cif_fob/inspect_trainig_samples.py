import duckdb

training_path = (
    "/storage/vacoda/data/interim/cif_fob/target_h5/"
    "target_h5_cif_fob_training_sample.parquet"
)

duckdb.sql(
    f"""
    SELECT
        factor_source,
        COUNT(*) AS n_rows,

        MEDIAN(observed_cif_fob_factor) AS median_factor,

        QUANTILE_CONT(
            observed_cif_fob_factor,
            0.01
        ) AS p01,

        QUANTILE_CONT(
            observed_cif_fob_factor,
            0.05
        ) AS p05,

        QUANTILE_CONT(
            observed_cif_fob_factor,
            0.95
        ) AS p95,

        QUANTILE_CONT(
            observed_cif_fob_factor,
            0.99
        ) AS p99,

        ROUND(
            100.0 * COUNT(*) FILTER (
                WHERE observed_cif_fob_factor < 1
            ) / COUNT(*),
            2
        ) AS share_below_1_percent,

        ROUND(
            100.0 * COUNT(*) FILTER (
                WHERE observed_cif_fob_factor > 1.5
            ) / COUNT(*),
            2
        ) AS share_above_1_5_percent

    FROM read_parquet('{training_path}')

    WHERE model_eligible

    GROUP BY factor_source
    ORDER BY factor_source
    """
).df()