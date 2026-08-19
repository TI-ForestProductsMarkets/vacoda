from pathlib import Path

import duckdb


PARQUET_PATH = Path(
    "/storage/vacoda/data/interim/quantity/target_h5/"
    "comtrade_target_h5_quantity_synced.parquet"
)


def main() -> None:
    if not PARQUET_PATH.is_file():
        raise FileNotFoundError(
            f"Quantity parquet not found: {PARQUET_PATH}"
        )

    con = duckdb.connect()

    try:
        parquet_sql = (
            "'"
            + str(PARQUET_PATH).replace("'", "''")
            + "'"
        )

        print("\n" + "=" * 90)
        print("1. COMPARE GROSS-WEIGHT FALLBACK DEFINITIONS")
        print("=" * 90)

        result = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE
                        NOT COALESCE(
                            has_quantity_kg,
                            FALSE
                        )
                        AND NOT COALESCE(
                            has_net_weight_kg,
                            FALSE
                        )
                        AND COALESCE(
                            has_gross_weight_kg,
                            FALSE
                        )
                ) AS flag_definition,

                COUNT(*) FILTER (
                    WHERE
                        gross_weight_kg > 0
                        AND quantity_kg_candidate IS NULL
                        AND net_weight_kg_candidate IS NULL
                ) AS candidate_definition

            FROM read_parquet({parquet_sql})
            """
        ).fetchdf()

        print(result.to_string(index=False))

        print("\n" + "=" * 90)
        print("2. CROSS-TAB OF BOTH DEFINITIONS")
        print("=" * 90)

        comparison = con.execute(
            f"""
            WITH base AS (
                SELECT
                    (
                        NOT COALESCE(
                            has_quantity_kg,
                            FALSE
                        )
                        AND NOT COALESCE(
                            has_net_weight_kg,
                            FALSE
                        )
                        AND COALESCE(
                            has_gross_weight_kg,
                            FALSE
                        )
                    ) AS by_flags,

                    (
                        gross_weight_kg > 0
                        AND quantity_kg_candidate IS NULL
                        AND net_weight_kg_candidate IS NULL
                    ) AS by_candidates

                FROM read_parquet({parquet_sql})
            )

            SELECT
                by_flags,
                by_candidates,
                COUNT(*) AS rows
            FROM base
            GROUP BY
                by_flags,
                by_candidates
            ORDER BY
                by_flags,
                by_candidates
            """
        ).fetchdf()

        print(comparison.to_string(index=False))

        print("\n" + "=" * 90)
        print("3. WHY DO THE DEFINITIONS DISAGREE?")
        print("=" * 90)

        disagreement_summary = con.execute(
            f"""
            WITH base AS (
                SELECT
                    *,

                    (
                        NOT COALESCE(
                            has_quantity_kg,
                            FALSE
                        )
                        AND NOT COALESCE(
                            has_net_weight_kg,
                            FALSE
                        )
                        AND COALESCE(
                            has_gross_weight_kg,
                            FALSE
                        )
                    ) AS by_flags,

                    (
                        gross_weight_kg > 0
                        AND quantity_kg_candidate IS NULL
                        AND net_weight_kg_candidate IS NULL
                    ) AS by_candidates

                FROM read_parquet({parquet_sql})
            )

            SELECT
                by_flags,
                by_candidates,

                COUNT(*) AS rows,

                COUNT(*) FILTER (
                    WHERE gross_weight_kg IS NULL
                ) AS gross_missing,

                COUNT(*) FILTER (
                    WHERE gross_weight_kg = 0
                ) AS gross_zero,

                COUNT(*) FILTER (
                    WHERE gross_weight_kg > 0
                ) AS gross_positive,

                COUNT(*) FILTER (
                    WHERE quantity_kg_candidate IS NOT NULL
                ) AS quantity_candidate_present,

                COUNT(*) FILTER (
                    WHERE net_weight_kg_candidate IS NOT NULL
                ) AS net_candidate_present,

                COUNT(*) FILTER (
                    WHERE alt_quantity_kg_candidate IS NOT NULL
                ) AS alt_candidate_present

            FROM base

            WHERE by_flags <> by_candidates

            GROUP BY
                by_flags,
                by_candidates

            ORDER BY
                by_flags,
                by_candidates
            """
        ).fetchdf()

        print(disagreement_summary.to_string(index=False))

        print("\n" + "=" * 90)
        print("4. BREAKDOWN OF DISAGREEING ROWS")
        print("=" * 90)

        breakdown = con.execute(
            f"""
            WITH base AS (
                SELECT
                    *,

                    (
                        NOT COALESCE(
                            has_quantity_kg,
                            FALSE
                        )
                        AND NOT COALESCE(
                            has_net_weight_kg,
                            FALSE
                        )
                        AND COALESCE(
                            has_gross_weight_kg,
                            FALSE
                        )
                    ) AS by_flags,

                    (
                        gross_weight_kg > 0
                        AND quantity_kg_candidate IS NULL
                        AND net_weight_kg_candidate IS NULL
                    ) AS by_candidates

                FROM read_parquet({parquet_sql})
            )

            SELECT
                has_quantity_kg,
                has_net_weight_kg,
                has_gross_weight_kg,

                CASE
                    WHEN quantity_kg_candidate IS NULL
                        THEN 'NULL'
                    ELSE 'present'
                END AS quantity_candidate,

                CASE
                    WHEN net_weight_kg_candidate IS NULL
                        THEN 'NULL'
                    ELSE 'present'
                END AS net_candidate,

                CASE
                    WHEN gross_weight_kg IS NULL
                        THEN 'NULL'
                    WHEN gross_weight_kg = 0
                        THEN 'zero'
                    WHEN gross_weight_kg > 0
                        THEN 'positive'
                    ELSE 'negative'
                END AS gross_value_state,

                by_flags,
                by_candidates,

                COUNT(*) AS rows

            FROM base

            WHERE by_flags <> by_candidates

            GROUP BY
                has_quantity_kg,
                has_net_weight_kg,
                has_gross_weight_kg,
                quantity_candidate,
                net_candidate,
                gross_value_state,
                by_flags,
                by_candidates

            ORDER BY rows DESC
            """
        ).fetchdf()

        print(breakdown.to_string(index=False))

        print("\n" + "=" * 90)
        print("5. SAMPLE OF DISAGREEING ROWS")
        print("=" * 90)

        sample = con.execute(
            f"""
            WITH base AS (
                SELECT
                    *,

                    (
                        NOT COALESCE(
                            has_quantity_kg,
                            FALSE
                        )
                        AND NOT COALESCE(
                            has_net_weight_kg,
                            FALSE
                        )
                        AND COALESCE(
                            has_gross_weight_kg,
                            FALSE
                        )
                    ) AS by_flags,

                    (
                        gross_weight_kg > 0
                        AND quantity_kg_candidate IS NULL
                        AND net_weight_kg_candidate IS NULL
                    ) AS by_candidates

                FROM read_parquet({parquet_sql})
            )

            SELECT
                year,
                target_cmd,

                quantity,
                quantity_unit,
                quantity_unit_code,
                has_quantity_kg,
                quantity_kg_candidate,

                alt_quantity,
                alt_quantity_unit,
                alt_quantity_unit_code,
                alt_quantity_kg_candidate,

                net_weight_kg,
                has_net_weight_kg,
                net_weight_kg_candidate,

                gross_weight_kg,
                has_gross_weight_kg,

                is_net_weight_estimated,
                is_gross_weight_estimated,

                by_flags,
                by_candidates

            FROM base

            WHERE by_flags <> by_candidates

            LIMIT 50
            """
        ).fetchdf()

        print(sample.to_string(index=False))

        print("\n" + "=" * 90)
        print("6. FUTURE GROSS-ONLY COVERAGE AFTER ALT-KG")
        print("=" * 90)

        future = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE
                        gross_weight_kg > 0
                        AND quantity_kg_candidate IS NULL
                        AND net_weight_kg_candidate IS NULL
                ) AS gross_after_current_sources,

                COUNT(*) FILTER (
                    WHERE
                        gross_weight_kg > 0
                        AND quantity_kg_candidate IS NULL
                        AND net_weight_kg_candidate IS NULL
                        AND alt_quantity_kg_candidate IS NULL
                ) AS gross_after_alt_kg_too

            FROM read_parquet({parquet_sql})
            """
        ).fetchdf()

        print(future.to_string(index=False))

    finally:
        con.close()


if __name__ == "__main__":
    main()