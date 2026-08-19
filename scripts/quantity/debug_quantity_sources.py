from pathlib import Path

import duckdb


PATH = Path(
    "/storage/vacoda/data/interim/hs_sync/target_h5/applied/"
    "comtrade_target_h5_hs_synced.parquet"
)

con = duckdb.connect()
con.execute("SET threads = 50")


# ------------------------------------------------------------
# 1. ALT-QUANTITY: POTENTIAL KG SOURCES
# ------------------------------------------------------------

print("\nALT-QUANTITY KG-LIKE UNITS")
print("=" * 90)

print(
    con.execute(
        """
        SELECT
            alt_quantity_unit_code,
            alt_quantity_unit,
            COUNT(*) AS rows,
            COUNT(*) FILTER (
                WHERE alt_quantity > 0
            ) AS positive_rows,
            COUNT(*) FILTER (
                WHERE net_weight_kg > 0
            ) AS with_net_weight,
            COUNT(*) FILTER (
                WHERE NOT (
                    quantity_unit_code = 8
                    AND quantity > 0
                )
                AND NOT (net_weight_kg > 0)
            ) AS potential_extra_rows
        FROM read_parquet(?)
        WHERE alt_quantity_unit_code IN (8, 21, 34)
        GROUP BY 1, 2
        ORDER BY 1
        """,
        [str(PATH)],
    ).fetchdf().to_string(index=False)
)


# ------------------------------------------------------------
# 2. NET WEIGHT + NON-KG QUANTITY
# ------------------------------------------------------------

print("\nNET WEIGHT WITH NON-KG QUANTITY")
print("=" * 90)

print(
    con.execute(
        """
        SELECT
            quantity_unit_code,
            quantity_unit,
            COUNT(*) AS rows
        FROM read_parquet(?)
        WHERE net_weight_kg > 0
          AND quantity > 0
          AND quantity_unit_code <> 8
        GROUP BY 1, 2
        ORDER BY rows DESC
        """,
        [str(PATH)],
    ).fetchdf().to_string(index=False)
)


print("\nNET WEIGHT WITH NON-KG ALT QUANTITY")
print("=" * 90)

print(
    con.execute(
        """
        SELECT
            alt_quantity_unit_code,
            alt_quantity_unit,
            COUNT(*) AS rows
        FROM read_parquet(?)
        WHERE net_weight_kg > 0
          AND alt_quantity > 0
          AND alt_quantity_unit_code NOT IN (8, 21)
        GROUP BY 1, 2
        ORDER BY rows DESC
        """,
        [str(PATH)],
    ).fetchdf().to_string(index=False)
)


# ------------------------------------------------------------
# 3. WHAT ALT-QUANTITY COULD ADD TO KG COVERAGE
# ------------------------------------------------------------

print("\nALT-QUANTITY INCREMENTAL KG COVERAGE")
print("=" * 90)

print(
    con.execute(
        """
        SELECT
            CASE
                WHEN alt_quantity_unit_code = 8
                    THEN 'alt_kg'
                WHEN alt_quantity_unit_code = 21
                    THEN 'alt_1000kg'
                ELSE 'other'
            END AS alt_source,

            COUNT(*) AS rows,

            COUNT(*) FILTER (
                WHERE alt_quantity > 0
            ) AS positive_rows,

            COUNT(*) FILTER (
                WHERE alt_quantity > 0
                  AND NOT (
                      quantity_unit_code = 8
                      AND quantity > 0
                  )
                  AND NOT (net_weight_kg > 0)
            ) AS incremental_rows

        FROM read_parquet(?)

        WHERE alt_quantity_unit_code IN (8, 21)

        GROUP BY 1
        ORDER BY 1
        """,
        [str(PATH)],
    ).fetchdf().to_string(index=False)
)


# ------------------------------------------------------------
# 4. GROSS-WEIGHT DEEP DIVE
# ------------------------------------------------------------

print("\nGROSS WEIGHT AVAILABILITY")
print("=" * 90)

print(
    con.execute(
        """
        SELECT
            COUNT(*) AS total_rows,

            COUNT(*) FILTER (
                WHERE gross_weight_kg > 0
            ) AS gross_positive,

            COUNT(*) FILTER (
                WHERE gross_weight_kg > 0
                  AND net_weight_kg > 0
            ) AS gross_and_net,

            COUNT(*) FILTER (
                WHERE gross_weight_kg > 0
                  AND NOT (net_weight_kg > 0)
            ) AS gross_without_net,

            COUNT(*) FILTER (
                WHERE gross_weight_kg > 0
                  AND NOT (
                      quantity_unit_code = 8
                      AND quantity > 0
                  )
                  AND NOT (net_weight_kg > 0)
            ) AS gross_after_current_sources

        FROM read_parquet(?)
        """,
        [str(PATH)],
    ).fetchdf().to_string(index=False)
)


print("\nGROSS / NET-WEIGHT RELATIONSHIP")
print("=" * 90)

print(
    con.execute(
        """
        SELECT
            COUNT(*) AS comparable_rows,

            median(gross_weight_kg / net_weight_kg) AS median_ratio,

            quantile_cont(
                gross_weight_kg / net_weight_kg,
                0.10
            ) AS p10_ratio,

            quantile_cont(
                gross_weight_kg / net_weight_kg,
                0.90
            ) AS p90_ratio,

            COUNT(*) FILTER (
                WHERE gross_weight_kg < net_weight_kg
            ) AS gross_less_than_net

        FROM read_parquet(?)

        WHERE gross_weight_kg > 0
          AND net_weight_kg > 0
        """,
        [str(PATH)],
    ).fetchdf().to_string(index=False)
)

con.close()