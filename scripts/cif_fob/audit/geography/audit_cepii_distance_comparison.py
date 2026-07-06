"""
audit_cepii_distance_comparison.py

Compare VACODA's current legacy CEPII GeoDist distance with the
CEPII Gravity v202211 2021 distance measures.

This is a standalone audit. It does not modify any production reference.
"""

from pathlib import Path

import duckdb


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CURRENT_PAIR_GEOGRAPHY_PATH = (
    PROJECT_ROOT
    / "references"
    / "geography"
    / "prepared"
    / "country_pair_geography.parquet"
)

GRAVITY_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "output"
    / "reports"
    / "cif_fob"
    / "distance_audit"
    / "gravity_recent"
    / "gravity_2021_pair_snapshot.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "reports"
    / "cif_fob"
    / "distance_audit"
    / "distance_comparison"
)


LARGE_COUNTRIES = [
    "CHN",
    "USA",
    "BRA",
    "AUS",
    "CAN",
    "RUS",
    "IDN",
    "IND",
]


def sql_literal(path: Path) -> str:
    """Return a safely quoted SQL path literal."""
    return "'" + str(path).replace("'", "''") + "'"


def main() -> None:
    for label, path in [
        (
            "Current VACODA pair geography",
            CURRENT_PAIR_GEOGRAPHY_PATH,
        ),
        (
            "Gravity 2021 snapshot",
            GRAVITY_SNAPSHOT_PATH,
        ),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_sql = sql_literal(
        CURRENT_PAIR_GEOGRAPHY_PATH
    )
    gravity_sql = sql_literal(
        GRAVITY_SNAPSHOT_PATH
    )

    comparison_path = (
        OUTPUT_DIR
        / "current_vs_gravity_2021_pair_comparison.parquet"
    )
    comparison_sql = sql_literal(comparison_path)

    con = duckdb.connect()

    try:
        con.execute("SET threads = 16")

        # ==============================================================
        # Validate uniqueness
        # ==============================================================

        uniqueness = con.execute(
            f"""
            SELECT
                'current' AS source,
                COUNT(*) AS n_rows,
                COUNT(
                    DISTINCT origin_iso3 || '|' || destination_iso3
                ) AS n_pairs

            FROM read_parquet({current_sql})

            UNION ALL

            SELECT
                'gravity' AS source,
                COUNT(*) AS n_rows,
                COUNT(
                    DISTINCT origin_iso3 || '|' || destination_iso3
                ) AS n_pairs

            FROM read_parquet({gravity_sql})
            """
        ).fetchdf()

        print("\nReference uniqueness:")
        print(uniqueness.to_string(index=False))

        uniqueness.to_csv(
            OUTPUT_DIR / "reference_uniqueness.csv",
            index=False,
        )

        if not (
            uniqueness["n_rows"]
            == uniqueness["n_pairs"]
        ).all():
            raise RuntimeError(
                "At least one pair reference contains duplicate keys."
            )

        # ==============================================================
        # Build full outer-join comparison
        # ==============================================================

        con.execute(
            f"""
            COPY (
                WITH current AS (
                    SELECT
                        origin_iso3,
                        destination_iso3,

                        distance_simple_km
                            AS legacy_dist_km,

                        distance_capital_km
                            AS legacy_distcap_km,

                        distance_weighted_km
                            AS legacy_distw_km,

                        distance_weighted_ces_km
                            AS legacy_distwces_km,

                        distance_model_km
                            AS current_distance_model_km,

                        distance_model_source
                            AS current_distance_source,

                        log_distance_model_km
                            AS current_log_distance,

                        CAST(contiguous AS INTEGER)
                            AS current_contiguous

                    FROM read_parquet({current_sql})
                ),

                gravity AS (
                    SELECT
                        origin_iso3,
                        destination_iso3,

                        gravity_dist_km,
                        gravity_distcap_km,
                        gravity_distw_harmonic_km,
                        gravity_distw_arithmetic_km,
                        gravity_contiguous

                    FROM read_parquet({gravity_sql})
                ),

                joined AS (
                    SELECT
                        COALESCE(
                            c.origin_iso3,
                            g.origin_iso3
                        ) AS origin_iso3,

                        COALESCE(
                            c.destination_iso3,
                            g.destination_iso3
                        ) AS destination_iso3,

                        c.origin_iso3 IS NOT NULL
                            AS found_current,

                        g.origin_iso3 IS NOT NULL
                            AS found_gravity,

                        c.legacy_dist_km,
                        c.legacy_distcap_km,
                        c.legacy_distw_km,
                        c.legacy_distwces_km,

                        c.current_distance_model_km,
                        c.current_distance_source,
                        c.current_log_distance,
                        c.current_contiguous,

                        g.gravity_dist_km,
                        g.gravity_distcap_km,
                        g.gravity_distw_harmonic_km,
                        g.gravity_distw_arithmetic_km,
                        g.gravity_contiguous

                    FROM current AS c

                    FULL OUTER JOIN gravity AS g
                      ON c.origin_iso3 = g.origin_iso3
                     AND c.destination_iso3 = g.destination_iso3
                )

                SELECT
                    *,
                    CASE 
                        WHEN gravity_dist_km > 0
                            THEN LN(gravity_dist_km)
                        ELSE NULL
                    END AS gravity_log_dist,

                    CASE
                        WHEN gravity_distw_harmonic_km > 0
                            THEN LN(gravity_distw_harmonic_km)
                        ELSE NULL
                    END AS gravity_log_distw_harmonic,

                    gravity_distw_harmonic_km
                    - current_distance_model_km
                        AS harmonic_absolute_difference_km,

                    ABS(
                        gravity_distw_harmonic_km
                        - current_distance_model_km
                    ) AS harmonic_absolute_difference_km_abs,

                    CASE
                        WHEN current_distance_model_km > 0
                            THEN (
                                ABS(
                                    gravity_distw_harmonic_km
                                    - current_distance_model_km
                                )
                                / current_distance_model_km
                            )
                        ELSE NULL
                    END AS harmonic_absolute_percentage_difference,

                    gravity_dist_km
                    - legacy_dist_km
                        AS simple_absolute_difference_km,

                    CASE
                        WHEN legacy_dist_km > 0
                            THEN (
                                ABS(
                                    gravity_dist_km
                                    - legacy_dist_km
                                )
                                / legacy_dist_km
                            )
                        ELSE NULL
                    END AS simple_absolute_percentage_difference,

                    CASE
                        WHEN current_contiguous IS NULL
                          OR gravity_contiguous IS NULL
                            THEN NULL
                        ELSE (
                            current_contiguous
                            <> gravity_contiguous
                        )
                    END AS contiguity_differs,

                    (
                        origin_iso3 IN (
                            'CHN',
                            'USA',
                            'BRA',
                            'AUS',
                            'CAN',
                            'RUS',
                            'IDN',
                            'IND'
                        )
                        OR destination_iso3 IN (
                            'CHN',
                            'USA',
                            'BRA',
                            'AUS',
                            'CAN',
                            'RUS',
                            'IDN',
                            'IND'
                        )
                    ) AS involves_large_country

                FROM joined

                ORDER BY
                    origin_iso3,
                    destination_iso3
            )
            TO {comparison_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        # ==============================================================
        # Coverage summary
        # ==============================================================

        coverage = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_union_pairs,

                COUNT(*) FILTER (
                    WHERE found_current
                ) AS n_current_pairs,

                COUNT(*) FILTER (
                    WHERE found_gravity
                ) AS n_gravity_pairs,

                COUNT(*) FILTER (
                    WHERE found_current
                      AND found_gravity
                ) AS n_matched_pairs,

                COUNT(*) FILTER (
                    WHERE found_current
                      AND NOT found_gravity
                ) AS n_current_only_pairs,

                COUNT(*) FILTER (
                    WHERE found_gravity
                      AND NOT found_current
                ) AS n_gravity_only_pairs,

                COUNT(*) FILTER (
                    WHERE found_current
                      AND found_gravity
                      AND origin_iso3 <> destination_iso3
                ) AS n_matched_international_pairs,

                COUNT(*) FILTER (
                    WHERE found_current
                      AND found_gravity
                      AND current_distance_model_km > 0
                      AND gravity_distw_harmonic_km > 0
                ) AS n_valid_harmonic_comparisons

            FROM read_parquet({comparison_sql})
            """
        ).fetchdf()

        coverage.to_csv(
            OUTPUT_DIR / "coverage_summary.csv",
            index=False,
        )

        print("\nCoverage summary:")
        print(coverage.to_string(index=False))

        # ==============================================================
        # Missing-pair lists
        # ==============================================================

        current_only = con.execute(
            f"""
            SELECT
                origin_iso3,
                destination_iso3,
                current_distance_model_km,
                current_distance_source

            FROM read_parquet({comparison_sql})

            WHERE found_current
              AND NOT found_gravity

            ORDER BY
                origin_iso3,
                destination_iso3
            """
        ).fetchdf()

        current_only.to_csv(
            OUTPUT_DIR / "pairs_missing_from_gravity.csv",
            index=False,
        )

        gravity_only = con.execute(
            f"""
            SELECT
                origin_iso3,
                destination_iso3,
                gravity_dist_km,
                gravity_distw_harmonic_km

            FROM read_parquet({comparison_sql})

            WHERE found_gravity
              AND NOT found_current

            ORDER BY
                origin_iso3,
                destination_iso3
            """
        ).fetchdf()

        gravity_only.to_csv(
            OUTPUT_DIR / "pairs_missing_from_current_geodist.csv",
            index=False,
        )

        # ==============================================================
        # Correlation and difference summary
        #
        # Exclude self-pairs from the main comparison.
        # ==============================================================

        summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_pairs,

                CORR(
                    current_distance_model_km,
                    gravity_distw_harmonic_km
                ) AS correlation_raw_current_vs_harmonic,

                CORR(
                    current_log_distance,
                    gravity_log_distw_harmonic
                ) AS correlation_log_current_vs_harmonic,

                CORR(
                    legacy_dist_km,
                    gravity_dist_km
                ) AS correlation_raw_legacy_dist_vs_gravity_dist,

                CORR(
                    CASE 
                        WHEN legacy_dist_km > 0
                            THEN LN(legacy_dist_km)
                        ELSE NULL
                        END, 
                    gravity_log_dist
                ) AS correlation_log_legacy_dist_vs_gravity_dist,

                MEDIAN(
                    harmonic_absolute_percentage_difference
                ) AS median_harmonic_absolute_percentage_difference,

                QUANTILE_CONT(
                    harmonic_absolute_percentage_difference,
                    0.95
                ) AS p95_harmonic_absolute_percentage_difference,

                QUANTILE_CONT(
                    harmonic_absolute_percentage_difference,
                    0.99
                ) AS p99_harmonic_absolute_percentage_difference,

                MAX(
                    harmonic_absolute_percentage_difference
                ) AS maximum_harmonic_absolute_percentage_difference,

                MEDIAN(
                    harmonic_absolute_difference_km_abs
                ) AS median_harmonic_absolute_difference_km,

                QUANTILE_CONT(
                    harmonic_absolute_difference_km_abs,
                    0.95
                ) AS p95_harmonic_absolute_difference_km

            FROM read_parquet({comparison_sql})

            WHERE found_current
              AND found_gravity
              AND origin_iso3 <> destination_iso3
              AND current_distance_model_km > 0
              AND gravity_distw_harmonic_km > 0
            """
        ).fetchdf()

        summary.to_csv(
            OUTPUT_DIR / "distance_comparison_summary.csv",
            index=False,
        )

        print("\nDistance comparison summary:")
        print(summary.to_string(index=False))

        # ==============================================================
        # Separate current weighted values from capital fallbacks
        # ==============================================================

        by_current_source = con.execute(
            f"""
            SELECT
                current_distance_source,

                COUNT(*) AS n_pairs,

                CORR(
                    current_distance_model_km,
                    gravity_distw_harmonic_km
                ) AS raw_correlation,

                CORR(
                    current_log_distance,
                    gravity_log_distw_harmonic
                ) AS log_correlation,

                MEDIAN(
                    harmonic_absolute_percentage_difference
                ) AS median_absolute_percentage_difference,

                QUANTILE_CONT(
                    harmonic_absolute_percentage_difference,
                    0.95
                ) AS p95_absolute_percentage_difference

            FROM read_parquet({comparison_sql})

            WHERE found_current
              AND found_gravity
              AND origin_iso3 <> destination_iso3
              AND current_distance_model_km > 0
              AND gravity_distw_harmonic_km > 0

            GROUP BY current_distance_source

            ORDER BY current_distance_source
            """
        ).fetchdf()

        by_current_source.to_csv(
            OUTPUT_DIR
            / "distance_comparison_by_current_source.csv",
            index=False,
        )

        print("\nComparison by current distance source:")
        print(by_current_source.to_string(index=False))

        # ==============================================================
        # Largest discrepancies
        # ==============================================================

        largest = con.execute(
            f"""
            SELECT
                origin_iso3,
                destination_iso3,

                current_distance_model_km,
                current_distance_source,

                legacy_dist_km,
                legacy_distcap_km,
                legacy_distw_km,

                gravity_dist_km,
                gravity_distcap_km,
                gravity_distw_harmonic_km,

                harmonic_absolute_difference_km,
                harmonic_absolute_difference_km_abs,
                harmonic_absolute_percentage_difference,

                current_contiguous,
                gravity_contiguous,
                contiguity_differs

            FROM read_parquet({comparison_sql})

            WHERE found_current
              AND found_gravity
              AND origin_iso3 <> destination_iso3
              AND current_distance_model_km > 0
              AND gravity_distw_harmonic_km > 0

            ORDER BY
                harmonic_absolute_percentage_difference DESC,
                harmonic_absolute_difference_km_abs DESC

            LIMIT 500
            """
        ).fetchdf()

        largest.to_csv(
            OUTPUT_DIR / "largest_distance_discrepancies.csv",
            index=False,
        )

        print("\nLargest 20 distance discrepancies:")
        print(largest.head(20).to_string(index=False))

        # ==============================================================
        # Large-country discrepancies
        # ==============================================================

        large_country = con.execute(
            f"""
            SELECT
                origin_iso3,
                destination_iso3,

                current_distance_model_km,
                current_distance_source,

                gravity_dist_km,
                gravity_distw_harmonic_km,

                harmonic_absolute_difference_km,
                harmonic_absolute_percentage_difference,

                current_contiguous,
                gravity_contiguous,
                contiguity_differs

            FROM read_parquet({comparison_sql})

            WHERE found_current
              AND found_gravity
              AND origin_iso3 <> destination_iso3
              AND involves_large_country
              AND current_distance_model_km > 0
              AND gravity_distw_harmonic_km > 0

            ORDER BY
                harmonic_absolute_percentage_difference DESC,
                origin_iso3,
                destination_iso3
            """
        ).fetchdf()

        large_country.to_csv(
            OUTPUT_DIR / "large_country_discrepancies.csv",
            index=False,
        )

        # Summary by focal large country
        large_country_summary_parts = []

        for country in LARGE_COUNTRIES:
            result = con.execute(
                f"""
                SELECT
                    '{country}' AS focal_country,

                    COUNT(*) AS n_pairs,

                    MEDIAN(
                        harmonic_absolute_percentage_difference
                    ) AS median_absolute_percentage_difference,

                    QUANTILE_CONT(
                        harmonic_absolute_percentage_difference,
                        0.95
                    ) AS p95_absolute_percentage_difference,

                    MAX(
                        harmonic_absolute_percentage_difference
                    ) AS maximum_absolute_percentage_difference

                FROM read_parquet({comparison_sql})

                WHERE found_current
                  AND found_gravity
                  AND origin_iso3 <> destination_iso3
                  AND (
                        origin_iso3 = '{country}'
                     OR destination_iso3 = '{country}'
                  )
                  AND current_distance_model_km > 0
                  AND gravity_distw_harmonic_km > 0
                """
            ).fetchdf()

            large_country_summary_parts.append(result)

        import pandas as pd

        large_country_summary = pd.concat(
            large_country_summary_parts,
            ignore_index=True,
        )

        large_country_summary.to_csv(
            OUTPUT_DIR / "large_country_summary.csv",
            index=False,
        )

        print("\nLarge-country summary:")
        print(large_country_summary.to_string(index=False))

        # ==============================================================
        # Contiguity comparison
        # ==============================================================

        contiguity_summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_matched_pairs,

                COUNT(*) FILTER (
                    WHERE current_contiguous
                          = gravity_contiguous
                ) AS n_contiguity_same,

                COUNT(*) FILTER (
                    WHERE current_contiguous
                          <> gravity_contiguous
                ) AS n_contiguity_different,

                COUNT(*) FILTER (
                    WHERE current_contiguous = 0
                      AND gravity_contiguous = 1
                ) AS n_current_zero_gravity_one,

                COUNT(*) FILTER (
                    WHERE current_contiguous = 1
                      AND gravity_contiguous = 0
                ) AS n_current_one_gravity_zero

            FROM read_parquet({comparison_sql})

            WHERE found_current
              AND found_gravity
              AND origin_iso3 <> destination_iso3
              AND current_contiguous IS NOT NULL
              AND gravity_contiguous IS NOT NULL
            """
        ).fetchdf()

        contiguity_summary.to_csv(
            OUTPUT_DIR / "contiguity_summary.csv",
            index=False,
        )

        print("\nContiguity summary:")
        print(contiguity_summary.to_string(index=False))

        contiguity_differences = con.execute(
            f"""
            SELECT
                origin_iso3,
                destination_iso3,
                current_contiguous,
                gravity_contiguous

            FROM read_parquet({comparison_sql})

            WHERE found_current
              AND found_gravity
              AND origin_iso3 <> destination_iso3
              AND current_contiguous IS NOT NULL
              AND gravity_contiguous IS NOT NULL
              AND current_contiguous
                  <> gravity_contiguous

            ORDER BY
                origin_iso3,
                destination_iso3
            """
        ).fetchdf()

        contiguity_differences.to_csv(
            OUTPUT_DIR / "contiguity_differences.csv",
            index=False,
        )

        print(
            "\nNumber of contiguity differences:",
            len(contiguity_differences),
        )

    finally:
        con.close()

    print(f"\nSaved comparison output to: {comparison_path}")
    print(f"Saved audit reports to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()