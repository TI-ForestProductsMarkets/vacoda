"""
audit_cepii_gravity_structure.py

Standalone structural audit of CEPII Gravity v202211.

This script does not modify any VACODA production reference or model
input. It inspects the raw Gravity CSV and saves audit tables separately.
"""

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAVITY_PATH = (
    PROJECT_ROOT
    / "references"
    / "cepii"
    / "gravity"
    / "audit_source"
    / "Gravity_V202211.csv"
)

COUNTRIES_PATH = (
    PROJECT_ROOT
    / "references"
    / "cepii"
    / "gravity"
    / "audit_source"
    / "Countries_V202211.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "reports"
    / "cif_fob"
    / "distance_audit"
    / "gravity_structure"
)


def main() -> None:
    if not GRAVITY_PATH.exists():
        raise FileNotFoundError(
            f"Gravity CSV not found: {GRAVITY_PATH}"
        )

    if not COUNTRIES_PATH.exists():
        raise FileNotFoundError(
            f"Countries CSV not found: {COUNTRIES_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect()

    try:
        con.execute("SET threads = 16")

        gravity_sql = """
            read_csv(
                ?,
                header = true,
                auto_detect = true,
                sample_size = 200000,
                null_padding = true,
                ignore_errors = false
            )
        """

        # ============================================================
        # Schema
        # ============================================================

        schema = con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM {gravity_sql}
            """,
            [str(GRAVITY_PATH)],
        ).fetchdf()

        schema.to_csv(
            OUTPUT_DIR / "gravity_schema.csv",
            index=False,
        )

        print("\nGravity schema:")
        print(schema.to_string(index=False))

        # ============================================================
        # Overall coverage
        # ============================================================

        overall = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,
                MIN(year) AS minimum_year,
                MAX(year) AS maximum_year,

                COUNT(DISTINCT iso3_o) AS n_origins,
                COUNT(DISTINCT iso3_d) AS n_destinations,

                COUNT(
                    DISTINCT
                    iso3_o || '|' || iso3_d
                ) AS n_directed_pairs,

                COUNT(*) FILTER (
                    WHERE iso3_o = iso3_d
                ) AS n_self_pair_rows,

                COUNT(*) FILTER (
                    WHERE iso3_o <> iso3_d
                ) AS n_international_pair_rows,

                COUNT(*) FILTER (
                    WHERE country_exists_o = 1
                      AND country_exists_d = 1
                ) AS n_both_countries_exist_rows,

                COUNT(*) FILTER (
                    WHERE iso3_o IS NULL
                       OR iso3_d IS NULL
                ) AS n_missing_iso_keys

            FROM {gravity_sql}
            """,
            [str(GRAVITY_PATH)],
        ).fetchdf()

        overall.to_csv(
            OUTPUT_DIR / "gravity_overall_coverage.csv",
            index=False,
        )

        print("\nOverall coverage:")
        print(overall.to_string(index=False))

        # ============================================================
        # Coverage by year
        # ============================================================

        yearly = con.execute(
            f"""
            SELECT
                year,
                COUNT(*) AS n_rows,

                COUNT(
                    DISTINCT
                    iso3_o || '|' || iso3_d
                ) AS n_directed_pairs,

                COUNT(*) FILTER (
                    WHERE iso3_o <> iso3_d
                ) AS n_international_rows,

                COUNT(*) FILTER (
                    WHERE country_exists_o = 1
                      AND country_exists_d = 1
                ) AS n_both_countries_exist,

                COUNT(*) FILTER (
                    WHERE dist IS NOT NULL
                ) AS n_dist_available,

                COUNT(*) FILTER (
                    WHERE distcap IS NOT NULL
                ) AS n_distcap_available,

                COUNT(*) FILTER (
                    WHERE distw_harmonic IS NOT NULL
                ) AS n_distw_harmonic_available,

                COUNT(*) FILTER (
                    WHERE contig IS NOT NULL
                ) AS n_contig_available

            FROM {gravity_sql}

            GROUP BY year
            ORDER BY year
            """,
            [str(GRAVITY_PATH)],
        ).fetchdf()

        yearly.to_csv(
            OUTPUT_DIR / "gravity_coverage_by_year.csv",
            index=False,
        )

        print("\nCoverage by year:")
        print(yearly.to_string(index=False))

        # ============================================================
        # Duplicate annual pair keys
        # ============================================================

        duplicates = con.execute(
            f"""
            SELECT
                year,
                iso3_o,
                iso3_d,
                COUNT(*) AS n_rows

            FROM {gravity_sql}

            GROUP BY
                year,
                iso3_o,
                iso3_d

            HAVING COUNT(*) > 1

            ORDER BY
                n_rows DESC,
                year,
                iso3_o,
                iso3_d
            """,
            [str(GRAVITY_PATH)],
        ).fetchdf()

        duplicates.to_csv(
            OUTPUT_DIR / "gravity_duplicate_year_pairs.csv",
            index=False,
        )

        print(
            "\nDuplicate year-country-pair keys:",
            len(duplicates),
        )

        # ============================================================
        # Distance validity
        # ============================================================

        distance_quality = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE dist IS NULL
                ) AS n_dist_missing,

                COUNT(*) FILTER (
                    WHERE dist <= 0
                ) AS n_dist_nonpositive,

                COUNT(*) FILTER (
                    WHERE distcap IS NULL
                ) AS n_distcap_missing,

                COUNT(*) FILTER (
                    WHERE distcap <= 0
                ) AS n_distcap_nonpositive,

                COUNT(*) FILTER (
                    WHERE distw_harmonic IS NULL
                ) AS n_distw_harmonic_missing,

                COUNT(*) FILTER (
                    WHERE distw_harmonic <= 0
                ) AS n_distw_harmonic_nonpositive,

                COUNT(*) FILTER (
                    WHERE contig NOT IN (0, 1)
                      AND contig IS NOT NULL
                ) AS n_invalid_contig

            FROM {gravity_sql}
            """,
            [str(GRAVITY_PATH)],
        ).fetchdf()

        distance_quality.to_csv(
            OUTPUT_DIR / "gravity_distance_quality.csv",
            index=False,
        )

        print("\nDistance quality:")
        print(distance_quality.to_string(index=False))

        # ============================================================
        # Does geography vary by year?
        #
        # Exclude self-pairs because their distance treatment may differ.
        # ============================================================

        pair_variation = con.execute(
            f"""
            SELECT
                iso3_o,
                iso3_d,

                COUNT(DISTINCT year) AS n_years,

                COUNT(DISTINCT dist)
                    AS n_dist_values,

                COUNT(DISTINCT distcap)
                    AS n_distcap_values,

                COUNT(DISTINCT distw_harmonic)
                    AS n_distw_harmonic_values,

                COUNT(DISTINCT distw_arithmetic)
                    AS n_distw_arithmetic_values,

                COUNT(DISTINCT contig)
                    AS n_contig_values,

                MIN(distw_harmonic)
                    AS minimum_distw_harmonic,

                MAX(distw_harmonic)
                    AS maximum_distw_harmonic

            FROM {gravity_sql}

            WHERE iso3_o IS NOT NULL
              AND iso3_d IS NOT NULL
              AND iso3_o <> iso3_d

            GROUP BY
                iso3_o,
                iso3_d
            """,
            [str(GRAVITY_PATH)],
        ).fetchdf()

        pair_variation.to_parquet(
            OUTPUT_DIR / "gravity_pair_time_variation.parquet",
            index=False,
        )

        con.register("pair_variation", pair_variation)

        variation_summary = con.execute(
            """
            SELECT
                COUNT(*) AS n_pairs,

                COUNT(*) FILTER (
                    WHERE n_dist_values > 1
                ) AS n_pairs_dist_changes,

                COUNT(*) FILTER (
                    WHERE n_distcap_values > 1
                ) AS n_pairs_distcap_changes,

                COUNT(*) FILTER (
                    WHERE n_distw_harmonic_values > 1
                ) AS n_pairs_distw_harmonic_changes,

                COUNT(*) FILTER (
                    WHERE n_distw_arithmetic_values > 1
                ) AS n_pairs_distw_arithmetic_changes,

                COUNT(*) FILTER (
                    WHERE n_contig_values > 1
                ) AS n_pairs_contig_changes,

                MAX(n_distw_harmonic_values)
                    AS maximum_distw_harmonic_values_per_pair

            FROM pair_variation
            """
        ).fetchdf()

        variation_summary.to_csv(
            OUTPUT_DIR / "gravity_time_variation_summary.csv",
            index=False,
        )

        print("\nTime-variation summary:")
        print(variation_summary.to_string(index=False))

        # ============================================================
        # Recent-year coverage
        # ============================================================

        recent = con.execute(
            f"""
            SELECT
                year,
                COUNT(
                    DISTINCT
                    iso3_o || '|' || iso3_d
                ) FILTER (
                    WHERE iso3_o <> iso3_d
                      AND distw_harmonic > 0
                ) AS n_usable_international_pairs

            FROM {gravity_sql}

            WHERE year >= 2010

            GROUP BY year
            ORDER BY year
            """,
            [str(GRAVITY_PATH)],
        ).fetchdf()

        recent.to_csv(
            OUTPUT_DIR / "gravity_recent_year_coverage.csv",
            index=False,
        )

        print("\nRecent-year usable pair coverage:")
        print(recent.to_string(index=False))

    finally:
        con.close()

    print(f"\nSaved audit outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()