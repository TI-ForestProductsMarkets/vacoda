"""
audit_cepii_gravity_recent.py

Audit the recent-period CEPII Gravity v202211 geography variables.

Purpose
-------
1. Restrict the Gravity file to country pairs for which both countries
   exist in each year.
2. Verify uniqueness by year and ISO3 pair.
3. Test whether distance and contiguity change during 2017-2021.
4. Create a clean 2021 pair-level snapshot for later comparison with
   VACODA's current legacy GeoDist reference.

This script does not modify any production VACODA reference.
"""

from pathlib import Path

import duckdb


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    / "gravity_recent"
)

SNAPSHOT_PATH = (
    OUTPUT_DIR
    / "gravity_2021_pair_snapshot.parquet"
)


# ======================================================================
# Helpers
# ======================================================================

def sql_literal(path: Path) -> str:
    """Return a safely quoted SQL path literal."""
    return "'" + str(path).replace("'", "''") + "'"


# ======================================================================
# Main audit
# ======================================================================

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

    gravity_sql = sql_literal(GRAVITY_PATH)
    countries_sql = sql_literal(COUNTRIES_PATH)
    snapshot_sql = sql_literal(SNAPSHOT_PATH)

    con = duckdb.connect()

    try:
        con.execute("SET threads = 16")

        # Read only the fields needed for this audit.
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW gravity AS
            SELECT
                CAST(year AS INTEGER) AS year,

                TRIM(UPPER(country_id_o)) AS country_id_o,
                TRIM(UPPER(country_id_d)) AS country_id_d,

                TRIM(UPPER(iso3_o)) AS iso3_o,
                TRIM(UPPER(iso3_d)) AS iso3_d,

                CAST(country_exists_o AS INTEGER)
                    AS country_exists_o,

                CAST(country_exists_d AS INTEGER)
                    AS country_exists_d,

                CAST(dist AS DOUBLE) AS dist,
                CAST(distcap AS DOUBLE) AS distcap,

                CAST(distw_harmonic AS DOUBLE)
                    AS distw_harmonic,

                CAST(distw_arithmetic AS DOUBLE)
                    AS distw_arithmetic,

                CAST(distw_harmonic_jh AS DOUBLE)
                    AS distw_harmonic_jh,

                CAST(distw_arithmetic_jh AS DOUBLE)
                    AS distw_arithmetic_jh,

                CAST(contig AS INTEGER) AS contig

            FROM read_csv(
                {gravity_sql},
                header = true,
                auto_detect = true,
                sample_size = 200000,
                null_padding = true,
                ignore_errors = false
            )
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW countries AS
            SELECT
                TRIM(UPPER(country_id)) AS country_id,
                TRIM(UPPER(iso3)) AS iso3,
                CAST(iso3num AS INTEGER) AS iso3num,
                country,
                countrylong,
                CAST(first_year AS INTEGER) AS first_year,
                CAST(last_year AS INTEGER) AS last_year,
                TRIM(UPPER(countrygroup_iso3))
                    AS countrygroup_iso3,
                TRIM(UPPER(iso2)) AS iso2

            FROM read_csv(
                {countries_sql},
                header = true,
                auto_detect = true,
                sample_size = -1,
                null_padding = true,
                ignore_errors = false
            )
            """
        )

        # ==============================================================
        # 1. Diagnose why ISO3 duplicates exist in the full file
        # ==============================================================

        country_id_mapping = con.execute(
            """
            SELECT
                iso3,
                COUNT(*) AS n_country_ids,
                LIST(
                    country_id
                    ORDER BY country_id
                ) AS country_ids,
                LIST(
                    country
                    ORDER BY country_id
                ) AS country_names

            FROM countries

            WHERE iso3 IS NOT NULL

            GROUP BY iso3

            HAVING COUNT(*) > 1

            ORDER BY
                n_country_ids DESC,
                iso3
            """
        ).fetchdf()

        country_id_mapping.to_csv(
            OUTPUT_DIR
            / "countries_multiple_country_ids_per_iso3.csv",
            index=False,
        )

        print("\nISO3 codes represented by multiple country IDs:")
        print(country_id_mapping.to_string(index=False))

        # ==============================================================
        # 2. Recent existing-country coverage
        # ==============================================================

        recent_coverage = con.execute(
            """
            SELECT
                year,

                COUNT(*) AS n_rows,

                COUNT(DISTINCT country_id_o)
                    AS n_origin_country_ids,

                COUNT(DISTINCT country_id_d)
                    AS n_destination_country_ids,

                COUNT(DISTINCT iso3_o)
                    AS n_origin_iso3,

                COUNT(DISTINCT iso3_d)
                    AS n_destination_iso3,

                COUNT(
                    DISTINCT iso3_o || '|' || iso3_d
                ) AS n_distinct_iso3_pairs,

                COUNT(*) FILTER (
                    WHERE iso3_o = iso3_d
                ) AS n_self_pairs,

                COUNT(*) FILTER (
                    WHERE iso3_o <> iso3_d
                ) AS n_international_pairs,

                COUNT(*) FILTER (
                    WHERE distw_harmonic > 0
                ) AS n_valid_distw_harmonic,

                COUNT(*) FILTER (
                    WHERE dist > 0
                ) AS n_valid_dist,

                COUNT(*) FILTER (
                    WHERE distcap > 0
                ) AS n_valid_distcap,

                COUNT(*) FILTER (
                    WHERE contig IN (0, 1)
                ) AS n_valid_contig

            FROM gravity

            WHERE year BETWEEN 2017 AND 2021
              AND country_exists_o = 1
              AND country_exists_d = 1

            GROUP BY year
            ORDER BY year
            """
        ).fetchdf()

        recent_coverage.to_csv(
            OUTPUT_DIR
            / "recent_existing_country_coverage.csv",
            index=False,
        )

        print("\nRecent existing-country coverage:")
        print(recent_coverage.to_string(index=False))

        # ==============================================================
        # 3. Verify uniqueness by year and ISO3 pair
        # ==============================================================

        recent_duplicates = con.execute(
            """
            SELECT
                year,
                iso3_o,
                iso3_d,

                COUNT(*) AS n_rows,

                LIST(
                    country_id_o
                    ORDER BY country_id_o
                ) AS origin_country_ids,

                LIST(
                    country_id_d
                    ORDER BY country_id_d
                ) AS destination_country_ids,

                LIST(
                    distw_harmonic
                    ORDER BY country_id_o, country_id_d
                ) AS distw_harmonic_values

            FROM gravity

            WHERE year BETWEEN 2017 AND 2021
              AND country_exists_o = 1
              AND country_exists_d = 1

            GROUP BY
                year,
                iso3_o,
                iso3_d

            HAVING COUNT(*) > 1

            ORDER BY
                year,
                iso3_o,
                iso3_d
            """
        ).fetchdf()

        recent_duplicates.to_csv(
            OUTPUT_DIR
            / "recent_duplicate_iso3_pairs.csv",
            index=False,
        )

        print(
            "\nDuplicate existing-country ISO3 pair-years:",
            len(recent_duplicates),
        )

        if not recent_duplicates.empty:
            print(recent_duplicates.head(30).to_string(index=False))

        # Do not continue to a pair-level analysis if ISO3 keys are
        # ambiguous in the relevant period.
        if not recent_duplicates.empty:
            raise RuntimeError(
                "Existing-country Gravity rows are not unique by "
                "year, iso3_o, iso3_d during 2017-2021. Inspect "
                "recent_duplicate_iso3_pairs.csv before proceeding."
            )

        # ==============================================================
        # 4. Recent-period time variation
        #
        # Now safe because year-ISO3 pair uniqueness was verified.
        # ==============================================================

        recent_variation = con.execute(
            """
            SELECT
                iso3_o,
                iso3_d,

                COUNT(*) AS n_year_rows,
                COUNT(DISTINCT year) AS n_years,

                MIN(year) AS first_year,
                MAX(year) AS last_year,

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
                    AS maximum_distw_harmonic,

                MAX(distw_harmonic)
                - MIN(distw_harmonic)
                    AS absolute_distw_harmonic_range,

                CASE
                    WHEN MIN(distw_harmonic) > 0
                        THEN (
                            MAX(distw_harmonic)
                            - MIN(distw_harmonic)
                        )
                        / MIN(distw_harmonic)
                    ELSE NULL
                END AS relative_distw_harmonic_range

            FROM gravity

            WHERE year BETWEEN 2017 AND 2021
              AND country_exists_o = 1
              AND country_exists_d = 1
              AND iso3_o <> iso3_d

            GROUP BY
                iso3_o,
                iso3_d

            ORDER BY
                relative_distw_harmonic_range DESC NULLS LAST,
                iso3_o,
                iso3_d
            """
        ).fetchdf()

        recent_variation.to_parquet(
            OUTPUT_DIR
            / "recent_pair_time_variation.parquet",
            index=False,
        )

        con.register(
            "recent_variation_result",
            recent_variation,
        )

        variation_summary = con.execute(
            """
            SELECT
                COUNT(*) AS n_international_pairs,

                COUNT(*) FILTER (
                    WHERE n_years = 5
                ) AS n_pairs_present_all_five_years,

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

                MEDIAN(
                    relative_distw_harmonic_range
                ) FILTER (
                    WHERE n_distw_harmonic_values > 1
                ) AS median_relative_range_among_changed_pairs,

                QUANTILE_CONT(
                    relative_distw_harmonic_range,
                    0.95
                ) FILTER (
                    WHERE n_distw_harmonic_values > 1
                ) AS p95_relative_range_among_changed_pairs,

                MAX(
                    relative_distw_harmonic_range
                ) AS maximum_relative_range

            FROM recent_variation_result
            """
        ).fetchdf()

        variation_summary.to_csv(
            OUTPUT_DIR
            / "recent_time_variation_summary.csv",
            index=False,
        )

        print("\nRecent time-variation summary:")
        print(variation_summary.to_string(index=False))

        # ==============================================================
        # 5. Save the largest recent changes
        # ==============================================================

        largest_changes = con.execute(
            """
            SELECT *
            FROM recent_variation_result

            WHERE n_distw_harmonic_values > 1

            ORDER BY
                relative_distw_harmonic_range DESC NULLS LAST,
                absolute_distw_harmonic_range DESC

            LIMIT 200
            """
        ).fetchdf()

        largest_changes.to_csv(
            OUTPUT_DIR
            / "largest_recent_distw_harmonic_changes.csv",
            index=False,
        )

        # ==============================================================
        # 6. Large-country variation
        # ==============================================================

        large_countries = [
            "CHN",
            "USA",
            "BRA",
            "AUS",
            "CAN",
            "RUS",
            "IDN",
            "IND",
        ]

        placeholders = ", ".join(
            f"'{country}'"
            for country in large_countries
        )

        large_country_variation = con.execute(
            f"""
            SELECT *
            FROM recent_variation_result

            WHERE iso3_o IN ({placeholders})
               OR iso3_d IN ({placeholders})

            ORDER BY
                relative_distw_harmonic_range DESC NULLS LAST,
                iso3_o,
                iso3_d
            """
        ).fetchdf()

        large_country_variation.to_csv(
            OUTPUT_DIR
            / "large_country_recent_variation.csv",
            index=False,
        )

        # ==============================================================
        # 7. Validate the 2021 snapshot
        # ==============================================================

        snapshot_audit = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,

                COUNT(DISTINCT iso3_o)
                    AS n_origins,

                COUNT(DISTINCT iso3_d)
                    AS n_destinations,

                COUNT(
                    DISTINCT iso3_o || '|' || iso3_d
                ) AS n_distinct_pairs,

                COUNT(*) FILTER (
                    WHERE iso3_o = iso3_d
                ) AS n_self_pairs,

                COUNT(*) FILTER (
                    WHERE iso3_o <> iso3_d
                ) AS n_international_pairs,

                COUNT(*) FILTER (
                    WHERE distw_harmonic IS NULL
                       OR distw_harmonic <= 0
                ) AS n_invalid_distw_harmonic,

                COUNT(*) FILTER (
                    WHERE dist IS NULL
                       OR dist <= 0
                ) AS n_invalid_dist,

                COUNT(*) FILTER (
                    WHERE contig NOT IN (0, 1)
                       OR contig IS NULL
                ) AS n_invalid_contig

            FROM gravity

            WHERE year = 2021
              AND country_exists_o = 1
              AND country_exists_d = 1
            """
        ).fetchdf()

        snapshot_audit.to_csv(
            OUTPUT_DIR
            / "gravity_2021_snapshot_audit.csv",
            index=False,
        )

        print("\n2021 snapshot audit:")
        print(snapshot_audit.to_string(index=False))

        audit_row = snapshot_audit.iloc[0]

        if int(audit_row["n_rows"]) != int(
            audit_row["n_distinct_pairs"]
        ):
            raise RuntimeError(
                "The 2021 existing-country subset is not unique "
                "by iso3_o and iso3_d."
            )

        # Self-pair distances may legitimately be zero or missing.
        # The later VACODA bilateral comparison will focus primarily
        # on international pairs.
        invalid_international = con.execute(
            """
            SELECT COUNT(*)
            FROM gravity

            WHERE year = 2021
              AND country_exists_o = 1
              AND country_exists_d = 1
              AND iso3_o <> iso3_d
              AND (
                    distw_harmonic IS NULL
                 OR distw_harmonic <= 0
              )
            """
        ).fetchone()[0]

        if invalid_international > 0:
            raise RuntimeError(
                "The 2021 snapshot contains "
                f"{invalid_international:,} international pairs "
                "with missing or nonpositive distw_harmonic."
            )

        # ==============================================================
        # 8. Write clean 2021 pair snapshot
        # ==============================================================

        con.execute(
            f"""
            COPY (
                SELECT
                    year AS gravity_year,

                    country_id_o,
                    country_id_d,

                    iso3_o AS origin_iso3,
                    iso3_d AS destination_iso3,

                    dist AS gravity_dist_km,
                    distcap AS gravity_distcap_km,

                    distw_harmonic
                        AS gravity_distw_harmonic_km,

                    distw_arithmetic
                        AS gravity_distw_arithmetic_km,

                    distw_harmonic_jh
                        AS gravity_distw_harmonic_jh_km,

                    distw_arithmetic_jh
                        AS gravity_distw_arithmetic_jh_km,

                    contig AS gravity_contiguous

                FROM gravity

                WHERE year = 2021
                  AND country_exists_o = 1
                  AND country_exists_d = 1

                ORDER BY
                    iso3_o,
                    iso3_d
            )
            TO {snapshot_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        print(f"\nSaved clean 2021 snapshot to: {SNAPSHOT_PATH}")

    finally:
        con.close()

    print(f"\nSaved recent-period audit outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()