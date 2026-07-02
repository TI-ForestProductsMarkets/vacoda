"""
prepare_gravity_pair_geography.py

Create an experimental CEPII Gravity v202211 bilateral-geography
reference compatible with VACODA's existing training-feature workflow.

The reference uses the audited 2021 Gravity snapshot:

    distance_model_km = distw_harmonic
    log_distance_model_km = ln(distw_harmonic)
    contiguous = Gravity contig

This script does not modify or overwrite the existing legacy reference:

    geography/prepared/country_pair_geography.parquet
"""

from pathlib import Path

import duckdb


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "reports"
    / "cif_fob"
    / "distance_audit"
    / "gravity_recent"
    / "gravity_2021_pair_snapshot.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "references"
    / "geography"
    / "prepared"
    / "country_pair_geography_gravity_2021.parquet"
)


# ======================================================================
# Helpers
# ======================================================================


def sql_literal(path: Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(path).replace("'", "''") + "'"


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Audited Gravity 2021 snapshot not found: "
            f"{INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_sql = sql_literal(INPUT_PATH)
    output_sql = sql_literal(OUTPUT_PATH)

    con = duckdb.connect()

    try:
        con.execute("SET threads = 16")

        # ==============================================================
        # Validate input schema
        # ==============================================================

        available_columns = {
            row[0]
            for row in con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet({input_sql})
                """
            ).fetchall()
        }

        required_columns = {
            "gravity_year",
            "origin_iso3",
            "destination_iso3",
            "gravity_dist_km",
            "gravity_distcap_km",
            "gravity_distw_harmonic_km",
            "gravity_distw_arithmetic_km",
            "gravity_contiguous",
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                "Gravity snapshot is missing required columns: "
                + ", ".join(missing_columns)
            )

        # ==============================================================
        # Audit source
        # ==============================================================

        source_audit = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                COUNT(
                    DISTINCT
                    origin_iso3 || '|' || destination_iso3
                ) AS n_distinct_pairs,

                COUNT(DISTINCT origin_iso3)
                    AS n_origins,

                COUNT(DISTINCT destination_iso3)
                    AS n_destinations,

                COUNT(*) FILTER (
                    WHERE origin_iso3 IS NULL
                       OR destination_iso3 IS NULL
                ) AS n_null_pair_keys,

                COUNT(*) FILTER (
                    WHERE gravity_distw_harmonic_km IS NULL
                       OR gravity_distw_harmonic_km <= 0
                ) AS n_invalid_model_distances,

                COUNT(*) FILTER (
                    WHERE gravity_contiguous IS NULL
                       OR gravity_contiguous NOT IN (0, 1)
                ) AS n_invalid_contiguous,

                COUNT(*) FILTER (
                    WHERE gravity_year <> 2021
                       OR gravity_year IS NULL
                ) AS n_non_2021_rows

            FROM read_parquet({input_sql})
            """
        ).fetchone()

        if source_audit is None:
            raise RuntimeError(
                "Could not audit Gravity snapshot."
            )

        (
            n_rows,
            n_distinct_pairs,
            n_origins,
            n_destinations,
            n_null_pair_keys,
            n_invalid_model_distances,
            n_invalid_contiguous,
            n_non_2021_rows,
        ) = source_audit

        print("\nSource audit:")
        print(f"Rows:                     {n_rows:,}")
        print(f"Distinct pairs:           {n_distinct_pairs:,}")
        print(f"Origins:                  {n_origins:,}")
        print(f"Destinations:             {n_destinations:,}")
        print(f"Null pair keys:           {n_null_pair_keys:,}")
        print(
            "Invalid harmonic distance:",
            f"{n_invalid_model_distances:,}",
        )
        print(
            "Invalid contiguity:       ",
            f"{n_invalid_contiguous:,}",
        )
        print(f"Non-2021 rows:             {n_non_2021_rows:,}")

        if n_rows != n_distinct_pairs:
            raise RuntimeError(
                "Gravity snapshot contains duplicate ISO3 pairs: "
                f"rows={n_rows:,}, distinct_pairs={n_distinct_pairs:,}."
            )

        if n_null_pair_keys > 0:
            raise RuntimeError(
                "Gravity snapshot contains null country-pair keys."
            )

        if n_invalid_model_distances > 0:
            raise RuntimeError(
                "Gravity snapshot contains missing or nonpositive "
                "distw_harmonic values."
            )

        if n_invalid_contiguous > 0:
            raise RuntimeError(
                "Gravity snapshot contains invalid contiguity values."
            )

        if n_non_2021_rows > 0:
            raise RuntimeError(
                "Gravity snapshot contains rows outside 2021."
            )

        # ==============================================================
        # Create VACODA-compatible pair reference
        #
        # Compatibility mapping:
        #
        # distance_simple_km
        #     Gravity dist
        #
        # distance_capital_km
        #     Gravity distcap
        #
        # distance_weighted_km
        #     Gravity distw_harmonic
        #
        # distance_weighted_ces_km
        #     NULL because Gravity does not provide the legacy distwces
        #
        # distance_model_km
        #     Gravity distw_harmonic
        #
        # distance_model_source
        #     Explicit Gravity provenance
        # ==============================================================

        con.execute(
            f"""
            COPY (
                SELECT
                    origin_iso3,
                    destination_iso3,

                    CAST(
                        gravity_dist_km
                        AS DOUBLE
                    ) AS distance_simple_km,

                    CAST(
                        gravity_distcap_km
                        AS DOUBLE
                    ) AS distance_capital_km,

                    CAST(
                        gravity_distw_harmonic_km
                        AS DOUBLE
                    ) AS distance_weighted_km,

                    CAST(
                        NULL
                        AS DOUBLE
                    ) AS distance_weighted_ces_km,

                    CAST(
                        gravity_contiguous
                        AS TINYINT
                    ) AS contiguous,

                    CAST(
                        gravity_distw_harmonic_km
                        AS DOUBLE
                    ) AS distance_model_km,

                    CAST(
                        'gravity_2021_distw_harmonic'
                        AS VARCHAR
                    ) AS distance_model_source,

                    CASE
                        WHEN gravity_distw_harmonic_km > 0
                            THEN LN(
                                CAST(
                                    gravity_distw_harmonic_km
                                    AS DOUBLE
                                )
                            )
                        ELSE NULL
                    END AS log_distance_model_km,

                    -- Explicit source metadata retained for auditing
                    CAST(
                        gravity_year
                        AS INTEGER
                    ) AS gravity_reference_year,

                    CAST(
                        gravity_distw_harmonic_km
                        AS DOUBLE
                    ) AS gravity_distw_harmonic_km,

                    CAST(
                        gravity_distw_arithmetic_km
                        AS DOUBLE
                    ) AS gravity_distw_arithmetic_km

                FROM read_parquet({input_sql})

                ORDER BY
                    origin_iso3,
                    destination_iso3
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        # ==============================================================
        # Validate prepared output
        # ==============================================================

        output_audit = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                COUNT(
                    DISTINCT
                    origin_iso3 || '|' || destination_iso3
                ) AS n_distinct_pairs,

                COUNT(*) FILTER (
                    WHERE distance_model_km IS NULL
                       OR distance_model_km <= 0
                ) AS n_invalid_model_distances,

                COUNT(*) FILTER (
                    WHERE log_distance_model_km IS NULL
                ) AS n_missing_log_distances,

                COUNT(*) FILTER (
                    WHERE contiguous IS NULL
                       OR contiguous NOT IN (0, 1)
                ) AS n_invalid_contiguous,

                COUNT(*) FILTER (
                    WHERE distance_model_source
                          <> 'gravity_2021_distw_harmonic'
                ) AS n_invalid_sources,

                COUNT(*) FILTER (
                    WHERE distance_weighted_km
                          <> distance_model_km
                ) AS n_weighted_model_mismatches,

                MIN(distance_model_km)
                    AS minimum_model_distance_km,

                MEDIAN(distance_model_km)
                    AS median_model_distance_km,

                MAX(distance_model_km)
                    AS maximum_model_distance_km

            FROM read_parquet({output_sql})
            """
        ).fetchone()

        if output_audit is None:
            raise RuntimeError(
                "Could not audit prepared Gravity reference."
            )

        (
            output_rows,
            output_distinct_pairs,
            output_invalid_distances,
            output_missing_logs,
            output_invalid_contiguous,
            output_invalid_sources,
            output_weighted_model_mismatches,
            minimum_distance,
            median_distance,
            maximum_distance,
        ) = output_audit

        print("\nPrepared-reference audit:")
        print(f"Rows:                     {output_rows:,}")
        print(
            f"Distinct pairs:           "
            f"{output_distinct_pairs:,}"
        )
        print(
            f"Invalid model distances:  "
            f"{output_invalid_distances:,}"
        )
        print(
            f"Missing log distances:    "
            f"{output_missing_logs:,}"
        )
        print(
            f"Invalid contiguity:        "
            f"{output_invalid_contiguous:,}"
        )
        print(
            f"Invalid source labels:     "
            f"{output_invalid_sources:,}"
        )
        print(
            f"Weighted/model mismatch:  "
            f"{output_weighted_model_mismatches:,}"
        )
        print(
            f"Minimum distance:          "
            f"{minimum_distance:,.3f} km"
        )
        print(
            f"Median distance:           "
            f"{median_distance:,.3f} km"
        )
        print(
            f"Maximum distance:          "
            f"{maximum_distance:,.3f} km"
        )

        if output_rows != output_distinct_pairs:
            raise RuntimeError(
                "Prepared Gravity reference contains duplicate pairs."
            )

        invalid_output_counts = {
            "invalid model distances": output_invalid_distances,
            "missing log distances": output_missing_logs,
            "invalid contiguity": output_invalid_contiguous,
            "invalid source labels": output_invalid_sources,
            "weighted/model mismatches": (
                output_weighted_model_mismatches
            ),
        }

        failures = {
            label: int(value)
            for label, value in invalid_output_counts.items()
            if int(value or 0) > 0
        }

        if failures:
            details = ", ".join(
                f"{label}={value:,}"
                for label, value in failures.items()
            )

            raise RuntimeError(
                "Prepared Gravity reference failed validation: "
                + details
            )

        print("\nPreview:")
        preview = con.execute(
            f"""
            SELECT *
            FROM read_parquet({output_sql})
            ORDER BY
                origin_iso3,
                destination_iso3
            LIMIT 10
            """
        ).fetchdf()

        print(preview.to_string(index=False))

    finally:
        con.close()

    print(f"\nSaved Gravity pair geography to:\n{OUTPUT_PATH}")
    print(
        "\nExisting legacy pair geography was not modified."
    )


if __name__ == "__main__":
    main()