"""
Prepare the CEPII Gravity v202211 bilateral geography reference.

This module reads the official CEPII Gravity CSV directly, selects one
reference year, validates the bilateral geography fields, and writes a
VACODA-compatible Parquet file.

No VACODA-authored crosswalk logic belongs in this module.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


REQUIRED_COLUMNS = {
    "year",
    "iso3_o",
    "iso3_d",
    "country_exists_o",
    "country_exists_d",
    "dist",
    "distcap",
    "distw_harmonic",
    "distw_arithmetic",
    "contig",
}


def _sql_literal(path: Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(path).replace("'", "''") + "'"


def prepare_gravity_pair_geography(
    input_path: Path,
    output_path: Path,
    *,
    reference_year: int = 2021,
    threads: int = 16,
    verbose: bool = True,
) -> None:
    """
    Prepare a VACODA-compatible CEPII Gravity bilateral geography file.

    Parameters
    ----------
    input_path:
        Official ``Gravity_V202211.csv`` source file.
    output_path:
        Destination Parquet file.
    reference_year:
        Gravity snapshot year to retain.
    threads:
        DuckDB thread count.
    verbose:
        Print diagnostics when true.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"CEPII Gravity source CSV not found: {input_path}"
        )

    if threads <= 0:
        raise ValueError("threads must be greater than zero.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_sql = _sql_literal(input_path)
    output_sql = _sql_literal(output_path)

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        available_columns = {
            str(row[0])
            for row in con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_csv_auto(
                    {input_sql},
                    HEADER=TRUE,
                    SAMPLE_SIZE=-1
                )
                """
            ).fetchall()
        }

        missing_columns = sorted(
            REQUIRED_COLUMNS - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                "CEPII Gravity source is missing required columns: "
                + ", ".join(missing_columns)
            )

        source_audit = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE TRY_CAST(year AS INTEGER) = {int(reference_year)}
                    AND TRY_CAST(country_exists_o AS INTEGER) = 1
                    AND TRY_CAST(country_exists_d AS INTEGER) = 1
                ) AS n_reference_year_rows,

                COUNT(*) FILTER (
                    WHERE TRY_CAST(year AS INTEGER) = {int(reference_year)}
                      AND TRY_CAST(country_exists_o AS INTEGER) = 1
                      AND TRY_CAST(country_exists_d AS INTEGER) = 1
                      AND (
                          iso3_o IS NULL
                          OR TRIM(CAST(iso3_o AS VARCHAR)) = ''
                          OR iso3_d IS NULL
                          OR TRIM(CAST(iso3_d AS VARCHAR)) = ''
                      )
                ) AS n_null_pair_keys,

                COUNT(*) FILTER (
                    WHERE TRY_CAST(year AS INTEGER) = {int(reference_year)}
                      AND TRY_CAST(country_exists_o AS INTEGER) = 1
                      AND TRY_CAST(country_exists_d AS INTEGER) = 1
                      AND (
                          TRY_CAST(distw_harmonic AS DOUBLE) IS NULL
                          OR TRY_CAST(distw_harmonic AS DOUBLE) <= 0
                      )
                ) AS n_invalid_model_distances,

                COUNT(*) FILTER (
                    WHERE TRY_CAST(year AS INTEGER) = {int(reference_year)}
                      AND TRY_CAST(country_exists_o AS INTEGER) = 1
                      AND TRY_CAST(country_exists_d AS INTEGER) = 1
                      AND (
                          TRY_CAST(contig AS INTEGER) IS NULL
                          OR TRY_CAST(contig AS INTEGER) NOT IN (0, 1)
                      )
                ) AS n_invalid_contiguous,

                COUNT(*) FILTER (
                    WHERE TRY_CAST(year AS INTEGER) = {int(reference_year)}
                      AND (
                      TRY_CAST(country_exists_o AS INTEGER) <> 1
                      OR TRY_CAST(country_exists_d AS INTEGER) <> 1
                      )
                    ) AS n_excluded_nonexistent_country_rows

            FROM read_csv_auto(
                {input_sql},
                HEADER=TRUE,
                SAMPLE_SIZE=-1
            )
            """
        ).fetchone()

        if source_audit is None:
            raise RuntimeError(
                "Could not audit CEPII Gravity source."
            )

        (
            n_reference_year_rows,
            n_null_pair_keys,
            n_invalid_model_distances,
            n_invalid_contiguous,
            n_excluded_nonexistent_country_rows
        ) = source_audit

        if n_reference_year_rows == 0:
            raise RuntimeError(
                f"No CEPII Gravity rows found for {reference_year}."
            )
        if verbose:
            print(f"Excluded nonexistent country rows: {n_excluded_nonexistent_country_rows}")
            
        source_failures = {
            "null country-pair keys": n_null_pair_keys,
            "invalid harmonic distances": (
                n_invalid_model_distances
            ),
            "invalid contiguity values": n_invalid_contiguous,
        }

        active_source_failures = {
            label: int(value)
            for label, value in source_failures.items()
            if int(value or 0) > 0
        }

        if active_source_failures:
            details = ", ".join(
                f"{label}={value:,}"
                for label, value
                in active_source_failures.items()
            )
            raise RuntimeError(
                "CEPII Gravity source validation failed: "
                + details
            )

        con.execute(
            f"""
            COPY (
                SELECT
                    UPPER(TRIM(CAST(iso3_o AS VARCHAR)))
                        AS origin_iso3,

                    UPPER(TRIM(CAST(iso3_d AS VARCHAR)))
                        AS destination_iso3,

                    TRY_CAST(dist AS DOUBLE)
                        AS distance_simple_km,

                    TRY_CAST(distcap AS DOUBLE)
                        AS distance_capital_km,

                    TRY_CAST(distw_harmonic AS DOUBLE)
                        AS distance_weighted_km,

                    CAST(NULL AS DOUBLE)
                        AS distance_weighted_ces_km,

                    CAST(
                        TRY_CAST(contig AS INTEGER)
                        AS TINYINT
                    ) AS contiguous,

                    TRY_CAST(distw_harmonic AS DOUBLE)
                        AS distance_model_km,

                    CAST(
                        'gravity_2021_distw_harmonic'
                        AS VARCHAR
                    ) AS distance_model_source,

                    LN(
                        TRY_CAST(distw_harmonic AS DOUBLE)
                    ) AS log_distance_model_km,

                    CAST(
                        TRY_CAST(year AS INTEGER)
                        AS INTEGER
                    ) AS gravity_reference_year,

                    TRY_CAST(distw_harmonic AS DOUBLE)
                        AS gravity_distw_harmonic_km,

                    TRY_CAST(distw_arithmetic AS DOUBLE)
                        AS gravity_distw_arithmetic_km

                FROM read_csv_auto(
                    {input_sql},
                    HEADER=TRUE,
                    SAMPLE_SIZE=-1
                )

                WHERE TRY_CAST(year AS INTEGER)
                    = {int(reference_year)}
                    AND TRY_CAST(country_exists_o AS INTEGER) = 1
                    AND TRY_CAST(country_exists_d AS INTEGER) = 1

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

        output_audit = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                COUNT(
                    DISTINCT
                    origin_iso3
                    || '|'
                    || destination_iso3
                ) AS n_distinct_pairs,

                COUNT(*) FILTER (
                    WHERE origin_iso3 IS NULL
                       OR destination_iso3 IS NULL
                ) AS n_null_pair_keys,

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
                    WHERE gravity_reference_year
                          <> {int(reference_year)}
                ) AS n_wrong_reference_year,

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
            n_rows,
            n_distinct_pairs,
            n_output_null_pair_keys,
            n_output_invalid_model_distances,
            n_missing_log_distances,
            n_output_invalid_contiguous,
            n_wrong_reference_year,
            n_weighted_model_mismatches,
            minimum_model_distance_km,
            median_model_distance_km,
            maximum_model_distance_km,
        ) = output_audit

        output_failures = {
            "duplicate country pairs": (
                n_rows - n_distinct_pairs
            ),
            "null country-pair keys": (
                n_output_null_pair_keys
            ),
            "invalid model distances": (
                n_output_invalid_model_distances
            ),
            "missing log distances": (
                n_missing_log_distances
            ),
            "invalid contiguity values": (
                n_output_invalid_contiguous
            ),
            "wrong reference year": (
                n_wrong_reference_year
            ),
            "weighted/model distance mismatches": (
                n_weighted_model_mismatches
            ),
        }

        active_output_failures = {
            label: int(value)
            for label, value in output_failures.items()
            if int(value or 0) > 0
        }

        if active_output_failures:
            details = ", ".join(
                f"{label}={value:,}"
                for label, value
                in active_output_failures.items()
            )
            raise RuntimeError(
                "Prepared CEPII Gravity reference failed "
                "validation: "
                + details
            )

        if verbose:
            print("\nPrepared CEPII Gravity pair geography")
            print(f"Reference year:  {reference_year}")
            print(f"Rows:            {n_rows:,}")
            print(f"Distinct pairs:  {n_distinct_pairs:,}")
            print(
                "Distance range:  "
                f"{minimum_model_distance_km:,.3f}–"
                f"{maximum_model_distance_km:,.3f} km"
            )
            print(
                "Median distance: "
                f"{median_model_distance_km:,.3f} km"
            )
            print(f"Saved to:        {output_path}")

    finally:
        con.close()
