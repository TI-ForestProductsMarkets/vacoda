"""
Create a Gravity-2021-compatible Comtrade-to-CEPII crosswalk.

The production legacy crosswalk is left unchanged.

The following legacy mappings are replaced by modern identity mappings
because these modern ISO3 codes exist directly in Gravity 2021:

    MNE -> MNE
    SRB -> SRB
    COD -> COD
    ROU -> ROU
    TLS -> TLS
"""

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_CROSSWALK_PATH = (
    PROJECT_ROOT
    / "references"
    / "geography"
    / "prepared"
    / "comtrade_cepii_crosswalk.parquet"
)

GRAVITY_PAIR_PATH = (
    PROJECT_ROOT
    / "references"
    / "geography"
    / "prepared"
    / "country_pair_geography_gravity_2021.parquet"
)

OUTPUT_CROSSWALK_PATH = (
    PROJECT_ROOT
    / "references"
    / "geography"
    / "prepared"
    / "comtrade_cepii_crosswalk_gravity_2021.parquet"
)

IDENTITY_OVERRIDES = {
    "COD",
    "MNE",
    "ROU",
    "SRB",
    "TLS",
}


def sql_literal(path: Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(path).replace("'", "''") + "'"


def main() -> None:
    for label, path in [
        ("Existing geography crosswalk", INPUT_CROSSWALK_PATH),
        ("Gravity pair geography", GRAVITY_PAIR_PATH),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    OUTPUT_CROSSWALK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_sql = sql_literal(INPUT_CROSSWALK_PATH)
    gravity_sql = sql_literal(GRAVITY_PAIR_PATH)
    output_sql = sql_literal(OUTPUT_CROSSWALK_PATH)

    override_sql = ", ".join(
        f"'{code}'"
        for code in sorted(IDENTITY_OVERRIDES)
    )

    con = duckdb.connect()

    try:
        con.execute("SET threads = 16")

        # ============================================================
        # Confirm required columns exist in the current crosswalk
        # ============================================================

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
            "trade_iso3",
            "cepii_iso3",
            "mapping_type",
            "usable_country",
            "usable_pair",
            "is_proxy",
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                "Existing crosswalk is missing required columns: "
                + ", ".join(missing_columns)
            )

        # ============================================================
        # Confirm all five modern ISO3 codes exist in Gravity
        # ============================================================

        missing_gravity_codes = con.execute(
            f"""
            WITH gravity_countries AS (
                SELECT origin_iso3 AS iso3
                FROM read_parquet({gravity_sql})

                UNION

                SELECT destination_iso3 AS iso3
                FROM read_parquet({gravity_sql})
            ),

            required_codes AS (
                SELECT *
                FROM (
                    VALUES
                        ('COD'),
                        ('MNE'),
                        ('ROU'),
                        ('SRB'),
                        ('TLS')
                ) AS t(iso3)
            )

            SELECT
                r.iso3

            FROM required_codes AS r

            LEFT JOIN gravity_countries AS g
              ON r.iso3 = g.iso3

            WHERE g.iso3 IS NULL

            ORDER BY r.iso3
            """
        ).fetchdf()

        if not missing_gravity_codes.empty:
            raise RuntimeError(
                "Some identity override codes are absent from "
                "Gravity 2021:\n"
                + missing_gravity_codes.to_string(index=False)
            )

        # ============================================================
        # Show the legacy mappings being replaced
        # ============================================================

        legacy_rows = con.execute(
            f"""
            SELECT
                trade_iso3,
                cepii_iso3,
                mapping_type,
                usable_country,
                usable_pair,
                is_proxy

            FROM read_parquet({input_sql})

            WHERE trade_iso3 IN ({override_sql})

            ORDER BY trade_iso3
            """
        ).fetchdf()

        print("\nLegacy mappings to replace:")
        print(legacy_rows.to_string(index=False))

        found_override_codes = set(
            legacy_rows["trade_iso3"].tolist()
        )

        missing_crosswalk_codes = sorted(
            IDENTITY_OVERRIDES - found_override_codes
        )

        if missing_crosswalk_codes:
            raise RuntimeError(
                "The current crosswalk does not contain the expected "
                "rows: "
                + ", ".join(missing_crosswalk_codes)
            )

        # ============================================================
        # Create the Gravity-specific crosswalk
        # ============================================================

        con.execute(
            f"""
            COPY (
                SELECT
                    trade_iso3,

                    CASE
                        WHEN trade_iso3 IN ({override_sql})
                            THEN trade_iso3
                        ELSE cepii_iso3
                    END AS cepii_iso3,

                    CASE
                        WHEN trade_iso3 IN ({override_sql})
                            THEN 'gravity_2021_identity'
                        ELSE mapping_type
                    END AS mapping_type,

                    usable_country,
                    usable_pair,

                    CASE
                        WHEN trade_iso3 IN ({override_sql})
                            THEN FALSE
                        ELSE is_proxy
                    END AS is_proxy

                FROM read_parquet({input_sql})

                ORDER BY trade_iso3
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        # ============================================================
        # Validate uniqueness and Gravity coverage
        # ============================================================

        audit = con.execute(
            f"""
            WITH gravity_countries AS (
                SELECT origin_iso3 AS iso3
                FROM read_parquet({gravity_sql})

                UNION

                SELECT destination_iso3 AS iso3
                FROM read_parquet({gravity_sql})
            )

            SELECT
                COUNT(*) AS n_rows,

                COUNT(DISTINCT trade_iso3)
                    AS n_distinct_trade_iso3,

                COUNT(*) FILTER (
                    WHERE trade_iso3 IS NULL
                       OR mapping_type IS NULL
                ) AS n_null_required,

                COUNT(*) FILTER (
                    WHERE usable_country IS NULL
                       OR usable_pair IS NULL
                       OR is_proxy IS NULL
                ) AS n_null_flags,

                COUNT(*) FILTER (
                    WHERE (usable_country OR usable_pair)
                      AND cepii_iso3 IS NULL
                ) AS n_usable_without_target,

                COUNT(*) FILTER (
                    WHERE usable_pair
                      AND NOT is_proxy
                      AND g.iso3 IS NULL
                ) AS n_nonproxy_pair_targets_missing_from_gravity,

                COUNT(*) FILTER (
                    WHERE trade_iso3 IN ({override_sql})
                      AND cepii_iso3 = trade_iso3
                      AND mapping_type = 'gravity_2021_identity'
                      AND NOT is_proxy
                ) AS n_valid_identity_overrides

            FROM read_parquet({output_sql}) AS x

            LEFT JOIN gravity_countries AS g
              ON x.cepii_iso3 = g.iso3
            """
        ).fetchone()

        if audit is None:
            raise RuntimeError(
                "Could not audit the prepared Gravity crosswalk."
            )

        (
            n_rows,
            n_distinct_trade_iso3,
            n_null_required,
            n_null_flags,
            n_usable_without_target,
            n_missing_gravity_targets,
            n_valid_identity_overrides,
        ) = audit

        print("\nPrepared Gravity crosswalk audit:")
        print(f"Rows:                              {n_rows:,}")
        print(
            f"Distinct trade ISO3:               "
            f"{n_distinct_trade_iso3:,}"
        )
        print(f"Null required fields:              {n_null_required:,}")
        print(f"Null flags:                        {n_null_flags:,}")
        print(
            f"Usable rows without target:        "
            f"{n_usable_without_target:,}"
        )
        print(
            f"Nonproxy targets missing Gravity:  "
            f"{n_missing_gravity_targets:,}"
        )
        print(
            f"Valid identity overrides:          "
            f"{n_valid_identity_overrides:,}"
        )

        failures = {
            "duplicate trade ISO3 keys": (
                n_rows - n_distinct_trade_iso3
            ),
            "null required fields": n_null_required,
            "null flags": n_null_flags,
            "usable rows without target": n_usable_without_target,
            "nonproxy targets missing from Gravity": (
                n_missing_gravity_targets
            ),
        }

        active_failures = {
            label: int(value)
            for label, value in failures.items()
            if int(value or 0) > 0
        }

        if active_failures:
            details = ", ".join(
                f"{label}={value:,}"
                for label, value in active_failures.items()
            )

            raise RuntimeError(
                "Gravity crosswalk validation failed: "
                + details
            )

        if n_valid_identity_overrides != len(
            IDENTITY_OVERRIDES
        ):
            raise RuntimeError(
                "Not all five Gravity identity overrides were created."
            )

        preview = con.execute(
            f"""
            SELECT
                trade_iso3,
                cepii_iso3,
                mapping_type,
                usable_country,
                usable_pair,
                is_proxy

            FROM read_parquet({output_sql})

            WHERE trade_iso3 IN ({override_sql})

            ORDER BY trade_iso3
            """
        ).fetchdf()

        print("\nGravity-specific mappings:")
        print(preview.to_string(index=False))

    finally:
        con.close()

    print(
        "\nSaved Gravity-specific crosswalk to:\n"
        f"{OUTPUT_CROSSWALK_PATH}"
    )

    print(
        "\nThe existing production crosswalk was not modified."
    )


if __name__ == "__main__":
    main()