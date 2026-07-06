#!/usr/bin/env python3
"""
Second-stage CIF/FOB geography audit.

Purpose
-------
Take the problematic direct-factor rows identified by the first audit and
check whether their exporter/importer codes and bilateral pairs exist in:

1. country geography crosswalk
2. country geography reference
3. pair geography crosswalk
4. bilateral pair geography reference

Machine-readable outputs are written under:

    /storage/vacoda/data/interim/cif_fob/target_<hs>/geography_audit/reference_resolution/

No Markdown report is written.

Expected reference schemas
--------------------------
Country and pair crosswalks:
    trade_iso3
    cepii_iso3
    mapping_type
    usable_country
    usable_pair
    is_proxy

Country geography:
    country_iso3
    area_km2
    internal_distance_km
    landlocked
    continent
    log_area_km2
    log_internal_distance_km

Pair geography:
    origin_iso3
    destination_iso3
    distance_model_km
    distance_model_source
    log_distance_model_km
    contiguous
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb


DEFAULT_STORAGE_ROOT = Path("/storage/vacoda")
DEFAULT_PROJECT_ROOT = Path("/home/saxena/Projects/vacoda")
DEFAULT_THREADS = 16

CROSSWALK_REQUIRED = {
    "trade_iso3",
    "cepii_iso3",
    "mapping_type",
    "usable_country",
    "usable_pair",
    "is_proxy",
}

COUNTRY_REQUIRED = {
    "country_iso3",
    "area_km2",
    "internal_distance_km",
    "landlocked",
    "continent",
    "log_area_km2",
    "log_internal_distance_km",
}

PAIR_REQUIRED = {
    "origin_iso3",
    "destination_iso3",
    "distance_model_km",
    "distance_model_source",
    "log_distance_model_km",
    "contiguous",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve missing CIF/FOB geography codes and pairs against "
            "country and bilateral geography references."
        )
    )
    parser.add_argument(
        "--target-hs",
        default="H5",
        help="Target HS revision, for example H5.",
    )
    parser.add_argument(
        "--failure-rows",
        type=Path,
        default=None,
        help=(
            "Row-level Parquet from the first geography audit. Default: "
            ".../geography_audit/direct_geography_failure_rows.parquet"
        ),
    )
    parser.add_argument(
        "--country-crosswalk",
        type=Path,
        default=None,
        help="Country-geography crosswalk Parquet.",
    )
    parser.add_argument(
        "--pair-crosswalk",
        type=Path,
        default=None,
        help="Pair-geography crosswalk Parquet.",
    )
    parser.add_argument(
        "--country-geography",
        type=Path,
        default=None,
        help="Country geography reference Parquet.",
    )
    parser.add_argument(
        "--pair-geography",
        type=Path,
        default=None,
        help="Bilateral pair geography reference Parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Machine-readable output directory. Default: "
            ".../geography_audit/reference_resolution/"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="DuckDB thread count.",
    )
    parser.add_argument(
        "--allow-country-proxies",
        action="store_true",
        help=(
            "Treat proxy country-crosswalk rows as usable. By default "
            "proxy rows are classified but not treated as active."
        ),
    )
    parser.add_argument(
        "--allow-pair-proxies",
        action="store_true",
        help=(
            "Treat proxy pair-crosswalk rows as usable. By default "
            "proxy rows are classified but not treated as active."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("CIF_FOB_REFERENCE_RESOLUTION_AUDIT")


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def resolve_paths(
    *,
    target_hs: str,
    failure_rows: Path | None,
    country_crosswalk: Path | None,
    pair_crosswalk: Path | None,
    country_geography: Path | None,
    pair_geography: Path | None,
    output_dir: Path | None,
) -> dict[str, Path]:
    target = target_hs.strip().lower()

    cif_fob_dir = (
        DEFAULT_STORAGE_ROOT
        / "data"
        / "interim"
        / "cif_fob"
        / f"target_{target}"
    )

    reference_dir = (
        DEFAULT_PROJECT_ROOT
        / "references"
        / "geography"
        / "prepared"
    )

    return {
        "failure_rows": failure_rows
        or (
            cif_fob_dir
            / "geography_audit"
            / "direct_geography_failure_rows.parquet"
        ),
        "country_crosswalk": country_crosswalk
        or (
            reference_dir
            / "comtrade_cepii_crosswalk.parquet"
        ),
        "pair_crosswalk": pair_crosswalk
        or (
            reference_dir
            / "comtrade_cepii_crosswalk_gravity_2021.parquet"
        ),
        "country_geography": country_geography
        or (
            reference_dir
            / "country_geography.parquet"
        ),
        "pair_geography": pair_geography
        or (
            reference_dir
            / "country_pair_geography_gravity_2021.parquet"
        ),
        "output_dir": output_dir
        or (
            cif_fob_dir
            / "geography_audit"
            / "reference_resolution"
        ),
    }


def read_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> set[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_literal(path)})"
    ).fetchall()
    return {str(row[0]) for row in rows}


def require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    path: Path,
    required: set[str],
    label: str,
) -> None:
    available = read_columns(con, path)
    missing = sorted(required - available)

    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
            + f". Path: {path}"
        )


def validate_inputs(
    *,
    con: duckdb.DuckDBPyConnection,
    paths: dict[str, Path],
) -> None:
    for key, path in paths.items():
        if key == "output_dir":
            continue
        if not path.is_file():
            raise FileNotFoundError(f"{key} not found: {path}")

    require_columns(
        con=con,
        path=paths["failure_rows"],
        required={
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "direct_evidence_value_usd",
            "observed_cif_fob_factor",
            "pair_geography_missing",
            "exporter_geography_missing",
            "importer_geography_missing",
        },
        label="First-audit failure rows",
    )
    require_columns(
        con=con,
        path=paths["country_crosswalk"],
        required=CROSSWALK_REQUIRED,
        label="Country crosswalk",
    )
    require_columns(
        con=con,
        path=paths["pair_crosswalk"],
        required=CROSSWALK_REQUIRED,
        label="Pair crosswalk",
    )
    require_columns(
        con=con,
        path=paths["country_geography"],
        required=COUNTRY_REQUIRED,
        label="Country geography",
    )
    require_columns(
        con=con,
        path=paths["pair_geography"],
        required=PAIR_REQUIRED,
        label="Pair geography",
    )


def ensure_outputs(
    *,
    output_dir: Path,
    filenames: Iterable[str],
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        output_dir / name
        for name in filenames
        if (output_dir / name).exists()
    ]

    if existing and not force:
        raise FileExistsError(
            "Reference-resolution outputs already exist. "
            "Use --force to overwrite:\n"
            + "\n".join(f"  {path}" for path in existing)
        )


def copy_query(
    con: duckdb.DuckDBPyConnection,
    *,
    query: str,
    output_path: Path,
    format_name: str,
) -> None:
    if format_name == "CSV":
        options = "FORMAT CSV, HEADER TRUE"
    elif format_name == "PARQUET":
        options = "FORMAT PARQUET, COMPRESSION ZSTD"
    else:
        raise ValueError(f"Unsupported format: {format_name}")

    con.execute(
        f"""
        COPY (
            {query}
        )
        TO {sql_literal(output_path)}
        ({options})
        """
    )


def build_audit(
    *,
    paths: dict[str, Path],
    threads: int,
    allow_country_proxies: bool,
    allow_pair_proxies: bool,
    force: bool,
    logger: logging.Logger,
) -> dict[str, object]:
    output_dir = paths["output_dir"]

    filenames = [
        "country_code_resolution.csv",
        "pair_resolution.csv",
        "row_level_reference_resolution.parquet",
        "resolution_class_summary.csv",
        "country_resolution_summary.csv",
        "pair_resolution_summary.csv",
        "reference_resolution_manifest.json",
    ]

    ensure_outputs(
        output_dir=output_dir,
        filenames=filenames,
        force=force,
    )

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")
        validate_inputs(con=con, paths=paths)

        failures = sql_literal(paths["failure_rows"])
        country_xwalk = sql_literal(paths["country_crosswalk"])
        pair_xwalk = sql_literal(paths["pair_crosswalk"])
        country_geo = sql_literal(paths["country_geography"])
        pair_geo = sql_literal(paths["pair_geography"])

        logger.info("Failure rows: %s", paths["failure_rows"])
        logger.info(
            "Country crosswalk: %s",
            paths["country_crosswalk"],
        )
        logger.info(
            "Pair crosswalk: %s",
            paths["pair_crosswalk"],
        )
        logger.info(
            "Country geography: %s",
            paths["country_geography"],
        )
        logger.info(
            "Pair geography: %s",
            paths["pair_geography"],
        )
        logger.info("Output directory: %s", output_dir)

        country_code_query = f"""
            WITH problematic_codes AS (
                SELECT
                    TRIM(exporter_iso) AS trade_iso3,
                    'exporter' AS role,
                    COUNT(*) AS n_rows,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd
                FROM read_parquet({failures})
                WHERE exporter_geography_missing
                GROUP BY 1

                UNION ALL

                SELECT
                    TRIM(importer_iso) AS trade_iso3,
                    'importer' AS role,
                    COUNT(*) AS n_rows,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd
                FROM read_parquet({failures})
                WHERE importer_geography_missing
                GROUP BY 1
            ),
            codes AS (
                SELECT
                    trade_iso3,
                    STRING_AGG(
                        DISTINCT role,
                        ','
                        ORDER BY role
                    ) AS affected_roles,
                    SUM(n_rows) AS n_affected_role_rows,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd
                FROM problematic_codes
                GROUP BY trade_iso3
            )
            SELECT
                c.trade_iso3,
                c.affected_roles,
                c.n_affected_role_rows,
                c.direct_evidence_value_usd,

                cx.trade_iso3 IS NOT NULL
                    AS country_crosswalk_found,
                cx.cepii_iso3
                    AS country_cepii_iso3,
                cx.mapping_type
                    AS country_mapping_type,
                cx.usable_country
                    AS country_usable_country,
                cx.usable_pair
                    AS country_usable_pair,
                cx.is_proxy
                    AS country_is_proxy,

                (
                    cx.trade_iso3 IS NOT NULL
                    AND cx.usable_country
                    AND (
                        NOT cx.is_proxy
                        OR {str(bool(allow_country_proxies)).upper()}
                    )
                    AND cx.cepii_iso3 IS NOT NULL
                ) AS country_crosswalk_active,

                cg.country_iso3 IS NOT NULL
                    AS country_geography_row_found,
                cg.area_km2,
                cg.internal_distance_km,
                cg.landlocked,
                cg.continent,
                cg.log_area_km2,
                cg.log_internal_distance_km,

                (
                    cg.country_iso3 IS NOT NULL
                    AND cg.landlocked IS NOT NULL
                    AND cg.area_km2 IS NOT NULL
                    AND cg.internal_distance_km IS NOT NULL
                ) AS country_required_fields_complete,

                px.trade_iso3 IS NOT NULL
                    AS pair_crosswalk_found,
                px.cepii_iso3
                    AS pair_cepii_iso3,
                px.mapping_type
                    AS pair_mapping_type,
                px.usable_country
                    AS pair_usable_country,
                px.usable_pair
                    AS pair_usable_pair,
                px.is_proxy
                    AS pair_is_proxy,

                (
                    px.trade_iso3 IS NOT NULL
                    AND px.usable_pair
                    AND (
                        NOT px.is_proxy
                        OR {str(bool(allow_pair_proxies)).upper()}
                    )
                    AND px.cepii_iso3 IS NOT NULL
                ) AS pair_crosswalk_active,

                CASE
                    WHEN cx.trade_iso3 IS NULL
                        THEN 'missing_country_crosswalk'
                    WHEN NOT cx.usable_country
                        THEN 'country_crosswalk_not_usable'
                    WHEN cx.is_proxy
                         AND NOT {str(bool(allow_country_proxies)).upper()}
                        THEN 'country_proxy_disallowed'
                    WHEN cx.cepii_iso3 IS NULL
                        THEN 'country_crosswalk_target_null'
                    WHEN cg.country_iso3 IS NULL
                        THEN 'mapped_country_absent_from_country_reference'
                    WHEN cg.landlocked IS NULL
                      OR cg.area_km2 IS NULL
                      OR cg.internal_distance_km IS NULL
                        THEN 'country_reference_required_fields_incomplete'
                    ELSE 'country_reference_resolved'
                END AS country_resolution_class,

                CASE
                    WHEN px.trade_iso3 IS NULL
                        THEN 'missing_pair_crosswalk'
                    WHEN NOT px.usable_pair
                        THEN 'pair_crosswalk_not_usable'
                    WHEN px.is_proxy
                         AND NOT {str(bool(allow_pair_proxies)).upper()}
                        THEN 'pair_proxy_disallowed'
                    WHEN px.cepii_iso3 IS NULL
                        THEN 'pair_crosswalk_target_null'
                    ELSE 'pair_identity_resolved'
                END AS pair_identity_resolution_class

            FROM codes AS c

            LEFT JOIN read_parquet({country_xwalk}) AS cx
              ON TRIM(c.trade_iso3) = TRIM(cx.trade_iso3)

            LEFT JOIN read_parquet({country_geo}) AS cg
              ON cx.cepii_iso3 = cg.country_iso3

            LEFT JOIN read_parquet({pair_xwalk}) AS px
              ON TRIM(c.trade_iso3) = TRIM(px.trade_iso3)

            ORDER BY
                c.n_affected_role_rows DESC,
                c.trade_iso3
        """

        copy_query(
            con,
            query=country_code_query,
            output_path=output_dir / "country_code_resolution.csv",
            format_name="CSV",
        )

        pair_resolution_query = f"""
            WITH problematic_pairs AS (
                SELECT
                    TRIM(exporter_iso) AS exporter_trade_iso3,
                    TRIM(importer_iso) AS importer_trade_iso3,
                    COUNT(*) AS n_rows,
                    COUNT(DISTINCT target_cmd)
                        AS n_distinct_products,
                    MIN(year) AS first_year,
                    MAX(year) AS last_year,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd
                FROM read_parquet({failures})
                WHERE pair_geography_missing
                GROUP BY 1, 2
            )
            SELECT
                p.*,

                ex.trade_iso3 IS NOT NULL
                    AS exporter_pair_crosswalk_found,
                ex.cepii_iso3
                    AS exporter_pair_cepii_iso3,
                ex.mapping_type
                    AS exporter_pair_mapping_type,
                ex.usable_pair
                    AS exporter_pair_usable,
                ex.is_proxy
                    AS exporter_pair_is_proxy,

                im.trade_iso3 IS NOT NULL
                    AS importer_pair_crosswalk_found,
                im.cepii_iso3
                    AS importer_pair_cepii_iso3,
                im.mapping_type
                    AS importer_pair_mapping_type,
                im.usable_pair
                    AS importer_pair_usable,
                im.is_proxy
                    AS importer_pair_is_proxy,

                (
                    ex.trade_iso3 IS NOT NULL
                    AND ex.usable_pair
                    AND (
                        NOT ex.is_proxy
                        OR {str(bool(allow_pair_proxies)).upper()}
                    )
                    AND ex.cepii_iso3 IS NOT NULL
                ) AS exporter_pair_identity_active,

                (
                    im.trade_iso3 IS NOT NULL
                    AND im.usable_pair
                    AND (
                        NOT im.is_proxy
                        OR {str(bool(allow_pair_proxies)).upper()}
                    )
                    AND im.cepii_iso3 IS NOT NULL
                ) AS importer_pair_identity_active,

                pg.origin_iso3 IS NOT NULL
                    AS exact_pair_reference_found,

                rev.origin_iso3 IS NOT NULL
                    AS reverse_pair_reference_found,

                pg.distance_model_km,
                pg.distance_model_source,
                pg.log_distance_model_km,
                pg.contiguous,

                CASE
                    WHEN ex.trade_iso3 IS NULL
                        THEN 'exporter_missing_pair_crosswalk'
                    WHEN NOT ex.usable_pair
                        THEN 'exporter_pair_crosswalk_not_usable'
                    WHEN ex.is_proxy
                         AND NOT {str(bool(allow_pair_proxies)).upper()}
                        THEN 'exporter_pair_proxy_disallowed'
                    WHEN ex.cepii_iso3 IS NULL
                        THEN 'exporter_pair_target_null'
                    WHEN im.trade_iso3 IS NULL
                        THEN 'importer_missing_pair_crosswalk'
                    WHEN NOT im.usable_pair
                        THEN 'importer_pair_crosswalk_not_usable'
                    WHEN im.is_proxy
                         AND NOT {str(bool(allow_pair_proxies)).upper()}
                        THEN 'importer_pair_proxy_disallowed'
                    WHEN im.cepii_iso3 IS NULL
                        THEN 'importer_pair_target_null'
                    WHEN pg.origin_iso3 IS NOT NULL
                        THEN 'exact_pair_reference_resolved'
                    WHEN rev.origin_iso3 IS NOT NULL
                        THEN 'reverse_pair_only'
                    ELSE 'mapped_pair_absent_from_pair_reference'
                END AS pair_resolution_class

            FROM problematic_pairs AS p

            LEFT JOIN read_parquet({pair_xwalk}) AS ex
              ON p.exporter_trade_iso3 = TRIM(ex.trade_iso3)

            LEFT JOIN read_parquet({pair_xwalk}) AS im
              ON p.importer_trade_iso3 = TRIM(im.trade_iso3)

            LEFT JOIN read_parquet({pair_geo}) AS pg
              ON ex.cepii_iso3 = pg.origin_iso3
             AND im.cepii_iso3 = pg.destination_iso3

            LEFT JOIN read_parquet({pair_geo}) AS rev
              ON ex.cepii_iso3 = rev.destination_iso3
             AND im.cepii_iso3 = rev.origin_iso3

            ORDER BY
                p.n_rows DESC,
                p.exporter_trade_iso3,
                p.importer_trade_iso3
        """

        copy_query(
            con,
            query=pair_resolution_query,
            output_path=output_dir / "pair_resolution.csv",
            format_name="CSV",
        )

        row_level_query = f"""
            WITH country_resolution AS (
                {country_code_query}
            ),
            pair_resolution AS (
                {pair_resolution_query}
            )
            SELECT
                f.*,

                ex.country_resolution_class
                    AS exporter_country_resolution_class,
                ex.country_cepii_iso3
                    AS exporter_country_cepii_iso3,
                ex.country_crosswalk_found
                    AS exporter_country_crosswalk_found,
                ex.country_geography_row_found
                    AS exporter_country_reference_found,

                im.country_resolution_class
                    AS importer_country_resolution_class,
                im.country_cepii_iso3
                    AS importer_country_cepii_iso3,
                im.country_crosswalk_found
                    AS importer_country_crosswalk_found,
                im.country_geography_row_found
                    AS importer_country_reference_found,

                pr.pair_resolution_class,
                pr.exporter_pair_cepii_iso3,
                pr.importer_pair_cepii_iso3,
                pr.exact_pair_reference_found,
                pr.reverse_pair_reference_found,

                CASE
                    WHEN f.importer_geography_missing
                     AND im.country_resolution_class
                         = 'country_reference_resolved'
                        THEN 'unexpected_importer_feature_join_failure'

                    WHEN f.exporter_geography_missing
                     AND ex.country_resolution_class
                         = 'country_reference_resolved'
                        THEN 'unexpected_exporter_feature_join_failure'

                    WHEN f.pair_geography_missing
                     AND pr.pair_resolution_class
                         = 'exact_pair_reference_resolved'
                        THEN 'unexpected_pair_feature_join_failure'

                    WHEN (
                        f.importer_geography_missing
                        AND im.country_resolution_class
                            <> 'country_reference_resolved'
                    )
                      OR (
                        f.exporter_geography_missing
                        AND ex.country_resolution_class
                            <> 'country_reference_resolved'
                    )
                      OR (
                        f.pair_geography_missing
                        AND pr.pair_resolution_class
                            <> 'exact_pair_reference_resolved'
                    )
                        THEN 'reference_or_crosswalk_gap'

                    ELSE 'resolved_or_not_applicable'
                END AS overall_resolution_class

            FROM read_parquet({failures}) AS f

            LEFT JOIN country_resolution AS ex
              ON TRIM(f.exporter_iso) = ex.trade_iso3

            LEFT JOIN country_resolution AS im
              ON TRIM(f.importer_iso) = im.trade_iso3

            LEFT JOIN pair_resolution AS pr
              ON TRIM(f.exporter_iso) = pr.exporter_trade_iso3
             AND TRIM(f.importer_iso) = pr.importer_trade_iso3
        """

        copy_query(
            con,
            query=row_level_query,
            output_path=(
                output_dir
                / "row_level_reference_resolution.parquet"
            ),
            format_name="PARQUET",
        )

        row_level = sql_literal(
            output_dir / "row_level_reference_resolution.parquet"
        )
        country_codes = sql_literal(
            output_dir / "country_code_resolution.csv"
        )
        pair_res = sql_literal(
            output_dir / "pair_resolution.csv"
        )

        resolution_summary_query = f"""
            SELECT
                overall_resolution_class,
                COUNT(*) AS n_rows,
                100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                    AS share_of_audited_rows_percent,
                COUNT(DISTINCT TRIM(exporter_iso))
                    AS n_distinct_exporters,
                COUNT(DISTINCT TRIM(importer_iso))
                    AS n_distinct_importers,
                COUNT(
                    DISTINCT
                    TRIM(exporter_iso)
                    || '|'
                    || TRIM(importer_iso)
                ) AS n_distinct_pairs,
                SUM(direct_evidence_value_usd)
                    AS direct_evidence_value_usd

            FROM read_parquet({row_level})

            GROUP BY overall_resolution_class
            ORDER BY n_rows DESC
        """

        copy_query(
            con,
            query=resolution_summary_query,
            output_path=output_dir / "resolution_class_summary.csv",
            format_name="CSV",
        )

        country_summary_query = f"""
            SELECT
                country_resolution_class,
                COUNT(*) AS n_codes,
                SUM(n_affected_role_rows)
                    AS n_affected_role_rows,
                SUM(direct_evidence_value_usd)
                    AS direct_evidence_value_usd,
                STRING_AGG(
                    trade_iso3,
                    ', '
                    ORDER BY n_affected_role_rows DESC
                ) AS codes

            FROM read_csv_auto({country_codes}, HEADER=TRUE)

            GROUP BY country_resolution_class
            ORDER BY n_affected_role_rows DESC
        """

        copy_query(
            con,
            query=country_summary_query,
            output_path=output_dir / "country_resolution_summary.csv",
            format_name="CSV",
        )

        pair_summary_query = f"""
            SELECT
                pair_resolution_class,
                COUNT(*) AS n_pairs,
                SUM(n_rows) AS n_rows,
                SUM(direct_evidence_value_usd)
                    AS direct_evidence_value_usd

            FROM read_csv_auto({pair_res}, HEADER=TRUE)

            GROUP BY pair_resolution_class
            ORDER BY n_rows DESC
        """

        copy_query(
            con,
            query=pair_summary_query,
            output_path=output_dir / "pair_resolution_summary.csv",
            format_name="CSV",
        )

        summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,
                COUNT(*) FILTER (
                    WHERE overall_resolution_class
                        = 'reference_or_crosswalk_gap'
                ) AS n_reference_or_crosswalk_gap,
                COUNT(*) FILTER (
                    WHERE overall_resolution_class LIKE 'unexpected_%'
                ) AS n_unexpected_feature_join_failure,
                COUNT(DISTINCT TRIM(exporter_iso))
                    AS n_distinct_exporters,
                COUNT(DISTINCT TRIM(importer_iso))
                    AS n_distinct_importers,
                COUNT(
                    DISTINCT
                    TRIM(exporter_iso)
                    || '|'
                    || TRIM(importer_iso)
                ) AS n_distinct_pairs

            FROM read_parquet({row_level})
            """
        ).fetchone()

    finally:
        con.close()

    if summary is None:
        raise RuntimeError("Reference-resolution audit returned no summary.")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "threads": int(threads),
        "allow_country_proxies": bool(allow_country_proxies),
        "allow_pair_proxies": bool(allow_pair_proxies),
        "inputs": {
            key: str(value)
            for key, value in paths.items()
            if key != "output_dir"
        },
        "output_dir": str(output_dir),
        "metrics": {
            "n_rows": int(summary[0] or 0),
            "n_reference_or_crosswalk_gap": int(summary[1] or 0),
            "n_unexpected_feature_join_failure": int(summary[2] or 0),
            "n_distinct_exporters": int(summary[3] or 0),
            "n_distinct_importers": int(summary[4] or 0),
            "n_distinct_pairs": int(summary[5] or 0),
        },
        "artifacts": {
            name: str(output_dir / name)
            for name in filenames
        },
    }

    manifest_path = (
        output_dir / "reference_resolution_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest


def main() -> int:
    args = parse_args()
    logger = configure_logging()

    if args.threads <= 0:
        raise ValueError("--threads must be greater than zero.")

    paths = resolve_paths(
        target_hs=args.target_hs,
        failure_rows=args.failure_rows,
        country_crosswalk=args.country_crosswalk,
        pair_crosswalk=args.pair_crosswalk,
        country_geography=args.country_geography,
        pair_geography=args.pair_geography,
        output_dir=args.output_dir,
    )

    manifest = build_audit(
        paths=paths,
        threads=args.threads,
        allow_country_proxies=args.allow_country_proxies,
        allow_pair_proxies=args.allow_pair_proxies,
        force=args.force,
        logger=logger,
    )

    metrics = manifest["metrics"]

    logger.info("=" * 72)
    logger.info("Reference-resolution geography audit completed.")
    logger.info("Audited rows: %s", f"{metrics['n_rows']:,}")
    logger.info(
        "Reference/crosswalk gaps: %s",
        f"{metrics['n_reference_or_crosswalk_gap']:,}",
    )
    logger.info(
        "Unexpected feature-join failures: %s",
        f"{metrics['n_unexpected_feature_join_failure']:,}",
    )
    logger.info("Output directory: %s", paths["output_dir"])
    logger.info("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
