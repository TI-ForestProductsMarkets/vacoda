"""
vacoda.cif_fob.training_features

Enrich the CIF/FOB factor-candidate training sample with model
predictors from prepared Parquet references.

Reference inputs
----------------
Bilateral geography:
    country_pair_geography.parquet

Country geography:
    country_geography.parquet

Comtrade-to-CEPII crosswalk:
    comtrade_cepii_crosswalk.parquet

Annual oil prices:
    annual_oil_prices.parquet

Predictors created
------------------
Bilateral:
    distance_model_km
    distance_model_source
    log_distance
    contiguity
    exporter_landlocked
    importer_landlocked

Fuel and time:
    crude_oil_average_usd_per_bbl
    log_oil_price
    log_oil_price_x_log_distance
    year

Product:
    product_group
    product_median_unit_value
    product_weight_to_value_ratio
    product_median_unit_value_source

Product median hierarchy
------------------------
The configured method is currently expected to be:

    lagged_product_median

Resolution hierarchy:

    1. lagged HS6 product median, when cumulative prior observations
       meet the configured minimum;
    2. same-year product-group median;
    3. same-year global median.

The factor-candidate input is preserved exactly one row per canonical
bilateral key. Missing geography is retained and explicitly flagged.
Missing oil prices within the configured scope are a hard failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.cif_fob import CifFobConfig
from vacoda.cif_fob.prepare_input import valid_output


# ======================================================================
# Constants
# ======================================================================


SUPPORTED_PRODUCT_MEDIAN_METHODS = {
    "lagged_product_median",
}

SUPPORTED_PRODUCT_FALLBACKS = {
    "product_group_year",
    "global_year",
}


# ======================================================================
# SQL helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _quoted_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


# ======================================================================
# Parquet-contract helpers
# ======================================================================


def _read_parquet_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> set[str]:
    """Return column names from a Parquet file."""
    rows = con.execute(
        """
        DESCRIBE
        SELECT *
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchall()

    return {
        str(row[0])
        for row in rows
    }


def _require_parquet_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    path: Path,
    required: set[str],
    label: str,
) -> None:
    """Raise when a reference Parquet is missing required columns."""
    available = _read_parquet_columns(
        con=con,
        path=path,
    )

    missing = sorted(
        required - available
    )

    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
            + f". Path: {path}"
        )


def _validate_unique_reference_key(
    *,
    con: duckdb.DuckDBPyConnection,
    path: Path,
    key_columns: list[str],
    label: str,
) -> int:
    """Return duplicate-key count and raise when duplicates exist."""
    key_sql = ", ".join(
        _quoted_identifier(column)
        for column in key_columns
    )

    row = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                {key_sql},
                COUNT(*) AS n_rows

            FROM read_parquet(?)

            GROUP BY {key_sql}

            HAVING COUNT(*) > 1
        )
        """,
        [str(path)],
    ).fetchone()

    duplicate_count = int(
        row[0]
        if row is not None
        else 0
    )

    if duplicate_count > 0:
        raise RuntimeError(
            f"{label} contains {duplicate_count:,} duplicate "
            f"reference keys on {key_columns}."
        )

    return duplicate_count


def _validate_reference_files(
    *,
    con: duckdb.DuckDBPyConnection,
    pair_geography_path: Path,
    country_geography_path: Path,
    geography_crosswalk_path: Path,
    oil_price_path: Path,
    minimum_year: int,
    maximum_year: int,
) -> dict[str, int]:
    """Validate schemas, uniqueness, crosswalk integrity, and oil coverage."""

    _require_parquet_columns(
        con=con,
        path=pair_geography_path,
        label="Bilateral geography reference",
        required={
            "origin_iso3",
            "destination_iso3",
            "distance_simple_km",
            "distance_capital_km",
            "distance_weighted_km",
            "distance_weighted_ces_km",
            "contiguous",
            "distance_model_km",
            "distance_model_source",
            "log_distance_model_km",
        },
    )

    _require_parquet_columns(
        con=con,
        path=country_geography_path,
        label="Country geography reference",
        required={
            "country_iso3",
            "area_km2",
            "internal_distance_km",
            "landlocked",
            "continent",
            "log_area_km2",
            "log_internal_distance_km",
        },
    )

    _require_parquet_columns(
        con=con,
        path=geography_crosswalk_path,
        label="Comtrade-to-CEPII geography crosswalk",
        required={
            "trade_iso3",
            "cepii_iso3",
            "mapping_type",
            "usable_country",
            "usable_pair",
            "is_proxy",
        },
    )

    _require_parquet_columns(
        con=con,
        path=oil_price_path,
        label="Annual oil-price reference",
        required={
            "year",
            "crude_oil_average_usd_per_bbl",
            "log_crude_oil_average",
        },
    )

    _validate_unique_reference_key(
        con=con,
        path=pair_geography_path,
        key_columns=["origin_iso3", "destination_iso3"],
        label="Bilateral geography reference",
    )
    _validate_unique_reference_key(
        con=con,
        path=country_geography_path,
        key_columns=["country_iso3"],
        label="Country geography reference",
    )
    _validate_unique_reference_key(
        con=con,
        path=geography_crosswalk_path,
        key_columns=["trade_iso3"],
        label="Comtrade-to-CEPII geography crosswalk",
    )
    _validate_unique_reference_key(
        con=con,
        path=oil_price_path,
        key_columns=["year"],
        label="Annual oil-price reference",
    )

    pair_audit = con.execute(
        """
        SELECT
            COUNT(*) AS n_rows,
            COUNT(*) FILTER (
                WHERE origin_iso3 IS NULL OR destination_iso3 IS NULL
            ) AS n_null_keys,
            COUNT(*) FILTER (
                WHERE distance_model_km IS NULL OR distance_model_km <= 0
            ) AS n_invalid_model_distances,
            COUNT(*) FILTER (
                WHERE log_distance_model_km IS NULL
            ) AS n_missing_log_distances,
            COUNT(*) FILTER (
                WHERE contiguous NOT IN (0, 1) OR contiguous IS NULL
            ) AS n_invalid_contiguous,
            COUNT(*) FILTER (
                WHERE distance_model_source = 'weighted'
            ) AS n_weighted_distance,
            COUNT(*) FILTER (
                WHERE distance_model_source = 'capital'
            ) AS n_capital_fallback
        FROM read_parquet(?)
        """,
        [str(pair_geography_path)],
    ).fetchone()

    if pair_audit is None:
        raise RuntimeError("Could not audit bilateral geography reference.")

    (
        n_pair_rows,
        n_pair_null_keys,
        n_invalid_model_distances,
        n_missing_log_distances,
        n_invalid_contiguous,
        n_weighted_distance,
        n_capital_fallback,
    ) = pair_audit

    if n_pair_null_keys > 0:
        raise RuntimeError(
            "Bilateral geography reference contains null pair keys: "
            f"{n_pair_null_keys:,}."
        )
    if n_invalid_model_distances > 0:
        raise RuntimeError(
            "Bilateral geography reference contains invalid model "
            f"distances: {n_invalid_model_distances:,}."
        )
    if n_missing_log_distances > 0:
        raise RuntimeError(
            "Bilateral geography reference contains missing log "
            f"distances: {n_missing_log_distances:,}."
        )
    if n_invalid_contiguous > 0:
        raise RuntimeError(
            "Bilateral geography reference contains invalid contiguous "
            f"values: {n_invalid_contiguous:,}."
        )

    country_audit = con.execute(
        """
        SELECT
            COUNT(*) AS n_rows,
            COUNT(*) FILTER (WHERE country_iso3 IS NULL) AS n_null_keys,
            COUNT(*) FILTER (
                WHERE landlocked NOT IN (0, 1) OR landlocked IS NULL
            ) AS n_invalid_landlocked
        FROM read_parquet(?)
        """,
        [str(country_geography_path)],
    ).fetchone()

    if country_audit is None:
        raise RuntimeError("Could not audit country geography reference.")

    n_country_rows, n_country_null_keys, n_invalid_landlocked = country_audit

    if n_country_null_keys > 0:
        raise RuntimeError(
            "Country geography reference contains null ISO3 keys: "
            f"{n_country_null_keys:,}."
        )
    if n_invalid_landlocked > 0:
        raise RuntimeError(
            "Country geography reference contains invalid landlocked "
            f"values: {n_invalid_landlocked:,}."
        )

    crosswalk_audit = con.execute(
        """
        WITH pair_universe AS (
            SELECT origin_iso3 AS country_iso3
            FROM read_parquet(?)
            UNION
            SELECT destination_iso3 AS country_iso3
            FROM read_parquet(?)
        )
        SELECT
            COUNT(*) AS n_rows,
            COUNT(*) FILTER (
                WHERE trade_iso3 IS NULL OR mapping_type IS NULL
            ) AS n_null_required,
            COUNT(*) FILTER (
                WHERE usable_country IS NULL
                   OR usable_pair IS NULL
                   OR is_proxy IS NULL
            ) AS n_null_flags,
            COUNT(*) FILTER (
                WHERE (usable_country OR usable_pair) AND cepii_iso3 IS NULL
            ) AS n_usable_without_target,
            COUNT(*) FILTER (
                WHERE NOT usable_country
                  AND NOT usable_pair
                  AND cepii_iso3 IS NOT NULL
            ) AS n_unusable_with_target,
            COUNT(*) FILTER (
                WHERE usable_country
                  AND cg.country_iso3 IS NULL
            ) AS n_missing_country_targets,
            COUNT(*) FILTER (
                WHERE usable_pair
                  AND pu.country_iso3 IS NULL
            ) AS n_missing_pair_targets,
            COUNT(*) FILTER (WHERE is_proxy) AS n_proxy_rows,
            COUNT(*) FILTER (
                WHERE NOT is_proxy AND (usable_country OR usable_pair)
            ) AS n_nonproxy_usable_rows
        FROM read_parquet(?) AS x
        LEFT JOIN read_parquet(?) AS cg
          ON x.cepii_iso3 = cg.country_iso3
        LEFT JOIN pair_universe AS pu
          ON x.cepii_iso3 = pu.country_iso3
        """,
        [
            str(pair_geography_path),
            str(pair_geography_path),
            str(geography_crosswalk_path),
            str(country_geography_path),
        ],
    ).fetchone()

    if crosswalk_audit is None:
        raise RuntimeError("Could not audit geography crosswalk.")

    (
        n_crosswalk_rows,
        n_crosswalk_null_required,
        n_crosswalk_null_flags,
        n_usable_without_target,
        n_unusable_with_target,
        n_missing_country_targets,
        n_missing_pair_targets,
        n_proxy_crosswalk_rows,
        n_nonproxy_usable_crosswalk_rows,
    ) = crosswalk_audit

    crosswalk_errors = {
        "null required values": n_crosswalk_null_required,
        "null boolean flags": n_crosswalk_null_flags,
        "usable rows without CEPII target": n_usable_without_target,
        "unusable rows with CEPII target": n_unusable_with_target,
        "country targets absent from country geography": n_missing_country_targets,
        "pair targets absent from pair geography": n_missing_pair_targets,
    }
    invalid_crosswalk = {
        key: int(value)
        for key, value in crosswalk_errors.items()
        if int(value or 0) > 0
    }
    if invalid_crosswalk:
        details = ", ".join(
            f"{key}={value:,}" for key, value in invalid_crosswalk.items()
        )
        raise RuntimeError(f"Invalid geography crosswalk: {details}.")

    oil_audit = con.execute(
        """
        SELECT
            COUNT(*) AS n_rows,
            MIN(year) AS minimum_year,
            MAX(year) AS maximum_year,
            COUNT(*) FILTER (
                WHERE year IS NULL
                   OR crude_oil_average_usd_per_bbl IS NULL
                   OR crude_oil_average_usd_per_bbl <= 0
                   OR log_crude_oil_average IS NULL
            ) AS n_invalid_rows
        FROM read_parquet(?)
        """,
        [str(oil_price_path)],
    ).fetchone()

    if oil_audit is None:
        raise RuntimeError("Could not audit annual oil-price reference.")

    n_oil_rows, oil_minimum_year, oil_maximum_year, n_invalid_oil_rows = oil_audit

    if n_invalid_oil_rows > 0:
        raise RuntimeError(
            "Annual oil-price reference contains invalid rows: "
            f"{n_invalid_oil_rows:,}."
        )

    missing_scope_years = con.execute(
        """
        WITH expected AS (
            SELECT range AS year
            FROM range(?, ? + 1)
        )
        SELECT LIST(e.year ORDER BY e.year)
        FROM expected AS e
        LEFT JOIN read_parquet(?) AS o
          ON e.year = o.year
        WHERE o.year IS NULL
        """,
        [int(minimum_year), int(maximum_year), str(oil_price_path)],
    ).fetchone()

    missing_years = (
        missing_scope_years[0]
        if missing_scope_years and missing_scope_years[0] is not None
        else []
    )
    if missing_years:
        raise RuntimeError(
            "Annual oil-price reference does not cover configured "
            "scope years: " + ", ".join(str(year) for year in missing_years)
        )

    return {
        "n_pair_reference_rows": int(n_pair_rows),
        "n_pair_weighted_distance": int(n_weighted_distance),
        "n_pair_capital_fallback": int(n_capital_fallback),
        "n_country_reference_rows": int(n_country_rows),
        "n_crosswalk_rows": int(n_crosswalk_rows),
        "n_proxy_crosswalk_rows": int(n_proxy_crosswalk_rows),
        "n_nonproxy_usable_crosswalk_rows": int(
            n_nonproxy_usable_crosswalk_rows
        ),
        "n_oil_reference_rows": int(n_oil_rows),
        "oil_reference_minimum_year": int(oil_minimum_year),
        "oil_reference_maximum_year": int(oil_maximum_year),
    }


# ======================================================================
# Product-feature configuration
# ======================================================================


def _validate_product_feature_config(
    cfg: CifFobConfig,
) -> None:
    """Validate product-feature method and fallback names."""
    product_cfg = cfg.features.product
    median_cfg = product_cfg.median_unit_value

    method = (
        median_cfg.method
        .strip()
        .lower()
    )

    if method not in SUPPORTED_PRODUCT_MEDIAN_METHODS:
        raise ValueError(
            "Unsupported product median-unit-value method "
            f"{median_cfg.method!r}. Supported methods: "
            f"{sorted(SUPPORTED_PRODUCT_MEDIAN_METHODS)}."
        )

    unsupported_fallbacks = sorted(
        {
            fallback.strip().lower()
            for fallback in median_cfg.fallback
        }
        - SUPPORTED_PRODUCT_FALLBACKS
    )

    if unsupported_fallbacks:
        raise ValueError(
            "Unsupported product median fallback values: "
            + ", ".join(unsupported_fallbacks)
            + ". Supported values: "
            + ", ".join(
                sorted(SUPPORTED_PRODUCT_FALLBACKS)
            )
        )

    if not (
        1
        <= product_cfg.group_digits
        <= 6
    ):
        raise ValueError(
            "Product group digits must be between 1 and 6, "
            f"got {product_cfg.group_digits}."
        )


def _product_fallback_sql(
    cfg: CifFobConfig,
) -> tuple[str, str, str]:
    """
    Build configured SQL expressions for product-feature fallback.

    Returns
    -------
    tuple[str, str, str]
        value expression, source expression, observation-count expression
    """
    minimum_observations = int(
        cfg.features.product
        .median_unit_value
        .minimum_observations
    )

    fallbacks = [
        fallback.strip().lower()
        for fallback in (
            cfg.features.product
            .median_unit_value
            .fallback
        )
    ]

    value_cases = [
        f"""
        WHEN lagged_product_observations
             >= {minimum_observations}
         AND lagged_product_median_unit_value
             IS NOT NULL
            THEN lagged_product_median_unit_value
        """
    ]

    source_cases = [
        f"""
        WHEN lagged_product_observations
             >= {minimum_observations}
         AND lagged_product_median_unit_value
             IS NOT NULL
            THEN 'lagged_product'
        """
    ]

    count_cases = [
        f"""
        WHEN lagged_product_observations
             >= {minimum_observations}
         AND lagged_product_median_unit_value
             IS NOT NULL
            THEN lagged_product_observations
        """
    ]

    for fallback in fallbacks:
        if fallback == "product_group_year":
            value_cases.append(
                f"""
                WHEN product_group_year_observations
                     >= {minimum_observations}
                 AND product_group_year_median_unit_value
                     IS NOT NULL
                    THEN product_group_year_median_unit_value
                """
            )

            source_cases.append(
                f"""
                WHEN product_group_year_observations
                     >= {minimum_observations}
                 AND product_group_year_median_unit_value
                     IS NOT NULL
                    THEN 'product_group_year'
                """
            )

            count_cases.append(
                f"""
                WHEN product_group_year_observations
                     >= {minimum_observations}
                 AND product_group_year_median_unit_value
                     IS NOT NULL
                    THEN product_group_year_observations
                """
            )

        elif fallback == "global_year":
            value_cases.append(
                f"""
                WHEN global_year_observations
                     >= {minimum_observations}
                 AND global_year_median_unit_value
                     IS NOT NULL
                    THEN global_year_median_unit_value
                """
            )

            source_cases.append(
                f"""
                WHEN global_year_observations
                     >= {minimum_observations}
                 AND global_year_median_unit_value
                     IS NOT NULL
                    THEN 'global_year'
                """
            )

            count_cases.append(
                f"""
                WHEN global_year_observations
                     >= {minimum_observations}
                 AND global_year_median_unit_value
                     IS NOT NULL
                    THEN global_year_observations
                """
            )

    value_expression = (
        "CASE\n"
        + "\n".join(value_cases)
        + "\nELSE NULL\nEND"
    )

    source_expression = (
        "CASE\n"
        + "\n".join(source_cases)
        + "\nELSE NULL\nEND"
    )

    count_expression = (
        "CASE\n"
        + "\n".join(count_cases)
        + "\nELSE NULL\nEND"
    )

    return (
        value_expression,
        source_expression,
        count_expression,
    )


# ======================================================================
# Existing-output inspection
# ======================================================================


def _inspect_existing_training_features(
    *,
    output_path: Path,
    threads: int,
) -> dict[str, Any]:
    """Inspect a previously created feature-enriched Parquet."""
    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        row = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE pair_geography_found
                ) AS n_pair_geography_found,

                COUNT(*) FILTER (
                    WHERE exporter_geography_found
                ) AS n_exporter_geography_found,

                COUNT(*) FILTER (
                    WHERE importer_geography_found
                ) AS n_importer_geography_found,

                COUNT(*) FILTER (
                    WHERE oil_price_found
                ) AS n_oil_price_found,

                COUNT(*) FILTER (
                    WHERE product_median_unit_value
                          IS NOT NULL
                ) AS n_product_feature_found,

                COUNT(*) FILTER (
                    WHERE initial_predictors_complete
                ) AS n_initial_predictors_complete,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND initial_predictors_complete
                ) AS n_model_candidates_predictors_complete,

                COUNT(*) FILTER (
                    WHERE model_eligible
                ) AS n_model_candidates,

                COUNT(*) FILTER (
                    WHERE lagged_pair_median_factor IS NOT NULL
                ) AS n_lagged_pair_factor_found,

                COUNT(*) FILTER (
                    WHERE model_eligible
                        AND lagged_pair_median_factor IS NOT NULL
                ) AS n_model_candidates_lagged_pair_factor_found,
                
                COUNT(DISTINCT year) AS n_years

            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()

    finally:
        con.close()

    if row is None:
        raise RuntimeError(
            "Could not inspect existing training-feature output: "
            f"{output_path}"
        )

    metric_names = [
        "n_rows",
        "n_pair_geography_found",
        "n_exporter_geography_found",
        "n_importer_geography_found",
        "n_oil_price_found",
        "n_product_feature_found",
        "n_initial_predictors_complete",
        "n_model_candidates_predictors_complete",
        "n_model_candidates",
        "n_lagged_pair_factor_found",
        "n_model_candidates_lagged_pair_factor_found",
        "n_years",
    ]
    
    metrics = dict(
        zip(
            metric_names,
            row,
            strict=True,
        )
    )
    model_candidates = int(
        metrics["n_model_candidates"]
        or 0
    )

    metrics[
    "model_candidate_lagged_pair_factor_coverage_percent"
    ] = (
        100.0
        * int(
            metrics[
                "n_model_candidates_lagged_pair_factor_found"
            ]
            or 0
        )
        / model_candidates
        if model_candidates
        else 0.0
    )

    return {
        "status": "reused",
        "audit_passed": True,
        "ready_for_rate_estimation": (
            int(
                metrics[
                    "n_model_candidates_predictors_complete"
                ]
                or 0
            )
            > 0
        ),
        "output_path": output_path,
        "training_features_path": output_path,
        **metrics,
    }


# ======================================================================
# Main feature builder
# ======================================================================


def build_training_features(
    *,
    cfg: CifFobConfig,
    input_path: Path,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build the predictor-enriched CIF/FOB training Parquet.

    Input grain
    -----------
    year, exporter_iso, importer_iso, target_cmd

    Output grain
    ------------
    Exactly the same grain and row count as the input.
    """
    output_path = cfg.product_path(
        "training_features"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            "CIF/FOB training-sample Parquet not found: "
            f"{input_path}"
        )

    if (
        not force
        and valid_output(output_path)
    ):
        logger.info(
            "REUSE — CIF/FOB training features already exist: %s",
            output_path,
        )

        return _inspect_existing_training_features(
            output_path=output_path,
            threads=cfg.engine.threads,
        )

    pair_geography_path = cfg.features.pair_geography.resolve_path()
    country_geography_path = cfg.features.geography.resolve_path()
    geography_crosswalk_path = (
        cfg.features.geography_crosswalk.resolve_path()
    )
    allow_proxy_mappings = (
        cfg.features.geography_crosswalk.allow_proxy_mappings
    )
    oil_price_path = cfg.features.oil_price.resolve_path()

    for label, path in [
        ("Bilateral geography reference", pair_geography_path),
        ("Country geography reference", country_geography_path),
        ("Geography crosswalk", geography_crosswalk_path),
        ("Annual oil-price reference", oil_price_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    _validate_product_feature_config(
        cfg
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_sql = _sql_literal(
        input_path
    )
    pair_geography_sql = _sql_literal(
        pair_geography_path
    )
    country_geography_sql = _sql_literal(
        country_geography_path
    )
    geography_crosswalk_sql = _sql_literal(
        geography_crosswalk_path
    )
    oil_price_sql = _sql_literal(
        oil_price_path
    )
    output_sql = _sql_literal(
        output_path
    )

    group_digits = int(
        cfg.features.product.group_digits
    )

    (
        product_value_expression,
        product_source_expression,
        product_count_expression,
    ) = _product_fallback_sql(cfg)

    logger.info(
        "Building CIF/FOB predictor-enriched training sample."
    )
    logger.info(
        "  Input: %s",
        input_path,
    )
    logger.info(
        "  Output: %s",
        output_path,
    )
    logger.info(
        "  Pair geography: %s",
        pair_geography_path,
    )
    logger.info(
        "  Country geography: %s",
        country_geography_path,
    )
    logger.info(
        "  Geography crosswalk: %s",
        geography_crosswalk_path,
    )
    logger.info(
        "  Allow proxy geography mappings: %s",
        allow_proxy_mappings,
    )
    logger.info(
        "  Annual oil prices: %s",
        oil_price_path,
    )
    logger.info(
        "  Product group digits: %s",
        group_digits,
    )
    logger.info(
        "  Product median method: %s",
        cfg.features.product
        .median_unit_value.method,
    )
    logger.info(
        "  Product median minimum observations: %s",
        cfg.features.product
        .median_unit_value.minimum_observations,
    )
    logger.info(
        "  Product median fallback order: %s",
        (
            ", ".join(
                cfg.features.product
                .median_unit_value.fallback
            )
            or "none"
        ),
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # ==============================================================
        # INPUT AND REFERENCE CONTRACTS
        # ==============================================================

        _require_parquet_columns(
            con=con,
            path=input_path,
            label="CIF/FOB training sample",
            required={
                "year",
                "exporter_iso",
                "importer_iso",
                "target_cmd",
                "export_unit_value_usd_per_kg",
                "observed_cif_fob_factor",
                "factor_source",
                "model_eligible",
            },
        )

        reference_metrics = _validate_reference_files(
            con=con,
            pair_geography_path=pair_geography_path,
            country_geography_path=country_geography_path,
            geography_crosswalk_path=geography_crosswalk_path,
            oil_price_path=oil_price_path,
            minimum_year=cfg.scope.min_year,
            maximum_year=cfg.scope.max_year,
        )

        input_audit = con.execute(
            f"""
            WITH duplicate_keys AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    COUNT(*) AS n_rows

                FROM read_parquet({input_sql})

                GROUP BY
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd

                HAVING COUNT(*) > 1
            )

            SELECT
                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                ) AS n_input_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_input_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                    WHERE year IS NULL
                       OR exporter_iso IS NULL
                       OR importer_iso IS NULL
                       OR target_cmd IS NULL
                ) AS n_null_input_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                    WHERE model_eligible
                ) AS n_model_candidates

            """
        ).fetchone()

        if input_audit is None:
            raise RuntimeError(
                "Could not audit CIF/FOB training-sample input."
            )

        (
            n_input_rows,
            n_duplicate_input_keys,
            n_null_input_keys,
            n_model_candidates,
        ) = input_audit

        if n_input_rows == 0:
            raise RuntimeError(
                "Training-sample input contains no rows."
            )

        if n_duplicate_input_keys > 0:
            raise RuntimeError(
                "Training-sample input contains duplicate canonical "
                f"keys: {n_duplicate_input_keys:,}."
            )

        if n_null_input_keys > 0:
            raise RuntimeError(
                "Training-sample input contains null canonical keys: "
                f"{n_null_input_keys:,}."
            )

        # ==============================================================
        # FEATURE CONSTRUCTION
        # ==============================================================

        query = f"""
        WITH base AS (
            SELECT
                t.*,

                LPAD(
                    CAST(target_cmd AS VARCHAR),
                    6,
                    '0'
                ) AS target_cmd_text,

                SUBSTR(
                    LPAD(
                        CAST(target_cmd AS VARCHAR),
                        6,
                        '0'
                    ),
                    1,
                    {group_digits}
                ) AS product_group

            FROM read_parquet({input_sql}) AS t
        ),

        harmonized AS (
            SELECT
                b.*,

                COALESCE(ex.mapping_type, 'identity')
                    AS exporter_geography_mapping_type,
                COALESCE(im.mapping_type, 'identity')
                    AS importer_geography_mapping_type,

                COALESCE(ex.is_proxy, FALSE)
                    AS exporter_geography_mapping_is_proxy,
                COALESCE(im.is_proxy, FALSE)
                    AS importer_geography_mapping_is_proxy,

                (
                    ex.trade_iso3 IS NOT NULL
                    AND ex.usable_country
                    AND (NOT ex.is_proxy OR {str(allow_proxy_mappings).upper()})
                ) AS exporter_country_mapping_applied,

                (
                    im.trade_iso3 IS NOT NULL
                    AND im.usable_country
                    AND (NOT im.is_proxy OR {str(allow_proxy_mappings).upper()})
                ) AS importer_country_mapping_applied,

                (
                    ex.trade_iso3 IS NOT NULL
                    AND ex.usable_pair
                    AND (NOT ex.is_proxy OR {str(allow_proxy_mappings).upper()})
                ) AS exporter_pair_mapping_applied,

                (
                    im.trade_iso3 IS NOT NULL
                    AND im.usable_pair
                    AND (NOT im.is_proxy OR {str(allow_proxy_mappings).upper()})
                ) AS importer_pair_mapping_applied,

                CASE
                    WHEN ex.trade_iso3 IS NOT NULL
                     AND ex.usable_country
                     AND (NOT ex.is_proxy OR {str(allow_proxy_mappings).upper()})
                        THEN ex.cepii_iso3
                    ELSE b.exporter_iso
                END AS exporter_country_geography_iso3,

                CASE
                    WHEN im.trade_iso3 IS NOT NULL
                     AND im.usable_country
                     AND (NOT im.is_proxy OR {str(allow_proxy_mappings).upper()})
                        THEN im.cepii_iso3
                    ELSE b.importer_iso
                END AS importer_country_geography_iso3,

                CASE
                    WHEN ex.trade_iso3 IS NOT NULL
                     AND ex.usable_pair
                     AND (NOT ex.is_proxy OR {str(allow_proxy_mappings).upper()})
                        THEN ex.cepii_iso3
                    ELSE b.exporter_iso
                END AS exporter_pair_geography_iso3,

                CASE
                    WHEN im.trade_iso3 IS NOT NULL
                     AND im.usable_pair
                     AND (NOT im.is_proxy OR {str(allow_proxy_mappings).upper()})
                        THEN im.cepii_iso3
                    ELSE b.importer_iso
                END AS importer_pair_geography_iso3

            FROM base AS b

            LEFT JOIN read_parquet({geography_crosswalk_sql}) AS ex
              ON b.exporter_iso = ex.trade_iso3

            LEFT JOIN read_parquet({geography_crosswalk_sql}) AS im
              ON b.importer_iso = im.trade_iso3
        ),

        valid_export_unit_values AS (
            SELECT
                year,

                LPAD(
                    CAST(target_cmd AS VARCHAR),
                    6,
                    '0'
                ) AS target_cmd_text,

                SUBSTR(
                    LPAD(
                        CAST(target_cmd AS VARCHAR),
                        6,
                        '0'
                    ),
                    1,
                    {group_digits}
                ) AS product_group,

                CAST(
                    export_unit_value_usd_per_kg
                    AS DOUBLE
                ) AS export_unit_value_usd_per_kg

            FROM read_parquet({input_sql})

            WHERE export_unit_value_usd_per_kg IS NOT NULL
              AND export_unit_value_usd_per_kg > 0
        ),

        direct_pair_year_factors AS (
            SELECT
                year,
                exporter_iso,
                importer_iso,
                MEDIAN(
                    observed_cif_fob_factor
                ) AS pair_year_median_factor,

                COUNT(*) AS pair_year_factor_observations

            FROM base
            WHERE model_eligible
                AND factor_source = 'direct'
                AND observed_cif_fob_factor IS NOT NULL
                AND observed_cif_fob_factor >= 1.0
                AND observed_cif_fob_factor <= 3.0
            GROUP BY
                year,
                exporter_iso,
                importer_iso
        ),

        candidate_pair_years AS (
            SELECT DISTINCT
                year,
                exporter_iso,
                importer_iso

            FROM base

            WHERE exporter_iso IS NOT NULL
              AND importer_iso IS NOT NULL
        ),

        direct_pair_lagged_history AS (
            SELECT
                c.year,
                c.exporter_iso,
                c.importer_iso,

                MEDIAN(
                    h.pair_year_median_factor
                ) AS lagged_pair_median_factor,

                SUM(
                    h.pair_year_factor_observations
                ) AS lagged_pair_factor_observations,

                COUNT(
                    h.year
                ) AS lagged_pair_factor_years

            FROM candidate_pair_years AS c

            LEFT JOIN direct_pair_year_factors AS h
              ON c.exporter_iso = h.exporter_iso
             AND c.importer_iso = h.importer_iso
             AND h.year < c.year

            GROUP BY
                c.year,
                c.exporter_iso,
                c.importer_iso
        ),



        product_year_medians AS (
            SELECT
                year,
                target_cmd_text,

                MEDIAN(
                    export_unit_value_usd_per_kg
                ) AS product_year_median_unit_value,

                COUNT(*) AS product_year_observations

            FROM valid_export_unit_values

            GROUP BY
                year,
                target_cmd_text
        ),

        product_year_with_lag AS (
            SELECT
                year,
                target_cmd_text,
                product_year_median_unit_value,
                product_year_observations,

                MEDIAN(
                    product_year_median_unit_value
                ) OVER (
                    PARTITION BY target_cmd_text
                    ORDER BY year

                    ROWS BETWEEN
                        UNBOUNDED PRECEDING
                        AND 1 PRECEDING
                ) AS lagged_product_median_unit_value,

                SUM(
                    product_year_observations
                ) OVER (
                    PARTITION BY target_cmd_text
                    ORDER BY year

                    ROWS BETWEEN
                        UNBOUNDED PRECEDING
                        AND 1 PRECEDING
                ) AS lagged_product_observations

            FROM product_year_medians
        ),

        product_group_year_medians AS (
            SELECT
                year,
                product_group,

                MEDIAN(
                    export_unit_value_usd_per_kg
                ) AS product_group_year_median_unit_value,

                COUNT(*) AS product_group_year_observations

            FROM valid_export_unit_values

            GROUP BY
                year,
                product_group
        ),

        global_year_medians AS (
            SELECT
                year,

                MEDIAN(
                    export_unit_value_usd_per_kg
                ) AS global_year_median_unit_value,

                COUNT(*) AS global_year_observations

            FROM valid_export_unit_values

            GROUP BY year
        ),

        joined AS (
            SELECT
                b.*,

                -- ----------------------------------------------
                -- Pair geography
                -- ----------------------------------------------

                pg.distance_simple_km,
                pg.distance_capital_km,
                pg.distance_weighted_km,
                pg.distance_weighted_ces_km,
                pg.distance_model_km,
                pg.distance_model_source,

                pg.log_distance_model_km
                    AS log_distance,

                CAST(
                    pg.contiguous
                    AS INTEGER
                ) AS contiguity,

                (
                    pg.origin_iso3 IS NOT NULL
                ) AS pair_geography_found,

                -- ----------------------------------------------
                -- Exporter geography
                -- ----------------------------------------------

                eg.area_km2
                    AS exporter_area_km2,

                eg.internal_distance_km
                    AS exporter_internal_distance_km,

                CAST(
                    eg.landlocked
                    AS INTEGER
                ) AS exporter_landlocked,

                eg.continent
                    AS exporter_continent,

                eg.log_area_km2
                    AS exporter_log_area_km2,

                eg.log_internal_distance_km
                    AS exporter_log_internal_distance_km,

                (
                    eg.country_iso3 IS NOT NULL
                ) AS exporter_geography_found,

                -- ----------------------------------------------
                -- Importer geography
                -- ----------------------------------------------

                ig.area_km2
                    AS importer_area_km2,

                ig.internal_distance_km
                    AS importer_internal_distance_km,

                CAST(
                    ig.landlocked
                    AS INTEGER
                ) AS importer_landlocked,

                ig.continent
                    AS importer_continent,

                ig.log_area_km2
                    AS importer_log_area_km2,

                ig.log_internal_distance_km
                    AS importer_log_internal_distance_km,

                (
                    ig.country_iso3 IS NOT NULL
                ) AS importer_geography_found,

                -- ----------------------------------------------
                -- Oil price
                -- ----------------------------------------------

                op.crude_oil_average_usd_per_bbl,

                op.log_crude_oil_average
                    AS log_oil_price,

                (
                    op.year IS NOT NULL
                ) AS oil_price_found,

                -- ----------------------------------------------
                -- Product feature candidates
                -- ----------------------------------------------

                py.lagged_product_median_unit_value,

                ph.lagged_pair_median_factor,
                ph.lagged_pair_factor_observations,
                ph.lagged_pair_factor_years,

                (
                ph.lagged_pair_median_factor IS NOT NULL    
                ) AS lagged_pair_factor_found,

                py.lagged_product_observations,

                pgy.product_group_year_median_unit_value,
                pgy.product_group_year_observations,

                gy.global_year_median_unit_value,
                gy.global_year_observations

            FROM harmonized AS b

            LEFT JOIN read_parquet(
                {pair_geography_sql}
            ) AS pg
              ON b.exporter_pair_geography_iso3
                    = pg.origin_iso3
             AND b.importer_pair_geography_iso3
                    = pg.destination_iso3

            LEFT JOIN read_parquet(
                {country_geography_sql}
            ) AS eg
              ON b.exporter_country_geography_iso3
                    = eg.country_iso3

            LEFT JOIN read_parquet(
                {country_geography_sql}
            ) AS ig
              ON b.importer_country_geography_iso3
                    = ig.country_iso3

            LEFT JOIN read_parquet(
                {oil_price_sql}
            ) AS op
              ON b.year = op.year

            LEFT JOIN direct_pair_lagged_history  AS ph
              ON b.year = ph.year
             AND b.exporter_iso = ph.exporter_iso
             AND b.importer_iso = ph.importer_iso

            LEFT JOIN product_year_with_lag AS py
              ON b.year = py.year
             AND b.target_cmd_text
                    = py.target_cmd_text

            LEFT JOIN product_group_year_medians AS pgy
              ON b.year = pgy.year
             AND b.product_group
                    = pgy.product_group

            LEFT JOIN global_year_medians AS gy
              ON b.year = gy.year
        ),

        product_resolved AS (
            SELECT
                *,

                {product_value_expression}
                    AS product_median_unit_value,

                {product_source_expression}
                    AS product_median_unit_value_source,

                {product_count_expression}
                    AS product_median_unit_value_observations

            FROM joined
        ),

        derived AS (
            SELECT
                *,

                CASE
                    WHEN log_oil_price IS NOT NULL
                     AND log_distance IS NOT NULL
                        THEN (
                            log_oil_price
                            * log_distance
                        )
                END AS log_oil_price_x_log_distance,

                CASE
                    WHEN product_median_unit_value > 0
                        THEN LN(product_median_unit_value)
                END AS log_product_median_unit_value,

            FROM product_resolved
        )

        SELECT
            *,

            (
                pair_geography_found
                AND exporter_geography_found
                AND importer_geography_found
                AND oil_price_found
                AND log_distance IS NOT NULL
                AND contiguity IS NOT NULL
                AND exporter_landlocked IS NOT NULL
                AND importer_landlocked IS NOT NULL
                AND log_oil_price_x_log_distance IS NOT NULL
                AND log_product_median_unit_value IS NOT NULL
                AND product_group IS NOT NULL
                AND year IS NOT NULL
            ) AS initial_predictors_complete,

            CASE
                WHEN NOT pair_geography_found
                    THEN 'missing_pair_geography'

                WHEN NOT exporter_geography_found
                    THEN 'missing_exporter_geography'

                WHEN NOT importer_geography_found
                    THEN 'missing_importer_geography'

                WHEN NOT oil_price_found
                    THEN 'missing_oil_price'

                WHEN product_median_unit_value IS NULL
                    THEN 'missing_product_feature'

                WHEN log_distance IS NULL
                    THEN 'missing_log_distance'

                WHEN contiguity IS NULL
                    THEN 'missing_contiguity'

                WHEN exporter_landlocked IS NULL
                    THEN 'missing_exporter_landlocked'

                WHEN importer_landlocked IS NULL
                    THEN 'missing_importer_landlocked'

                WHEN log_oil_price_x_log_distance IS NULL
                    THEN 'missing_oil_distance_interaction'

                ELSE 'complete'
            END AS predictor_completeness_reason

        FROM derived
        """

        con.execute(
            f"""
            COPY (
                {query}
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        # ==============================================================
        # OUTPUT AUDIT
        # ==============================================================

        audit_row = con.execute(
            f"""
            WITH duplicate_keys AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    COUNT(*) AS n_rows

                FROM read_parquet({output_sql})

                GROUP BY
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd

                HAVING COUNT(*) > 1
            )

            SELECT
                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                ) AS n_output_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_output_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE pair_geography_found
                ) AS n_pair_geography_found,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE NOT pair_geography_found
                ) AS n_pair_geography_missing,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE exporter_geography_found
                ) AS n_exporter_geography_found,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE NOT exporter_geography_found
                ) AS n_exporter_geography_missing,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE importer_geography_found
                ) AS n_importer_geography_found,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE NOT importer_geography_found
                ) AS n_importer_geography_missing,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE oil_price_found
                ) AS n_oil_price_found,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE NOT oil_price_found
                ) AS n_oil_price_missing,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE product_median_unit_value
                          IS NOT NULL
                ) AS n_product_feature_found,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE product_median_unit_value
                          IS NULL
                ) AS n_product_feature_missing,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE product_median_unit_value_source
                          = 'lagged_product'
                ) AS n_product_feature_lagged,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE product_median_unit_value_source
                          = 'product_group_year'
                ) AS n_product_feature_group_fallback,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE product_median_unit_value_source
                          = 'global_year'
                ) AS n_product_feature_global_fallback,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE lagged_pair_median_factor IS NOT NULL
                ) AS n_lagged_pair_factor_found,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                        AND lagged_pair_median_factor IS NOT NULL
                ) AS n_model_candidates_lagged_pair_factor_found,
                
                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE exporter_country_mapping_applied
                       OR importer_country_mapping_applied
                       OR exporter_pair_mapping_applied
                       OR importer_pair_mapping_applied
                ) AS n_rows_with_geography_mapping,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE (exporter_country_mapping_applied
                           AND exporter_geography_mapping_is_proxy)
                       OR (importer_country_mapping_applied
                           AND importer_geography_mapping_is_proxy)
                       OR (exporter_pair_mapping_applied
                           AND exporter_geography_mapping_is_proxy)
                       OR (importer_pair_mapping_applied
                           AND importer_geography_mapping_is_proxy)
                ) AS n_rows_with_proxy_mapping,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE distance_model_source
                          = 'weighted'
                ) AS n_weighted_distance_used,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE distance_model_source
                          = 'capital'
                ) AS n_capital_distance_fallback_used,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE initial_predictors_complete
                ) AS n_initial_predictors_complete,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                      AND initial_predictors_complete
                ) AS n_model_candidates_predictors_complete,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                      AND NOT initial_predictors_complete
                ) AS n_model_candidates_predictors_incomplete

            """
        ).fetchone()

    finally:
        con.close()

    if audit_row is None:
        raise RuntimeError(
            "Could not audit feature-enriched training output."
        )

    metric_names = [
        "n_output_rows",
        "n_duplicate_output_keys",
        "n_pair_geography_found",
        "n_pair_geography_missing",
        "n_exporter_geography_found",
        "n_exporter_geography_missing",
        "n_importer_geography_found",
        "n_importer_geography_missing",
        "n_oil_price_found",
        "n_oil_price_missing",
        "n_product_feature_found",
        "n_product_feature_missing",
        "n_product_feature_lagged",
        "n_product_feature_group_fallback",
        "n_product_feature_global_fallback",
        "n_lagged_pair_factor_found",
        "n_model_candidates_lagged_pair_factor_found",
        "n_rows_with_geography_mapping",
        "n_rows_with_proxy_mapping",
        "n_weighted_distance_used",
        "n_capital_distance_fallback_used",
        "n_initial_predictors_complete",
        "n_model_candidates_predictors_complete",
        "n_model_candidates_predictors_incomplete",
    ]

    metrics = {
        **reference_metrics,
        **dict(
            zip(
                metric_names,
                audit_row,
                strict=True,
            )
        ),
    }

    metrics["n_input_rows"] = int(
        n_input_rows
    )
    metrics["n_model_candidates"] = int(
        n_model_candidates
    )

    # ==================================================================
    # Hard audits
    # ==================================================================

    if (
        metrics["n_output_rows"]
        != metrics["n_input_rows"]
    ):
        raise RuntimeError(
            "Feature enrichment did not preserve every input row "
            "exactly once: "
            f"input={metrics['n_input_rows']:,}, "
            f"output={metrics['n_output_rows']:,}."
        )

    if metrics["n_duplicate_output_keys"] > 0:
        raise RuntimeError(
            "Feature-enriched output contains duplicate canonical "
            f"keys: {metrics['n_duplicate_output_keys']:,}."
        )

    if metrics["n_oil_price_missing"] > 0:
        raise RuntimeError(
            "Feature-enriched output contains rows without an annual "
            f"oil price: {metrics['n_oil_price_missing']:,}."
        )

    if metrics["n_product_feature_found"] == 0:
        raise RuntimeError(
            "No product median-unit-value predictors were constructed."
        )

    if (
        metrics["n_model_candidates_predictors_complete"]
        == 0
    ):
        raise RuntimeError(
            "No model-eligible observations have a complete initial "
            "predictor set."
        )

    # ==================================================================
    # Coverage metrics
    # ==================================================================

    output_rows = int(
        metrics["n_output_rows"]
        or 0
    )

    model_candidates = int(
        metrics["n_model_candidates"]
        or 0
    )

    metrics["pair_geography_coverage_percent"] = (
        100.0
        * int(
            metrics["n_pair_geography_found"]
            or 0
        )
        / output_rows
        if output_rows
        else 0.0
    )

    metrics["exporter_geography_coverage_percent"] = (
        100.0
        * int(
            metrics["n_exporter_geography_found"]
            or 0
        )
        / output_rows
        if output_rows
        else 0.0
    )

    metrics["importer_geography_coverage_percent"] = (
        100.0
        * int(
            metrics["n_importer_geography_found"]
            or 0
        )
        / output_rows
        if output_rows
        else 0.0
    )

    metrics["product_feature_coverage_percent"] = (
        100.0
        * int(
            metrics["n_product_feature_found"]
            or 0
        )
        / output_rows
        if output_rows
        else 0.0
    )

    metrics["initial_predictor_coverage_percent"] = (
        100.0
        * int(
            metrics["n_initial_predictors_complete"]
            or 0
        )
        / output_rows
        if output_rows
        else 0.0
    )

    metrics[
        "model_candidate_predictor_coverage_percent"
    ] = (
        100.0
        * int(
            metrics[
                "n_model_candidates_predictors_complete"
            ]
            or 0
        )
        / model_candidates
        if model_candidates
        else 0.0
    )
    metrics[
        "model_candidate_lagged_pair_factor_coverage_percent"
    ] = (
        100.0
        * int(
            metrics[
                "n_model_candidates_lagged_pair_factor_found"
            ]
            or 0
        )
        / model_candidates
        if model_candidates
        else 0.0
    )
    

    audit_passed = True

    ready_for_rate_estimation = (
        metrics[
            "n_model_candidates_predictors_complete"
        ]
        > 0
    )

    # ==================================================================
    # Logging
    # ==================================================================

    logger.info(
        "CIF/FOB predictor-enriched training sample built."
    )
    logger.info(
        "  Input rows: %s",
        metrics["n_input_rows"],
    )
    logger.info(
        "  Output rows: %s",
        metrics["n_output_rows"],
    )
    logger.info(
        "  Duplicate output keys: %s",
        metrics["n_duplicate_output_keys"],
    )

    logger.info(
        "  Pair geography found: %s",
        metrics["n_pair_geography_found"],
    )
    logger.info(
        "  Pair geography coverage: %.4f%%",
        metrics["pair_geography_coverage_percent"],
    )
    logger.info(
        "  Pair geography missing: %s",
        metrics["n_pair_geography_missing"],
    )

    logger.info(
        "  Exporter geography coverage: %.4f%%",
        metrics[
            "exporter_geography_coverage_percent"
        ],
    )
    logger.info(
        "  Importer geography coverage: %.4f%%",
        metrics[
            "importer_geography_coverage_percent"
        ],
    )

    logger.info(
        "  Annual oil-price rows found: %s",
        metrics["n_oil_price_found"],
    )
    logger.info(
        "  Annual oil-price rows missing: %s",
        metrics["n_oil_price_missing"],
    )

    logger.info(
        "  Product feature found: %s",
        metrics["n_product_feature_found"],
    )
    logger.info(
        "  Product-feature coverage: %.4f%%",
        metrics["product_feature_coverage_percent"],
    )
    logger.info(
        "  Product feature from lagged product: %s",
        metrics["n_product_feature_lagged"],
    )
    logger.info(
        "  Product feature from group-year fallback: %s",
        metrics[
            "n_product_feature_group_fallback"
        ],
    )
    logger.info(
        "  Product feature from global-year fallback: %s",
        metrics[
            "n_product_feature_global_fallback"
        ],
    )

    logger.info(
        "  Rows with applied geography mapping: %s",
        metrics["n_rows_with_geography_mapping"],
    )
    logger.info(
        "  Rows with applied proxy mapping: %s",
        metrics["n_rows_with_proxy_mapping"],
    )
    logger.info(
        "  Weighted distances used: %s",
        metrics["n_weighted_distance_used"],
    )
    logger.info(
        "  Capital-distance fallbacks used: %s",
        metrics[
            "n_capital_distance_fallback_used"
        ],
    )

    logger.info(
        "  Rows with complete initial predictors: %s",
        metrics["n_initial_predictors_complete"],
    )
    logger.info(
        "  Overall predictor coverage: %.4f%%",
        metrics[
            "initial_predictor_coverage_percent"
        ],
    )
    
    logger.info(
        "  Rows with lagged pair-factor history: %s",
        metrics["n_lagged_pair_factor_found"],
    )
    
    logger.info(
        "  Model candidates with lagged pair-factor history: %s",
        metrics[
            "n_model_candidates_lagged_pair_factor_found"
        ],
    )
    logger.info(
        "  Model-candidate lagged pair-factor coverage: %.4f%%",
        metrics[
            "model_candidate_lagged_pair_factor_coverage_percent"
        ],
    )
    logger.info(
        "  Model candidates with complete predictors: %s",
        metrics[
            "n_model_candidates_predictors_complete"
        ],
    )
    logger.info(
        "  Model-candidate predictor coverage: %.4f%%",
        metrics[
            "model_candidate_predictor_coverage_percent"
        ],
    )

    logger.info(
        "  Step 4 audit passed: %s",
        audit_passed,
    )
    logger.info(
        "  Ready for rate estimation: %s",
        ready_for_rate_estimation,
    )
    logger.info(
        "  Output: %s",
        output_path,
    )

    return {
        "status": "completed",
        "audit_passed": audit_passed,
        "ready_for_rate_estimation": (
            ready_for_rate_estimation
        ),
        "output_path": output_path,
        "training_features_path": output_path,
        **metrics,
    }

