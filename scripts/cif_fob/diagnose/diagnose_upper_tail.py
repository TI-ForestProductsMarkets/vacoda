
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from vacoda.config.base import (
    get_config_path,
    get_step_config_path,
    load_yaml_config,
    require_dict,
)
from vacoda.config.steps.cif_fob import CifFobConfig
from vacoda.config.pipeline import (
    PipelineEngineConfig,
    PipelineScopeConfig,
)
from vacoda.steps.cif_fob.estimate_rates import (
    _evidence_value,
    _source_weight,
)


# ======================================================================
# Configuration
# ======================================================================


def load_runtime_config() -> CifFobConfig:
    """
    Load scope and engine from pipeline.yaml and CIF/FOB settings from
    configs/steps/cif_fob.yaml.
    """
    pipeline_path = get_config_path("pipeline")
    cif_fob_path = get_step_config_path("cif_fob")

    pipeline_raw = load_yaml_config(
        pipeline_path
    )

    scope = PipelineScopeConfig.from_dict(
        require_dict(
            pipeline_raw.get("scope"),
            key="scope",
        )
    )

    engine = PipelineEngineConfig.from_dict(
        require_dict(
            pipeline_raw.get("engine"),
            key="engine",
        )
    )

    return CifFobConfig.from_yaml(
        cif_fob_path,
        scope=scope,
        engine=engine,
    )


# ======================================================================
# Generic helpers
# ======================================================================


def quote_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


def read_parquet_columns(
    *,
    path: Path,
    threads: int,
) -> set[str]:
    """Return the columns available in a Parquet artifact."""
    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        rows = con.execute(
            """
            DESCRIBE
            SELECT *
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchall()

    finally:
        con.close()

    return {
        str(row[0])
        for row in rows
    }


def numeric_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a numeric column or an all-null fallback series."""
    if column not in frame.columns:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype="float64",
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def boolean_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a boolean column or an all-false fallback series."""
    if column not in frame.columns:
        return pd.Series(
            False,
            index=frame.index,
            dtype="bool",
        )

    return (
        frame[column]
        .fillna(False)
        .astype(bool)
    )


def safe_median(
    values: pd.Series,
) -> float:
    """Return a numeric median or NaN when no numeric values exist."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return float("nan")

    return float(
        numeric.median()
    )


def safe_mean(
    values: pd.Series,
) -> float:
    """Return a numeric mean or NaN when no numeric values exist."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return float("nan")

    return float(
        numeric.mean()
    )


# ======================================================================
# Data loading and clean-estimation rules
# ======================================================================


CORE_COLUMNS = {
    "year",
    "exporter_iso",
    "importer_iso",
    "target_cmd",
    "model_eligible",
    "initial_predictors_complete",
    "factor_source",
    "observed_cif_fob_factor",
    "import_direct_covered_primary_value",
    "export_value_fob",
    "import_value_cif",
    "pair_uses_weight_proxy",
    "pair_uses_estimated_weight",
    "import_weight_estimated",
}


OPTIONAL_AUDIT_COLUMNS = {
    "export_weight_kg",
    "import_weight_kg",
    "weight_concordance",
    "raw_value_ratio",
    "unit_value_ratio",
    "export_unit_value_usd_per_kg",
    "import_unit_value_usd_per_kg",
    "export_source_rows",
    "import_source_rows",
    "export_positive_value_rows",
    "import_positive_value_rows",
    "export_positive_weight_rows",
    "import_positive_weight_rows",
    "export_proxy_weight_rows",
    "import_proxy_weight_rows",
    "export_estimated_weight_rows",
    "import_estimated_weight_rows",
    "import_direct_cif_value",
    "import_direct_fob_value",
    "export_direct_cif_value",
    "export_direct_fob_value",
    "lagged_pair_median_factor",
    "latest_pair_median_factor",
    "years_since_latest_pair_factor",
    "lagged_pair_factor_observations",
    "lagged_pair_factor_years",
    "pair_geography_found",
    "distance_model",
    "log_distance",
    "contiguity",
    "exporter_landlocked",
    "importer_landlocked",
    "product_group",
}


def load_training_features(
    *,
    cfg: CifFobConfig,
) -> pd.DataFrame:
    """
    Load required and available optional columns from training features.
    """
    input_path = cfg.product_path(
        "training_features"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            "Training-features Parquet not found: "
            f"{input_path}"
        )

    available_columns = read_parquet_columns(
        path=input_path,
        threads=cfg.engine.threads,
    )

    missing_core = sorted(
        CORE_COLUMNS
        - available_columns
    )

    if missing_core:
        raise RuntimeError(
            "Training-features Parquet is missing required "
            "upper-tail audit columns: "
            + ", ".join(missing_core)
        )

    selected_columns = sorted(
        CORE_COLUMNS
        | (
            OPTIONAL_AUDIT_COLUMNS
            & available_columns
        )
    )

    select_sql = ",\n".join(
        quote_identifier(column)
        for column in selected_columns
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        data = con.execute(
            f"""
            SELECT
                {select_sql}

            FROM read_parquet(?)
            """,
            [str(input_path)],
        ).df()

    finally:
        con.close()

    if data.empty:
        raise RuntimeError(
            "Training-features Parquet contains no rows."
        )

    return data


def build_clean_estimation_sample(
    *,
    data: pd.DataFrame,
    cfg: CifFobConfig,
) -> pd.DataFrame:
    """
    Apply the same clean-estimation rules used by estimate_rates.py.
    """
    estimation_cfg = cfg.method.estimation

    minimum_evidence_value = float(
        estimation_cfg.minimum_evidence_value_usd
    )
    minimum_factor = float(
        estimation_cfg.minimum_factor
    )
    maximum_factor = float(
        estimation_cfg.maximum_factor
    )
    exclude_proxy_weights = bool(
        estimation_cfg.exclude_proxy_weights
    )

    training_factor_sources = {
        str(source).strip().lower()
        for source
        in estimation_cfg.training_factor_sources
    }

    working = data.copy()

    working["year"] = pd.to_numeric(
        working["year"],
        errors="raise",
    ).astype(int)

    working["observed_cif_fob_factor"] = pd.to_numeric(
        working["observed_cif_fob_factor"],
        errors="coerce",
    )

    working["estimation_evidence_value"] = (
        _evidence_value(
            working
        )
    )

    working["estimation_source_weight"] = (
        _source_weight(
            frame=working,
            cfg=cfg,
        )
    )

    working["passes_minimum_value"] = (
        working["estimation_evidence_value"]
        .ge(minimum_evidence_value)
        .fillna(False)
    )

    working["passes_factor_lower_bound"] = (
        working["observed_cif_fob_factor"]
        .ge(minimum_factor)
        .fillna(False)
    )

    working["passes_factor_upper_bound"] = (
        working["observed_cif_fob_factor"]
        .le(maximum_factor)
        .fillna(False)
    )

    working["passes_factor_source_rule"] = (
        working["factor_source"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(training_factor_sources)
    )

    if exclude_proxy_weights:
        working["passes_proxy_rule"] = (
            ~boolean_series(
                working,
                "pair_uses_weight_proxy",
            )
        )
    else:
        working["passes_proxy_rule"] = True

    working["prediction_eligible"] = (
        boolean_series(
            working,
            "initial_predictors_complete",
        )
    )

    working["clean_estimation_eligible"] = (
        boolean_series(
            working,
            "model_eligible",
        )
        & working["prediction_eligible"]
        & working["passes_factor_source_rule"]
        & working["passes_minimum_value"]
        & working["passes_factor_lower_bound"]
        & working["passes_factor_upper_bound"]
        & working["passes_proxy_rule"]
        & working["estimation_source_weight"]
        .gt(0)
        .fillna(False)
        & working["observed_cif_fob_factor"].notna()
    )

    clean = working.loc[
        working["clean_estimation_eligible"]
    ].copy()

    if clean.empty:
        raise RuntimeError(
            "No observations passed the configured "
            "clean-estimation rules."
        )

    duplicate_mask = clean.duplicated(
        subset=[
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise RuntimeError(
            "Clean training features contain duplicate canonical keys "
            "(year, exporter_iso, importer_iso, target_cmd). "
            f"Duplicate rows: {int(duplicate_mask.sum()):,}."
        )

    return clean


# ======================================================================
# Evidence-quality fields
# ======================================================================


def derive_quantity_agreement(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Return mirror quantity agreement.

    Prefer the pipeline's existing weight_concordance field. If it is
    absent, derive min(export weight, import weight) divided by max.
    """
    if "weight_concordance" in frame.columns:
        existing = numeric_series(
            frame,
            "weight_concordance",
        )

        if existing.notna().any():
            return existing

    export_weight = numeric_series(
        frame,
        "export_weight_kg",
    )
    import_weight = numeric_series(
        frame,
        "import_weight_kg",
    )

    minimum_weight = pd.concat(
        [
            export_weight,
            import_weight,
        ],
        axis=1,
    ).min(
        axis=1,
        skipna=False,
    )

    maximum_weight = pd.concat(
        [
            export_weight,
            import_weight,
        ],
        axis=1,
    ).max(
        axis=1,
        skipna=False,
    )

    valid = (
        minimum_weight.gt(0)
        & maximum_weight.gt(0)
    )

    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype="float64",
    )

    result.loc[valid] = (
        minimum_weight.loc[valid]
        / maximum_weight.loc[valid]
    )

    return result


def derive_any_proxy_weight(
    frame: pd.DataFrame,
) -> pd.Series:
    """Return whether any evidence source uses a proxy weight."""
    proxy = (
        boolean_series(
            frame,
            "pair_uses_weight_proxy",
        )
    )

    if "export_proxy_weight_rows" in frame.columns:
        proxy |= (
            numeric_series(
                frame,
                "export_proxy_weight_rows",
            )
            .fillna(0)
            .gt(0)
        )

    if "import_proxy_weight_rows" in frame.columns:
        proxy |= (
            numeric_series(
                frame,
                "import_proxy_weight_rows",
            )
            .fillna(0)
            .gt(0)
        )

    return proxy


def derive_any_estimated_weight(
    frame: pd.DataFrame,
) -> pd.Series:
    """Return whether any evidence source uses an estimated weight."""
    estimated = (
        boolean_series(
            frame,
            "pair_uses_estimated_weight",
        )
        | boolean_series(
            frame,
            "import_weight_estimated",
        )
    )

    if "export_estimated_weight_rows" in frame.columns:
        estimated |= (
            numeric_series(
                frame,
                "export_estimated_weight_rows",
            )
            .fillna(0)
            .gt(0)
        )

    if "import_estimated_weight_rows" in frame.columns:
        estimated |= (
            numeric_series(
                frame,
                "import_estimated_weight_rows",
            )
            .fillna(0)
            .gt(0)
        )

    return estimated


def make_numeric_band(
    *,
    values: pd.Series,
    cut_points: tuple[float, ...],
    prefix: str,
    missing_label: str,
) -> pd.Series:
    """
    Create deterministic bands from configured cut points.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    sorted_points = tuple(
        sorted(
            set(
                float(point)
                for point in cut_points
            )
        )
    )

    if not sorted_points:
        return pd.Series(
            np.where(
                numeric.notna(),
                f"{prefix}_available",
                missing_label,
            ),
            index=values.index,
            dtype="string",
        )

    bins = [
        float("-inf"),
        *sorted_points,
        float("inf"),
    ]

    labels: list[str] = []

    first = sorted_points[0]

    labels.append(
        f"{prefix}_below_{first:g}"
    )

    for lower, upper in zip(
        sorted_points[:-1],
        sorted_points[1:],
        strict=True,
    ):
        labels.append(
            f"{prefix}_{lower:g}_to_{upper:g}"
        )

    labels.append(
        f"{prefix}_{sorted_points[-1]:g}_and_above"
    )

    band = pd.cut(
        numeric,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("string")

    return band.fillna(
        missing_label
    )


# ======================================================================
# Historical persistence
# ======================================================================


def build_pair_year_history(
    *,
    clean: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Build one observation per bilateral pair and year.

    The pair-year factor is the median across product rows for that pair
    and year.
    """
    pair_year = (
        clean.groupby(
            [
                "exporter_iso",
                "importer_iso",
                "year",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            pair_year_median_factor=(
                "observed_cif_fob_factor",
                "median",
            ),
            pair_year_n_products=(
                "target_cmd",
                "nunique",
            ),
            pair_year_evidence_value=(
                "estimation_evidence_value",
                "sum",
            ),
        )
        .sort_values(
            [
                "exporter_iso",
                "importer_iso",
                "year",
            ]
        )
        .reset_index(drop=True)
    )

    pair_year["pair_year_is_high"] = (
        pair_year["pair_year_median_factor"]
        .ge(threshold)
    )

    groups = pair_year.groupby(
        [
            "exporter_iso",
            "importer_iso",
        ],
        dropna=False,
        sort=False,
    )

    pair_year["prior_pair_years"] = (
        groups.cumcount()
        .astype("int64")
    )

    pair_year["prior_pair_high_years"] = (
        groups["pair_year_is_high"]
        .transform(
            lambda values: (
                values.astype("int64")
                .cumsum()
                .shift(1)
                .fillna(0)
            )
        )
        .astype("int64")
    )

    pair_year["prior_pair_high_share"] = np.where(
        pair_year["prior_pair_years"] > 0,
        (
            pair_year["prior_pair_high_years"]
            / pair_year["prior_pair_years"]
        ),
        np.nan,
    )

    pair_year["latest_prior_pair_factor"] = (
        groups["pair_year_median_factor"]
        .shift(1)
    )

    pair_year["prior_pair_median_factor"] = (
        groups["pair_year_median_factor"]
        .transform(
            lambda values: (
                values.expanding()
                .median()
                .shift(1)
            )
        )
    )

    return pair_year


def add_pair_product_history(
    *,
    clean: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Add strictly prior pair-product persistence statistics.

    The input grain is already one row per pair-product-year.
    """
    result = clean.sort_values(
        [
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "year",
        ]
    ).copy()

    result["current_factor_is_high"] = (
        result["observed_cif_fob_factor"]
        .ge(threshold)
    )

    groups = result.groupby(
        [
            "exporter_iso",
            "importer_iso",
            "target_cmd",
        ],
        dropna=False,
        sort=False,
    )

    result["prior_pair_product_years"] = (
        groups.cumcount()
        .astype("int64")
    )

    result["prior_pair_product_high_years"] = (
        groups["current_factor_is_high"]
        .transform(
            lambda values: (
                values.astype("int64")
                .cumsum()
                .shift(1)
                .fillna(0)
            )
        )
        .astype("int64")
    )

    result["prior_pair_product_high_share"] = np.where(
        result["prior_pair_product_years"] > 0,
        (
            result["prior_pair_product_high_years"]
            / result["prior_pair_product_years"]
        ),
        np.nan,
    )

    result["latest_prior_pair_product_factor"] = (
        groups["observed_cif_fob_factor"]
        .shift(1)
    )

    result["prior_pair_product_median_factor"] = (
        groups["observed_cif_fob_factor"]
        .transform(
            lambda values: (
                values.expanding()
                .median()
                .shift(1)
            )
        )
    )

    return result


def add_persistence_fields(
    *,
    clean: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add pair and pair-product histories to the clean sample."""
    pair_year = build_pair_year_history(
        clean=clean,
        threshold=threshold,
    )

    enriched = add_pair_product_history(
        clean=clean,
        threshold=threshold,
    )

    pair_columns = [
        "exporter_iso",
        "importer_iso",
        "year",
        "pair_year_median_factor",
        "pair_year_n_products",
        "pair_year_evidence_value",
        "prior_pair_years",
        "prior_pair_high_years",
        "prior_pair_high_share",
        "latest_prior_pair_factor",
        "prior_pair_median_factor",
    ]

    enriched = enriched.merge(
        pair_year[pair_columns],
        on=[
            "exporter_iso",
            "importer_iso",
            "year",
        ],
        how="left",
        validate="many_to_one",
    )

    return enriched, pair_year


# ======================================================================
# Credibility classification
# ======================================================================


def classify_tail_observations(
    *,
    tail: pd.DataFrame,
    minimum_prior_years: int,
    historical_high_share_threshold: float,
    quantity_agreement_threshold: float | None,
) -> pd.DataFrame:
    """
    Assign a diagnostic credibility class.

    This classification does not modify training weights.
    """
    result = tail.copy()

    pair_history_sufficient = (
        result["prior_pair_years"]
        .ge(minimum_prior_years)
    )

    pair_product_history_sufficient = (
        result["prior_pair_product_years"]
        .ge(minimum_prior_years)
    )

    persistent_pair = (
        pair_history_sufficient
        & result["prior_pair_high_share"]
        .ge(historical_high_share_threshold)
        .fillna(False)
    )

    persistent_pair_product = (
        pair_product_history_sufficient
        & result["prior_pair_product_high_share"]
        .ge(historical_high_share_threshold)
        .fillna(False)
    )

    result["persistent_high_pair"] = (
        persistent_pair
    )

    result["persistent_high_pair_product"] = (
        persistent_pair_product
    )

    result["persistent_high"] = (
        persistent_pair
        | persistent_pair_product
    )

    result["insufficient_history"] = (
        ~pair_history_sufficient
        & ~pair_product_history_sufficient
    )

    weak_weight_evidence = (
        result["any_proxy_weight"]
        | result["any_estimated_weight"]
    )

    if quantity_agreement_threshold is None:
        weak_quantity_agreement = pd.Series(
            False,
            index=result.index,
            dtype="bool",
        )
    else:
        weak_quantity_agreement = (
            result["quantity_agreement_ratio"]
            .notna()
            & result["quantity_agreement_ratio"]
            .lt(quantity_agreement_threshold)
        )

    result["weak_weight_evidence"] = (
        weak_weight_evidence
    )

    result["weak_quantity_agreement"] = (
        weak_quantity_agreement
    )

    result["weak_evidence"] = (
        weak_weight_evidence
        | weak_quantity_agreement
    )

    result["credibility_class"] = (
        "isolated_high_good_evidence"
    )

    result.loc[
        result["weak_evidence"],
        "credibility_class",
    ] = "isolated_high_weak_evidence"

    result.loc[
        result["insufficient_history"],
        "credibility_class",
    ] = "insufficient_history"

    result.loc[
        result["persistent_high"],
        "credibility_class",
    ] = "persistent_high"

    return result


# ======================================================================
# Audit summaries
# ======================================================================


def summary_row(
    *,
    section: str,
    group_name: str,
    group_value: str,
    frame: pd.DataFrame,
    denominator_rows: int,
    year: int | None = None,
) -> dict[str, Any]:
    """Create one normalized summary row."""
    n_rows = int(
        len(frame)
    )

    return {
        "section": section,
        "year": year,
        "group_name": group_name,
        "group_value": group_value,
        "n_rows": n_rows,
        "row_share": (
            n_rows / denominator_rows
            if denominator_rows > 0
            else np.nan
        ),
        "median_observed_factor": safe_median(
            frame["observed_cif_fob_factor"]
        ),
        "median_evidence_value": safe_median(
            frame["estimation_evidence_value"]
        ),
        "median_quantity_agreement": safe_median(
            frame["quantity_agreement_ratio"]
        ),
        "mean_prior_pair_high_share": safe_mean(
            frame["prior_pair_high_share"]
        ),
        "mean_prior_pair_product_high_share": safe_mean(
            frame["prior_pair_product_high_share"]
        ),
    }


def build_summary(
    *,
    clean: pd.DataFrame,
    tail: pd.DataFrame,
    threshold: float,
    extreme_threshold: float,
) -> pd.DataFrame:
    """Create annual and grouped upper-tail summaries."""
    rows: list[dict[str, Any]] = []

    for year in sorted(
        clean["year"].unique()
    ):
        year_clean = clean.loc[
            clean["year"].eq(year)
        ]

        year_tail = tail.loc[
            tail["year"].eq(year)
        ]

        year_extreme = year_tail.loc[
            year_tail[
                "observed_cif_fob_factor"
            ].ge(extreme_threshold)
        ]

        n_clean = int(
            len(year_clean)
        )
        n_tail = int(
            len(year_tail)
        )
        n_extreme = int(
            len(year_extreme)
        )

        rows.append(
            {
                "section": "annual_tail_composition",
                "year": int(year),
                "group_name": "factor_range",
                "group_value": "all_clean",
                "n_rows": n_clean,
                "row_share": 1.0,
                "median_observed_factor": safe_median(
                    year_clean[
                        "observed_cif_fob_factor"
                    ]
                ),
                "median_evidence_value": safe_median(
                    year_clean[
                        "estimation_evidence_value"
                    ]
                ),
                "median_quantity_agreement": safe_median(
                    year_clean[
                        "quantity_agreement_ratio"
                    ]
                ),
                "mean_prior_pair_high_share": safe_mean(
                    year_clean[
                        "prior_pair_high_share"
                    ]
                ),
                "mean_prior_pair_product_high_share": safe_mean(
                    year_clean[
                        "prior_pair_product_high_share"
                    ]
                ),
            }
        )

        rows.append(
            {
                "section": "annual_tail_composition",
                "year": int(year),
                "group_name": "factor_range",
                "group_value": (
                    f"upper_tail_ge_{threshold:g}"
                ),
                "n_rows": n_tail,
                "row_share": (
                    n_tail / n_clean
                    if n_clean > 0
                    else np.nan
                ),
                "median_observed_factor": safe_median(
                    year_tail[
                        "observed_cif_fob_factor"
                    ]
                ),
                "median_evidence_value": safe_median(
                    year_tail[
                        "estimation_evidence_value"
                    ]
                ),
                "median_quantity_agreement": safe_median(
                    year_tail[
                        "quantity_agreement_ratio"
                    ]
                ),
                "mean_prior_pair_high_share": safe_mean(
                    year_tail[
                        "prior_pair_high_share"
                    ]
                ),
                "mean_prior_pair_product_high_share": safe_mean(
                    year_tail[
                        "prior_pair_product_high_share"
                    ]
                ),
            }
        )

        rows.append(
            {
                "section": "annual_tail_composition",
                "year": int(year),
                "group_name": "factor_range",
                "group_value": (
                    f"extreme_tail_ge_{extreme_threshold:g}"
                ),
                "n_rows": n_extreme,
                "row_share": (
                    n_extreme / n_clean
                    if n_clean > 0
                    else np.nan
                ),
                "median_observed_factor": safe_median(
                    year_extreme[
                        "observed_cif_fob_factor"
                    ]
                ),
                "median_evidence_value": safe_median(
                    year_extreme[
                        "estimation_evidence_value"
                    ]
                ),
                "median_quantity_agreement": safe_median(
                    year_extreme[
                        "quantity_agreement_ratio"
                    ]
                ),
                "mean_prior_pair_high_share": safe_mean(
                    year_extreme[
                        "prior_pair_high_share"
                    ]
                ),
                "mean_prior_pair_product_high_share": safe_mean(
                    year_extreme[
                        "prior_pair_product_high_share"
                    ]
                ),
            }
        )

    grouped_dimensions = {
        "credibility_class": "credibility_class",
        "factor_source": "factor_source",
        "value_band": "value_band",
        "quantity_agreement_band": (
            "quantity_agreement_band"
        ),
        "proxy_weight": "any_proxy_weight",
        "estimated_weight": "any_estimated_weight",
    }

    n_tail_total = int(
        len(tail)
    )

    for section, column in grouped_dimensions.items():
        grouped = tail.groupby(
            column,
            dropna=False,
            sort=True,
        )

        for value, group in grouped:
            rows.append(
                summary_row(
                    section=section,
                    group_name=column,
                    group_value=str(value),
                    frame=group,
                    denominator_rows=n_tail_total,
                )
            )

    for year, year_tail in tail.groupby(
        "year",
        sort=True,
    ):
        denominator = int(
            len(year_tail)
        )

        for value, group in year_tail.groupby(
            "credibility_class",
            dropna=False,
            sort=True,
        ):
            rows.append(
                summary_row(
                    section="annual_credibility_class",
                    group_name="credibility_class",
                    group_value=str(value),
                    frame=group,
                    denominator_rows=denominator,
                    year=int(year),
                )
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "section",
                "year",
                "group_name",
                "group_value",
            ],
            na_position="first",
        )
        .reset_index(drop=True)
    )


def build_route_history_output(
    *,
    pair_year: pd.DataFrame,
    threshold: float,
    minimum_prior_years: int,
    historical_high_share_threshold: float,
) -> pd.DataFrame:
    """Create the bilateral route-history audit artifact."""
    result = pair_year.copy()

    result["persistent_high_pair"] = (
        result["prior_pair_years"]
        .ge(minimum_prior_years)
        & result["prior_pair_high_share"]
        .ge(historical_high_share_threshold)
        .fillna(False)
    )

    result["current_pair_year_is_high"] = (
        result["pair_year_median_factor"]
        .ge(threshold)
    )

    return result.sort_values(
        [
            "year",
            "exporter_iso",
            "importer_iso",
        ]
    ).reset_index(drop=True)


# ======================================================================
# Printing
# ======================================================================


def print_main_summary(
    *,
    clean: pd.DataFrame,
    tail: pd.DataFrame,
    threshold: float,
    extreme_threshold: float,
) -> None:
    """Print concise upper-tail audit results."""
    n_clean = int(
        len(clean)
    )
    n_tail = int(
        len(tail)
    )
    n_extreme = int(
        tail[
            "observed_cif_fob_factor"
        ].ge(extreme_threshold).sum()
    )

    print("\n" + "=" * 96)
    print("UPPER-TAIL AUDIT")
    print("=" * 96)
    print(f"Clean estimation rows: {n_clean:,}")
    print(
        f"Upper-tail rows (>= {threshold:g}): "
        f"{n_tail:,} "
        f"({100 * n_tail / n_clean:.2f}%)"
    )
    print(
        f"Extreme-tail rows (>= {extreme_threshold:g}): "
        f"{n_extreme:,} "
        f"({100 * n_extreme / n_clean:.2f}%)"
    )

    credibility = (
        tail["credibility_class"]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "credibility_class"
        )
        .reset_index(
            name="n_rows"
        )
    )

    credibility["row_share"] = (
        credibility["n_rows"]
        / max(n_tail, 1)
    )

    print("\n" + "=" * 96)
    print("TAIL CREDIBILITY CLASSIFICATION")
    print("=" * 96)

    display = credibility.copy()

    display["row_share"] = display[
        "row_share"
    ].map(
        lambda value: f"{value * 100:.2f}%"
    )

    print(
        display.to_string(
            index=False
        )
    )

    annual = (
        tail.groupby(
            "year",
            as_index=False,
        )
        .agg(
            n_upper_tail=(
                "observed_cif_fob_factor",
                "size",
            ),
            median_factor=(
                "observed_cif_fob_factor",
                "median",
            ),
            median_evidence_value=(
                "estimation_evidence_value",
                "median",
            ),
            median_quantity_agreement=(
                "quantity_agreement_ratio",
                "median",
            ),
            persistent_share=(
                "persistent_high",
                "mean",
            ),
            weak_evidence_share=(
                "weak_evidence",
                "mean",
            ),
        )
    )

    annual["persistent_share"] = (
        annual["persistent_share"]
        .map(
            lambda value: f"{value * 100:.2f}%"
        )
    )

    annual["weak_evidence_share"] = (
        annual["weak_evidence_share"]
        .map(
            lambda value: f"{value * 100:.2f}%"
        )
    )

    annual["median_factor"] = (
        annual["median_factor"]
        .map(
            lambda value: f"{value:.4f}"
        )
    )

    annual["median_evidence_value"] = (
        annual["median_evidence_value"]
        .map(
            lambda value: f"{value:,.2f}"
        )
    )

    annual["median_quantity_agreement"] = (
        annual["median_quantity_agreement"]
        .map(
            lambda value: (
                f"{value:.4f}"
                if pd.notna(value)
                else ""
            )
        )
    )

    print("\n" + "=" * 96)
    print("UPPER TAIL BY YEAR")
    print("=" * 96)
    print(
        annual.to_string(
            index=False
        )
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    cfg = load_runtime_config()

    audit_cfg = (
        cfg.method.estimation
        .diagnostics
        .upper_tail_audit
    )

    if not audit_cfg.enabled:
        raise RuntimeError(
            "Upper-tail audit is disabled. Set "
            "method.estimation.diagnostics."
            "upper_tail_audit.enabled: true"
        )

    threshold = float(
        audit_cfg.threshold
    )

    extreme_threshold = float(
        audit_cfg.extreme_threshold
    )

    minimum_prior_years = int(
        audit_cfg.persistence.minimum_prior_years
    )

    historical_high_share_threshold = float(
        audit_cfg.persistence
        .historical_high_share_threshold
    )

    value_bands = tuple(
        float(value)
        for value
        in audit_cfg.value_bands_usd
    )

    quantity_agreement_bands = tuple(
        float(value)
        for value
        in audit_cfg.quantity_agreement_bands
    )

    if threshold <= 0:
        raise ValueError(
            "Upper-tail threshold must be positive."
        )

    if extreme_threshold < threshold:
        raise ValueError(
            "extreme_threshold must be greater than or equal to "
            "threshold."
        )

    if minimum_prior_years < 1:
        raise ValueError(
            "minimum_prior_years must be at least 1."
        )

    if not (
        0
        <= historical_high_share_threshold
        <= 1
    ):
        raise ValueError(
            "historical_high_share_threshold must be between "
            "0 and 1."
        )

    if quantity_agreement_bands:
        quantity_agreement_threshold: float | None = max(
            quantity_agreement_bands
        )
    else:
        quantity_agreement_threshold = None

    input_path = cfg.product_path(
        "training_features"
    )

    summary_path = cfg.product_path(
        audit_cfg.products.summary
    )

    route_history_path = cfg.product_path(
        audit_cfg.products.route_history
    )

    observations_path = cfg.product_path(
        audit_cfg.products.observations
    )

    print(
        f"Target HS:                  "
        f"{cfg.scope.target_hs}"
    )
    print(
        f"Configured scope:           "
        f"{cfg.scope.min_year}-"
        f"{cfg.scope.max_year}"
    )
    print(
        f"Engine threads:             "
        f"{cfg.engine.threads}"
    )
    print(
        f"Training features:          "
        f"{input_path}"
    )
    print(
        f"Upper-tail threshold:       "
        f"{threshold:g}"
    )
    print(
        f"Extreme-tail threshold:     "
        f"{extreme_threshold:g}"
    )
    print(
        f"Minimum prior years:        "
        f"{minimum_prior_years}"
    )
    print(
        f"Historical high-share rule: "
        f"{historical_high_share_threshold:.4f}"
    )
    print(
        "Quantity evidence cutoff:  "
        + (
            f"{quantity_agreement_threshold:.4f}"
            if quantity_agreement_threshold
            is not None
            else "not configured"
        )
    )

    data = load_training_features(
        cfg=cfg
    )

    clean = build_clean_estimation_sample(
        data=data,
        cfg=cfg,
    )

    clean["quantity_agreement_ratio"] = (
        derive_quantity_agreement(
            clean
        )
    )

    clean["any_proxy_weight"] = (
        derive_any_proxy_weight(
            clean
        )
    )

    clean["any_estimated_weight"] = (
        derive_any_estimated_weight(
            clean
        )
    )

    clean["value_band"] = make_numeric_band(
        values=clean[
            "estimation_evidence_value"
        ],
        cut_points=value_bands,
        prefix="value_usd",
        missing_label="value_missing",
    )

    clean["quantity_agreement_band"] = (
        make_numeric_band(
            values=clean[
                "quantity_agreement_ratio"
            ],
            cut_points=quantity_agreement_bands,
            prefix="agreement",
            missing_label="agreement_missing",
        )
    )

    clean, pair_year = add_persistence_fields(
        clean=clean,
        threshold=threshold,
    )

    tail = clean.loc[
        clean[
            "observed_cif_fob_factor"
        ].ge(threshold)
    ].copy()

    if tail.empty:
        raise RuntimeError(
            "No clean estimation observations meet the configured "
            f"upper-tail threshold of {threshold:g}."
        )

    tail["is_extreme_tail"] = (
        tail[
            "observed_cif_fob_factor"
        ].ge(extreme_threshold)
    )

    tail = classify_tail_observations(
        tail=tail,
        minimum_prior_years=(
            minimum_prior_years
        ),
        historical_high_share_threshold=(
            historical_high_share_threshold
        ),
        quantity_agreement_threshold=(
            quantity_agreement_threshold
        ),
    )

    summary = build_summary(
        clean=clean,
        tail=tail,
        threshold=threshold,
        extreme_threshold=extreme_threshold,
    )

    route_history = build_route_history_output(
        pair_year=pair_year,
        threshold=threshold,
        minimum_prior_years=(
            minimum_prior_years
        ),
        historical_high_share_threshold=(
            historical_high_share_threshold
        ),
    )

    print_main_summary(
        clean=clean,
        tail=tail,
        threshold=threshold,
        extreme_threshold=extreme_threshold,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    route_history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    observations_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    route_history.to_csv(
        route_history_path,
        index=False,
    )

    tail.to_parquet(
        observations_path,
        index=False,
    )

    print("\n" + "=" * 96)
    print("UPPER-TAIL AUDIT COMPLETE")
    print("=" * 96)
    print(
        f"Summary:       "
        f"{summary_path}"
    )
    print(
        f"Route history: "
        f"{route_history_path}"
    )
    print(
        f"Observations:  "
        f"{observations_path}"
    )


if __name__ == "__main__":
    main()
