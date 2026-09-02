from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


# ======================================================================
# Configuration
# ======================================================================


def load_runtime_config() -> CifFobConfig:
    """
    Load the pipeline scope/engine and CIF/FOB step configuration.
    """
    pipeline_raw = load_yaml_config(
        get_config_path("pipeline")
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
        get_step_config_path("cif_fob"),
        scope=scope,
        engine=engine,
    )


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the CIF/FOB upper tail from the already-written "
            "estimated-rates Parquet. No model refitting is performed."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Estimated-rates Parquet. Defaults to the configured "
            "estimated_rates product."
        ),
    )

    parser.add_argument(
        "--training-features",
        type=Path,
        default=None,
        help=(
            "Optional training-features Parquet used to restore audit "
            "fields not retained in estimated_rates. Defaults to the "
            "configured training_features product."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for audit artifacts. Defaults to "
            "<cif_fob output>/tail_audit_saved."
        ),
    )

    parser.add_argument(
        "--tail-threshold",
        type=float,
        default=None,
        help=(
            "Upper-tail factor threshold. Defaults to the configured "
            "upper_tail_audit.threshold."
        ),
    )

    parser.add_argument(
        "--extreme-threshold",
        type=float,
        default=None,
        help=(
            "Extreme-tail threshold. Defaults to the configured "
            "upper_tail_audit.extreme_threshold."
        ),
    )

    parser.add_argument(
        "--minimum-group-rows",
        type=int,
        default=100,
        help=(
            "Minimum clean observations required for a concentration "
            "group to be retained."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Maximum rows retained per concentration dimension.",
    )

    return parser.parse_args()


# ======================================================================
# Generic helpers
# ======================================================================


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def read_parquet_columns(
    *,
    path: Path,
    threads: int,
) -> set[str]:
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

    return {str(row[0]) for row in rows}


def numeric_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
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


def safe_spearman(
    left: pd.Series,
    right: pd.Series,
) -> float:
    pair = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()

    if len(pair) < 3:
        return float("nan")

    if pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")

    result = spearmanr(
        pair["left"],
        pair["right"],
        nan_policy="omit",
    )

    statistic = (
        result.statistic
        if hasattr(result, "statistic")
        else result[0]
    )

    return float(statistic)


def safe_quantile(
    values: pd.Series,
    quantile: float,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return float("nan")

    return float(numeric.quantile(quantile))


# ======================================================================
# Input loading
# ======================================================================


BASE_COLUMNS = {
    "year",
    "exporter_iso",
    "importer_iso",
    "target_cmd",
    "product_group",
    "factor_source",
    "observed_cif_fob_factor",
    "clean_estimation_eligible",
    "prediction_eligible",
    "estimation_evidence_value",
    "estimation_source_weight",
    "import_direct_covered_primary_value",
    "export_value_fob",
    "import_value_cif",
    "pair_uses_weight_proxy",
    "pair_uses_estimated_weight",
    "import_weight_estimated",
    "weight_concordance",
    "export_weight_kg",
    "import_weight_kg",
    "lagged_pair_median_factor",
    "years_since_latest_pair_factor",
    "lagged_pair_factor_observations",
    "lagged_pair_factor_years",
    "log_distance",
    "log_distance_squared",
    "contiguity",
    "exporter_landlocked",
    "importer_landlocked",
    "log_oil_price_x_log_distance",
    "log_product_median_unit_value",
    "primary_model_name",
    "estimation_holdout_year",
}

TRAINING_FEATURE_AUDIT_COLUMNS = {
    "weight_concordance",
    "export_weight_kg",
    "import_weight_kg",
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
    "latest_pair_median_factor",
    "pair_geography_found",
    "distance_model",
}



def detect_prediction_columns(
    available_columns: set[str],
) -> tuple[list[str], list[str]]:
    """
    Return final full-sample and genuine holdout prediction columns.

    The holdout columns are the important ones for residual evaluation,
    because they are out of sample. Final prediction columns are useful
    for distribution comparison only.
    """
    final_factor_columns = sorted(
        column
        for column in available_columns
        if column.endswith(
            "_estimated_cif_fob_factor"
        )
        and column != "estimated_cif_fob_factor"
    )

    holdout_factor_columns = sorted(
        column
        for column in available_columns
        if column.endswith(
            "_holdout_predicted_factor"
        )
    )

    return final_factor_columns, holdout_factor_columns


def load_estimated_rates(
    *,
    input_path: Path,
    threads: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Estimated-rates Parquet not found: {input_path}"
        )

    available = read_parquet_columns(
        path=input_path,
        threads=threads,
    )

    required = {
        "year",
        "exporter_iso",
        "importer_iso",
        "target_cmd",
        "observed_cif_fob_factor",
        "clean_estimation_eligible",
    }

    missing = sorted(required - available)

    if missing:
        raise RuntimeError(
            "Estimated-rates Parquet is missing required columns: "
            + ", ".join(missing)
        )

    final_factor_columns, holdout_factor_columns = (
        detect_prediction_columns(available)
    )

    selected = sorted(
        (BASE_COLUMNS & available)
        | set(final_factor_columns)
        | set(holdout_factor_columns)
    )

    select_sql = ",\n".join(
        quote_identifier(column)
        for column in selected
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        frame = con.execute(
            f"""
            SELECT
                {select_sql}
            FROM read_parquet({sql_literal(input_path)})
            """
        ).df()

    finally:
        con.close()

    if frame.empty:
        raise RuntimeError(
            "Estimated-rates Parquet contains no rows."
        )

    return (
        frame,
        final_factor_columns,
        holdout_factor_columns,
    )


def enrich_from_training_features(
    *,
    data: pd.DataFrame,
    training_features_path: Path,
    threads: int,
) -> pd.DataFrame:
    """
    Restore evidence-quality columns that Step 5 may not retain.

    The join uses the canonical rate-estimation key and does not alter
    model predictions. When no additional fields are available, the
    original frame is returned unchanged.
    """
    if not training_features_path.exists():
        print(
            "Training-features artifact not found; continuing with "
            "estimated_rates columns only: "
            f"{training_features_path}"
        )
        return data

    available = read_parquet_columns(
        path=training_features_path,
        threads=threads,
    )

    join_keys = [
        "year",
        "exporter_iso",
        "importer_iso",
        "target_cmd",
    ]

    missing_keys = [
        key
        for key in join_keys
        if key not in available
    ]

    if missing_keys:
        raise RuntimeError(
            "Training-features Parquet is missing join keys: "
            + ", ".join(missing_keys)
        )

    additional_columns = sorted(
        (
            TRAINING_FEATURE_AUDIT_COLUMNS
            & available
        )
        - set(data.columns)
    )

    if not additional_columns:
        return data

    selected = join_keys + additional_columns

    select_sql = ",\n".join(
        quote_identifier(column)
        for column in selected
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        audit_fields = con.execute(
            f"""
            SELECT
                {select_sql}
            FROM read_parquet(
                {sql_literal(training_features_path)}
            )
            """
        ).df()

    finally:
        con.close()

    duplicate_mask = audit_fields.duplicated(
        subset=join_keys,
        keep=False,
    )

    if duplicate_mask.any():
        raise RuntimeError(
            "Training-features artifact contains duplicate canonical "
            f"keys: {int(duplicate_mask.sum()):,} rows."
        )

    enriched = data.merge(
        audit_fields,
        on=join_keys,
        how="left",
        validate="one_to_one",
    )

    print(
        "Restored audit columns from training features: "
        + ", ".join(additional_columns)
    )

    return enriched


# ======================================================================
# Clean sample and evidence fields
# ======================================================================


def derive_quantity_agreement(
    frame: pd.DataFrame,
) -> pd.Series:
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

    minimum = pd.concat(
        [export_weight, import_weight],
        axis=1,
    ).min(axis=1, skipna=False)

    maximum = pd.concat(
        [export_weight, import_weight],
        axis=1,
    ).max(axis=1, skipna=False)

    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype="float64",
    )

    valid = minimum.gt(0) & maximum.gt(0)

    result.loc[valid] = (
        minimum.loc[valid]
        / maximum.loc[valid]
    )

    return result


def prepare_clean_sample(
    *,
    data: pd.DataFrame,
    tail_threshold: float,
    extreme_threshold: float,
) -> pd.DataFrame:
    clean = data.loc[
        boolean_series(
            data,
            "clean_estimation_eligible",
        )
    ].copy()

    if clean.empty:
        raise RuntimeError(
            "No clean estimation observations were found."
        )

    clean["year"] = pd.to_numeric(
        clean["year"],
        errors="raise",
    ).astype(int)

    clean["observed_cif_fob_factor"] = pd.to_numeric(
        clean["observed_cif_fob_factor"],
        errors="coerce",
    )

    clean = clean.loc[
        clean["observed_cif_fob_factor"].notna()
    ].copy()

    clean["is_upper_tail"] = (
        clean["observed_cif_fob_factor"]
        .ge(tail_threshold)
    )

    clean["is_extreme_tail"] = (
        clean["observed_cif_fob_factor"]
        .ge(extreme_threshold)
    )

    clean["quantity_agreement_ratio"] = (
        derive_quantity_agreement(clean)
    )

    clean["any_proxy_weight"] = (
        boolean_series(
            clean,
            "pair_uses_weight_proxy",
        )
    )

    clean["any_estimated_weight"] = (
        boolean_series(
            clean,
            "pair_uses_estimated_weight",
        )
        | boolean_series(
            clean,
            "import_weight_estimated",
        )
    )

    clean["both_landlocked"] = (
        numeric_series(
            clean,
            "exporter_landlocked",
        ).fillna(0).gt(0)
        & numeric_series(
            clean,
            "importer_landlocked",
        ).fillna(0).gt(0)
    )

    clean["route"] = (
        clean["exporter_iso"].astype("string")
        + "->"
        + clean["importer_iso"].astype("string")
    )

    return clean


# ======================================================================
# Strictly prior persistence
# ======================================================================


def add_pair_product_persistence(
    *,
    clean: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    result = clean.sort_values(
        [
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "year",
        ]
    ).copy()

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
        groups.cumcount().astype("int64")
    )

    result["prior_pair_product_high_years"] = (
        groups["is_upper_tail"]
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

    result["prior_pair_product_max_factor"] = (
        groups["observed_cif_fob_factor"]
        .transform(
            lambda values: (
                values.expanding()
                .max()
                .shift(1)
            )
        )
    )

    return result


def build_pair_year_history(
    *,
    clean: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
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
        groups.cumcount().astype("int64")
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


def add_persistence_fields(
    *,
    clean: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = add_pair_product_persistence(
        clean=clean,
        threshold=threshold,
    )

    pair_year = build_pair_year_history(
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


def classify_tail_credibility(
    *,
    clean: pd.DataFrame,
    minimum_prior_years: int,
    historical_high_share_threshold: float,
    quantity_agreement_threshold: float,
) -> pd.DataFrame:
    result = clean.copy()

    persistent_pair = (
        result["prior_pair_years"]
        .ge(minimum_prior_years)
        & result["prior_pair_high_share"]
        .ge(historical_high_share_threshold)
        .fillna(False)
    )

    persistent_pair_product = (
        result["prior_pair_product_years"]
        .ge(minimum_prior_years)
        & result["prior_pair_product_high_share"]
        .ge(historical_high_share_threshold)
        .fillna(False)
    )

    result["persistent_high_pair"] = persistent_pair
    result["persistent_high_pair_product"] = (
        persistent_pair_product
    )

    result["persistent_high"] = (
        persistent_pair
        | persistent_pair_product
    )

    result["weak_quantity_agreement"] = (
        result["quantity_agreement_ratio"]
        .notna()
        & result["quantity_agreement_ratio"]
        .lt(quantity_agreement_threshold)
    )

    result["weak_evidence"] = (
        result["any_proxy_weight"]
        | result["any_estimated_weight"]
        | result["weak_quantity_agreement"]
    )

    result["credibility_class"] = "not_upper_tail"

    upper = result["is_upper_tail"]

    result.loc[
        upper,
        "credibility_class",
    ] = "isolated_high_good_evidence"

    result.loc[
        upper & result["weak_evidence"],
        "credibility_class",
    ] = "isolated_high_weak_evidence"

    insufficient_history = (
        result["prior_pair_years"]
        .lt(minimum_prior_years)
        & result["prior_pair_product_years"]
        .lt(minimum_prior_years)
    )

    result.loc[
        upper & insufficient_history,
        "credibility_class",
    ] = "insufficient_history"

    result.loc[
        upper & result["persistent_high"],
        "credibility_class",
    ] = "persistent_high"

    return result


# ======================================================================
# Overall and evidence summaries
# ======================================================================


def build_overall_summary(
    *,
    clean: pd.DataFrame,
    tail_threshold: float,
    extreme_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    groups = {
        "all_clean": pd.Series(
            True,
            index=clean.index,
        ),
        f"upper_tail_ge_{tail_threshold:g}": (
            clean["is_upper_tail"]
        ),
        f"extreme_tail_ge_{extreme_threshold:g}": (
            clean["is_extreme_tail"]
        ),
        f"non_tail_lt_{tail_threshold:g}": (
            ~clean["is_upper_tail"]
        ),
    }

    for name, mask in groups.items():
        subset = clean.loc[mask]

        rows.append(
            {
                "group": name,
                "n_rows": int(len(subset)),
                "row_share": (
                    len(subset) / len(clean)
                    if len(clean)
                    else np.nan
                ),
                "median_factor": safe_quantile(
                    subset["observed_cif_fob_factor"],
                    0.50,
                ),
                "p90_factor": safe_quantile(
                    subset["observed_cif_fob_factor"],
                    0.90,
                ),
                "p95_factor": safe_quantile(
                    subset["observed_cif_fob_factor"],
                    0.95,
                ),
                "median_evidence_value": safe_quantile(
                    numeric_series(
                        subset,
                        "estimation_evidence_value",
                    ),
                    0.50,
                ),
                "median_quantity_agreement": safe_quantile(
                    subset["quantity_agreement_ratio"],
                    0.50,
                ),
                "proxy_weight_share": (
                    subset["any_proxy_weight"].mean()
                    if len(subset)
                    else np.nan
                ),
                "estimated_weight_share": (
                    subset["any_estimated_weight"].mean()
                    if len(subset)
                    else np.nan
                ),
                "persistent_high_share": (
                    subset["persistent_high"].mean()
                    if len(subset)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_credibility_summary(
    *,
    clean: pd.DataFrame,
) -> pd.DataFrame:
    tail = clean.loc[
        clean["is_upper_tail"]
    ].copy()

    if tail.empty:
        return pd.DataFrame()

    result = (
        tail.groupby(
            "credibility_class",
            dropna=False,
            as_index=False,
        )
        .agg(
            n_rows=(
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
            median_prior_pair_high_share=(
                "prior_pair_high_share",
                "median",
            ),
            median_prior_pair_product_high_share=(
                "prior_pair_product_high_share",
                "median",
            ),
        )
    )

    result["row_share"] = (
        result["n_rows"]
        / max(len(tail), 1)
    )

    return result.sort_values(
        "n_rows",
        ascending=False,
    ).reset_index(drop=True)


# ======================================================================
# Concentration analysis
# ======================================================================


def summarize_dimension(
    *,
    clean: pd.DataFrame,
    dimension_name: str,
    column: str,
    minimum_group_rows: int,
    top_n: int,
) -> pd.DataFrame:
    if column not in clean.columns:
        return pd.DataFrame()

    overall_tail_share = float(
        clean["is_upper_tail"].mean()
    )

    grouped = (
        clean.groupby(
            column,
            dropna=False,
            as_index=False,
        )
        .agg(
            n_rows=(
                "observed_cif_fob_factor",
                "size",
            ),
            n_upper_tail=(
                "is_upper_tail",
                "sum",
            ),
            tail_share=(
                "is_upper_tail",
                "mean",
            ),
            median_factor=(
                "observed_cif_fob_factor",
                "median",
            ),
            p95_factor=(
                "observed_cif_fob_factor",
                lambda values: values.quantile(0.95),
            ),
            median_evidence_value=(
                "estimation_evidence_value",
                "median",
            ),
            persistent_tail_share=(
                "persistent_high",
                "mean",
            ),
        )
    )

    grouped = grouped.loc[
        grouped["n_rows"].ge(
            minimum_group_rows
        )
    ].copy()

    if grouped.empty:
        return grouped

    grouped["dimension"] = dimension_name
    grouped["group_value"] = (
        grouped[column].astype("string")
    )

    grouped["tail_lift"] = np.where(
        overall_tail_share > 0,
        grouped["tail_share"]
        / overall_tail_share,
        np.nan,
    )

    grouped = grouped.sort_values(
        [
            "tail_lift",
            "n_upper_tail",
            "n_rows",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).head(top_n)

    return grouped[
        [
            "dimension",
            "group_value",
            "n_rows",
            "n_upper_tail",
            "tail_share",
            "tail_lift",
            "median_factor",
            "p95_factor",
            "median_evidence_value",
            "persistent_tail_share",
        ]
    ]


def build_concentration_summary(
    *,
    clean: pd.DataFrame,
    minimum_group_rows: int,
    top_n: int,
) -> pd.DataFrame:
    dimensions = {
        "year": "year",
        "exporter": "exporter_iso",
        "importer": "importer_iso",
        "route": "route",
        "product": "target_cmd",
        "product_group": "product_group",
        "factor_source": "factor_source",
        "both_landlocked": "both_landlocked",
        "contiguity": "contiguity",
    }

    frames: list[pd.DataFrame] = []

    for dimension_name, column in dimensions.items():
        summary = summarize_dimension(
            clean=clean,
            dimension_name=dimension_name,
            column=column,
            minimum_group_rows=minimum_group_rows,
            top_n=top_n,
        )

        if not summary.empty:
            frames.append(summary)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ======================================================================
# Candidate-predictor screening
# ======================================================================


NUMERIC_CANDIDATE_COLUMNS = [
    "log_distance",
    "log_distance_squared",
    "contiguity",
    "exporter_landlocked",
    "importer_landlocked",
    "both_landlocked",
    "lagged_pair_median_factor",
    "years_since_latest_pair_factor",
    "lagged_pair_factor_observations",
    "lagged_pair_factor_years",
    "log_oil_price_x_log_distance",
    "log_product_median_unit_value",
    "estimation_evidence_value",
    "quantity_agreement_ratio",
    "prior_pair_high_share",
    "prior_pair_product_high_share",
    "prior_pair_product_median_factor",
    "latest_prior_pair_product_factor",
    "prior_pair_median_factor",
    "latest_prior_pair_factor",
]


def build_numeric_predictor_screen(
    *,
    clean: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    tail_indicator = (
        clean["is_upper_tail"]
        .astype(float)
    )

    observed_factor = clean[
        "observed_cif_fob_factor"
    ]

    for column in NUMERIC_CANDIDATE_COLUMNS:
        if column not in clean.columns:
            continue

        values = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

        available = values.notna()

        if available.sum() < 3:
            continue

        tail_values = values.loc[
            clean["is_upper_tail"]
        ].dropna()

        non_tail_values = values.loc[
            ~clean["is_upper_tail"]
        ].dropna()

        rows.append(
            {
                "predictor": column,
                "n_available": int(available.sum()),
                "missing_share": float(
                    1.0 - available.mean()
                ),
                "n_unique": int(
                    values.loc[available].nunique()
                ),
                "spearman_with_factor": safe_spearman(
                    values,
                    observed_factor,
                ),
                "spearman_with_tail_indicator": (
                    safe_spearman(
                        values,
                        tail_indicator,
                    )
                ),
                "median_non_tail": (
                    float(non_tail_values.median())
                    if not non_tail_values.empty
                    else np.nan
                ),
                "median_upper_tail": (
                    float(tail_values.median())
                    if not tail_values.empty
                    else np.nan
                ),
                "median_difference_tail_minus_non_tail": (
                    (
                        float(tail_values.median())
                        - float(non_tail_values.median())
                    )
                    if (
                        not tail_values.empty
                        and not non_tail_values.empty
                    )
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["absolute_tail_spearman"] = (
        result["spearman_with_tail_indicator"]
        .abs()
    )

    return result.sort_values(
        [
            "absolute_tail_spearman",
            "n_available",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


def build_numeric_predictor_bands(
    *,
    clean: pd.DataFrame,
    minimum_rows_per_band: int = 100,
) -> pd.DataFrame:
    """
    Show nonlinear tail incidence by predictor quintile.

    This is often more informative than a single correlation because
    HGB relationships need not be monotonic.
    """
    rows: list[dict[str, Any]] = []

    overall_tail_share = float(
        clean["is_upper_tail"].mean()
    )

    for column in NUMERIC_CANDIDATE_COLUMNS:
        if column not in clean.columns:
            continue

        values = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

        available = values.notna()

        if available.sum() < 5:
            continue

        unique_count = int(
            values.loc[available].nunique()
        )

        if unique_count < 2:
            continue

        if unique_count <= 10:
            band = values.astype("string")
        else:
            try:
                band = pd.qcut(
                    values,
                    q=5,
                    duplicates="drop",
                ).astype("string")
            except ValueError:
                continue

        working = pd.DataFrame(
            {
                "band": band,
                "value": values,
                "factor": clean[
                    "observed_cif_fob_factor"
                ],
                "is_upper_tail": clean[
                    "is_upper_tail"
                ],
                "persistent_high": clean[
                    "persistent_high"
                ],
            }
        ).dropna(
            subset=["band"]
        )

        grouped = (
            working.groupby(
                "band",
                dropna=False,
                as_index=False,
            )
            .agg(
                n_rows=(
                    "factor",
                    "size",
                ),
                value_min=(
                    "value",
                    "min",
                ),
                value_median=(
                    "value",
                    "median",
                ),
                value_max=(
                    "value",
                    "max",
                ),
                median_factor=(
                    "factor",
                    "median",
                ),
                p95_factor=(
                    "factor",
                    lambda values: values.quantile(0.95),
                ),
                tail_share=(
                    "is_upper_tail",
                    "mean",
                ),
                persistent_tail_share=(
                    "persistent_high",
                    "mean",
                ),
            )
        )

        grouped = grouped.loc[
            grouped["n_rows"].ge(
                minimum_rows_per_band
            )
        ].copy()

        if grouped.empty:
            continue

        grouped["predictor"] = column
        grouped["tail_lift"] = np.where(
            overall_tail_share > 0,
            grouped["tail_share"]
            / overall_tail_share,
            np.nan,
        )

        rows.extend(
            grouped[
                [
                    "predictor",
                    "band",
                    "n_rows",
                    "value_min",
                    "value_median",
                    "value_max",
                    "median_factor",
                    "p95_factor",
                    "tail_share",
                    "tail_lift",
                    "persistent_tail_share",
                ]
            ].to_dict("records")
        )

    return pd.DataFrame(rows)


# ======================================================================
# Genuine holdout residual audit
# ======================================================================


def model_name_from_holdout_column(
    column: str,
) -> str:
    suffix = "_holdout_predicted_factor"

    if column.endswith(suffix):
        return column[: -len(suffix)]

    return column


def build_oos_residual_observations(
    *,
    clean: pd.DataFrame,
    holdout_factor_columns: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    observed = pd.to_numeric(
        clean["observed_cif_fob_factor"],
        errors="coerce",
    )

    for column in holdout_factor_columns:
        predictions = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

        valid = (
            observed.gt(0)
            & predictions.gt(0)
        )

        if not valid.any():
            continue

        model_name = model_name_from_holdout_column(
            column
        )

        subset_columns = [
            column_name
            for column_name in [
                "year",
                "exporter_iso",
                "importer_iso",
                "route",
                "target_cmd",
                "product_group",
                "observed_cif_fob_factor",
                "estimation_evidence_value",
                "quantity_agreement_ratio",
                "any_proxy_weight",
                "any_estimated_weight",
                "is_upper_tail",
                "is_extreme_tail",
                "persistent_high",
                "credibility_class",
                "log_distance",
                "lagged_pair_median_factor",
                "years_since_latest_pair_factor",
                "log_product_median_unit_value",
                "prior_pair_high_share",
                "prior_pair_product_high_share",
            ]
            if column_name in clean.columns
        ]

        subset = clean.loc[
            valid,
            subset_columns,
        ].copy()

        subset["model_name"] = model_name
        subset["predicted_factor"] = (
            predictions.loc[valid]
            .to_numpy(dtype=float)
        )

        subset["log_residual"] = (
            np.log(
                subset[
                    "observed_cif_fob_factor"
                ].to_numpy(dtype=float)
            )
            - np.log(
                subset[
                    "predicted_factor"
                ].to_numpy(dtype=float)
            )
        )

        subset["absolute_percentage_error"] = (
            np.abs(
                subset["predicted_factor"]
                - subset[
                    "observed_cif_fob_factor"
                ]
            )
            / subset[
                "observed_cif_fob_factor"
            ].clip(lower=1e-12)
        )

        subset["underpredicted"] = (
            subset["log_residual"] > 0
        )

        rows.append(subset)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


def build_oos_model_summary(
    *,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    if residuals.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for (
        model_name,
        subset_name,
    ), group in residuals.assign(
        subset=np.where(
            residuals["is_upper_tail"],
            "upper_tail",
            "non_tail",
        )
    ).groupby(
        [
            "model_name",
            "subset",
        ],
        sort=True,
    ):
        rows.append(
            {
                "model_name": model_name,
                "subset": subset_name,
                "n_rows": int(len(group)),
                "median_ape": float(
                    group[
                        "absolute_percentage_error"
                    ].median()
                ),
                "p90_ape": float(
                    group[
                        "absolute_percentage_error"
                    ].quantile(0.90)
                ),
                "p95_ape": float(
                    group[
                        "absolute_percentage_error"
                    ].quantile(0.95)
                ),
                "median_log_residual": float(
                    group["log_residual"].median()
                ),
                "mean_log_residual": float(
                    group["log_residual"].mean()
                ),
                "underprediction_share": float(
                    group["underpredicted"].mean()
                ),
                "observed_p95": float(
                    group[
                        "observed_cif_fob_factor"
                    ].quantile(0.95)
                ),
                "predicted_p95": float(
                    group[
                        "predicted_factor"
                    ].quantile(0.95)
                ),
            }
        )

    return pd.DataFrame(rows)


def build_oos_residual_concentration(
    *,
    residuals: pd.DataFrame,
    minimum_group_rows: int,
    top_n: int,
) -> pd.DataFrame:
    if residuals.empty:
        return pd.DataFrame()

    dimensions = {
        "exporter": "exporter_iso",
        "importer": "importer_iso",
        "route": "route",
        "product": "target_cmd",
        "product_group": "product_group",
    }

    frames: list[pd.DataFrame] = []

    tail_residuals = residuals.loc[
        residuals["is_upper_tail"]
    ].copy()

    for model_name, model_frame in tail_residuals.groupby(
        "model_name",
        sort=True,
    ):
        for dimension, column in dimensions.items():
            if column not in model_frame.columns:
                continue

            grouped = (
                model_frame.groupby(
                    column,
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    n_rows=(
                        "log_residual",
                        "size",
                    ),
                    median_log_residual=(
                        "log_residual",
                        "median",
                    ),
                    mean_log_residual=(
                        "log_residual",
                        "mean",
                    ),
                    p90_log_residual=(
                        "log_residual",
                        lambda values: values.quantile(0.90),
                    ),
                    median_ape=(
                        "absolute_percentage_error",
                        "median",
                    ),
                    underprediction_share=(
                        "underpredicted",
                        "mean",
                    ),
                    persistent_tail_share=(
                        "persistent_high",
                        "mean",
                    ),
                )
            )

            grouped = grouped.loc[
                grouped["n_rows"].ge(
                    minimum_group_rows
                )
            ].copy()

            if grouped.empty:
                continue

            grouped["model_name"] = model_name
            grouped["dimension"] = dimension
            grouped["group_value"] = (
                grouped[column].astype("string")
            )

            grouped = grouped.sort_values(
                [
                    "median_log_residual",
                    "n_rows",
                ],
                ascending=[
                    False,
                    False,
                ],
            ).head(top_n)

            frames.append(
                grouped[
                    [
                        "model_name",
                        "dimension",
                        "group_value",
                        "n_rows",
                        "median_log_residual",
                        "mean_log_residual",
                        "p90_log_residual",
                        "median_ape",
                        "underprediction_share",
                        "persistent_tail_share",
                    ]
                ]
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_oos_predictor_residual_screen(
    *,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    if residuals.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for model_name, model_frame in residuals.groupby(
        "model_name",
        sort=True,
    ):
        tail = model_frame.loc[
            model_frame["is_upper_tail"]
        ]

        if tail.empty:
            continue

        for predictor in NUMERIC_CANDIDATE_COLUMNS:
            if predictor not in tail.columns:
                continue

            values = pd.to_numeric(
                tail[predictor],
                errors="coerce",
            )

            rows.append(
                {
                    "model_name": model_name,
                    "predictor": predictor,
                    "n_available": int(
                        values.notna().sum()
                    ),
                    "spearman_with_log_residual": (
                        safe_spearman(
                            values,
                            tail["log_residual"],
                        )
                    ),
                    "spearman_with_ape": (
                        safe_spearman(
                            values,
                            tail[
                                "absolute_percentage_error"
                            ],
                        )
                    ),
                    "median_value_underpredicted": (
                        float(
                            values.loc[
                                tail["underpredicted"]
                            ].median()
                        )
                        if values.loc[
                            tail["underpredicted"]
                        ].notna().any()
                        else np.nan
                    ),
                    "median_value_not_underpredicted": (
                        float(
                            values.loc[
                                ~tail["underpredicted"]
                            ].median()
                        )
                        if values.loc[
                            ~tail["underpredicted"]
                        ].notna().any()
                        else np.nan
                    ),
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["absolute_residual_spearman"] = (
        result["spearman_with_log_residual"]
        .abs()
    )

    return result.sort_values(
        [
            "model_name",
            "absolute_residual_spearman",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


# ======================================================================
# Final prediction distribution comparison
# ======================================================================


def build_final_prediction_distribution(
    *,
    data: pd.DataFrame,
    final_factor_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in final_factor_columns:
        values = pd.to_numeric(
            data[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        model_name = column[
            : -len(
                "_estimated_cif_fob_factor"
            )
        ]

        rows.append(
            {
                "model_name": model_name,
                "n_predictions": int(len(values)),
                "minimum": float(values.min()),
                "p05": float(values.quantile(0.05)),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "maximum": float(values.max()),
                "share_at_or_below_1": float(
                    values.le(1.0 + 1e-12).mean()
                ),
                "share_at_or_above_3": float(
                    values.ge(3.0 - 1e-12).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# Output
# ======================================================================


def write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    frame.to_csv(
        path,
        index=False,
    )


def write_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    frame.to_parquet(
        path,
        index=False,
    )


def print_key_results(
    *,
    clean: pd.DataFrame,
    credibility_summary: pd.DataFrame,
    oos_model_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    n_clean = int(len(clean))
    n_tail = int(clean["is_upper_tail"].sum())
    n_extreme = int(clean["is_extreme_tail"].sum())

    print("\n" + "=" * 96)
    print("CIF/FOB SAVED-DATA UPPER-TAIL AUDIT")
    print("=" * 96)
    print(f"Clean observations:   {n_clean:,}")
    print(
        f"Upper-tail rows:      {n_tail:,} "
        f"({100.0 * n_tail / max(n_clean, 1):.2f}%)"
    )
    print(
        f"Extreme-tail rows:    {n_extreme:,} "
        f"({100.0 * n_extreme / max(n_clean, 1):.2f}%)"
    )

    if not credibility_summary.empty:
        print("\nTAIL CREDIBILITY")
        print(
            credibility_summary[
                [
                    "credibility_class",
                    "n_rows",
                    "row_share",
                    "median_factor",
                    "median_evidence_value",
                ]
            ].to_string(index=False)
        )

    if not oos_model_summary.empty:
        print("\nGENUINE HOLDOUT RESIDUAL SUMMARY")
        print(
            oos_model_summary.to_string(
                index=False
            )
        )

    print("\nArtifacts written to:")
    print(output_dir)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config()

    audit_cfg = (
        cfg.method.estimation
        .diagnostics
        .upper_tail_audit
    )

    input_path = (
        args.input
        if args.input is not None
        else cfg.product_path(
            "estimated_rates"
        )
    )

    training_features_path = (
        args.training_features
        if args.training_features is not None
        else cfg.product_path(
            "training_features"
        )
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else cfg.output_dir()
        / "tail_audit_saved"
    )

    tail_threshold = (
        float(args.tail_threshold)
        if args.tail_threshold is not None
        else float(audit_cfg.threshold)
    )

    extreme_threshold = (
        float(args.extreme_threshold)
        if args.extreme_threshold is not None
        else float(audit_cfg.extreme_threshold)
    )

    if tail_threshold <= 0:
        raise ValueError(
            "tail_threshold must be positive."
        )

    if extreme_threshold < tail_threshold:
        raise ValueError(
            "extreme_threshold must be greater than or equal to "
            "tail_threshold."
        )

    minimum_prior_years = int(
        audit_cfg.persistence.minimum_prior_years
    )

    historical_high_share_threshold = float(
        audit_cfg.persistence
        .historical_high_share_threshold
    )

    quantity_bands = tuple(
        float(value)
        for value
        in audit_cfg.quantity_agreement_bands
    )

    quantity_agreement_threshold = (
        max(quantity_bands)
        if quantity_bands
        else 0.95
    )

    print(f"Input:               {input_path}")
    print(f"Training features:   {training_features_path}")
    print(f"Output directory:    {output_dir}")
    print(f"Tail threshold:      {tail_threshold:g}")
    print(f"Extreme threshold:   {extreme_threshold:g}")
    print(
        f"Minimum group rows:  "
        f"{args.minimum_group_rows:,}"
    )
    print(f"Top groups:          {args.top_n:,}")

    (
        data,
        final_factor_columns,
        holdout_factor_columns,
    ) = load_estimated_rates(
        input_path=input_path,
        threads=cfg.engine.threads,
    )

    data = enrich_from_training_features(
        data=data,
        training_features_path=training_features_path,
        threads=cfg.engine.threads,
    )

    clean = prepare_clean_sample(
        data=data,
        tail_threshold=tail_threshold,
        extreme_threshold=extreme_threshold,
    )

    clean, pair_year = add_persistence_fields(
        clean=clean,
        threshold=tail_threshold,
    )

    clean = classify_tail_credibility(
        clean=clean,
        minimum_prior_years=minimum_prior_years,
        historical_high_share_threshold=(
            historical_high_share_threshold
        ),
        quantity_agreement_threshold=(
            quantity_agreement_threshold
        ),
    )

    tail = clean.loc[
        clean["is_upper_tail"]
    ].copy()

    if tail.empty:
        raise RuntimeError(
            "No clean observations meet the configured upper-tail "
            f"threshold of {tail_threshold:g}."
        )

    overall_summary = build_overall_summary(
        clean=clean,
        tail_threshold=tail_threshold,
        extreme_threshold=extreme_threshold,
    )

    credibility_summary = (
        build_credibility_summary(
            clean=clean,
        )
    )

    concentration_summary = (
        build_concentration_summary(
            clean=clean,
            minimum_group_rows=(
                args.minimum_group_rows
            ),
            top_n=args.top_n,
        )
    )

    numeric_predictor_screen = (
        build_numeric_predictor_screen(
            clean=clean,
        )
    )

    numeric_predictor_bands = (
        build_numeric_predictor_bands(
            clean=clean,
            minimum_rows_per_band=max(
                25,
                args.minimum_group_rows,
            ),
        )
    )

    oos_residuals = (
        build_oos_residual_observations(
            clean=clean,
            holdout_factor_columns=(
                holdout_factor_columns
            ),
        )
    )

    oos_model_summary = (
        build_oos_model_summary(
            residuals=oos_residuals,
        )
    )

    oos_residual_concentration = (
        build_oos_residual_concentration(
            residuals=oos_residuals,
            minimum_group_rows=max(
                10,
                args.minimum_group_rows // 4,
            ),
            top_n=args.top_n,
        )
    )

    oos_predictor_residual_screen = (
        build_oos_predictor_residual_screen(
            residuals=oos_residuals,
        )
    )

    final_prediction_distribution = (
        build_final_prediction_distribution(
            data=data,
            final_factor_columns=(
                final_factor_columns
            ),
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        overall_summary,
        output_dir
        / "tail_overall_summary.csv",
    )

    write_csv(
        credibility_summary,
        output_dir
        / "tail_credibility_summary.csv",
    )

    write_csv(
        concentration_summary,
        output_dir
        / "tail_concentration_summary.csv",
    )

    write_csv(
        numeric_predictor_screen,
        output_dir
        / "tail_numeric_predictor_screen.csv",
    )

    write_csv(
        numeric_predictor_bands,
        output_dir
        / "tail_numeric_predictor_bands.csv",
    )

    write_csv(
        pair_year,
        output_dir
        / "tail_pair_year_history.csv",
    )

    write_parquet(
        tail,
        output_dir
        / "tail_observations.parquet",
    )

    write_csv(
        oos_model_summary,
        output_dir
        / "tail_oos_model_summary.csv",
    )

    write_csv(
        oos_residual_concentration,
        output_dir
        / "tail_oos_residual_concentration.csv",
    )

    write_csv(
        oos_predictor_residual_screen,
        output_dir
        / "tail_oos_predictor_residual_screen.csv",
    )

    if not oos_residuals.empty:
        write_parquet(
            oos_residuals,
            output_dir
            / "tail_oos_residual_observations.parquet",
        )

    write_csv(
        final_prediction_distribution,
        output_dir
        / "tail_final_prediction_distribution.csv",
    )

    print_key_results(
        clean=clean,
        credibility_summary=(
            credibility_summary
        ),
        oos_model_summary=oos_model_summary,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
