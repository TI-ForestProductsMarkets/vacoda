from __future__ import annotations

import argparse
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
from vacoda.config.cif_fob import CifFobConfig
from vacoda.config.pipeline import (
    PipelineEngineConfig,
    PipelineScopeConfig,
)


KEY_COLUMNS = [
    "year",
    "exporter_iso",
    "importer_iso",
    "target_cmd",
]

TRAINING_COLUMNS = {
    "product_group",
    "observed_cif_fob_factor",
    "clean_estimation_eligible",
    "estimation_evidence_value",
    "raw_value_ratio",
    "unit_value_ratio",
    "weight_concordance",
    "export_weight_kg",
    "import_weight_kg",
    "export_value_fob",
    "import_value_cif",
    "export_unit_value_usd_per_kg",
    "import_unit_value_usd_per_kg",
    "pair_uses_weight_proxy",
    "pair_uses_estimated_weight",
    "import_weight_estimated",
    "log_distance",
    "contiguity",
    "exporter_landlocked",
    "importer_landlocked",
}


def load_runtime_config() -> CifFobConfig:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate where the unusually large 2022 CIF/FOB-factor "
            "tail is concentrated."
        )
    )

    parser.add_argument(
        "--estimated-rates",
        type=Path,
        default=None,
        help=(
            "Estimated-rates Parquet containing clean_estimation_eligible "
            "and estimation_evidence_value."
        ),
    )

    parser.add_argument(
        "--training-features",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--target-year",
        type=int,
        default=2022,
    )

    parser.add_argument(
        "--tail-threshold",
        type=float,
        default=1.25,
    )

    parser.add_argument(
        "--extreme-threshold",
        type=float,
        default=1.50,
    )

    parser.add_argument(
        "--reference-start-year",
        type=int,
        default=2021,
        help=(
            "First comparison year. The target year itself is excluded."
        ),
    )

    parser.add_argument(
        "--reference-end-year",
        type=int,
        default=2025,
        help=(
            "Last comparison year. The target year itself is excluded."
        ),
    )

    parser.add_argument(
        "--minimum-target-rows",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--minimum-reference-rows",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
    )

    return parser.parse_args()


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def read_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> set[str]:
    rows = con.execute(
        """
        DESCRIBE
        SELECT *
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchall()

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


def safe_quantile(
    values: pd.Series,
    q: float,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return float("nan")

    return float(numeric.quantile(q))


def load_clean_audit_frame(
    *,
    estimated_rates_path: Path,
    training_features_path: Path,
    threads: int,
) -> pd.DataFrame:
    """
    Load clean-sample fields from estimated_rates and audit dimensions
    from training_features, then join them on the canonical key.

    clean_estimation_eligible is produced by estimate_rates.py and is
    therefore not expected to exist in training_features.parquet.
    """
    if not estimated_rates_path.exists():
        raise FileNotFoundError(
            f"Estimated-rates Parquet not found: {estimated_rates_path}"
        )

    if not training_features_path.exists():
        raise FileNotFoundError(
            "Training-features Parquet not found: "
            f"{training_features_path}"
        )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        estimated_available = read_columns(
            con=con,
            path=estimated_rates_path,
        )

        training_available = read_columns(
            con=con,
            path=training_features_path,
        )

        estimated_required = {
            *KEY_COLUMNS,
            "observed_cif_fob_factor",
            "clean_estimation_eligible",
        }

        missing_estimated = sorted(
            estimated_required
            - estimated_available
        )

        if missing_estimated:
            raise RuntimeError(
                "Estimated-rates Parquet is missing required columns: "
                + ", ".join(missing_estimated)
            )

        training_required = set(
            KEY_COLUMNS
        )

        missing_training = sorted(
            training_required
            - training_available
        )

        if missing_training:
            raise RuntimeError(
                "Training-features Parquet is missing canonical keys: "
                + ", ".join(missing_training)
            )

        estimated_selected = sorted(
            estimated_required
            | (
                {
                    "estimation_evidence_value",
                    "estimation_source_weight",
                }
                & estimated_available
            )
        )

        training_selected = sorted(
            set(KEY_COLUMNS)
            | (
                TRAINING_COLUMNS
                & training_available
            )
            - {
                "observed_cif_fob_factor",
                "clean_estimation_eligible",
                "estimation_evidence_value",
            }
        )

        estimated_select_sql = ",\n".join(
            quote_identifier(column)
            for column in estimated_selected
        )

        training_select_sql = ",\n".join(
            quote_identifier(column)
            for column in training_selected
        )

        estimated = con.execute(
            f"""
            SELECT
                {estimated_select_sql}
            FROM read_parquet(?)
            """,
            [str(estimated_rates_path)],
        ).df()

        training = con.execute(
            f"""
            SELECT
                {training_select_sql}
            FROM read_parquet(?)
            """,
            [str(training_features_path)],
        ).df()

    finally:
        con.close()

    for label, frame in [
        ("Estimated rates", estimated),
        ("Training features", training),
    ]:
        duplicate_mask = frame.duplicated(
            subset=KEY_COLUMNS,
            keep=False,
        )

        if duplicate_mask.any():
            raise RuntimeError(
                f"{label} contains duplicate canonical keys: "
                f"{int(duplicate_mask.sum()):,}"
            )

    merged = estimated.merge(
        training,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    clean = merged.loc[
        boolean_series(
            merged,
            "clean_estimation_eligible",
        )
    ].copy()

    if clean.empty:
        raise RuntimeError(
            "No clean estimation observations found in estimated rates."
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

    return clean


def derive_fields(
    *,
    frame: pd.DataFrame,
    tail_threshold: float,
    extreme_threshold: float,
) -> pd.DataFrame:
    result = frame.copy()

    result["is_upper_tail"] = (
        result["observed_cif_fob_factor"]
        .ge(tail_threshold)
    )

    result["is_extreme_tail"] = (
        result["observed_cif_fob_factor"]
        .ge(extreme_threshold)
    )

    result["route"] = (
        result["exporter_iso"].astype("string")
        + "->"
        + result["importer_iso"].astype("string")
    )

    export_weight = numeric_series(
        result,
        "export_weight_kg",
    )
    import_weight = numeric_series(
        result,
        "import_weight_kg",
    )

    minimum_weight = pd.concat(
        [export_weight, import_weight],
        axis=1,
    ).min(
        axis=1,
        skipna=False,
    )

    maximum_weight = pd.concat(
        [export_weight, import_weight],
        axis=1,
    ).max(
        axis=1,
        skipna=False,
    )

    quantity_agreement = numeric_series(
        result,
        "weight_concordance",
    )

    missing_existing = quantity_agreement.isna()
    valid_weights = (
        minimum_weight.gt(0)
        & maximum_weight.gt(0)
    )

    quantity_agreement.loc[
        missing_existing
        & valid_weights
    ] = (
        minimum_weight.loc[
            missing_existing
            & valid_weights
        ]
        / maximum_weight.loc[
            missing_existing
            & valid_weights
        ]
    )

    result["quantity_agreement_ratio"] = (
        quantity_agreement
    )

    result["any_proxy_weight"] = (
        boolean_series(
            result,
            "pair_uses_weight_proxy",
        )
    )

    result["any_estimated_weight"] = (
        boolean_series(
            result,
            "pair_uses_estimated_weight",
        )
        | boolean_series(
            result,
            "import_weight_estimated",
        )
    )

    evidence = numeric_series(
        result,
        "estimation_evidence_value",
    )

    if evidence.notna().any():
        result["log_evidence_value"] = np.where(
            evidence.gt(0),
            np.log1p(evidence),
            np.nan,
        )

    raw_ratio = numeric_series(
        result,
        "raw_value_ratio",
    )

    result["raw_value_ratio_distance_from_one"] = (
        np.abs(
            raw_ratio
            - 1.0
        )
    )

    unit_ratio = numeric_series(
        result,
        "unit_value_ratio",
    )

    result["unit_value_ratio_distance_from_one"] = (
        np.abs(
            unit_ratio
            - 1.0
        )
    )

    result["both_landlocked"] = (
        numeric_series(
            result,
            "exporter_landlocked",
        ).fillna(0).gt(0)
        & numeric_series(
            result,
            "importer_landlocked",
        ).fillna(0).gt(0)
    )

    return result


def build_annual_outcome_summary(
    *,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return (
        frame.groupby(
            "year",
            as_index=False,
            sort=True,
        )
        .agg(
            n_rows=(
                "observed_cif_fob_factor",
                "size",
            ),
            upper_tail_share=(
                "is_upper_tail",
                "mean",
            ),
            extreme_tail_share=(
                "is_extreme_tail",
                "mean",
            ),
            median_factor=(
                "observed_cif_fob_factor",
                "median",
            ),
            p90_factor=(
                "observed_cif_fob_factor",
                lambda values: values.quantile(0.90),
            ),
            p95_factor=(
                "observed_cif_fob_factor",
                lambda values: values.quantile(0.95),
            ),
            p99_factor=(
                "observed_cif_fob_factor",
                lambda values: values.quantile(0.99),
            ),
            median_evidence_value=(
                "estimation_evidence_value",
                "median",
            ),
            median_quantity_agreement=(
                "quantity_agreement_ratio",
                "median",
            ),
            proxy_weight_share=(
                "any_proxy_weight",
                "mean",
            ),
            estimated_weight_share=(
                "any_estimated_weight",
                "mean",
            ),
        )
    )


def concentration_summary(
    *,
    frame: pd.DataFrame,
    column: str,
    dimension: str,
    target_year: int,
    reference_start_year: int,
    reference_end_year: int,
    minimum_target_rows: int,
    minimum_reference_rows: int,
    top_n: int,
) -> pd.DataFrame:
    if column not in frame.columns:
        return pd.DataFrame()

    target = frame.loc[
        frame["year"].eq(target_year)
    ].copy()

    reference = frame.loc[
        frame["year"].between(
            reference_start_year,
            reference_end_year,
        )
        & ~frame["year"].eq(target_year)
    ].copy()

    target_grouped = (
        target.groupby(
            column,
            dropna=False,
            as_index=False,
        )
        .agg(
            target_n_rows=(
                "observed_cif_fob_factor",
                "size",
            ),
            target_n_upper_tail=(
                "is_upper_tail",
                "sum",
            ),
            target_upper_tail_share=(
                "is_upper_tail",
                "mean",
            ),
            target_extreme_tail_share=(
                "is_extreme_tail",
                "mean",
            ),
            target_median_factor=(
                "observed_cif_fob_factor",
                "median",
            ),
            target_p95_factor=(
                "observed_cif_fob_factor",
                lambda values: values.quantile(0.95),
            ),
            target_median_evidence=(
                "estimation_evidence_value",
                "median",
            ),
            target_median_quantity_agreement=(
                "quantity_agreement_ratio",
                "median",
            ),
        )
    )

    reference_grouped = (
        reference.groupby(
            column,
            dropna=False,
            as_index=False,
        )
        .agg(
            reference_n_rows=(
                "observed_cif_fob_factor",
                "size",
            ),
            reference_n_upper_tail=(
                "is_upper_tail",
                "sum",
            ),
            reference_upper_tail_share=(
                "is_upper_tail",
                "mean",
            ),
            reference_extreme_tail_share=(
                "is_extreme_tail",
                "mean",
            ),
            reference_median_factor=(
                "observed_cif_fob_factor",
                "median",
            ),
            reference_p95_factor=(
                "observed_cif_fob_factor",
                lambda values: values.quantile(0.95),
            ),
            reference_median_evidence=(
                "estimation_evidence_value",
                "median",
            ),
            reference_median_quantity_agreement=(
                "quantity_agreement_ratio",
                "median",
            ),
        )
    )

    result = target_grouped.merge(
        reference_grouped,
        on=column,
        how="left",
        validate="one_to_one",
    )

    result = result.loc[
        result["target_n_rows"].ge(
            minimum_target_rows
        )
        & result["reference_n_rows"]
        .fillna(0)
        .ge(
            minimum_reference_rows
        )
    ].copy()

    if result.empty:
        return result

    result["dimension"] = dimension
    result["group_value"] = (
        result[column].astype("string")
    )

    result["tail_share_difference"] = (
        result["target_upper_tail_share"]
        - result["reference_upper_tail_share"]
    )

    result["tail_lift"] = np.where(
        result["reference_upper_tail_share"]
        .gt(0),
        (
            result["target_upper_tail_share"]
            / result["reference_upper_tail_share"]
        ),
        np.nan,
    )

    result["expected_target_tail_rows"] = (
        result["target_n_rows"]
        * result["reference_upper_tail_share"]
    )

    result["excess_target_tail_rows"] = (
        result["target_n_upper_tail"]
        - result["expected_target_tail_rows"]
    )

    result["factor_p95_difference"] = (
        result["target_p95_factor"]
        - result["reference_p95_factor"]
    )

    result = result.sort_values(
        [
            "excess_target_tail_rows",
            "tail_share_difference",
            "target_n_rows",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).head(top_n)

    return result[
        [
            "dimension",
            "group_value",
            "target_n_rows",
            "reference_n_rows",
            "target_n_upper_tail",
            "target_upper_tail_share",
            "reference_upper_tail_share",
            "tail_share_difference",
            "tail_lift",
            "expected_target_tail_rows",
            "excess_target_tail_rows",
            "target_extreme_tail_share",
            "reference_extreme_tail_share",
            "target_median_factor",
            "reference_median_factor",
            "target_p95_factor",
            "reference_p95_factor",
            "factor_p95_difference",
            "target_median_evidence",
            "reference_median_evidence",
            "target_median_quantity_agreement",
            "reference_median_quantity_agreement",
        ]
    ]


def build_concentration_outputs(
    *,
    frame: pd.DataFrame,
    target_year: int,
    reference_start_year: int,
    reference_end_year: int,
    minimum_target_rows: int,
    minimum_reference_rows: int,
    top_n: int,
) -> pd.DataFrame:
    dimensions = {
        "exporter": "exporter_iso",
        "importer": "importer_iso",
        "route": "route",
        "product": "target_cmd",
        "product_group": "product_group",
        "contiguity": "contiguity",
        "both_landlocked": "both_landlocked",
    }

    frames: list[pd.DataFrame] = []

    for dimension, column in dimensions.items():
        summary = concentration_summary(
            frame=frame,
            column=column,
            dimension=dimension,
            target_year=target_year,
            reference_start_year=reference_start_year,
            reference_end_year=reference_end_year,
            minimum_target_rows=minimum_target_rows,
            minimum_reference_rows=minimum_reference_rows,
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


def make_quantile_band(
    *,
    target: pd.Series,
    reference: pd.Series,
    q: int = 5,
) -> tuple[pd.Series, pd.Series]:
    combined = pd.concat(
        [
            pd.to_numeric(
                target,
                errors="coerce",
            ),
            pd.to_numeric(
                reference,
                errors="coerce",
            ),
        ],
        ignore_index=True,
    )

    available = combined.dropna()

    if (
        available.empty
        or available.nunique() < 2
    ):
        return (
            pd.Series(
                "missing",
                index=target.index,
                dtype="string",
            ),
            pd.Series(
                "missing",
                index=reference.index,
                dtype="string",
            ),
        )

    try:
        _, edges = pd.qcut(
            available,
            q=q,
            duplicates="drop",
            retbins=True,
        )
    except ValueError:
        edges = np.array(
            [
                available.min(),
                available.max(),
            ],
            dtype=float,
        )

    edges = np.unique(edges)

    if len(edges) < 2:
        return (
            pd.Series(
                "available",
                index=target.index,
                dtype="string",
            ),
            pd.Series(
                "available",
                index=reference.index,
                dtype="string",
            ),
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    target_numeric = pd.to_numeric(
        target,
        errors="coerce",
    )
    reference_numeric = pd.to_numeric(
        reference,
        errors="coerce",
    )

    target_band = pd.cut(
        target_numeric,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    ).astype("string").fillna("missing")

    reference_band = pd.cut(
        reference_numeric,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    ).astype("string").fillna("missing")

    return target_band, reference_band


def build_numeric_band_summary(
    *,
    frame: pd.DataFrame,
    target_year: int,
    reference_start_year: int,
    reference_end_year: int,
) -> pd.DataFrame:
    numeric_dimensions = [
        "estimation_evidence_value",
        "raw_value_ratio",
        "unit_value_ratio",
        "raw_value_ratio_distance_from_one",
        "unit_value_ratio_distance_from_one",
        "quantity_agreement_ratio",
        "log_distance",
        "export_unit_value_usd_per_kg",
        "import_unit_value_usd_per_kg",
        "export_weight_kg",
        "import_weight_kg",
    ]

    target = frame.loc[
        frame["year"].eq(target_year)
    ].copy()

    reference = frame.loc[
        frame["year"].between(
            reference_start_year,
            reference_end_year,
        )
        & ~frame["year"].eq(target_year)
    ].copy()

    rows: list[dict[str, Any]] = []

    for feature in numeric_dimensions:
        if feature not in frame.columns:
            continue

        target_band, reference_band = (
            make_quantile_band(
                target=target[feature],
                reference=reference[feature],
                q=5,
            )
        )

        target_working = pd.DataFrame(
            {
                "band": target_band,
                "factor": target[
                    "observed_cif_fob_factor"
                ],
                "upper_tail": target[
                    "is_upper_tail"
                ],
                "extreme_tail": target[
                    "is_extreme_tail"
                ],
            }
        )

        reference_working = pd.DataFrame(
            {
                "band": reference_band,
                "factor": reference[
                    "observed_cif_fob_factor"
                ],
                "upper_tail": reference[
                    "is_upper_tail"
                ],
                "extreme_tail": reference[
                    "is_extreme_tail"
                ],
            }
        )

        target_grouped = (
            target_working.groupby(
                "band",
                dropna=False,
                as_index=False,
            )
            .agg(
                target_n_rows=(
                    "factor",
                    "size",
                ),
                target_upper_tail_share=(
                    "upper_tail",
                    "mean",
                ),
                target_extreme_tail_share=(
                    "extreme_tail",
                    "mean",
                ),
                target_median_factor=(
                    "factor",
                    "median",
                ),
                target_p95_factor=(
                    "factor",
                    lambda values: values.quantile(0.95),
                ),
            )
        )

        reference_grouped = (
            reference_working.groupby(
                "band",
                dropna=False,
                as_index=False,
            )
            .agg(
                reference_n_rows=(
                    "factor",
                    "size",
                ),
                reference_upper_tail_share=(
                    "upper_tail",
                    "mean",
                ),
                reference_extreme_tail_share=(
                    "extreme_tail",
                    "mean",
                ),
                reference_median_factor=(
                    "factor",
                    "median",
                ),
                reference_p95_factor=(
                    "factor",
                    lambda values: values.quantile(0.95),
                ),
            )
        )

        merged = target_grouped.merge(
            reference_grouped,
            on="band",
            how="outer",
            validate="one_to_one",
        )

        merged["feature"] = feature
        merged["tail_share_difference"] = (
            merged["target_upper_tail_share"]
            - merged["reference_upper_tail_share"]
        )

        merged["tail_lift"] = np.where(
            merged[
                "reference_upper_tail_share"
            ].gt(0),
            (
                merged["target_upper_tail_share"]
                / merged[
                    "reference_upper_tail_share"
                ]
            ),
            np.nan,
        )

        rows.extend(
            merged[
                [
                    "feature",
                    "band",
                    "target_n_rows",
                    "reference_n_rows",
                    "target_upper_tail_share",
                    "reference_upper_tail_share",
                    "tail_share_difference",
                    "tail_lift",
                    "target_extreme_tail_share",
                    "reference_extreme_tail_share",
                    "target_median_factor",
                    "reference_median_factor",
                    "target_p95_factor",
                    "reference_p95_factor",
                ]
            ].to_dict("records")
        )

    return pd.DataFrame(rows)


def build_target_tail_observations(
    *,
    frame: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    target_tail = frame.loc[
        frame["year"].eq(target_year)
        & frame["is_upper_tail"]
    ].copy()

    sort_columns = [
        "observed_cif_fob_factor",
        "estimation_evidence_value",
    ]

    available_sort_columns = [
        column
        for column in sort_columns
        if column in target_tail.columns
    ]

    return target_tail.sort_values(
        available_sort_columns,
        ascending=[False] * len(
            available_sort_columns
        ),
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config()

    estimated_rates_path = (
        args.estimated_rates
        if args.estimated_rates is not None
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
        / f"targeted_{args.target_year}_outcome_audit"
    )

    if args.extreme_threshold < args.tail_threshold:
        raise ValueError(
            "extreme_threshold must be greater than or equal "
            "to tail_threshold."
        )

    if (
        args.reference_start_year
        > args.reference_end_year
    ):
        raise ValueError(
            "reference_start_year must not exceed "
            "reference_end_year."
        )

    print(
        f"Estimated rates:      "
        f"{estimated_rates_path}"
    )
    print(
        f"Training features:    "
        f"{training_features_path}"
    )
    print(
        f"Target year:          "
        f"{args.target_year}"
    )
    print(
        f"Reference years:      "
        f"{args.reference_start_year}-"
        f"{args.reference_end_year}, "
        f"excluding {args.target_year}"
    )
    print(
        f"Upper-tail threshold: "
        f"{args.tail_threshold:g}"
    )
    print(
        f"Extreme threshold:    "
        f"{args.extreme_threshold:g}"
    )
    print(
        f"Output directory:     "
        f"{output_dir}"
    )

    frame = load_clean_audit_frame(
        estimated_rates_path=estimated_rates_path,
        training_features_path=training_features_path,
        threads=cfg.engine.threads,
    )

    frame = derive_fields(
        frame=frame,
        tail_threshold=args.tail_threshold,
        extreme_threshold=args.extreme_threshold,
    )

    scoped = frame.loc[
        frame["year"].between(
            args.reference_start_year,
            args.reference_end_year,
        )
    ].copy()

    if scoped.loc[
        scoped["year"].eq(
            args.target_year
        )
    ].empty:
        raise RuntimeError(
            f"No clean observations found for {args.target_year}."
        )

    annual_summary = (
        build_annual_outcome_summary(
            frame=scoped,
        )
    )

    concentration = (
        build_concentration_outputs(
            frame=scoped,
            target_year=args.target_year,
            reference_start_year=(
                args.reference_start_year
            ),
            reference_end_year=(
                args.reference_end_year
            ),
            minimum_target_rows=(
                args.minimum_target_rows
            ),
            minimum_reference_rows=(
                args.minimum_reference_rows
            ),
            top_n=args.top_n,
        )
    )

    numeric_bands = (
        build_numeric_band_summary(
            frame=scoped,
            target_year=args.target_year,
            reference_start_year=(
                args.reference_start_year
            ),
            reference_end_year=(
                args.reference_end_year
            ),
        )
    )

    target_tail = (
        build_target_tail_observations(
            frame=scoped,
            target_year=args.target_year,
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    annual_summary.to_csv(
        output_dir
        / "annual_outcome_distribution.csv",
        index=False,
    )

    concentration.to_csv(
        output_dir
        / f"{args.target_year}_tail_concentration.csv",
        index=False,
    )

    numeric_bands.to_csv(
        output_dir
        / f"{args.target_year}_tail_by_numeric_band.csv",
        index=False,
    )

    target_tail.to_parquet(
        output_dir
        / f"{args.target_year}_upper_tail_observations.parquet",
        index=False,
    )

    print("\n" + "=" * 96)
    print(
        f"TARGETED {args.target_year} OUTCOME-CONCENTRATION AUDIT"
    )
    print("=" * 96)

    print("\nAnnual outcome distribution:")
    print(
        annual_summary.to_string(
            index=False
        )
    )

    if not concentration.empty:
        print(
            "\nLargest contributors to excess "
            f"{args.target_year} tail rows:"
        )

        display_columns = [
            "dimension",
            "group_value",
            "target_n_rows",
            "target_upper_tail_share",
            "reference_upper_tail_share",
            "tail_lift",
            "excess_target_tail_rows",
            "factor_p95_difference",
        ]

        print(
            concentration[
                display_columns
            ]
            .sort_values(
                "excess_target_tail_rows",
                ascending=False,
            )
            .head(30)
            .to_string(index=False)
        )

    print("\nArtifacts written to:")
    print(output_dir)


if __name__ == "__main__":
    main()
