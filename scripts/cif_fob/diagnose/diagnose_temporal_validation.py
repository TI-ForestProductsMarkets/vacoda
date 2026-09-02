from __future__ import annotations

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
    _build_model,
    _configured_predictors,
    _encode_predictors,
    _evidence_value,
    _learn_category_maps,
    _source_weight,
    _validate_categories,
)


# ======================================================================
# Configuration loading
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
# Small helpers
# ======================================================================


def quote_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


def load_training_feature_data(
    *,
    cfg: CifFobConfig,
    predictors: list[str],
) -> pd.DataFrame:
    """
    Load the columns needed to reproduce Step 5 estimation eligibility
    and temporal model fitting.
    """
    input_path = cfg.product_path(
        "training_features"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            "Training-features Parquet not found: "
            f"{input_path}"
        )

    required_columns = [
        "year",
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
        *predictors,
    ]

    required_columns = list(
        dict.fromkeys(required_columns)
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        available_columns = {
            str(row[0])
            for row in con.execute(
                """
                DESCRIBE
                SELECT *
                FROM read_parquet(?)
                """,
                [str(input_path)],
            ).fetchall()
        }

        missing_columns = sorted(
            set(required_columns)
            - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                "Training-features Parquet is missing required "
                "temporal-validation columns: "
                + ", ".join(missing_columns)
            )

        select_sql = ",\n".join(
            quote_identifier(column)
            for column in required_columns
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
            "Training-features input contains no rows."
        )

    return data


# ======================================================================
# Clean estimation sample
# ======================================================================


def build_clean_estimation_sample(
    *,
    data: pd.DataFrame,
    cfg: CifFobConfig,
) -> pd.DataFrame:
    """
    Reproduce the clean-estimation rules used by estimate_rates.py.
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

    working = data.copy()

    working["estimation_evidence_value"] = (
        _evidence_value(working)
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

    observed_factor = pd.to_numeric(
        working["observed_cif_fob_factor"],
        errors="coerce",
    )

    working["passes_factor_lower_bound"] = (
        observed_factor
        .ge(minimum_factor)
        .fillna(False)
    )

    working["passes_factor_upper_bound"] = (
        observed_factor
        .le(maximum_factor)
        .fillna(False)
    )

    if exclude_proxy_weights:
        working["passes_proxy_rule"] = (
            ~working["pair_uses_weight_proxy"]
            .fillna(False)
            .astype(bool)
        )
    else:
        working["passes_proxy_rule"] = True

    working["prediction_eligible"] = (
        working["initial_predictors_complete"]
        .fillna(False)
        .astype(bool)
    )

    training_factor_sources = {
        source.strip().lower()
        for source
        in estimation_cfg.training_factor_sources
    }

    working["passes_factor_source_rule"] = (
        working["factor_source"]
        .astype("string")
        .str.lower()
        .isin(training_factor_sources)
    )

    working["clean_estimation_eligible"] = (
        working["model_eligible"]
        .fillna(False)
        .astype(bool)
        & working["prediction_eligible"]
        & working["passes_factor_source_rule"]
        & working["passes_minimum_value"]
        & working["passes_factor_lower_bound"]
        & working["passes_factor_upper_bound"]
        & working["passes_proxy_rule"]
        & working["estimation_source_weight"]
        .gt(0)
        .fillna(False)
        & observed_factor.notna()
    )

    clean = working.loc[
        working["clean_estimation_eligible"]
    ].copy()

    if clean.empty:
        raise RuntimeError(
            "No observations passed the configured "
            "clean-estimation rules."
        )

    clean["year"] = pd.to_numeric(
        clean["year"],
        errors="raise",
    ).astype(int)

    clean["observed_cif_fob_factor"] = (
        pd.to_numeric(
            clean["observed_cif_fob_factor"],
            errors="raise",
        )
    )

    clean["log_observed_factor"] = np.log(
        clean["observed_cif_fob_factor"]
    )

    return clean


# ======================================================================
# Fold resolution
# ======================================================================


def resolve_validation_years(
    *,
    clean: pd.DataFrame,
    backtest_cfg,
) -> tuple[int, ...]:
    """
    Resolve expanding-window validation years from available data and
    YAML configuration.
    """
    available_years = tuple(
        sorted(
            int(year)
            for year
            in clean["year"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    if not available_years:
        raise RuntimeError(
            "No years are available in the clean estimation sample."
        )

    if backtest_cfg.validation_years:
        configured_years = tuple(
            sorted(
                int(year)
                for year
                in backtest_cfg.validation_years
            )
        )

        missing_years = sorted(
            set(configured_years)
            - set(available_years)
        )

        if missing_years:
            raise ValueError(
                "Configured temporal-validation years are absent "
                "from the clean estimation sample: "
                + ", ".join(
                    str(year)
                    for year in missing_years
                )
            )

        candidate_years = configured_years

    else:
        candidate_years = available_years

    resolved_years: list[int] = []

    for validation_year in candidate_years:
        earlier_years = [
            year
            for year in available_years
            if year < validation_year
        ]

        if (
            len(earlier_years)
            >= backtest_cfg.minimum_training_years
        ):
            resolved_years.append(
                validation_year
            )

    if not resolved_years:
        raise RuntimeError(
            "No valid temporal-validation folds were resolved. "
            "Inspect validation_years and minimum_training_years "
            "in cif_fob.yaml."
        )

    return tuple(resolved_years)


# ======================================================================
# Metric calculation
# ======================================================================


def calculate_fold_metrics(
    *,
    observed_factor: np.ndarray,
    predicted_factor: np.ndarray,
    upper_tail_threshold: float,
) -> dict[str, float | int]:
    """Calculate all supported temporal-validation metrics."""
    observed = np.asarray(
        observed_factor,
        dtype="float64",
    )

    predicted = np.asarray(
        predicted_factor,
        dtype="float64",
    )

    if observed.shape != predicted.shape:
        raise ValueError(
            "Observed and predicted arrays must have the same shape."
        )

    if observed.size == 0:
        raise ValueError(
            "Cannot calculate metrics for an empty validation fold."
        )

    ape = (
        np.abs(predicted - observed)
        / np.maximum(observed, 1e-12)
    )

    observed_rank = pd.Series(
        observed
    ).rank(
        method="average"
    )

    predicted_rank = pd.Series(
        predicted
    ).rank(
        method="average"
    )

    spearman_correlation = float(
        observed_rank.corr(
            predicted_rank,
            method="pearson",
        )
    )

    tail_mask = (
        observed
        >= upper_tail_threshold
    )

    n_upper_tail = int(
        tail_mask.sum()
    )

    if n_upper_tail > 0:
        upper_tail_ape = ape[
            tail_mask
        ]

        upper_tail_median_ape = float(
            np.median(
                upper_tail_ape
            )
        )

        upper_tail_p90_ape = float(
            np.quantile(
                upper_tail_ape,
                0.90,
            )
        )

        upper_tail_mean_ape = float(
            np.mean(
                upper_tail_ape
            )
        )

    else:
        upper_tail_median_ape = float("nan")
        upper_tail_p90_ape = float("nan")
        upper_tail_mean_ape = float("nan")

    return {
        "median_ape": float(
            np.median(ape)
        ),
        "mean_ape": float(
            np.mean(ape)
        ),
        "p90_ape": float(
            np.quantile(
                ape,
                0.90,
            )
        ),
        "p95_ape": float(
            np.quantile(
                ape,
                0.95,
            )
        ),
        "spearman_correlation": (
            spearman_correlation
        ),
        "observed_median": float(
            np.median(observed)
        ),
        "predicted_median": float(
            np.median(predicted)
        ),
        "observed_p95": float(
            np.quantile(
                observed,
                0.95,
            )
        ),
        "predicted_p95": float(
            np.quantile(
                predicted,
                0.95,
            )
        ),
        "observed_stddev": float(
            np.std(
                observed,
                ddof=1,
            )
        ),
        "predicted_stddev": float(
            np.std(
                predicted,
                ddof=1,
            )
        ),
        "n_upper_tail": n_upper_tail,
        "upper_tail_median_ape": (
            upper_tail_median_ape
        ),
        "upper_tail_mean_ape": (
            upper_tail_mean_ape
        ),
        "upper_tail_p90_ape": (
            upper_tail_p90_ape
        ),
    }


# ======================================================================
# Summary creation
# ======================================================================


ERROR_METRICS = {
    "median_ape",
    "mean_ape",
    "p90_ape",
    "p95_ape",
    "upper_tail_median_ape",
    "upper_tail_mean_ape",
    "upper_tail_p90_ape",
}

HIGHER_IS_BETTER_METRICS = {
    "spearman_correlation",
}

DISTRIBUTION_METRICS = {
    "observed_median",
    "predicted_median",
    "observed_p95",
    "predicted_p95",
    "observed_stddev",
    "predicted_stddev",
}


def build_summary(
    *,
    folds: pd.DataFrame,
    configured_metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Summarize configured metrics across validation folds."""
    supported_metrics = (
        ERROR_METRICS
        | HIGHER_IS_BETTER_METRICS
        | DISTRIBUTION_METRICS
    )

    unknown_metrics = sorted(
        set(configured_metrics)
        - supported_metrics
    )

    if unknown_metrics:
        raise ValueError(
            "Unsupported temporal-validation metric(s): "
            + ", ".join(unknown_metrics)
        )

    rows: list[dict[str, Any]] = []

    for metric in configured_metrics:
        metric_values = pd.to_numeric(
            folds[metric],
            errors="coerce",
        )

        valid_mask = (
            metric_values.notna()
        )

        valid_folds = folds.loc[
            valid_mask,
            [
                "validation_year",
                metric,
            ],
        ].copy()

        if valid_folds.empty:
            rows.append(
                {
                    "metric": metric,
                    "n_folds": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "minimum": np.nan,
                    "maximum": np.nan,
                    "best_validation_year": None,
                    "worst_validation_year": None,
                }
            )
            continue

        valid_folds[metric] = pd.to_numeric(
            valid_folds[metric],
            errors="raise",
        )

        if metric in ERROR_METRICS:
            best_index = valid_folds[
                metric
            ].idxmin()

            worst_index = valid_folds[
                metric
            ].idxmax()

        elif metric in HIGHER_IS_BETTER_METRICS:
            best_index = valid_folds[
                metric
            ].idxmax()

            worst_index = valid_folds[
                metric
            ].idxmin()

        else:
            best_index = None
            worst_index = None

        rows.append(
            {
                "metric": metric,
                "n_folds": int(
                    valid_folds.shape[0]
                ),
                "mean": float(
                    valid_folds[
                        metric
                    ].mean()
                ),
                "median": float(
                    valid_folds[
                        metric
                    ].median()
                ),
                "minimum": float(
                    valid_folds[
                        metric
                    ].min()
                ),
                "maximum": float(
                    valid_folds[
                        metric
                    ].max()
                ),
                "best_validation_year": (
                    int(
                        valid_folds.loc[
                            best_index,
                            "validation_year",
                        ]
                    )
                    if best_index is not None
                    else None
                ),
                "worst_validation_year": (
                    int(
                        valid_folds.loc[
                            worst_index,
                            "validation_year",
                        ]
                    )
                    if worst_index is not None
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# Printing
# ======================================================================


def print_fold_table(
    *,
    folds: pd.DataFrame,
    configured_metrics: tuple[str, ...],
) -> None:
    """Print fold-level temporal-validation results."""
    display_columns = [
        "validation_year",
        "training_min_year",
        "training_max_year",
        "n_training_years",
        "n_training_rows",
        "n_validation_rows",
        "n_upper_tail",
        *configured_metrics,
    ]

    display_columns = list(
        dict.fromkeys(display_columns)
    )

    display = folds[
        display_columns
    ].copy()

    percentage_columns = [
        column
        for column in display.columns
        if column in ERROR_METRICS
    ]

    factor_columns = [
        column
        for column in display.columns
        if column in DISTRIBUTION_METRICS
    ]

    for column in percentage_columns:
        display[column] = display[
            column
        ].map(
            lambda value: (
                f"{value * 100:.2f}%"
                if pd.notna(value)
                else ""
            )
        )

    for column in factor_columns:
        display[column] = display[
            column
        ].map(
            lambda value: (
                f"{value:.4f}"
                if pd.notna(value)
                else ""
            )
        )

    if (
        "spearman_correlation"
        in display.columns
    ):
        display[
            "spearman_correlation"
        ] = display[
            "spearman_correlation"
        ].map(
            lambda value: (
                f"{value:.4f}"
                if pd.notna(value)
                else ""
            )
        )

    print("\n" + "=" * 96)
    print("EXPANDING-WINDOW TEMPORAL VALIDATION")
    print("=" * 96)
    print(
        display.to_string(
            index=False
        )
    )


def print_summary_table(
    summary: pd.DataFrame,
) -> None:
    """Print cross-fold summary metrics."""
    display = summary.copy()

    # Pandas 2.x is strict about assigning strings into float columns.
    # Cast printable columns to object before applying string formatting.
    for column in (
        "mean",
        "median",
        "minimum",
        "maximum",
    ):
        if column in display.columns:
            display[column] = display[column].astype(object)

    for index, row in display.iterrows():
        metric = str(
            row["metric"]
        )

        if metric in ERROR_METRICS:
            for column in (
                "mean",
                "median",
                "minimum",
                "maximum",
            ):
                value = display.at[
                    index,
                    column,
                ]

                display.at[
                    index,
                    column,
                ] = (
                    f"{float(value) * 100:.2f}%"
                    if pd.notna(value)
                    else ""
                )

        elif (
            metric in HIGHER_IS_BETTER_METRICS
            or metric in DISTRIBUTION_METRICS
        ):
            for column in (
                "mean",
                "median",
                "minimum",
                "maximum",
            ):
                value = display.at[
                    index,
                    column,
                ]

                display.at[
                    index,
                    column,
                ] = (
                    f"{float(value):.4f}"
                    if pd.notna(value)
                    else ""
                )

    print("\n" + "=" * 96)
    print("TEMPORAL VALIDATION SUMMARY")
    print("=" * 96)
    print(
        display.to_string(
            index=False
        )
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    cfg = load_runtime_config()

    validation_cfg = (
        cfg.method.estimation.validation
    )

    backtest_cfg = (
        validation_cfg.temporal_backtest
    )

    if not backtest_cfg.enabled:
        raise RuntimeError(
            "Temporal validation is disabled. Set "
            "cif_fob.method.estimation.validation."
            "temporal_backtest.enabled: true"
        )

    if (
        backtest_cfg.strategy
        != "expanding_window"
    ):
        raise ValueError(
            "Unsupported temporal-backtest strategy "
            f"{backtest_cfg.strategy!r}. "
            "Supported: expanding_window."
        )

    predictors = (
        _configured_predictors(cfg)
    )

    categorical_predictors = list(
        cfg.method.estimation
        .categorical_predictors
    )

    unknown_categorical = sorted(
        set(categorical_predictors)
        - set(predictors)
    )

    if unknown_categorical:
        raise ValueError(
            "Categorical predictors are absent from the "
            "configured predictor list: "
            + ", ".join(
                unknown_categorical
            )
        )

    data = load_training_feature_data(
        cfg=cfg,
        predictors=predictors,
    )

    clean = build_clean_estimation_sample(
        data=data,
        cfg=cfg,
    )

    validation_years = (
        resolve_validation_years(
            clean=clean,
            backtest_cfg=backtest_cfg,
        )
    )

    available_years = tuple(
        sorted(
            int(year)
            for year
            in clean["year"]
            .unique()
            .tolist()
        )
    )

    print(
        f"Target HS:             "
        f"{cfg.scope.target_hs}"
    )
    print(
        f"Configured scope:      "
        f"{cfg.scope.min_year}-"
        f"{cfg.scope.max_year}"
    )
    print(
        f"Engine threads:        "
        f"{cfg.engine.threads}"
    )
    print(
        f"Training features:     "
        f"{cfg.product_path('training_features')}"
    )
    print(
        f"Available clean years: "
        f"{', '.join(map(str, available_years))}"
    )
    print(
        f"Validation years:      "
        f"{', '.join(map(str, validation_years))}"
    )
    print(
        f"Minimum train years:   "
        f"{backtest_cfg.minimum_training_years}"
    )
    print(
        f"Upper-tail threshold:  "
        f"{backtest_cfg.upper_tail_threshold:.4f}"
    )
    print(
        f"Predictors:            "
        f"{', '.join(predictors)}"
    )
    print(
        "Categorical:           "
        + (
            ", ".join(
                categorical_predictors
            )
            if categorical_predictors
            else "none"
        )
    )

    categorical_set = set(
        categorical_predictors
    )

    categorical_mask = [
        predictor
        in categorical_set
        for predictor in predictors
    ]

    fold_results: list[
        dict[str, Any]
    ] = []

    for validation_year in validation_years:
        train_mask = (
            clean["year"]
            .lt(validation_year)
        )

        validation_mask = (
            clean["year"]
            .eq(validation_year)
        )

        train = clean.loc[
            train_mask
        ].copy()

        validation = clean.loc[
            validation_mask
        ].copy()

        if train.empty:
            raise RuntimeError(
                "No training observations for "
                f"validation year {validation_year}."
            )

        if validation.empty:
            raise RuntimeError(
                "No validation observations for "
                f"year {validation_year}."
            )

        training_years = tuple(
            sorted(
                int(year)
                for year
                in train["year"]
                .unique()
                .tolist()
            )
        )

        print(
            "\n"
            + "-" * 96
        )
        print(
            f"Validation year: "
            f"{validation_year}"
        )
        print(
            f"Training years:  "
            f"{training_years[0]}-"
            f"{training_years[-1]}"
        )
        print(
            f"Training rows:   "
            f"{len(train):,}"
        )
        print(
            f"Validation rows: "
            f"{len(validation):,}"
        )

        category_maps = (
            _learn_category_maps(
                frame=train,
                categorical_predictors=(
                    categorical_predictors
                ),
            )
        )

        _validate_categories(
            category_maps=category_maps,
            max_bins=(
                cfg.method.estimation
                .model.max_bins
            ),
        )

        encoded_train = (
            _encode_predictors(
                frame=train,
                predictors=predictors,
                categorical_predictors=(
                    categorical_predictors
                ),
                category_maps=category_maps,
            )
        )

        encoded_validation = (
            _encode_predictors(
                frame=validation,
                predictors=predictors,
                categorical_predictors=(
                    categorical_predictors
                ),
                category_maps=category_maps,
            )
        )

        model = _build_model(
            cfg,
            categorical_mask=(
                categorical_mask
            ),
        )

        model.fit(
            encoded_train[
                predictors
            ],
            train[
                "log_observed_factor"
            ],
            sample_weight=train[
                "estimation_source_weight"
            ],
        )

        predicted_log_factor = (
            model.predict(
                encoded_validation[
                    predictors
                ]
            )
        )

        predicted_factor = np.exp(
            predicted_log_factor
        )

        observed_factor = (
            validation[
                "observed_cif_fob_factor"
            ]
            .to_numpy(
                dtype="float64"
            )
        )

        fold_metrics = (
            calculate_fold_metrics(
                observed_factor=(
                    observed_factor
                ),
                predicted_factor=(
                    predicted_factor
                ),
                upper_tail_threshold=float(
                    backtest_cfg
                    .upper_tail_threshold
                ),
            )
        )

        fold_results.append(
            {
                "validation_year": (
                    validation_year
                ),
                "training_min_year": (
                    int(
                        train[
                            "year"
                        ].min()
                    )
                ),
                "training_max_year": (
                    int(
                        train[
                            "year"
                        ].max()
                    )
                ),
                "n_training_years": (
                    int(
                        train[
                            "year"
                        ].nunique()
                    )
                ),
                "n_training_rows": (
                    int(len(train))
                ),
                "n_validation_rows": (
                    int(
                        len(validation)
                    )
                ),
                **fold_metrics,
            }
        )

        print(
            "Median APE:      "
            f"{fold_metrics['median_ape'] * 100:.2f}%"
        )
        print(
            "P90 APE:         "
            f"{fold_metrics['p90_ape'] * 100:.2f}%"
        )
        print(
            "Spearman:        "
            f"{fold_metrics['spearman_correlation']:.4f}"
        )
        print(
            "Observed P95:    "
            f"{fold_metrics['observed_p95']:.4f}"
        )
        print(
            "Predicted P95:   "
            f"{fold_metrics['predicted_p95']:.4f}"
        )
        print(
            "Upper-tail rows: "
            f"{fold_metrics['n_upper_tail']:,}"
        )
        print(
            "Tail median APE: "
            f"{fold_metrics['upper_tail_median_ape'] * 100:.2f}%"
        )

    folds = pd.DataFrame(
        fold_results
    ).sort_values(
        "validation_year"
    ).reset_index(
        drop=True
    )

    configured_metrics = tuple(
        backtest_cfg.metrics
    )

    summary = build_summary(
        folds=folds,
        configured_metrics=(
            configured_metrics
        ),
    )

    print_fold_table(
        folds=folds,
        configured_metrics=(
            configured_metrics
        ),
    )

    print_summary_table(
        summary
    )

    fold_path = cfg.product_path(
        backtest_cfg
        .output.fold_product
    )

    summary_path = cfg.product_path(
        backtest_cfg
        .output.summary_product
    )

    fold_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds.to_csv(
        fold_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 96)
    print(
        "TEMPORAL VALIDATION COMPLETE"
    )
    print("=" * 96)
    print(
        f"Fold results: "
        f"{fold_path}"
    )
    print(
        f"Summary:      "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()