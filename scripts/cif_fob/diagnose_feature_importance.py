"""
scripts.diagnose_feature_importance
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from vacoda.config.cif_fob import CifFobConfig
from vacoda.config.pipeline import (
    PipelineEngineConfig,
    PipelineScopeConfig,
)
from vacoda.cif_fob.estimate_rates import (
    _build_model,
    _configured_predictors,
    _encode_predictors,
    _evidence_value,
    _learn_category_maps,
    _resolve_holdout_year,
    _source_weight,
    _validate_categories,
)


# ======================================================================
# Runtime configuration
# ======================================================================

TARGET_HS = "H5"
MIN_YEAR = 2017
MAX_YEAR = 2025
THREADS = 48

N_REPEATS = 10
RANDOM_STATE = 42
UPPER_TAIL_THRESHOLD = 1.25

CONFIG_PATH = Path("configs/steps/cif_fob.yaml")


# ======================================================================
# Metric helpers
# ======================================================================


def absolute_percentage_errors(
    *,
    observed_factor: np.ndarray,
    predicted_factor: np.ndarray,
) -> np.ndarray:
    """Return row-level absolute percentage errors."""
    observed = np.asarray(
        observed_factor,
        dtype="float64",
    )
    predicted = np.asarray(
        predicted_factor,
        dtype="float64",
    )

    return (
        np.abs(predicted - observed)
        / np.maximum(observed, 1e-12)
    )


def calculate_metrics(
    *,
    observed_factor: np.ndarray,
    predicted_factor: np.ndarray,
) -> dict[str, float]:
    """Return the model metrics used by the CIF/FOB diagnostics."""
    ape = absolute_percentage_errors(
        observed_factor=observed_factor,
        predicted_factor=predicted_factor,
    )

    return {
        "median_ape": float(np.median(ape)),
        "mean_ape": float(np.mean(ape)),
        "p90_ape": float(np.quantile(ape, 0.90)),
        "p95_ape": float(np.quantile(ape, 0.95)),
    }


def print_metrics(
    *,
    title: str,
    metrics: dict[str, float],
    n_rows: int,
) -> None:
    """Print one metric group."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print(f"Rows:       {n_rows:,}")
    print(f"Median APE: {metrics['median_ape'] * 100:.4f}%")
    print(f"Mean APE:   {metrics['mean_ape'] * 100:.4f}%")
    print(f"P90 APE:    {metrics['p90_ape'] * 100:.4f}%")
    print(f"P95 APE:    {metrics['p95_ape'] * 100:.4f}%")


# ======================================================================
# Data preparation
# ======================================================================


def load_candidate_data(
    *,
    cfg: CifFobConfig,
    input_path: Path,
    predictors: list[str],
) -> pd.DataFrame:
    """
    Load only potential estimation observations from the Step 4 Parquet.

    The remaining clean-estimation filters are applied in pandas using
    the same logic as estimate_rates.py.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Training-features Parquet not found: {input_path}"
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

    # Preserve order and remove duplicates such as year.
    required_columns = list(
        dict.fromkeys(required_columns)
    )

    quoted_columns = ",\n".join(
        '"' + column.replace('"', '""') + '"'
        for column in required_columns
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        data = con.execute(
            f"""
            SELECT
                {quoted_columns}

            FROM read_parquet(?)

            WHERE model_eligible
              AND initial_predictors_complete
            """,
            [str(input_path)],
        ).df()

    finally:
        con.close()

    if data.empty:
        raise RuntimeError(
            "No model-eligible observations were loaded."
        )

    return data


def build_clean_estimation_sample(
    *,
    data: pd.DataFrame,
    cfg: CifFobConfig,
) -> pd.DataFrame:
    """Apply the same clean-estimation rules as estimate_rates.py."""
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

    observed_factor = pd.to_numeric(
        working["observed_cif_fob_factor"],
        errors="coerce",
    )

    training_factor_sources = {
        source.strip().lower()
        for source
        in estimation_cfg.training_factor_sources
    }

    clean_mask = (
        working["factor_source"]
        .astype("string")
        .str.lower()
        .isin(training_factor_sources)
        & working["estimation_evidence_value"]
        .ge(minimum_evidence_value)
        .fillna(False)
        & observed_factor
        .ge(minimum_factor)
        .fillna(False)
        & observed_factor
        .le(maximum_factor)
        .fillna(False)
        & working["estimation_source_weight"]
        .gt(0)
        .fillna(False)
        & observed_factor.notna()
    )

    if exclude_proxy_weights:
        clean_mask &= (
            ~working["pair_uses_weight_proxy"]
            .fillna(False)
            .astype(bool)
        )

    clean = working.loc[clean_mask].copy()

    if clean.empty:
        raise RuntimeError(
            "No observations passed the clean-estimation rules."
        )

    clean["observed_cif_fob_factor"] = (
        pd.to_numeric(
            clean["observed_cif_fob_factor"],
            errors="raise",
        )
    )

    clean["log_observed_factor"] = np.log(
        clean["observed_cif_fob_factor"]
    )

    clean["year"] = pd.to_numeric(
        clean["year"],
        errors="raise",
    ).astype(int)

    return clean


# ======================================================================
# Permutation importance
# ======================================================================


def calculate_permutation_importance(
    *,
    model,
    X_holdout: pd.DataFrame,
    observed_factor: np.ndarray,
    predictors: list[str],
    evaluation_mask: np.ndarray,
    n_repeats: int,
    random_state: int,
    evaluation_name: str,
) -> pd.DataFrame:
    """
    Calculate permutation importance in factor space.

    Importance is the increase in error after shuffling one predictor.
    Positive values mean the predictor is useful.
    Negative values mean shuffling improved the score on average.
    """
    if evaluation_mask.dtype != bool:
        evaluation_mask = evaluation_mask.astype(bool)

    n_evaluation_rows = int(evaluation_mask.sum())

    if n_evaluation_rows == 0:
        raise RuntimeError(
            f"No observations available for {evaluation_name!r}."
        )

    baseline_prediction_log = model.predict(
        X_holdout[predictors]
    )
    baseline_prediction_factor = np.exp(
        baseline_prediction_log
    )

    baseline_metrics = calculate_metrics(
        observed_factor=observed_factor[evaluation_mask],
        predicted_factor=baseline_prediction_factor[
            evaluation_mask
        ],
    )

    rng = np.random.default_rng(random_state)
    results: list[dict[str, float | int | str]] = []

    for predictor in predictors:
        median_increases: list[float] = []
        mean_increases: list[float] = []
        p90_increases: list[float] = []
        p95_increases: list[float] = []

        original_values = (
            X_holdout[predictor]
            .to_numpy(copy=True)
        )

        for _ in range(n_repeats):
            permuted = X_holdout.copy()

            permuted[predictor] = rng.permutation(
                original_values
            )

            permuted_prediction_log = model.predict(
                permuted[predictors]
            )
            permuted_prediction_factor = np.exp(
                permuted_prediction_log
            )

            permuted_metrics = calculate_metrics(
                observed_factor=observed_factor[
                    evaluation_mask
                ],
                predicted_factor=permuted_prediction_factor[
                    evaluation_mask
                ],
            )

            median_increases.append(
                permuted_metrics["median_ape"]
                - baseline_metrics["median_ape"]
            )
            mean_increases.append(
                permuted_metrics["mean_ape"]
                - baseline_metrics["mean_ape"]
            )
            p90_increases.append(
                permuted_metrics["p90_ape"]
                - baseline_metrics["p90_ape"]
            )
            p95_increases.append(
                permuted_metrics["p95_ape"]
                - baseline_metrics["p95_ape"]
            )

        results.append(
            {
                "evaluation_group": evaluation_name,
                "predictor": predictor,
                "n_evaluation_rows": n_evaluation_rows,
                "median_ape_importance_mean": float(
                    np.mean(median_increases)
                ),
                "median_ape_importance_std": float(
                    np.std(
                        median_increases,
                        ddof=1,
                    )
                    if n_repeats > 1
                    else 0.0
                ),
                "mean_ape_importance_mean": float(
                    np.mean(mean_increases)
                ),
                "mean_ape_importance_std": float(
                    np.std(
                        mean_increases,
                        ddof=1,
                    )
                    if n_repeats > 1
                    else 0.0
                ),
                "p90_ape_importance_mean": float(
                    np.mean(p90_increases)
                ),
                "p90_ape_importance_std": float(
                    np.std(
                        p90_increases,
                        ddof=1,
                    )
                    if n_repeats > 1
                    else 0.0
                ),
                "p95_ape_importance_mean": float(
                    np.mean(p95_increases)
                ),
                "p95_ape_importance_std": float(
                    np.std(
                        p95_increases,
                        ddof=1,
                    )
                    if n_repeats > 1
                    else 0.0
                ),
            }
        )

    result = pd.DataFrame(results)

    return result.sort_values(
        by=[
            "median_ape_importance_mean",
            "mean_ape_importance_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)


def print_importance_table(
    *,
    title: str,
    importance: pd.DataFrame,
) -> None:
    """Print permutation importance in percentage-point units."""
    display = importance.copy()

    percentage_columns = [
        "median_ape_importance_mean",
        "median_ape_importance_std",
        "mean_ape_importance_mean",
        "mean_ape_importance_std",
        "p90_ape_importance_mean",
        "p90_ape_importance_std",
        "p95_ape_importance_mean",
        "p95_ape_importance_std",
    ]

    for column in percentage_columns:
        display[column] = display[column].map(
            lambda value: f"{value * 100:.4f} pp"
        )

    display.insert(
        0,
        "rank",
        np.arange(1, len(display) + 1),
    )

    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

    print(
        display[
            [
                "rank",
                "predictor",
                "median_ape_importance_mean",
                "median_ape_importance_std",
                "mean_ape_importance_mean",
                "p90_ape_importance_mean",
                "p95_ape_importance_mean",
            ]
        ].to_string(index=False)
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    scope = PipelineScopeConfig(
        target_hs=TARGET_HS,
        min_year=MIN_YEAR,
        max_year=MAX_YEAR,
    )

    engine = PipelineEngineConfig(
        name="duckdb",
        threads=THREADS,
    )

    cfg = CifFobConfig.from_yaml(
        CONFIG_PATH,
        scope=scope,
        engine=engine,
    )

    predictors = _configured_predictors(cfg)
    categorical_predictors = list(
        cfg.method.estimation.categorical_predictors
    )

    unknown_categorical = sorted(
        set(categorical_predictors)
        - set(predictors)
    )

    if unknown_categorical:
        raise ValueError(
            "Categorical predictors are absent from the configured "
            "predictor list: "
            + ", ".join(unknown_categorical)
        )

    input_path = cfg.product_path(
        "training_features"
    )

    print(f"Using config: {CONFIG_PATH}")
    print(f"Using input:  {input_path}")
    print(f"Predictors:   {', '.join(predictors)}")
    print(
        "Categorical:  "
        + (
            ", ".join(categorical_predictors)
            if categorical_predictors
            else "none"
        )
    )
    print(f"Repeats:      {N_REPEATS}")
    print(
        "Tail cutoff:  "
        f"observed factor >= {UPPER_TAIL_THRESHOLD:.2f}"
    )

    data = load_candidate_data(
        cfg=cfg,
        input_path=input_path,
        predictors=predictors,
    )

    clean = build_clean_estimation_sample(
        data=data,
        cfg=cfg,
    )

    holdout_year = _resolve_holdout_year(
        clean=clean,
        cfg=cfg,
    )

    train_mask = clean["year"].lt(
        holdout_year
    )
    holdout_mask = clean["year"].eq(
        holdout_year
    )

    n_train = int(train_mask.sum())
    n_holdout = int(holdout_mask.sum())

    if n_train == 0 or n_holdout == 0:
        raise RuntimeError(
            "Temporal validation requires both training and holdout "
            f"observations. Training={n_train:,}, "
            f"holdout={n_holdout:,}."
        )

    train = clean.loc[train_mask].copy()
    holdout = clean.loc[holdout_mask].copy()

    category_maps = _learn_category_maps(
        frame=train,
        categorical_predictors=categorical_predictors,
    )

    _validate_categories(
        category_maps=category_maps,
        max_bins=cfg.method.estimation.model.max_bins,
    )

    encoded_train = _encode_predictors(
        frame=train,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        category_maps=category_maps,
    )

    encoded_holdout = _encode_predictors(
        frame=holdout,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        category_maps=category_maps,
    )

    categorical_set = set(
        categorical_predictors
    )

    categorical_mask = [
        predictor in categorical_set
        for predictor in predictors
    ]

    model = _build_model(
        cfg,
        categorical_mask=categorical_mask,
    )

    print("\nFitting temporal holdout model...")
    print(f"Training year(s): < {holdout_year}")
    print(f"Holdout year:     {holdout_year}")
    print(f"Training rows:    {n_train:,}")
    print(f"Holdout rows:     {n_holdout:,}")

    model.fit(
        encoded_train[predictors],
        train["log_observed_factor"],
        sample_weight=train[
            "estimation_source_weight"
        ],
    )

    observed_holdout_factor = (
        holdout["observed_cif_fob_factor"]
        .to_numpy(dtype="float64")
    )

    baseline_prediction_factor = np.exp(
        model.predict(
            encoded_holdout[predictors]
        )
    )

    all_rows_mask = np.ones(
        len(holdout),
        dtype=bool,
    )

    upper_tail_mask = (
        observed_holdout_factor
        >= UPPER_TAIL_THRESHOLD
    )

    baseline_all_metrics = calculate_metrics(
        observed_factor=observed_holdout_factor,
        predicted_factor=baseline_prediction_factor,
    )

    baseline_tail_metrics = calculate_metrics(
        observed_factor=observed_holdout_factor[
            upper_tail_mask
        ],
        predicted_factor=baseline_prediction_factor[
            upper_tail_mask
        ],
    )

    print_metrics(
        title="BASELINE HOLDOUT METRICS — ALL OBSERVATIONS",
        metrics=baseline_all_metrics,
        n_rows=len(holdout),
    )

    print_metrics(
        title=(
            "BASELINE HOLDOUT METRICS — "
            f"OBSERVED FACTOR >= {UPPER_TAIL_THRESHOLD:.2f}"
        ),
        metrics=baseline_tail_metrics,
        n_rows=int(upper_tail_mask.sum()),
    )

    overall_importance = (
        calculate_permutation_importance(
            model=model,
            X_holdout=encoded_holdout,
            observed_factor=observed_holdout_factor,
            predictors=predictors,
            evaluation_mask=all_rows_mask,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            evaluation_name="all_holdout",
        )
    )

    tail_importance = (
        calculate_permutation_importance(
            model=model,
            X_holdout=encoded_holdout,
            observed_factor=observed_holdout_factor,
            predictors=predictors,
            evaluation_mask=upper_tail_mask,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE + 1,
            evaluation_name=(
                f"observed_factor_ge_"
                f"{UPPER_TAIL_THRESHOLD:.2f}"
            ),
        )
    )

    print_importance_table(
        title=(
            "PERMUTATION IMPORTANCE — "
            "ALL HOLDOUT OBSERVATIONS"
        ),
        importance=overall_importance,
    )

    print_importance_table(
        title=(
            "PERMUTATION IMPORTANCE — "
            f"OBSERVED FACTOR >= {UPPER_TAIL_THRESHOLD:.2f}"
        ),
        importance=tail_importance,
    )

    combined = pd.concat(
        [
            overall_importance,
            tail_importance,
        ],
        ignore_index=True,
    )

    output_path = (
        input_path.parent
        / "target_h5_cif_fob_permutation_importance.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 88)
    print("FEATURE-IMPORTANCE DIAGNOSTIC COMPLETE")
    print("=" * 88)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()