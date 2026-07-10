"""
vacoda.cif_fob.estimate_rates

Estimate CIF/FOB conversion factors from the Step 4 training-features
artifact using multiple candidate estimators on one shared evidence sample.

Supported estimators
--------------------
1. hist_gradient_boosting
2. hist_gradient_boosting_squared
3. log_linear_regression
4. log_linear_regression_smearing
5. xgboost_log_squared

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import statsmodels.api as sm
except ImportError as exc:  # pragma: no cover - environment guard
    sm = None
    _STATSMODELS_IMPORT_ERROR = exc
else:
    _STATSMODELS_IMPORT_ERROR = None

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - environment guard
    XGBRegressor = None
    _XGBOOST_IMPORT_ERROR = exc
else:
    _XGBOOST_IMPORT_ERROR = None

from vacoda.config.cif_fob import CifFobConfig
from vacoda.cif_fob.prepare_input import valid_output


# ======================================================================
# Small containers
# ======================================================================


@dataclass
class FittedHGBModel:
    estimator: HistGradientBoostingRegressor
    predictors: list[str]
    categorical_predictors: list[str]
    category_maps: dict[str, dict[str, int]]
    target_transformation: str


@dataclass
class FittedXGBoostModel:
    estimator: Any
    predictors: list[str]
    categorical_predictors: list[str]
    category_levels: dict[str, list[str]]
    target_transformation: str
    n_estimators_used: int


@dataclass
class FittedLogLinearModel:
    result: Any
    predictors: list[str]
    categorical_predictors: list[str]
    design_columns: list[str]
    fit_intercept: bool
    numeric_fill_values: dict[str, float]
    inverse_transform: str
    smearing_factor: float


@dataclass(frozen=True)
class EstimationRunSpec:
    """One concrete algorithm × predictor-set estimation run."""

    run_name: str
    model_name: str
    predictor_set_name: str | None
    model_cfg: Any
    predictors: list[str]
    categorical_predictors: list[str]
    is_primary: bool


# ======================================================================
# Generic helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _quoted_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


def _safe_model_prefix(model_name: str) -> str:
    """Return a stable column prefix for a configured model name."""
    aliases = {
        "hist_gradient_boosting": "hgb",
        "hist_gradient_boosting_squared": "hgb_sq",
        "hist_gradient_boosting_squared_level": "hgb_sq_lvl",
        "hist_gradient_boosting_squared_regularized": "hgb_sq_reg",
        "log_linear_regression": "ols",
        "log_linear_regression_smearing": "ols_sm",
        "xgboost_log_squared": "xgb_log_sq",
    }
    return aliases.get(
        model_name,
        "".join(
            character if character.isalnum() else "_"
            for character in model_name.lower()
        ).strip("_"),
    )


def _configured_predictors(cfg: CifFobConfig) -> list[str]:
    """Return unique legacy/global predictors while preserving YAML order."""
    predictors = cfg.method.predictors.all()

    if not predictors:
        raise ValueError(
            "cif_fob.method.predictors must contain at least one predictor."
        )

    duplicates = {
        predictor
        for predictor in predictors
        if predictors.count(predictor) > 1
    }

    if duplicates:
        raise ValueError(
            "CIF/FOB predictor configuration contains duplicates: "
            + ", ".join(sorted(duplicates))
        )

    return predictors


def _configured_predictor_sets(
    cfg: CifFobConfig,
) -> dict[str, Any]:
    """Return configured predictor sets, or a legacy fallback set."""
    predictor_sets = getattr(
        cfg.method.estimation,
        "predictor_sets",
        {},
    )

    if predictor_sets:
        return dict(predictor_sets)

    # Backward-compatible fallback: one implicit predictor set using the
    # old global cif_fob.method.predictors block.
    class _LegacyPredictorSet:
        name = "global_predictors"
        categorical: tuple[str, ...] = tuple()

        @property
        def all_predictors(self) -> list[str]:
            return _configured_predictors(cfg)

    return {
        "global_predictors": _LegacyPredictorSet(),
    }


def _all_predictors_for_validation(
    predictor_sets: dict[str, Any],
) -> list[str]:
    """Return the union of all predictors used by all predictor sets."""
    predictors: list[str] = []

    for predictor_set in predictor_sets.values():
        predictors.extend(
            list(predictor_set.all_predictors)
        )

    duplicates_removed = list(dict.fromkeys(predictors))

    if not duplicates_removed:
        raise ValueError(
            "No predictors are configured for CIF/FOB estimation."
        )

    return duplicates_removed


def _resolve_run_specs(
    *,
    cfg: CifFobConfig,
    model_configs: dict[str, Any],
    predictor_sets: dict[str, Any],
    primary_model_name: str,
) -> list[EstimationRunSpec]:
    """Build concrete algorithm × predictor-set run specifications."""
    estimation_cfg = cfg.method.estimation
    primary_predictor_set_name = None

    if predictor_sets:
        resolver = getattr(
            estimation_cfg,
            "primary_predictor_set_name",
            None,
        )
        if callable(resolver):
            primary_predictor_set_name = resolver()
        else:
            primary_predictor_set_name = next(reversed(predictor_sets))

    run_specs: list[EstimationRunSpec] = []

    for predictor_set_name, predictor_set in predictor_sets.items():
        predictors = list(predictor_set.all_predictors)
        categorical_predictors = [
            str(value)
            for value in getattr(
                predictor_set,
                "categorical",
                (),
            )
        ]

        unknown_categorical = sorted(
            set(categorical_predictors) - set(predictors)
        )
        if unknown_categorical:
            raise ValueError(
                f"Predictor set {predictor_set_name!r} declares "
                "categorical predictors that are not in its full "
                "predictor list: "
                + ", ".join(unknown_categorical)
            )

        for model_name, model_cfg in model_configs.items():
            if predictor_set_name == "global_predictors":
                run_name = model_name
                categorical_for_run = _model_categorical_predictors(
                    cfg=cfg,
                    model_cfg=model_cfg,
                )
            else:
                run_name = f"{model_name}__{predictor_set_name}"
                categorical_for_run = categorical_predictors

            is_primary = (
                model_name == primary_model_name
                and (
                    predictor_set_name == "global_predictors"
                    or predictor_set_name == primary_predictor_set_name
                )
            )

            run_specs.append(
                EstimationRunSpec(
                    run_name=run_name,
                    model_name=model_name,
                    predictor_set_name=(
                        None
                        if predictor_set_name == "global_predictors"
                        else predictor_set_name
                    ),
                    model_cfg=model_cfg,
                    predictors=predictors,
                    categorical_predictors=categorical_for_run,
                    is_primary=is_primary,
                )
            )

    if not any(spec.is_primary for spec in run_specs):
        raise RuntimeError(
            "Could not resolve a primary CIF/FOB estimation run."
        )

    return run_specs


def _enabled_model_configs(
    cfg: CifFobConfig,
) -> tuple[str, dict[str, Any]]:
    """
    Return the configured primary model and enabled model candidates.

    Only the multi-model ``estimation.models`` configuration is accepted.
    """
    estimation_cfg = cfg.method.estimation
    models_cfg = getattr(estimation_cfg, "models", None)

    if models_cfg is None:
        raise ValueError(
            "CIF/FOB estimation configuration must define "
            "'cif_fob.method.estimation.models'. "
            "The legacy single 'model' block is not supported."
        )

    primary = str(models_cfg.primary).strip().lower()
    enabled = dict(models_cfg.enabled())

    if not enabled:
        raise ValueError(
            "No enabled CIF/FOB estimation models are configured."
        )

    if primary not in enabled:
        raise ValueError(
            f"Primary CIF/FOB model {primary!r} is not enabled."
        )

    return primary, enabled

def _estimation_run_checkpoint_dir(
    *,
    cfg: CifFobConfig,
    run_name: str,
) -> Path:
    """Return checkpoint directory for one algorithm × predictor-set run."""
    if "__" in run_name:
        model_algorithm, predictor_set_name = run_name.split("__", 1)
    else:
        model_algorithm = run_name
        predictor_set_name = "global_predictors"

    return (
        cfg.output_dir()
        / "model_runs"
        / model_algorithm
        / predictor_set_name
    )


def _estimation_run_checkpoint_paths(
    *,
    cfg: CifFobConfig,
    run_name: str,
) -> dict[str, Path]:
    """Return all checkpoint paths for one estimation run."""
    checkpoint_dir = _estimation_run_checkpoint_dir(
        cfg=cfg,
        run_name=run_name,
    )

    return {
        "folds": checkpoint_dir / "temporal_folds.parquet",
        "predictions": checkpoint_dir / "predictions.parquet",
    }


def _run_checkpoint_complete(
    *,
    cfg: CifFobConfig,
    run_name: str,
) -> bool:
    """Return whether all required checkpoint files exist."""
    paths = _estimation_run_checkpoint_paths(
        cfg=cfg,
        run_name=run_name,
    )

    return all(
        valid_output(path)
        for path in paths.values()
    )


def _write_run_checkpoint(
    *,
    cfg: CifFobConfig,
    run_name: str,
    fold_results: pd.DataFrame,
    predictions: pd.DataFrame,
    threads: int,
) -> None:
    """Write one completed estimation-run checkpoint."""
    paths = _estimation_run_checkpoint_paths(
        cfg=cfg,
        run_name=run_name,
    )

    for path in paths.values():
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    _write_dataframe(
        frame=fold_results,
        output_path=paths["folds"],
        threads=threads,
    )

    _write_dataframe(
        frame=predictions,
        output_path=paths["predictions"],
        threads=threads,
    )


def _read_run_checkpoint(
    *,
    cfg: CifFobConfig,
    run_name: str,
    threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one completed estimation-run checkpoint."""
    paths = _estimation_run_checkpoint_paths(
        cfg=cfg,
        run_name=run_name,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        folds = con.execute(
            """
            SELECT *
            FROM read_parquet(?)
            """,
            [str(paths["folds"])],
        ).df()

        predictions = con.execute(
            """
            SELECT *
            FROM read_parquet(?)
            """,
            [str(paths["predictions"])],
        ).df()

    finally:
        con.close()

    if folds.empty:
        raise RuntimeError(
            f"Checkpoint folds are empty for run {run_name!r}."
        )

    if predictions.empty:
        raise RuntimeError(
            f"Checkpoint predictions are empty for run {run_name!r}."
        )

    return folds, predictions


def _build_run_prediction_checkpoint(
    *,
    run_name: str,
    prediction_row_index: pd.Index,
    full_predicted_log: np.ndarray,
    full_predicted_factor: np.ndarray,
    latest_holdout_year: int,
    latest_holdout_predictions: tuple[pd.Index, np.ndarray, np.ndarray] | None,
) -> pd.DataFrame:
    """Build row-level prediction checkpoint for one estimation run."""
    prediction_checkpoint = pd.DataFrame(
        {
            "_row_index": prediction_row_index.to_numpy(),
            "model_name": run_name,
            "full_predicted_log_factor": full_predicted_log,
            "full_predicted_factor": full_predicted_factor,
        }
    )

    prediction_checkpoint[
        "holdout_predicted_log_factor"
    ] = np.nan

    prediction_checkpoint[
        "holdout_predicted_factor"
    ] = np.nan

    prediction_checkpoint[
        "is_latest_holdout"
    ] = False

    if latest_holdout_predictions is not None:
        (
            holdout_index,
            holdout_predicted_log,
            holdout_predicted_factor,
        ) = latest_holdout_predictions

        holdout_frame = pd.DataFrame(
            {
                "_row_index": holdout_index.to_numpy(),
                "holdout_predicted_log_factor": (
                    holdout_predicted_log
                ),
                "holdout_predicted_factor": (
                    holdout_predicted_factor
                ),
                "is_latest_holdout": True,
            }
        )

        prediction_checkpoint = prediction_checkpoint.merge(
            holdout_frame,
            on="_row_index",
            how="left",
            suffixes=("", "_holdout"),
        )

        for column in [
            "holdout_predicted_log_factor",
            "holdout_predicted_factor",
            "is_latest_holdout",
        ]:
            holdout_column = f"{column}_holdout"

            if holdout_column in prediction_checkpoint.columns:
                prediction_checkpoint[column] = (
                    prediction_checkpoint[holdout_column]
                    .combine_first(prediction_checkpoint[column])
                )
                prediction_checkpoint = prediction_checkpoint.drop(
                    columns=[holdout_column]
                )

        prediction_checkpoint["is_latest_holdout"] = (
            prediction_checkpoint["is_latest_holdout"]
            .fillna(False)
            .astype(bool)
        )

    prediction_checkpoint["latest_holdout_year"] = int(
        latest_holdout_year
    )

    return prediction_checkpoint

def _model_categorical_predictors(
    *,
    cfg: CifFobConfig,
    model_cfg: Any,
) -> list[str]:
    """Return model-specific categorical predictors."""
    configured = getattr(model_cfg, "categorical_predictors", None)

    if configured is not None:
        return [str(value) for value in configured]

    legacy = getattr(
        cfg.method.estimation,
        "categorical_predictors",
        [],
    )
    return [str(value) for value in legacy]


def _read_parquet_columns(
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


def _validate_input_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    predictors: list[str],
) -> None:
    """Validate estimation, diagnostics, and configured predictor columns."""
    available = _read_parquet_columns(
        con=con,
        path=input_path,
    )

    required = {
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
        "lagged_pair_factor_observations",
        "lagged_pair_factor_years",
        "lagged_pair_factor_found",
        *predictors,
    }

    missing = sorted(required - available)

    if missing:
        raise RuntimeError(
            "The CIF/FOB training-features input is missing required "
            "estimation columns: "
            + ", ".join(missing)
        )


def _source_weight(
    *,
    frame: pd.DataFrame,
    cfg: CifFobConfig,
) -> pd.Series:
    """Assign configured confidence weights to direct/mirror evidence."""
    weights = cfg.method.estimation.source_weights

    direct = frame["factor_source"].eq("direct")
    mirror = frame["factor_source"].eq("mirror")

    direct_estimated = (
        direct
        & frame["import_weight_estimated"].fillna(False).astype(bool)
    )
    direct_reported = direct & ~direct_estimated

    mirror_estimated = (
        mirror
        & frame["pair_uses_estimated_weight"].fillna(False).astype(bool)
    )
    mirror_reported = mirror & ~mirror_estimated

    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype="float64",
    )

    result.loc[direct_reported] = weights.direct_reported_weight
    result.loc[direct_estimated] = weights.direct_estimated_weight
    result.loc[mirror_reported] = weights.mirror_reported_weight
    result.loc[mirror_estimated] = weights.mirror_estimated_weight

    return result


def _evidence_value(frame: pd.DataFrame) -> pd.Series:
    """Return the source-appropriate monetary evidence value."""
    direct_value = pd.to_numeric(
        frame["import_direct_covered_primary_value"],
        errors="coerce",
    )

    mirror_value = pd.concat(
        [
            pd.to_numeric(frame["export_value_fob"], errors="coerce"),
            pd.to_numeric(frame["import_value_cif"], errors="coerce"),
        ],
        axis=1,
    ).min(axis=1, skipna=False)

    values = np.where(
        frame["factor_source"].eq("direct"),
        direct_value,
        np.where(
            frame["factor_source"].eq("mirror"),
            mirror_value,
            np.nan,
        ),
    )

    return pd.Series(
        values,
        index=frame.index,
        dtype="float64",
    )

def _resolve_maximum_factor(
    *,
    data: pd.DataFrame,
    cfg: CifFobConfig,
    minimum_factor: float,
) -> dict[str, Any]:
    """
    Resolve the effective upper CIF/FOB factor bound.

    Current rule:

        min(domain_cap, max(Q3 + k×IQR, P99))

    The distribution is calculated from positive selected direct factors,
    before applying the final upper-bound filter.
    """
    rule = cfg.method.estimation.maximum_factor_rule

    method = str(rule.method).strip().lower()

    if method != "robust_adaptive_cap":
        raise ValueError(
            "Unsupported maximum_factor_rule.method: "
            f"{method!r}."
        )

    observed_factor = pd.to_numeric(
        data["observed_cif_fob_factor"],
        errors="coerce",
    )

    factor_source = (
        data["factor_source"]
        .astype("string")
        .str.lower()
    )

    model_eligible = (
        data["model_eligible"]
        .fillna(False)
        .astype(bool)
    )

    direct_positive = observed_factor.loc[
        model_eligible
        & factor_source.eq("direct")
        & observed_factor.notna()
        & observed_factor.gt(0)
    ]

    if direct_positive.empty:
        raise RuntimeError(
            "Cannot resolve adaptive maximum CIF/FOB factor: "
            "no positive selected direct factors are available."
        )

    q1 = float(
        direct_positive.quantile(0.25)
    )
    median = float(
        direct_positive.quantile(0.50)
    )
    q3 = float(
        direct_positive.quantile(0.75)
    )
    iqr = q3 - q1

    iqr_limit = q3 + float(rule.iqr_multiplier) * iqr

    percentile_limit = float(
        direct_positive.quantile(
            float(rule.percentile)
        )
    )

    statistical_limit = max(
        iqr_limit,
        percentile_limit,
    )

    domain_cap = float(
        rule.domain_cap
    )

    maximum_factor = min(
        domain_cap,
        statistical_limit,
    )

    if maximum_factor < minimum_factor:
        raise RuntimeError(
            "Resolved maximum CIF/FOB factor is below the configured "
            "minimum factor: "
            f"minimum_factor={minimum_factor:.6f}, "
            f"maximum_factor={maximum_factor:.6f}."
        )

    return {
        "method": method,
        "minimum_factor": float(minimum_factor),
        "maximum_factor": float(maximum_factor),
        "domain_cap": domain_cap,
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "iqr_multiplier": float(rule.iqr_multiplier),
        "iqr_limit": float(iqr_limit),
        "percentile": float(rule.percentile),
        "percentile_limit": percentile_limit,
        "statistical_limit": float(statistical_limit),
        "n_direct_positive_factors": int(len(direct_positive)),
        "n_direct_factors_above_maximum": int(
            direct_positive.gt(maximum_factor).sum()
        ),
        "share_direct_factors_above_maximum_percent": (
            100.0
            * float(direct_positive.gt(maximum_factor).mean())
        ),
    }

def _resolve_latest_holdout_year(
    *,
    clean: pd.DataFrame,
    cfg: CifFobConfig,
) -> int:
    """Resolve the latest-year holdout used for preserved OOS columns."""
    validation_cfg = cfg.method.estimation.validation

    if validation_cfg.strategy != "latest_year":
        raise ValueError(
            "Unsupported validation strategy "
            f"{validation_cfg.strategy!r}. Supported: latest_year."
        )

    if validation_cfg.holdout_year is not None:
        holdout_year = int(validation_cfg.holdout_year)
    else:
        holdout_year = int(clean["year"].max())

    available_years = set(
        pd.to_numeric(clean["year"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    if holdout_year not in available_years:
        raise RuntimeError(
            f"Configured holdout year {holdout_year} is not available "
            "in the clean estimation sample."
        )

    return holdout_year


def _resolve_temporal_validation_years(
    *,
    clean: pd.DataFrame,
    cfg: CifFobConfig,
) -> list[int]:
    """Return valid expanding-window validation years."""
    backtest_cfg = cfg.method.estimation.validation.temporal_backtest

    if not backtest_cfg.enabled:
        return []

    available_years = sorted(
        pd.to_numeric(clean["year"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    requested_years = list(backtest_cfg.validation_years)

    if requested_years:
        missing = sorted(set(requested_years) - set(available_years))
        if missing:
            raise RuntimeError(
                "Temporal validation years are absent from the clean "
                "estimation sample: "
                + ", ".join(str(year) for year in missing)
            )
        candidate_years = sorted(requested_years)
    else:
        candidate_years = available_years

    minimum_training_years = int(
        backtest_cfg.minimum_training_years
    )

    valid_years: list[int] = []

    for year in candidate_years:
        earlier_years = [
            observed_year
            for observed_year in available_years
            if observed_year < year
        ]
        if len(earlier_years) >= minimum_training_years:
            valid_years.append(year)

    return valid_years


# ======================================================================
# HistGradientBoosting helpers
# ======================================================================


def _learn_category_maps(
    *,
    frame: pd.DataFrame,
    categorical_predictors: list[str],
) -> dict[str, dict[str, int]]:
    """Learn deterministic category-to-integer maps."""
    category_maps: dict[str, dict[str, int]] = {}

    for column in categorical_predictors:
        values = frame[column].astype("string")
        unique_values = sorted(
            str(value)
            for value in values.dropna().unique().tolist()
        )

        category_maps[column] = {
            value: index
            for index, value in enumerate(unique_values)
        }

    return category_maps


def _encode_hgb_predictors(
    *,
    frame: pd.DataFrame,
    predictors: list[str],
    categorical_predictors: list[str],
    category_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Encode predictors for HistGradientBoostingRegressor."""
    encoded = frame[predictors].copy()
    categorical_set = set(categorical_predictors)

    for column in predictors:
        if column in categorical_set:
            values = encoded[column].astype("string")
            encoded[column] = (
                values
                .map(category_maps[column])
                .astype("float64")
            )
        else:
            encoded[column] = pd.to_numeric(
                encoded[column],
                errors="coerce",
            )

    return encoded


def _validate_hgb_categories(
    *,
    category_maps: dict[str, dict[str, int]],
    max_bins: int,
) -> None:
    """Validate category cardinality against HGB bin limits."""
    maximum_categories = max_bins - 1

    for predictor, mapping in category_maps.items():
        n_categories = len(mapping)

        if n_categories > maximum_categories:
            raise RuntimeError(
                f"Categorical predictor {predictor!r} has "
                f"{n_categories:,} levels, but max_bins={max_bins} "
                f"supports at most {maximum_categories:,} non-missing "
                "categories."
            )


def _fit_hgb(
    *,
    train_frame: pd.DataFrame,
    predictors: list[str],
    categorical_predictors: list[str],
    sample_weight: pd.Series,
    model_cfg: Any,
) -> FittedHGBModel:
    category_maps = _learn_category_maps(
        frame=train_frame,
        categorical_predictors=categorical_predictors,
    )

    _validate_hgb_categories(
        category_maps=category_maps,
        max_bins=int(model_cfg.max_bins),
    )

    encoded = _encode_hgb_predictors(
        frame=train_frame,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        category_maps=category_maps,
    )

    categorical_mask = [
        predictor in set(categorical_predictors)
        for predictor in predictors
    ]

    target_transformation = str(
        getattr(model_cfg, "target_transformation", "log")).strip().lower()

    if target_transformation == "log":
        training_target = train_frame[
        "log_observed_factor"
    ].to_numpy(dtype=float)

    elif target_transformation == "none":
        training_target = train_frame[
        "observed_cif_fob_factor"
    ].to_numpy(dtype=float)

    else:
        raise ValueError(
            "HGB target_transformation must be 'log' or 'none', "
            f"got {target_transformation!r}."
    )

    estimator = HistGradientBoostingRegressor(
        loss=model_cfg.loss,
        learning_rate=float(model_cfg.learning_rate),
        max_iter=int(model_cfg.max_iter),
        max_leaf_nodes=int(model_cfg.max_leaf_nodes),
        min_samples_leaf=int(model_cfg.min_samples_leaf),
        l2_regularization=float(model_cfg.l2_regularization),
        max_bins=int(model_cfg.max_bins),
        categorical_features=categorical_mask,
        random_state=int(model_cfg.random_state),
    )

    estimator.fit(
        encoded[predictors],
        training_target,
        sample_weight=sample_weight,
    )

    return FittedHGBModel(
        estimator=estimator,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        category_maps=category_maps,
        target_transformation=target_transformation
    )


def _predict_hgb(
    fitted: FittedHGBModel,
    frame: pd.DataFrame,
) -> np.ndarray:
    encoded = _encode_hgb_predictors(
        frame=frame,
        predictors=fitted.predictors,
        categorical_predictors=fitted.categorical_predictors,
        category_maps=fitted.category_maps,
    )

    raw_prediction = np.asarray(
        fitted.estimator.predict(
            encoded[fitted.predictors]
        ),
        dtype=float,
    )

    if fitted.target_transformation == "log":
        return raw_prediction

    if fitted.target_transformation == "none":
        # The remaining estimation pipeline expects log-factor
        # predictions and later converts them using np.exp().
        positive_prediction = np.maximum(
            raw_prediction,
            1e-12,
        )
        return np.log(positive_prediction)

    raise ValueError(
        "Unsupported fitted HGB target transformation: "
        f"{fitted.target_transformation!r}."
    )

# ======================================================================
# XGBoost helpers
# ======================================================================


def _learn_xgb_category_levels(
    *,
    frame: pd.DataFrame,
    categorical_predictors: list[str],
) -> dict[str, list[str]]:
    """Learn deterministic category levels from the current training sample."""
    return {
        column: sorted(
            str(value)
            for value in frame[column]
            .astype("string")
            .dropna()
            .unique()
            .tolist()
        )
        for column in categorical_predictors
    }


def _encode_xgb_predictors(
    *,
    frame: pd.DataFrame,
    predictors: list[str],
    categorical_predictors: list[str],
    category_levels: dict[str, list[str]],
    enable_categorical: bool,
) -> pd.DataFrame:
    """Encode predictors consistently for XGBRegressor."""
    encoded = frame[predictors].copy()
    categorical_set = set(categorical_predictors)

    for column in predictors:
        if column not in categorical_set:
            encoded[column] = pd.to_numeric(
                encoded[column],
                errors="coerce",
            ).astype("float64")
            continue

        levels = category_levels[column]
        values = encoded[column].astype("string")

        if enable_categorical:
            encoded[column] = pd.Categorical(
                values,
                categories=levels,
            )
        else:
            mapping = {
                value: index
                for index, value in enumerate(levels)
            }
            encoded[column] = (
                values.map(mapping).astype("float64")
            )

    return encoded


def _xgb_target(
    *,
    frame: pd.DataFrame,
    target_transformation: str,
) -> np.ndarray:
    if target_transformation == "log":
        return frame["log_observed_factor"].to_numpy(dtype=float)

    if target_transformation == "none":
        return frame["observed_cif_fob_factor"].to_numpy(dtype=float)

    raise ValueError(
        "XGBoost target_transformation must be 'log' or 'none', "
        f"got {target_transformation!r}."
    )


def _make_xgb_estimator(
    *,
    model_cfg: Any,
    n_estimators: int,
    use_early_stopping: bool,
) -> Any:
    if XGBRegressor is None:  # pragma: no cover
        raise RuntimeError(
            "xgboost is required for xgboost_log_squared."
        ) from _XGBOOST_IMPORT_ERROR

    kwargs: dict[str, Any] = {
        "objective": str(model_cfg.objective),
        "eval_metric": str(model_cfg.eval_metric),
        "n_estimators": int(n_estimators),
        "learning_rate": float(model_cfg.learning_rate),
        "max_depth": int(model_cfg.max_depth),
        "min_child_weight": float(model_cfg.min_child_weight),
        "subsample": float(model_cfg.subsample),
        "colsample_bytree": float(model_cfg.colsample_bytree),
        "reg_alpha": float(model_cfg.reg_alpha),
        "reg_lambda": float(model_cfg.reg_lambda),
        "gamma": float(model_cfg.gamma),
        "tree_method": str(model_cfg.tree_method),
        "max_bin": int(model_cfg.max_bins),
        "enable_categorical": bool(model_cfg.enable_categorical),
        "random_state": int(model_cfg.random_state),
        "n_jobs": int(model_cfg.n_jobs),
    }

    if bool(model_cfg.enable_categorical):
        kwargs["max_cat_to_onehot"] = int(
            model_cfg.max_cat_to_onehot
        )

    if use_early_stopping:
        kwargs["early_stopping_rounds"] = int(
            model_cfg.early_stopping_rounds
        )

    return XGBRegressor(**kwargs)


def _fit_xgboost(
    *,
    train_frame: pd.DataFrame,
    predictors: list[str],
    categorical_predictors: list[str],
    sample_weight: pd.Series,
    model_cfg: Any,
) -> FittedXGBoostModel:
    """
    Fit XGBoost with leakage-safe internal early stopping.

    The latest year inside the supplied training sample is used only to
    choose the boosting-round count. The estimator is then refit on the
    complete supplied training sample using that selected count.
    """
    if XGBRegressor is None:  # pragma: no cover
        raise RuntimeError(
            "xgboost is required for xgboost_log_squared."
        ) from _XGBOOST_IMPORT_ERROR

    target_transformation = str(
        model_cfg.target_transformation
    ).strip().lower()

    category_levels = _learn_xgb_category_levels(
        frame=train_frame,
        categorical_predictors=categorical_predictors,
    )

    encoded_full = _encode_xgb_predictors(
        frame=train_frame,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        category_levels=category_levels,
        enable_categorical=bool(model_cfg.enable_categorical),
    )
    target_full = _xgb_target(
        frame=train_frame,
        target_transformation=target_transformation,
    )
    full_weights = sample_weight.reindex(
        train_frame.index
    ).to_numpy(dtype=float)

    training_years = sorted(
        pd.to_numeric(
            train_frame["year"],
            errors="coerce",
        ).dropna().astype(int).unique().tolist()
    )

    selected_estimators = int(model_cfg.n_estimators)

    if len(training_years) >= 2:
        internal_validation_year = training_years[-1]
        fit_mask = train_frame["year"].lt(
            internal_validation_year
        )
        eval_mask = train_frame["year"].eq(
            internal_validation_year
        )

        if fit_mask.any() and eval_mask.any():
            tuning_estimator = _make_xgb_estimator(
                model_cfg=model_cfg,
                n_estimators=int(model_cfg.n_estimators),
                use_early_stopping=True,
            )
            tuning_estimator.fit(
                encoded_full.loc[fit_mask, predictors],
                target_full[fit_mask.to_numpy()],
                sample_weight=full_weights[fit_mask.to_numpy()],
                eval_set=[(
                    encoded_full.loc[eval_mask, predictors],
                    target_full[eval_mask.to_numpy()],
                )],
                sample_weight_eval_set=[
                    full_weights[eval_mask.to_numpy()]
                ],
                verbose=False,
            )

            best_iteration = getattr(
                tuning_estimator,
                "best_iteration",
                None,
            )
            if best_iteration is not None:
                selected_estimators = max(
                    1,
                    int(best_iteration) + 1,
                )

    final_estimator = _make_xgb_estimator(
        model_cfg=model_cfg,
        n_estimators=selected_estimators,
        use_early_stopping=False,
    )
    final_estimator.fit(
        encoded_full[predictors],
        target_full,
        sample_weight=full_weights,
        verbose=False,
    )

    return FittedXGBoostModel(
        estimator=final_estimator,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        category_levels=category_levels,
        target_transformation=target_transformation,
        n_estimators_used=selected_estimators,
    )


def _predict_xgboost(
    fitted: FittedXGBoostModel,
    frame: pd.DataFrame,
) -> np.ndarray:
    encoded = _encode_xgb_predictors(
        frame=frame,
        predictors=fitted.predictors,
        categorical_predictors=fitted.categorical_predictors,
        category_levels=fitted.category_levels,
        enable_categorical=bool(
            getattr(fitted.estimator, "enable_categorical", False)
        ),
    )

    raw_prediction = np.asarray(
        fitted.estimator.predict(
            encoded[fitted.predictors]
        ),
        dtype=float,
    )

    if fitted.target_transformation == "log":
        return raw_prediction

    if fitted.target_transformation == "none":
        return np.log(
            np.maximum(raw_prediction, 1e-12)
        )

    raise ValueError(
        "Unsupported fitted XGBoost target transformation: "
        f"{fitted.target_transformation!r}."
    )


# ======================================================================
# Log-linear statistical model helpers
# ======================================================================


def _build_log_linear_design(
    *,
    frame: pd.DataFrame,
    predictors: list[str],
    categorical_predictors: list[str],
    fit_intercept: bool,
    numeric_fill_values: dict[str, float] | None = None,
    expected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Build a leakage-safe one-hot encoded statistical design matrix.

    Numeric missing values are imputed with medians learned from the
    training fold only. A missing-value indicator is added for every
    numeric predictor that is missing in the training fold.

    During prediction:
      - the training medians are reused;
      - the design is aligned to the training columns;
      - unseen categorical levels map to the training reference pattern.
    """
    design = frame[predictors].copy()
    categorical_set = set(categorical_predictors)

    numeric_predictors = [
        predictor
        for predictor in predictors
        if predictor not in categorical_set
    ]

    learned_fill_values: dict[str, float] = (
        {}
        if numeric_fill_values is None
        else dict(numeric_fill_values)
    )

    for column in numeric_predictors:
        numeric = pd.to_numeric(
            design[column],
            errors="coerce",
        )

        missing_indicator = numeric.isna().astype(float)
        indicator_column = f"{column}__missing"

        if numeric_fill_values is None:
            median_value = numeric.median(skipna=True)

            if pd.isna(median_value):
                raise RuntimeError(
                    "Log-linear regression cannot estimate a training "
                    f"median for numeric predictor {column!r}; the "
                    "training fold contains no finite values."
                )

            learned_fill_values[column] = float(median_value)

            if missing_indicator.any():
                design[indicator_column] = missing_indicator
        else:
            if column not in learned_fill_values:
                raise RuntimeError(
                    "Missing stored imputation value for statistical "
                    f"predictor {column!r}."
                )

            if (
                expected_columns is not None
                and indicator_column in expected_columns
            ):
                design[indicator_column] = missing_indicator

        design[column] = numeric.fillna(
            learned_fill_values[column]
        )

    if categorical_predictors:
        design = pd.get_dummies(
            design,
            columns=categorical_predictors,
            drop_first=True,
            dtype=float,
        )

    design = design.astype(float)

    if expected_columns is not None:
        expected_without_const = [
            column
            for column in expected_columns
            if column != "const"
        ]
        design = design.reindex(
            columns=expected_without_const,
            fill_value=0.0,
        )

    if fit_intercept:
        if sm is None:  # pragma: no cover
            raise RuntimeError(
                "statsmodels is required for log_linear_regression."
            ) from _STATSMODELS_IMPORT_ERROR

        design = sm.add_constant(
            design,
            has_constant="add",
        )

    if expected_columns is not None:
        design = design.reindex(
            columns=expected_columns,
            fill_value=0.0,
        )

    if not np.isfinite(design.to_numpy(dtype=float)).all():
        invalid_columns = [
            column
            for column in design.columns
            if not np.isfinite(
                pd.to_numeric(
                    design[column],
                    errors="coerce",
                ).to_numpy(dtype=float)
            ).all()
        ]
        raise RuntimeError(
            "The statistical design matrix still contains non-finite "
            "values after imputation. Columns: "
            + ", ".join(invalid_columns)
        )

    return design, learned_fill_values


def _fit_log_linear(
    *,
    train_frame: pd.DataFrame,
    predictors: list[str],
    categorical_predictors: list[str],
    sample_weight: pd.Series,
    model_cfg: Any,
) -> FittedLogLinearModel:
    if sm is None:  # pragma: no cover
        raise RuntimeError(
            "statsmodels is required for log_linear_regression."
        ) from _STATSMODELS_IMPORT_ERROR

    estimator_method = str(
        getattr(model_cfg, "estimator_method", "ols")
    ).strip().lower()

    if estimator_method not in {"ols", "wls"}:
        raise ValueError(
            "log_linear_regression estimator_method must be 'ols' or "
            f"'wls', got {estimator_method!r}."
        )

    fit_intercept = bool(
        getattr(model_cfg, "fit_intercept", True)
    )

    design, numeric_fill_values = _build_log_linear_design(
        frame=train_frame,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        fit_intercept=fit_intercept,
    )

    target = train_frame["log_observed_factor"].astype(float)

    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise RuntimeError(
            "The log-linear regression target contains non-finite values."
        )

    if estimator_method == "wls":
        model = sm.WLS(
            endog=target,
            exog=design,
            weights=sample_weight.astype(float),
            missing="raise",
        )
    else:
        model = sm.OLS(
            endog=target,
            exog=design,
            missing="raise",
        )

    covariance_method = str(
        getattr(model_cfg, "covariance_method", "hc3")
    ).strip().lower()

    if covariance_method == "nonrobust":
        result = model.fit()
    else:
        result = model.fit(
            cov_type=covariance_method.upper()
        )

    fitted_log = np.asarray(result.predict(design), dtype=float)
    residuals = target.to_numpy(dtype=float) - fitted_log

    inverse_transform = str(
        getattr(model_cfg, "inverse_transform", "exp")).strip().lower()
    
    if inverse_transform not in {"exp", "exp_smearing"}:
        raise ValueError(
            "log_linear_regression inverse_transform must be 'exp' or "
            f"'exp_smearing', got {inverse_transform!r}."
        )

    if inverse_transform == "exp_smearing":
        smearing_factor = float(
            np.mean(np.exp(residuals))
        )
    else:
        smearing_factor = 1.0

    return FittedLogLinearModel(
        result=result,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        design_columns=list(design.columns),
        fit_intercept=fit_intercept,
        numeric_fill_values=numeric_fill_values,
        inverse_transform=inverse_transform,
        smearing_factor = smearing_factor
    )


def _predict_log_linear(
    fitted: FittedLogLinearModel,
    frame: pd.DataFrame,
) -> np.ndarray:
    design, _ = _build_log_linear_design(
        frame=frame,
        predictors=fitted.predictors,
        categorical_predictors=fitted.categorical_predictors,
        fit_intercept=fitted.fit_intercept,
        numeric_fill_values=fitted.numeric_fill_values,
        expected_columns=fitted.design_columns,
    )

    predicted_log = np.asarray(
        fitted.result.predict(design),
        dtype=float,
    )

    if fitted.inverse_transform == "exp_smearing":
        return (
            predicted_log
            + np.log(fitted.smearing_factor)
        )

    return predicted_log


# ======================================================================
# Model dispatch
# ======================================================================


def _fit_model(
    *,
    model_name: str,
    model_cfg: Any,
    train_frame: pd.DataFrame,
    predictors: list[str],
    categorical_predictors: list[str],
    sample_weight: pd.Series,
) -> FittedHGBModel | FittedXGBoostModel | FittedLogLinearModel:

    if model_name in {
        "hist_gradient_boosting",
        "hist_gradient_boosting_squared",
        "hist_gradient_boosting_squared_level",
        "hist_gradient_boosting_squared_regularized",
    }:
        return _fit_hgb(
            train_frame=train_frame,
            predictors=predictors,
            categorical_predictors=categorical_predictors,
            sample_weight=sample_weight,
            model_cfg=model_cfg,
        )

    if model_name == "xgboost_log_squared":
        return _fit_xgboost(
            train_frame=train_frame,
            predictors=predictors,
            categorical_predictors=categorical_predictors,
            sample_weight=sample_weight,
            model_cfg=model_cfg,
        )

    if model_name in {
        "log_linear_regression",
        "log_linear_regression_smearing",
    }:
        return _fit_log_linear(
            train_frame=train_frame,
            predictors=predictors,
            categorical_predictors=categorical_predictors,
            sample_weight=sample_weight,
            model_cfg=model_cfg,
        )

    raise ValueError(
        f"Unsupported CIF/FOB estimation model: {model_name!r}."
    )


def _predict_model(
    fitted: FittedHGBModel | FittedXGBoostModel | FittedLogLinearModel,
    frame: pd.DataFrame,
) -> np.ndarray:
    if isinstance(fitted, FittedHGBModel):
        return _predict_hgb(fitted, frame)

    if isinstance(fitted, FittedXGBoostModel):
        return _predict_xgboost(fitted, frame)

    if isinstance(fitted, FittedLogLinearModel):
        return _predict_log_linear(fitted, frame)

    raise TypeError(
        f"Unsupported fitted CIF/FOB model: {type(fitted).__name__}"
    )


# ======================================================================
# Validation metrics
# ======================================================================

def _metric_values(
    *,
    observed_factor: np.ndarray,
    predicted_factor: np.ndarray,
    upper_tail_threshold: float,
) -> dict[str, float]:
    """Return the full configured metric family."""
    observed_factor = np.asarray(observed_factor, dtype=float)
    predicted_factor = np.asarray(predicted_factor, dtype=float)

    safe_observed_factor = np.maximum(observed_factor, 1e-12)
    safe_predicted_factor = np.maximum(predicted_factor, 1e-12)

    observed_log_factor = np.log(safe_observed_factor)
    predicted_log_factor = np.log(safe_predicted_factor)

    ape = (
        np.abs(predicted_factor - observed_factor)
        / safe_observed_factor
    )

    upper_mask = observed_factor >= upper_tail_threshold

    if len(observed_factor) >= 2:
        correlation_result = spearmanr(
            observed_factor,
            predicted_factor,
            nan_policy="omit",
        )
        spearman_correlation = float(
            correlation_result.statistic
            if hasattr(correlation_result, "statistic")
            else correlation_result[0]
        )
    else:
        spearman_correlation = np.nan

    metrics = {
        "mae_log_factor": float(
            mean_absolute_error(
                observed_log_factor,
                predicted_log_factor,
            )
        ),
        "rmse_log_factor": float(
            np.sqrt(
                mean_squared_error(
                    observed_log_factor,
                    predicted_log_factor,
                )
            )
        ),
        "r2_log_factor": float(
            r2_score(
                observed_log_factor,
                predicted_log_factor,
            )
        ),
        "median_ape": float(np.median(ape)),
        "mean_ape": float(np.mean(ape)),
        "p90_ape": float(np.quantile(ape, 0.90)),
        "p95_ape": float(np.quantile(ape, 0.95)),
        "spearman_correlation": spearman_correlation,
        "observed_p95": float(
            np.quantile(observed_factor, 0.95)
        ),
        "predicted_p95": float(
            np.quantile(predicted_factor, 0.95)
        ),
        "upper_tail_median_ape": (
            float(np.median(ape[upper_mask]))
            if upper_mask.any()
            else np.nan
        ),
        "upper_tail_p90_ape": (
            float(np.quantile(ape[upper_mask], 0.90))
            if upper_mask.any()
            else np.nan
        ),
        "n_upper_tail": int(upper_mask.sum()),
    }

    return metrics


def _run_temporal_backtest(
    *,
    clean: pd.DataFrame,
    cfg: CifFobConfig,
    run_spec: EstimationRunSpec,
    minimum_factor: float,
    maximum_factor: float,
    validation_years: list[int]) -> tuple[pd.DataFrame, dict[int, tuple[pd.Index, np.ndarray, np.ndarray]]]:
    """
    Run expanding-window folds for one model.

    Returns
    -------
    fold_results
        One row per model/year fold.
    predictions_by_year
        Mapping from validation year to
        (original row index, predicted log factor, predicted factor).
    """
    model_name = run_spec.model_name
    model_cfg = run_spec.model_cfg
    predictors = run_spec.predictors
    categorical_predictors = run_spec.categorical_predictors

    unknown_categorical = sorted(
        set(categorical_predictors) - set(predictors)
    )

    if unknown_categorical:
        raise ValueError(
            f"Run {run_spec.run_name!r} declares categorical predictors "
            "that are not configured predictors: "
            + ", ".join(unknown_categorical)
        )

    upper_tail_threshold = float(
        cfg.method.estimation.validation
        .temporal_backtest.upper_tail_threshold
    )

    fold_rows: list[dict[str, Any]] = []
    predictions_by_year: dict[
        int,
        tuple[pd.Index, np.ndarray, np.ndarray],
    ] = {}

    for validation_year in validation_years:
        train_mask = clean["year"].lt(validation_year)
        validation_mask = clean["year"].eq(validation_year)

        train_frame = clean.loc[train_mask]
        validation_frame = clean.loc[validation_mask]

        if train_frame.empty or validation_frame.empty:
            continue

        fitted = _fit_model(
            model_name=model_name,
            model_cfg=model_cfg,
            train_frame=train_frame,
            predictors=predictors,
            categorical_predictors=categorical_predictors,
            sample_weight=train_frame["estimation_source_weight"],
        )

        predicted_log = _predict_model(
            fitted,
            validation_frame,
        )
        predicted_factor = np.clip(
            np.exp(predicted_log),
            minimum_factor,
            maximum_factor,
        )

        observed_factor = validation_frame[
            "observed_cif_fob_factor"
        ].to_numpy(dtype=float)

        metrics = _metric_values(
            observed_factor=observed_factor,
            predicted_factor=predicted_factor,
            upper_tail_threshold=upper_tail_threshold,
        )

        fold_rows.append(
            {
                "model_name": run_spec.run_name,
                "model_algorithm": model_name,
                "predictor_set_name": run_spec.predictor_set_name,
                "validation_year": int(validation_year),
                "n_train": int(len(train_frame)),
                "n_holdout": int(len(validation_frame)),
                **metrics,
            }
        )

        predictions_by_year[int(validation_year)] = (
            validation_frame.index,
            predicted_log,
            predicted_factor,
        )

    return pd.DataFrame(fold_rows), predictions_by_year


def _summarise_backtests(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Create one compact summary row per model."""
    if fold_results.empty:
        return pd.DataFrame()

    metric_columns = [
        column
        for column in fold_results.columns
        if column not in {
            "model_name",
            "model_algorithm",
            "predictor_set_name",
            "validation_year",
            "n_train",
            "n_holdout",
        }
    ]

    rows: list[dict[str, Any]] = []

    for model_name, group in fold_results.groupby(
        "model_name",
        sort=False,
    ):
        row: dict[str, Any] = {
            "model_name": model_name,
            "model_algorithm": (
                group["model_algorithm"].iloc[0]
                if "model_algorithm" in group.columns
                else model_name
            ),
            "predictor_set_name": (
                group["predictor_set_name"].iloc[0]
                if "predictor_set_name" in group.columns
                else None
            ),
            "n_folds": int(len(group)),
            "first_validation_year": int(
                group["validation_year"].min()
            ),
            "last_validation_year": int(
                group["validation_year"].max()
            ),
            "total_holdout_rows": int(
                group["n_holdout"].sum()
            ),
        }

        for metric in metric_columns:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )
            row[f"mean_{metric}"] = float(values.mean())
            row[f"median_{metric}"] = float(values.median())
            row[f"worst_{metric}"] = float(values.max())

        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================
# Existing output inspection
# ======================================================================


def _inspect_existing_estimated_rates(
    *,
    output_path: Path,
    threads: int,
) -> dict[str, Any]:
    """Inspect a previously written Step 5 output."""
    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        row = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,

                ANY_VALUE(primary_model_name)
                    AS primary_model_name,

                ANY_VALUE(primary_estimation_run)
                    AS primary_estimation_run,

                ANY_VALUE(estimation_minimum_factor)
                    AS minimum_factor,

                ANY_VALUE(estimation_maximum_factor)
                    AS maximum_factor,

                ANY_VALUE(estimation_holdout_year)
                    AS holdout_year,

                ANY_VALUE(estimation_training_observations)
                    AS estimation_training_observations,

                COUNT(*) FILTER (
                    WHERE prediction_eligible
                ) AS n_prediction_eligible,

                COUNT(*) FILTER (
                    WHERE estimated_cif_fob_factor IS NOT NULL
                ) AS n_with_estimated_factor,

                COUNT(*) FILTER (
                    WHERE clean_estimation_eligible
                ) AS n_clean_estimation_observations,

                MEDIAN(estimated_cif_fob_factor)
                    FILTER (
                        WHERE estimated_cif_fob_factor IS NOT NULL
                    ) AS median_estimated_factor,

                QUANTILE_CONT(estimated_cif_fob_factor, 0.05)
                    FILTER (
                        WHERE estimated_cif_fob_factor IS NOT NULL
                    ) AS estimated_factor_p05,

                QUANTILE_CONT(estimated_cif_fob_factor, 0.95)
                    FILTER (
                        WHERE estimated_cif_fob_factor IS NOT NULL
                    ) AS estimated_factor_p95

            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()

    finally:
        con.close()

    if row is None:
        raise RuntimeError(
            f"Could not inspect estimated-rate output: {output_path}"
        )

    (
        n_rows,
        primary_model_name,
        primary_estimation_run,
        minimum_factor,
        maximum_factor,
        holdout_year,
        estimation_training_observations,
        n_prediction_eligible,
        n_with_estimated_factor,
        n_clean_estimation_observations,
        median_estimated_factor,
        estimated_factor_p05,
        estimated_factor_p95,
    ) = row

    if not primary_estimation_run:
        raise RuntimeError(
            "Existing estimated-rates output does not contain "
            "'primary_estimation_run'. Re-run Step 5 with force=True "
            "so the estimated-rates artifact is written with primary "
            "model metadata."
        )

    n_prediction_eligible = int(n_prediction_eligible or 0)
    n_with_estimated_factor = int(n_with_estimated_factor or 0)

    prediction_coverage_percent = (
        100.0 * n_with_estimated_factor / n_prediction_eligible
        if n_prediction_eligible > 0
        else 0.0
    )

    return {
        "status": "reused",
        "audit_passed": True,
        "ready_for_trade_correction": (
            n_with_estimated_factor > 0
        ),
        "output_path": output_path,
        "estimated_rates_path": output_path,

        # Critical metadata needed downstream.
        "primary_model_name": primary_model_name,
        "primary_estimation_run": primary_estimation_run,
        "minimum_factor": minimum_factor,
        "maximum_factor": maximum_factor,
        "holdout_year": holdout_year,

        # Keep both names if other parts of the code expect either.
        "n_rows": int(n_rows or 0),
        "n_output_rows": int(n_rows or 0),

        "n_prediction_eligible": n_prediction_eligible,
        "n_with_estimated_factor": n_with_estimated_factor,
        "prediction_coverage_percent": prediction_coverage_percent,

        "n_clean_estimation_observations": (
            int(n_clean_estimation_observations or 0)
        ),
        "estimation_training_observations": (
            int(estimation_training_observations or 0)
            if estimation_training_observations is not None
            else None
        ),

        "median_estimated_factor": median_estimated_factor,
        "estimated_factor_p05": estimated_factor_p05,
        "estimated_factor_p95": estimated_factor_p95,
    }

# ======================================================================
# Output helpers
# ======================================================================


def _write_dataframe(
    *,
    frame: pd.DataFrame,
    output_path: Path,
    threads: int,
) -> None:
    """Write a DataFrame through DuckDB using ZSTD Parquet."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")
        con.register("output_df", frame)

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM output_df
            )
            TO {_sql_literal(output_path)}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )
    finally:
        con.close()


def _write_validation_outputs(
    *,
    cfg: CifFobConfig,
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Write configured temporal-validation CSV products."""
    if cfg.products.temporal_validation_folds.enable:
        fold_path = cfg.product_path(
            "temporal_validation_folds"
        )
        fold_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fold_results.to_csv(
            fold_path,
            index=False,
        )

    if cfg.products.temporal_validation_summary.enable:
        summary_path = cfg.product_path(
            "temporal_validation_summary"
        )
        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        summary.to_csv(
            summary_path,
            index=False,
        )


# ======================================================================
# Main estimator
# ======================================================================


def estimate_cif_fob_rates(
    *,
    cfg: CifFobConfig,
    input_path: Path,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Estimate CIF/FOB factors from the Step 4 feature artifact.

    The workflow signature remains unchanged.

    This implementation checkpoints each algorithm × predictor-set run
    separately, so completed model runs can be reused if a later run or
    final assembly fails.
    """
    output_path = cfg.product_path("estimated_rates")

    if not input_path.exists():
        raise FileNotFoundError(
            f"CIF/FOB training-features Parquet not found: {input_path}"
        )

    if not force and valid_output(output_path):
        logger.info(
            "REUSE — CIF/FOB estimated rates already exist: %s",
            output_path,
        )
        return _inspect_existing_estimated_rates(
            output_path=output_path,
            threads=cfg.engine.threads,
        )

    estimation_cfg = cfg.method.estimation

    primary_model_name, model_configs = _enabled_model_configs(cfg)
    predictor_sets = _configured_predictor_sets(cfg)

    run_specs = _resolve_run_specs(
        cfg=cfg,
        model_configs=model_configs,
        predictor_sets=predictor_sets,
        primary_model_name=primary_model_name,
    )

    predictors = _all_predictors_for_validation(
        predictor_sets
    )

    primary_run_name = next(
        spec.run_name
        for spec in run_specs
        if spec.is_primary
    )

    minimum_evidence_value = float(
        estimation_cfg.minimum_evidence_value_usd
    )
    minimum_factor = float(
        estimation_cfg.minimum_factor
    )
    exclude_proxy_weights = bool(
        estimation_cfg.exclude_proxy_weights
    )

    logger.info("Estimating and comparing CIF/FOB rates.")
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)
    logger.info("  Predictor union: %s", ", ".join(predictors))
    logger.info(
        "  Predictor sets: %s",
        ", ".join(predictor_sets),
    )
    logger.info(
        "  Enabled models: %s",
        ", ".join(model_configs),
    )
    logger.info("  Primary model: %s", primary_model_name)
    logger.info("  Primary estimation run: %s", primary_run_name)
    logger.info(
        "  Minimum evidence value: %.2f USD",
        minimum_evidence_value,
    )

    logger.info(
        "  Exclude proxy weights: %s",
        exclude_proxy_weights,
    )
    logger.info(
        "  Training factor sources: %s",
        ", ".join(estimation_cfg.training_factor_sources),
    )

    # ==================================================================
    # Read and validate Step 4 feature artifact
    # ==================================================================

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        _validate_input_columns(
            con=con,
            input_path=input_path,
            predictors=predictors,
        )

        select_columns = [
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
            "lagged_pair_factor_observations",
            "lagged_pair_factor_years",
            "lagged_pair_factor_found",
            *[
                predictor
                for predictor in predictors
                if predictor != "year"
            ],
        ]

        select_columns = list(
            dict.fromkeys(select_columns)
        )

        quoted_select = ",\n".join(
            _quoted_identifier(column)
            for column in select_columns
        )

        data = con.execute(
            f"""
            SELECT
                {quoted_select}
            FROM read_parquet({_sql_literal(input_path)})
            """
        ).df()

    finally:
        con.close()

    if data.empty:
        raise RuntimeError(
            "Rate-estimation input contains no rows."
        )
    
    maximum_factor_metrics = _resolve_maximum_factor(
        data=data,
        cfg=cfg,
        minimum_factor=minimum_factor,
        )

    maximum_factor = float(maximum_factor_metrics["maximum_factor"])

    logger.info(
        "  Direct positive factors above resolved maximum: %s "
        "(%.4f%%)",
        maximum_factor_metrics["n_direct_factors_above_maximum"],
        maximum_factor_metrics[
            "share_direct_factors_above_maximum_percent"
        ],
    )
    
    logger.info(
        "  Permitted observed-factor range: %.3f-%.3f",
        minimum_factor,
        maximum_factor,
    )

    logger.info(
        "  Maximum-factor rule: %s",
        maximum_factor_metrics["method"],
    )

    logger.info(
        "  Maximum-factor components: "
        "Q1=%.6f, median=%.6f, Q3=%.6f, IQR=%.6f, "
        "IQR limit=%.6f, P%.2f=%.6f, domain cap=%.6f, "
        "resolved maximum=%.6f",
        maximum_factor_metrics["q1"],
        maximum_factor_metrics["median"],
        maximum_factor_metrics["q3"],
        maximum_factor_metrics["iqr"],
        maximum_factor_metrics["iqr_limit"],
        100.0 * maximum_factor_metrics["percentile"],
        maximum_factor_metrics["percentile_limit"],
        maximum_factor_metrics["domain_cap"],
        maximum_factor_metrics["maximum_factor"],
    )

    # ==================================================================
    # Shared clean estimation sample
    # ==================================================================

    data["estimation_evidence_value"] = _evidence_value(
        data
    )

    data["estimation_source_weight"] = _source_weight(
        frame=data,
        cfg=cfg,
    )

    data["passes_minimum_value"] = (
        data["estimation_evidence_value"]
        .gt(minimum_evidence_value)
        .fillna(False)
    )

    observed_factor = pd.to_numeric(
        data["observed_cif_fob_factor"],
        errors="coerce",
    )

    data["passes_factor_lower_bound"] = (
        observed_factor
        .ge(minimum_factor)
        .fillna(False)
    )

    data["passes_factor_upper_bound"] = (
        observed_factor
        .le(maximum_factor)
        .fillna(False)
    )

    if exclude_proxy_weights:
        is_direct = (
            data["factor_source"]
            .astype("string")
            .str.lower()
            .eq("direct")
        )

        data["passes_proxy_rule"] = (
            is_direct
            | (
                ~data["pair_uses_weight_proxy"]
                .fillna(False)
                .astype(bool)
            )
        )
    else:
        data["passes_proxy_rule"] = True

    data["prediction_eligible"] = (
        data["initial_predictors_complete"]
        .fillna(False)
        .astype(bool)
    )

    training_factor_sources = [
        source.strip().lower()
        for source in estimation_cfg.training_factor_sources
    ]

    data["passes_factor_source_rule"] = (
        data["factor_source"]
        .astype("string")
        .str.lower()
        .isin(training_factor_sources)
    )

    data["clean_estimation_eligible"] = (
        data["model_eligible"].fillna(False).astype(bool)
        & data["prediction_eligible"]
        & data["passes_factor_source_rule"]
        & data["passes_minimum_value"]
        & data["passes_factor_lower_bound"]
        & data["passes_factor_upper_bound"]
        & data["passes_proxy_rule"]
        & data["estimation_source_weight"].gt(0)
        & observed_factor.notna()
    )

    clean = data.loc[
        data["clean_estimation_eligible"]
    ].copy()

    if clean.empty:
        raise RuntimeError(
            "No observations passed the configured clean-estimation rules."
        )

    clean["observed_cif_fob_factor"] = pd.to_numeric(
        clean["observed_cif_fob_factor"],
        errors="raise",
    )

    clean["log_observed_factor"] = np.log(
        clean["observed_cif_fob_factor"]
    )

    latest_holdout_year = _resolve_latest_holdout_year(
        clean=clean,
        cfg=cfg,
    )

    validation_years = _resolve_temporal_validation_years(
        clean=clean,
        cfg=cfg,
    )

    if latest_holdout_year not in validation_years:
        validation_years = sorted(
            set(validation_years + [latest_holdout_year])
        )

    logger.info(
        "  Temporal validation years: %s",
        ", ".join(str(year) for year in validation_years),
    )

    prediction_rows = data.loc[
        data["prediction_eligible"]
    ].copy()

    if prediction_rows.empty:
        raise RuntimeError(
            "No rows have complete predictors for final estimation."
        )

    # ==================================================================
    # Run or reuse one checkpoint per model × predictor-set run
    # ==================================================================

    all_fold_results: list[pd.DataFrame] = []

    data["is_temporal_holdout"] = (
        data["year"].eq(latest_holdout_year)
        & data["clean_estimation_eligible"]
    )

    primary_factor_column: str | None = None
    primary_log_column: str | None = None

    for run_spec in run_specs:
        prefix = _safe_model_prefix(
            run_spec.run_name
        )

        log_column = (
            f"{prefix}_predicted_log_cif_fob_factor"
        )
        factor_column = (
            f"{prefix}_estimated_cif_fob_factor"
        )
        holdout_log_column = (
            f"{prefix}_holdout_predicted_log_factor"
        )
        holdout_factor_column = (
            f"{prefix}_holdout_predicted_factor"
        )

        data[log_column] = np.nan
        data[factor_column] = np.nan
        data[holdout_log_column] = np.nan
        data[holdout_factor_column] = np.nan

        if (
            not force
            and _run_checkpoint_complete(
                cfg=cfg,
                run_name=run_spec.run_name,
            )
        ):
            logger.info(
                "  REUSE estimation run checkpoint: %s",
                run_spec.run_name,
            )

            fold_results, prediction_checkpoint = (
                _read_run_checkpoint(
                    cfg=cfg,
                    run_name=run_spec.run_name,
                    threads=cfg.engine.threads,
                )
            )

        else:
            logger.info(
                "  RUN estimation checkpoint: %s",
                run_spec.run_name,
            )

            fold_results, predictions_by_year = (
                _run_temporal_backtest(
                    clean=clean,
                    cfg=cfg,
                    run_spec=run_spec,
                    minimum_factor=minimum_factor,
                    maximum_factor=maximum_factor,
                    validation_years=validation_years,
                )
            )

            if fold_results.empty:
                raise RuntimeError(
                    f"Run {run_spec.run_name!r} produced no "
                    "temporal folds."
                )

            latest_holdout_predictions = (
                predictions_by_year.get(latest_holdout_year)
            )

            logger.info(
                "  Final full-sample fit: %s",
                run_spec.run_name,
            )

            fitted = _fit_model(
                model_name=run_spec.model_name,
                model_cfg=run_spec.model_cfg,
                train_frame=clean,
                predictors=run_spec.predictors,
                categorical_predictors=(
                    run_spec.categorical_predictors
                ),
                sample_weight=clean["estimation_source_weight"],
            )

            full_predicted_log = _predict_model(
                fitted,
                prediction_rows,
            )

            full_predicted_factor = np.clip(
                np.exp(full_predicted_log),
                minimum_factor,
                maximum_factor,
            )

            prediction_checkpoint = (
                _build_run_prediction_checkpoint(
                    run_name=run_spec.run_name,
                    prediction_row_index=prediction_rows.index,
                    full_predicted_log=full_predicted_log,
                    full_predicted_factor=full_predicted_factor,
                    latest_holdout_year=latest_holdout_year,
                    latest_holdout_predictions=(
                        latest_holdout_predictions
                    ),
                )
            )

            _write_run_checkpoint(
                cfg=cfg,
                run_name=run_spec.run_name,
                fold_results=fold_results,
                predictions=prediction_checkpoint,
                threads=cfg.engine.threads,
            )

            logger.info(
                "  WROTE estimation run checkpoint: %s",
                run_spec.run_name,
            )

        if fold_results.empty:
            raise RuntimeError(
                f"Run {run_spec.run_name!r} has empty fold results."
            )

        all_fold_results.append(
            fold_results
        )

        required_prediction_columns = {
            "_row_index",
            "full_predicted_log_factor",
            "full_predicted_factor",
            "holdout_predicted_log_factor",
            "holdout_predicted_factor",
        }

        missing_prediction_columns = sorted(
            required_prediction_columns
            - set(prediction_checkpoint.columns)
        )

        if missing_prediction_columns:
            raise RuntimeError(
                f"Checkpoint for run {run_spec.run_name!r} is "
                "missing prediction columns: "
                + ", ".join(missing_prediction_columns)
            )

        prediction_checkpoint = prediction_checkpoint.set_index(
            "_row_index"
        )

        unknown_checkpoint_rows = sorted(
            set(prediction_checkpoint.index)
            - set(data.index)
        )

        if unknown_checkpoint_rows:
            raise RuntimeError(
                f"Checkpoint for run {run_spec.run_name!r} contains "
                "row indexes that are absent from the current input. "
                f"Example: {unknown_checkpoint_rows[:5]}"
            )

        data.loc[
            prediction_checkpoint.index,
            log_column,
        ] = prediction_checkpoint[
            "full_predicted_log_factor"
        ].to_numpy(dtype=float)

        data.loc[
            prediction_checkpoint.index,
            factor_column,
        ] = prediction_checkpoint[
            "full_predicted_factor"
        ].to_numpy(dtype=float)

        holdout_mask = prediction_checkpoint[
            "holdout_predicted_factor"
        ].notna()

        if holdout_mask.any():
            holdout_index = prediction_checkpoint.index[
                holdout_mask
            ]

            data.loc[
                holdout_index,
                holdout_log_column,
            ] = prediction_checkpoint.loc[
                holdout_index,
                "holdout_predicted_log_factor",
            ].to_numpy(dtype=float)

            data.loc[
                holdout_index,
                holdout_factor_column,
            ] = prediction_checkpoint.loc[
                holdout_index,
                "holdout_predicted_factor",
            ].to_numpy(dtype=float)

        latest_row = fold_results.loc[
            fold_results["validation_year"].eq(
                latest_holdout_year
            )
        ]

        if not latest_row.empty:
            row = latest_row.iloc[0]
            logger.info(
                "    Latest holdout %s — median APE %.4f%%, "
                "p90 APE %.4f%%, p95 APE %.4f%%",
                latest_holdout_year,
                100.0 * float(row["median_ape"]),
                100.0 * float(row["p90_ape"]),
                100.0 * float(row["p95_ape"]),
            )

        if run_spec.is_primary:
            primary_log_column = log_column
            primary_factor_column = factor_column

    if primary_factor_column is None or primary_log_column is None:
        raise RuntimeError(
            "Primary model predictions were not produced."
        )

    fold_results = pd.concat(
        all_fold_results,
        ignore_index=True,
    )

    validation_summary = _summarise_backtests(
        fold_results
    )

    _write_validation_outputs(
        cfg=cfg,
        fold_results=fold_results,
        summary=validation_summary,
    )

    # ==================================================================
    # Stable downstream aliases
    # ==================================================================

    data["predicted_log_cif_fob_factor"] = data[
        primary_log_column
    ]

    data["estimated_cif_fob_factor"] = data[
        primary_factor_column
    ]

    data["estimated_factor_method"] = np.where(
        data["prediction_eligible"],
        primary_run_name,
        None,
    )

    data["primary_model_name"] = primary_model_name
    data["primary_estimation_run"] = primary_run_name
    data["estimation_holdout_year"] = latest_holdout_year
    data["estimation_training_observations"] = len(clean)
    data["estimation_minimum_factor"] = minimum_factor
    data["estimation_maximum_factor"] = maximum_factor
    data["maximum_factor_rule_method"] = maximum_factor_metrics["method"]
    data["maximum_factor_domain_cap"] = maximum_factor_metrics["domain_cap"]
    data["maximum_factor_q1"] = maximum_factor_metrics["q1"]
    data["maximum_factor_median"] = maximum_factor_metrics["median"]
    data["maximum_factor_q3"] = maximum_factor_metrics["q3"]
    data["maximum_factor_iqr"] = maximum_factor_metrics["iqr"]
    data["maximum_factor_iqr_multiplier"] = (maximum_factor_metrics["iqr_multiplier"])
    data["maximum_factor_iqr_limit"] = maximum_factor_metrics["iqr_limit"]
    data["maximum_factor_percentile"] = maximum_factor_metrics["percentile"]
    data["maximum_factor_percentile_limit"] = maximum_factor_metrics["percentile_limit"]
    data["maximum_factor_statistical_limit"] = maximum_factor_metrics["statistical_limit"]
    data["factor_for_application"] = data["estimated_cif_fob_factor"]

    # ==================================================================
    # Write and audit main output
    # ==================================================================

    _write_dataframe(
        frame=data,
        output_path=output_path,
        threads=cfg.engine.threads,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        output_sql = _sql_literal(output_path)

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
                ) AS n_duplicate_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE clean_estimation_eligible
                ) AS n_clean_estimation_observations,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE prediction_eligible
                ) AS n_prediction_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE estimated_cif_fob_factor IS NOT NULL
                ) AS n_with_estimated_factor,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE prediction_eligible
                      AND estimated_cif_fob_factor IS NULL
                ) AS n_missing_prediction_when_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE NOT prediction_eligible
                      AND estimated_cif_fob_factor IS NOT NULL
                ) AS n_prediction_when_ineligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE estimated_cif_fob_factor IS NOT NULL
                      AND (
                            estimated_cif_fob_factor < {minimum_factor}
                         OR estimated_cif_fob_factor > {maximum_factor}
                      )
                ) AS n_prediction_bound_failures,

                (
                    SELECT MEDIAN(estimated_cif_fob_factor)
                    FROM read_parquet({output_sql})
                    WHERE estimated_cif_fob_factor IS NOT NULL
                ) AS median_estimated_factor,

                (
                    SELECT QUANTILE_CONT(
                        estimated_cif_fob_factor,
                        0.05
                    )
                    FROM read_parquet({output_sql})
                    WHERE estimated_cif_fob_factor IS NOT NULL
                ) AS estimated_factor_p05,

                (
                    SELECT QUANTILE_CONT(
                        estimated_cif_fob_factor,
                        0.95
                    )
                    FROM read_parquet({output_sql})
                    WHERE estimated_cif_fob_factor IS NOT NULL
                ) AS estimated_factor_p95
            """
        ).fetchone()

    finally:
        con.close()

    if audit_row is None:
        raise RuntimeError(
            "Could not audit estimated-rate output."
        )

    (
        n_output_rows,
        n_duplicate_keys,
        n_clean_estimation_observations,
        n_prediction_eligible,
        n_with_estimated_factor,
        n_missing_prediction_when_eligible,
        n_prediction_when_ineligible,
        n_prediction_bound_failures,
        median_estimated_factor,
        estimated_factor_p05,
        estimated_factor_p95,
    ) = audit_row

    if n_output_rows != len(data):
        raise RuntimeError(
            "Estimated-rate output row count does not match input: "
            f"output={n_output_rows:,}, input={len(data):,}."
        )

    if n_duplicate_keys > 0:
        raise RuntimeError(
            "Estimated-rate output contains duplicate bilateral keys: "
            f"{n_duplicate_keys:,}."
        )

    if n_missing_prediction_when_eligible > 0:
        raise RuntimeError(
            "Prediction-eligible rows are missing primary-model "
            f"estimates: {n_missing_prediction_when_eligible:,}."
        )

    if n_prediction_when_ineligible > 0:
        raise RuntimeError(
            "Prediction-ineligible rows unexpectedly received primary "
            f"model estimates: {n_prediction_when_ineligible:,}."
        )

    if n_prediction_bound_failures > 0:
        raise RuntimeError(
            "Estimated factors fall outside configured limits: "
            f"{n_prediction_bound_failures:,}."
        )

    ready_for_trade_correction = (
        int(n_with_estimated_factor or 0) > 0
    )

    prediction_coverage_percent = (
        100.0 * n_with_estimated_factor / n_output_rows
        if n_output_rows
        else 0.0
    )

    logger.info("CIF/FOB rates estimated and compared.")
    logger.info("  Primary model: %s", primary_model_name)
    logger.info("  Primary estimation run: %s", primary_run_name)
    logger.info("  Output rows: %s", n_output_rows)
    logger.info(
        "  Clean estimation observations: %s",
        n_clean_estimation_observations,
    )
    logger.info(
        "  Prediction-eligible rows: %s",
        n_prediction_eligible,
    )
    logger.info(
        "  Rows with primary estimated factor: %s",
        n_with_estimated_factor,
    )
    logger.info(
        "  Estimated-factor coverage: %.4f%%",
        prediction_coverage_percent,
    )
    logger.info(
        "  Median primary estimated factor: %s",
        median_estimated_factor,
    )
    logger.info(
        "  Primary estimated factor p05/p95: %s / %s",
        estimated_factor_p05,
        estimated_factor_p95,
    )
    logger.info(
        "  Temporal validation folds: %s",
        cfg.product_path("temporal_validation_folds"),
    )
    logger.info(
        "  Temporal validation summary: %s",
        cfg.product_path("temporal_validation_summary"),
    )
    logger.info("  Output: %s", output_path)

    latest_primary_row = fold_results.loc[
        (fold_results["model_name"] == primary_run_name)
        & (
            fold_results["validation_year"]
            == latest_holdout_year
        )
    ]

    latest_primary_metrics: dict[str, Any] = {}

    if not latest_primary_row.empty:
        latest_primary_metrics = latest_primary_row.iloc[0].to_dict()

    return {
        "status": "completed",
        "audit_passed": True,
        "ready_for_trade_correction": (
            ready_for_trade_correction
        ),
        "output_path": output_path,
        "estimated_rates_path": output_path,
        "primary_model_name": primary_model_name,
        "primary_estimation_run": primary_run_name,
        "minimum_factor": minimum_factor,
        "maximum_factor": maximum_factor,
        "maximum_factor_rule":maximum_factor_metrics,
        "enabled_models": list(model_configs),
        "enabled_estimation_runs": [
            spec.run_name
            for spec in run_specs
        ],
        "predictor_sets": list(predictor_sets),
        "n_output_rows": n_output_rows,
        "n_clean_estimation_observations": (
            n_clean_estimation_observations
        ),
        "n_prediction_eligible": n_prediction_eligible,
        "n_with_estimated_factor": n_with_estimated_factor,
        "prediction_coverage_percent": (
            prediction_coverage_percent
        ),
        "n_duplicate_keys": n_duplicate_keys,
        "median_estimated_factor": median_estimated_factor,
        "estimated_factor_p05": estimated_factor_p05,
        "estimated_factor_p95": estimated_factor_p95,
        "holdout_year": latest_holdout_year,
        "temporal_validation_years": validation_years,
        "temporal_validation_fold_count": int(
            len(fold_results)
        ),
        "temporal_validation_summary_path": (
            cfg.product_path(
                "temporal_validation_summary"
            )
        ),
        "temporal_validation_folds_path": (
            cfg.product_path(
                "temporal_validation_folds"
            )
        ),
        "latest_primary_holdout_metrics": (
            latest_primary_metrics
        ),
    }



