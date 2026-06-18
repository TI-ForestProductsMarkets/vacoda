"""
vacoda.cif_fob.estimate_rates

Estimate CIF/FOB conversion factors from the Step 4 training-features
artifact.

Design
------
1. Read predictors, thresholds, source weights, validation policy, and
   model hyperparameters from the parsed CIF/FOB configuration.
2. Build the clean estimation sample from model-eligible rows with a
   complete predictor set.
3. Fit the configured histogram gradient-boosting model to the log of
   the observed CIF/FOB factor.
4. Evaluate on a temporal holdout.
5. Refit on all clean observations.
6. Predict only rows with complete predictors. Rows without complete
   predictors remain in the output with a null estimate and an explicit
   availability flag; they can be handled by a later fallback stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from altair.datasets import data
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from vacoda.config.cif_fob import CifFobConfig
from vacoda.cif_fob.prepare_input import valid_output


# ======================================================================
# Helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _quoted_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


def _configured_predictors(cfg: CifFobConfig) -> list[str]:
    """Return unique configured predictors while preserving YAML order."""
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
    """Validate estimation, audit, and configured predictor columns."""
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


def _encode_predictors(
    *,
    frame: pd.DataFrame,
    predictors: list[str],
    categorical_predictors: list[str],
    category_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Encode configured predictors for scikit-learn."""
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


def _validate_categories(
    *,
    category_maps: dict[str, dict[str, int]],
    max_bins: int,
) -> None:
    """
    HistGradientBoosting reserves one bin for missing values, so the
    number of non-missing categories must be below max_bins.
    """
    maximum_categories = max_bins - 1

    for predictor, mapping in category_maps.items():
        n_categories = len(mapping)

        if n_categories > maximum_categories:
            raise RuntimeError(
                f"Categorical predictor {predictor!r} has "
                f"{n_categories:,} levels, but max_bins={max_bins} "
                f"supports at most {maximum_categories:,} non-missing "
                "categories. Use a broader grouping or increase max_bins "
                "within scikit-learn's supported limit."
            )


def _build_model(cfg: CifFobConfig, *, categorical_mask: list[bool]):
    """Construct the configured estimator."""
    model_cfg = cfg.method.estimation.model

    if model_cfg.name != "hist_gradient_boosting":
        raise ValueError(
            "Unsupported CIF/FOB estimation model "
            f"{model_cfg.name!r}. Supported: hist_gradient_boosting."
        )

    return HistGradientBoostingRegressor(
        loss=model_cfg.loss,
        learning_rate=model_cfg.learning_rate,
        max_iter=model_cfg.max_iter,
        max_leaf_nodes=model_cfg.max_leaf_nodes,
        min_samples_leaf=model_cfg.min_samples_leaf,
        l2_regularization=model_cfg.l2_regularization,
        max_bins=model_cfg.max_bins,
        categorical_features=categorical_mask,
        random_state=model_cfg.random_state,
    )


def _resolve_holdout_year(
    *,
    clean: pd.DataFrame,
    cfg: CifFobConfig,
) -> int:
    """Resolve the configured temporal holdout year."""
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


def _prediction_error_metrics(
    *,
    observed_log: np.ndarray,
    predicted_log: np.ndarray,
) -> dict[str, float]:
    """Return temporal-holdout error metrics."""
    observed_factor = np.exp(observed_log)
    predicted_factor = np.exp(predicted_log)

    absolute_percentage_error = (
        np.abs(predicted_factor - observed_factor)
        / np.maximum(observed_factor, 1e-12)
    )

    return {
        "holdout_mae_log_factor": float(
            mean_absolute_error(observed_log, predicted_log)
        ),
        "holdout_median_absolute_percentage_error": float(
            np.median(absolute_percentage_error)
        ),
        "holdout_p90_absolute_percentage_error": float(
            np.quantile(absolute_percentage_error, 0.90)
        ),
    }


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
                    ) AS median_estimated_factor

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
        n_prediction_eligible,
        n_with_estimated_factor,
        n_clean_estimation_observations,
        median_estimated_factor,
    ) = row

    return {
        "status": "reused",
        "audit_passed": True,
        "ready_for_trade_correction": (
            int(n_with_estimated_factor or 0) > 0
        ),
        "output_path": output_path,
        "estimated_rates_path": output_path,
        "n_rows": n_rows,
        "n_prediction_eligible": n_prediction_eligible,
        "n_with_estimated_factor": n_with_estimated_factor,
        "n_clean_estimation_observations": (
            n_clean_estimation_observations
        ),
        "median_estimated_factor": median_estimated_factor,
    }


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
    """Estimate CIF/FOB factors from the Step 4 feature artifact."""
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictors = _configured_predictors(cfg)
    estimation_cfg = cfg.method.estimation
    categorical_predictors = list(
        estimation_cfg.categorical_predictors
    )

    unknown_categorical = sorted(
        set(categorical_predictors) - set(predictors)
    )

    if unknown_categorical:
        raise ValueError(
            "Configured categorical predictors are not present in the "
            "predictor list: "
            + ", ".join(unknown_categorical)
        )

    minimum_evidence_value = float(
        estimation_cfg.minimum_evidence_value_usd
    )
    minimum_factor = float(estimation_cfg.minimum_factor)
    maximum_factor = float(estimation_cfg.maximum_factor)
    exclude_proxy_weights = bool(
        estimation_cfg.exclude_proxy_weights
    )

    logger.info("Estimating CIF/FOB rates.")
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)
    logger.info("  Predictors: %s", ", ".join(predictors))
    logger.info(
        "  Categorical predictors: %s",
        ", ".join(categorical_predictors) or "none",
    )
    logger.info(
        "  Minimum evidence value: %.2f USD",
        minimum_evidence_value,
    )
    logger.info(
        "  Permitted observed-factor range: %.3f-%.3f",
        minimum_factor,
        maximum_factor,
    )
    logger.info(
        "  Exclude proxy weights: %s",
        exclude_proxy_weights,
    )
    logger.info(
        "  Model: %s",
        estimation_cfg.model.name,
    )
    logger.info(
        "  Validation strategy: %s",
        estimation_cfg.validation.strategy,
    )

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

        input_sql = _sql_literal(input_path)

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
            *[
                predictor
                for predictor in predictors
                if predictor not in {
                    "year",
                }
            ],
        ]

        # Preserve deterministic order and avoid duplicate selected fields.
        select_columns = list(dict.fromkeys(select_columns))

        quoted_select = ",\n".join(
            _quoted_identifier(column)
            for column in select_columns
        )

        data = con.execute(
            f"""
            SELECT
                {quoted_select}
            FROM read_parquet({input_sql})
            """
        ).df()

    finally:
        con.close()

    if data.empty:
        raise RuntimeError(
            "Rate-estimation input contains no rows."
        )

    # ==================================================================
    # Clean estimation sample
    # ==================================================================

    data["estimation_evidence_value"] = _evidence_value(data)
    data["estimation_source_weight"] = _source_weight(
        frame=data,
        cfg=cfg,
    )

    data["passes_minimum_value"] = (
        data["estimation_evidence_value"]
        .ge(minimum_evidence_value)
        .fillna(False)
    )

    data["passes_factor_lower_bound"] = (
        pd.to_numeric(
            data["observed_cif_fob_factor"],
            errors="coerce",
        )
        .ge(minimum_factor)
        .fillna(False)
    )

    data["passes_factor_upper_bound"] = (
        pd.to_numeric(
            data["observed_cif_fob_factor"],
            errors="coerce",
        )
        .le(maximum_factor)
        .fillna(False)
    )

    if exclude_proxy_weights:
        data["passes_proxy_rule"] = (
            ~data["pair_uses_weight_proxy"]
            .fillna(False)
            .astype(bool)
        )
    else:
        data["passes_proxy_rule"] = True

    data["prediction_eligible"] = (
        data["initial_predictors_complete"]
        .fillna(False)
        .astype(bool)
    )

    data["clean_estimation_eligible"] = (
        data["model_eligible"].fillna(False).astype(bool)
        & data["prediction_eligible"]
        & data["passes_minimum_value"]
        & data["passes_factor_lower_bound"]
        & data["passes_factor_upper_bound"]
        & data["passes_proxy_rule"]
        & data["estimation_source_weight"].gt(0)
        & data["observed_cif_fob_factor"].notna()
    )

    clean = data.loc[
        data["clean_estimation_eligible"]
    ].copy()

    if clean.empty:
        raise RuntimeError(
            "No observations passed the configured clean-estimation rules."
        )

    clean["log_observed_factor"] = np.log(
        pd.to_numeric(
            clean["observed_cif_fob_factor"],
            errors="raise",
        )
    )

    # ==================================================================
    # Temporal holdout
    # ==================================================================

    holdout_year = _resolve_holdout_year(
        clean=clean,
        cfg=cfg,
    )

    train_mask = clean["year"].lt(holdout_year)
    holdout_mask = clean["year"].eq(holdout_year)

    n_train = int(train_mask.sum())
    n_holdout = int(holdout_mask.sum())

    if n_train == 0 or n_holdout == 0:
        raise RuntimeError(
            "Temporal validation requires observations before and during "
            f"holdout year {holdout_year}. "
            f"Training rows={n_train:,}, holdout rows={n_holdout:,}."
        )

    holdout_category_maps = _learn_category_maps(
        frame=clean.loc[train_mask],
        categorical_predictors=categorical_predictors,
    )

    _validate_categories(
        category_maps=holdout_category_maps,
        max_bins=estimation_cfg.model.max_bins,
    )

    encoded_train = _encode_predictors(
        frame=clean.loc[train_mask],
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        category_maps=holdout_category_maps,
    )

    encoded_holdout = _encode_predictors(
        frame=clean.loc[holdout_mask],
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        category_maps=holdout_category_maps,
    )

    categorical_mask = [
        predictor in set(categorical_predictors)
        for predictor in predictors
    ]

    holdout_model = _build_model(
        cfg,
        categorical_mask=categorical_mask,
    )

    holdout_model.fit(
        encoded_train[predictors],
        clean.loc[train_mask, "log_observed_factor"],
        sample_weight=clean.loc[
            train_mask,
            "estimation_source_weight",
        ],
    )

    holdout_prediction_log = holdout_model.predict(
        encoded_holdout[predictors]
    )

    holdout_metrics = _prediction_error_metrics(
        observed_log=clean.loc[
            holdout_mask,
            "log_observed_factor",
        ].to_numpy(),
        predicted_log=holdout_prediction_log,
    )

    # --------------------------------------------------------------
    # Preserve genuine out-of-sample holdout predictions.
    #  --------------------------------------------------------------
    
    holdout_prediction_factor = np.exp(
    holdout_prediction_log)

    data["is_temporal_holdout"] = False
    data["holdout_predicted_log_factor"] = np.nan
    data["holdout_predicted_factor"] = np.nan

    holdout_indices = clean.loc[
        holdout_mask
    ].index

    data.loc[
        holdout_indices,
        "is_temporal_holdout",
    ] = True

    data.loc[
        holdout_indices,
        "holdout_predicted_log_factor",
    ] = holdout_prediction_log

    data.loc[
        holdout_indices,
        "holdout_predicted_factor",
    ] = holdout_prediction_factor


    logger.info("  Temporal holdout year: %s", holdout_year)
    logger.info("  Training observations: %s", n_train)
    logger.info("  Holdout observations: %s", n_holdout)
    logger.info(
        "  Holdout MAE, log factor: %.6f",
        holdout_metrics["holdout_mae_log_factor"],
    )
    logger.info(
        "  Holdout median absolute percentage error: %.4f%%",
        100.0
        * holdout_metrics[
            "holdout_median_absolute_percentage_error"
        ],
    )
    logger.info(
        "  Holdout p90 absolute percentage error: %.4f%%",
        100.0
        * holdout_metrics[
            "holdout_p90_absolute_percentage_error"
        ],
    )

    # ==================================================================
    # Final fit and prediction
    # ==================================================================

    final_category_maps = _learn_category_maps(
        frame=clean,
        categorical_predictors=categorical_predictors,
    )

    _validate_categories(
        category_maps=final_category_maps,
        max_bins=estimation_cfg.model.max_bins,
    )

    encoded_clean = _encode_predictors(
        frame=clean,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        category_maps=final_category_maps,
    )

    prediction_rows = data.loc[
        data["prediction_eligible"]
    ].copy()

    encoded_prediction = _encode_predictors(
        frame=prediction_rows,
        predictors=predictors,
        categorical_predictors=categorical_predictors,
        category_maps=final_category_maps,
    )

    final_model = _build_model(
        cfg,
        categorical_mask=categorical_mask,
    )

    final_model.fit(
        encoded_clean[predictors],
        clean["log_observed_factor"],
        sample_weight=clean["estimation_source_weight"],
    )

    predicted_log_factor = final_model.predict(
        encoded_prediction[predictors]
    )

    predicted_factor = np.clip(
        np.exp(predicted_log_factor),
        minimum_factor,
        maximum_factor,
    )

    data["predicted_log_cif_fob_factor"] = np.nan
    data["estimated_cif_fob_factor"] = np.nan

    data.loc[
        prediction_rows.index,
        "predicted_log_cif_fob_factor",
    ] = predicted_log_factor

    data.loc[
        prediction_rows.index,
        "estimated_cif_fob_factor",
    ] = predicted_factor

    data["estimated_factor_method"] = np.where(
        data["prediction_eligible"],
        estimation_cfg.model.name,
        None,
    )

    data["estimation_holdout_year"] = holdout_year
    data["estimation_training_observations"] = len(clean)

    # A later correction/fallback stage can decide how to handle rows
    # without complete model predictors.
    data["factor_for_application"] = data[
        "estimated_cif_fob_factor"
    ]

    # ==================================================================
    # Write and audit output
    # ==================================================================

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        con.register(
            "estimated_rates_df",
            data,
        )

        output_sql = _sql_literal(output_path)

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM estimated_rates_df
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

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
            "Prediction-eligible rows are missing estimated factors: "
            f"{n_missing_prediction_when_eligible:,}."
        )

    if n_prediction_when_ineligible > 0:
        raise RuntimeError(
            "Prediction-ineligible rows unexpectedly received model "
            f"estimates: {n_prediction_when_ineligible:,}."
        )

    if n_prediction_bound_failures > 0:
        raise RuntimeError(
            "Estimated factors fall outside configured limits: "
            f"{n_prediction_bound_failures:,}."
        )

    audit_passed = True
    ready_for_trade_correction = (
        n_with_estimated_factor > 0
    )

    prediction_coverage_percent = (
        100.0 * n_with_estimated_factor / n_output_rows
        if n_output_rows
        else 0.0
    )

    logger.info("CIF/FOB rates estimated.")
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
        "  Rows with estimated factor: %s",
        n_with_estimated_factor,
    )
    logger.info(
        "  Estimated-factor coverage: %.4f%%",
        prediction_coverage_percent,
    )
    logger.info(
        "  Duplicate bilateral keys: %s",
        n_duplicate_keys,
    )
    logger.info(
        "  Median estimated factor: %s",
        median_estimated_factor,
    )
    logger.info(
        "  Estimated factor p05/p95: %s / %s",
        estimated_factor_p05,
        estimated_factor_p95,
    )
    logger.info("  Step 5 audit passed: %s", audit_passed)
    logger.info(
        "  Ready for trade correction: %s",
        ready_for_trade_correction,
    )
    logger.info("  Output: %s", output_path)

    return {
        "status": "completed",
        "audit_passed": audit_passed,
        "ready_for_trade_correction": ready_for_trade_correction,
        "output_path": output_path,
        "estimated_rates_path": output_path,
        "n_output_rows": n_output_rows,
        "n_clean_estimation_observations": (
            n_clean_estimation_observations
        ),
        "n_prediction_eligible": n_prediction_eligible,
        "n_with_estimated_factor": n_with_estimated_factor,
        "prediction_coverage_percent": prediction_coverage_percent,
        "n_duplicate_keys": n_duplicate_keys,
        "median_estimated_factor": median_estimated_factor,
        "estimated_factor_p05": estimated_factor_p05,
        "estimated_factor_p95": estimated_factor_p95,
        "holdout_year": holdout_year,
        "n_holdout_training_observations": n_train,
        "n_holdout_observations": n_holdout,
        **holdout_metrics,
    }
