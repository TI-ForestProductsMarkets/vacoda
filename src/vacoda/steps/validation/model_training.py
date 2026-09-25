"""
vacoda.steps.validation.model_training

Cross-validate candidate quantity models.

The step:

1. reads the Validation training-sample artifact;
2. uses only rows marked ``training_selected`` for model fitting;
3. performs leave-one-year-out validation;
4. fits every enabled candidate model on the remaining years;
5. predicts the held-out year;
6. writes row-level out-of-fold predictions for downstream Diagnostics.

This step does not select a production model.
Model comparison and methodological interpretation belong to Diagnostics
and Reporting.
"""

from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from time import perf_counter
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    TargetEncoder,
)

from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal


# ======================================================================
# Helpers
# ======================================================================


def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    required_columns: set[str],
) -> None:
    """Require all columns needed for model development."""

    input_sql = sql_literal(input_path)

    available = {
        str(row[0])
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet({input_sql})
            """
        ).fetchall()
    }

    missing = sorted(
        required_columns - available
    )

    if missing:
        raise RuntimeError(
            "Validation model-training input is missing required "
            "column(s): "
            + ", ".join(missing)
        )


def _ordered_unique(
    values: list[str],
) -> list[str]:
    """Return values without duplicates while preserving order."""

    return list(
        dict.fromkeys(values)
    )


def _build_feature_frame(
    *,
    frame: pd.DataFrame,
    cfg: ValidationConfig,
) -> pd.DataFrame:
    """
    Build configured model predictors.

    Numeric transformations are applied here. Categorical variables are
    retained as strings and encoded later using training-fold evidence only.
    """

    result = pd.DataFrame(
        index=frame.index
    )

    # ------------------------------------------------------------------
    # Numeric features
    # ------------------------------------------------------------------

    for (
        column,
        feature_cfg,
    ) in cfg.model.features.numeric.items():

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if values.isna().any():
            raise RuntimeError(
                f"Numeric model feature {column!r} contains "
                "missing or non-numeric values."
            )

        if feature_cfg.transform == "log":

            invalid = (
                (~np.isfinite(values))
                | (values <= 0)
            )

            if invalid.any():
                raise RuntimeError(
                    f"Numeric model feature {column!r} requires "
                    "strictly positive finite values for log transform."
                )

            result[column] = np.log(
                values.astype(float)
            )

        elif feature_cfg.transform == "none":

            if not np.isfinite(
                values.astype(float)
            ).all():
                raise RuntimeError(
                    f"Numeric model feature {column!r} contains "
                    "non-finite values."
                )

            result[column] = (
                values.astype(float)
            )

        else:
            raise RuntimeError(
                "Unsupported numeric feature transform: "
                f"{feature_cfg.transform!r}"
            )

    # ------------------------------------------------------------------
    # Categorical features
    # ------------------------------------------------------------------

    for column in cfg.model.features.categorical:

        result[column] = (
            frame[column]
            .astype("string")
            .fillna("__MISSING__")
        )

    return result


def _build_target(
    *,
    frame: pd.DataFrame,
    cfg: ValidationConfig,
) -> np.ndarray:
    """Build the configured prediction target."""

    target_col = cfg.model.target.column

    values = pd.to_numeric(
        frame[target_col],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Model target {target_col!r} contains non-finite values."
        )

    if cfg.model.target.transform == "log":

        if np.any(values <= 0):
            raise RuntimeError(
                f"Model target {target_col!r} must be strictly "
                "positive for log transform."
            )

        return np.log(
            values
        )

    if cfg.model.target.transform == "none":
        return values

    raise RuntimeError(
        "Unsupported model target transform: "
        f"{cfg.model.target.transform!r}"
    )


def _inverse_target(
    *,
    values: np.ndarray,
    cfg: ValidationConfig,
) -> np.ndarray:
    """Return predictions on the original quantity scale."""

    if cfg.model.target.transform == "log":
        return np.exp(
            values
        )

    return values


def _encode_features(
    *,
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    y_train: np.ndarray,
    cfg: ValidationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode categorical features using training-fold information only.

    TargetEncoder.fit_transform performs internal cross-fitting on the
    training fold. The held-out year is transformed only after the
    encoder has been fitted on the other years.
    """

    numeric_columns = list(
        cfg.model.features.numeric.keys()
    )

    categorical_columns = list(
        cfg.model.features.categorical
    )

    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Numeric predictors
    # ------------------------------------------------------------------

    if numeric_columns:

        train_parts.append(
            x_train[
                numeric_columns
            ].to_numpy(
                dtype=float
            )
        )

        validation_parts.append(
            x_validation[
                numeric_columns
            ].to_numpy(
                dtype=float
            )
        )

    # ------------------------------------------------------------------
    # Categorical predictors
    # ------------------------------------------------------------------

    if categorical_columns:

        encoder = TargetEncoder(
            target_type="continuous",
            smooth="auto",
            cv=5,
            random_state=cfg.model.random_seed,
        )

        encoded_train = encoder.fit_transform(
            x_train[
                categorical_columns
            ],
            y_train,
        )

        encoded_validation = encoder.transform(
            x_validation[
                categorical_columns
            ]
        )

        train_parts.append(
            np.asarray(
                encoded_train,
                dtype=float,
            )
        )

        validation_parts.append(
            np.asarray(
                encoded_validation,
                dtype=float,
            )
        )

    if not train_parts:
        raise RuntimeError(
            "No configured model features are available."
        )

    return (
        np.column_stack(
            train_parts
        ),
        np.column_stack(
            validation_parts
        ),
    )


def _build_estimator(
    *,
    model_type: str,
    parameters: dict[str, Any],
    cfg: ValidationConfig,
):
    """Construct one configured candidate estimator."""

    params = dict(
        parameters
    )

    if model_type == "ridge":

        estimator = Ridge(
            **params
        )

        # Ridge benefits from feature scaling.
        return make_pipeline(
            StandardScaler(),
            estimator,
        )

    if model_type == "hist_gradient_boosting":

        params.setdefault(
            "random_state",
            cfg.model.random_seed,
        )

        return HistGradientBoostingRegressor(
            **params
        )

    if model_type == "extra_trees":

        params.setdefault(
            "random_state",
            cfg.model.random_seed,
        )

        params.setdefault(
            "n_jobs",
            int(cfg.engine.threads),
        )

        return ExtraTreesRegressor(
            **params
        )

    if model_type == "random_forest":

        params.setdefault(
            "random_state",
            cfg.model.random_seed,
        )

        params.setdefault(
            "n_jobs",
            int(cfg.engine.threads),
        )

        return RandomForestRegressor(
            **params
        )

    raise RuntimeError(
        f"Unsupported Validation model type: {model_type!r}"
    )


# ======================================================================
# Public step
# ======================================================================


def train_validation_models(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    run_id: str,
    output_path: Path | None = None,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Cross-validate all enabled Validation quantity models.

    Only rows with ``training_selected = TRUE`` are used.

    Every configured year becomes one validation fold. The held-out
    year is never used for fitting either the feature encoder or the
    regression model.
    """

    start = perf_counter()

    input_path = Path(
        input_path
    )

    output_path = (
        Path(output_path)
        if output_path is not None
        else cfg.product_path(
            "model_predictions"
        )
    )

    # ------------------------------------------------------------------
    # Input / output
    # ------------------------------------------------------------------

    if not input_path.is_file():
        raise FileNotFoundError(
            "Validation model-training input not found: "
            f"{input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Reuse
    # ------------------------------------------------------------------

    if (
        not force
        and output_path.is_file()
        and output_path.stat().st_size > 0
    ):
        logger.info(
            "%s REUSE — Validation model predictions: %s",
            get_symbols("reuse"),
            output_path,
        )

        con = duckdb.connect()

        try:
            output_sql = sql_literal(
                output_path
            )

            summary = con.execute(
                f"""
                SELECT
                    COUNT(*) AS n_rows,
                    COUNT(DISTINCT model_name) AS n_models,
                    COUNT(DISTINCT held_out_year) AS n_years
                FROM read_parquet({output_sql})
                """
            ).fetchone()

        finally:
            con.close()

        return {
            "status": "completed",
            "run_id": run_id,
            "output_path": output_path,
            "model_predictions_path": output_path,
            "n_prediction_rows": int(summary[0]),
            "n_models": int(summary[1]),
            "n_validation_years": int(summary[2]),
            "reused": True,
        }

    logger.info(
        "%s REBUILD — Validation model cross-validation",
        get_symbols("rebuilt"),
    )

    logger.info(
        "  Input: %s",
        input_path,
    )

    logger.info(
        "  Output: %s",
        output_path,
    )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    target_col = (
        cfg.model.target.column
    )

    validation_year_col = (
        cfg.model.validation.year_column
    )

    numeric_columns = list(
        cfg.model.features.numeric.keys()
    )

    categorical_columns = list(
        cfg.model.features.categorical
    )

    feature_columns = _ordered_unique(
        numeric_columns
        + categorical_columns
    )

    key_columns = [
        validation_year_col,
        "exporter_iso",
        "importer_iso",
        "target_cmd",
    ]

    selected_columns = _ordered_unique(
        key_columns
        + feature_columns
        + [
            target_col,
            "training_selected",
        ]
    )

    enabled_models = (
        cfg.model.models.enabled()
    )

    if not enabled_models:
        raise RuntimeError(
            "No Validation candidate models are enabled."
        )

    # ------------------------------------------------------------------
    # Read selected training population
    # ------------------------------------------------------------------

    con = duckdb.connect()

    parts_dir = (
        output_path.parent
        / f".{output_path.stem}_parts"
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        con.execute(
            f"PRAGMA threads={int(cfg.engine.threads)}"
        )

        _require_columns(
            con=con,
            input_path=input_path,
            required_columns=set(
                selected_columns
            ),
        )

        input_sql = sql_literal(
            input_path
        )

        select_sql = ",\n".join(
            f'"{column}"'
            for column in selected_columns
        )

        frame = con.execute(
            f"""
            SELECT
                {select_sql}

            FROM read_parquet({input_sql})

            WHERE training_selected
            """
        ).df()

        if frame.empty:
            raise RuntimeError(
                "Validation model training contains no rows with "
                "training_selected = TRUE."
            )

        logger.info(
            "  Selected model-training rows: %s",
            f"{len(frame):,}",
        )

        # --------------------------------------------------------------
        # Years / validation folds
        # --------------------------------------------------------------

        years = sorted(
            int(value)
            for value in frame[
                validation_year_col
            ]
            .dropna()
            .unique()
        )

        if (
            len(years) - 1
            < cfg.model.validation.minimum_training_years
        ):
            raise RuntimeError(
                "Leave-one-year-out validation does not provide "
                "enough training years. "
                f"Available years: {len(years)}; "
                "minimum training years required per fold: "
                f"{cfg.model.validation.minimum_training_years}."
            )

        logger.info(
            "  Leave-one-year-out folds: %s",
            ", ".join(
                str(year)
                for year in years
            ),
        )

        # --------------------------------------------------------------
        # Prepare output parts directory
        # --------------------------------------------------------------

        if parts_dir.exists():
            rmtree(
                parts_dir
            )

        parts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if temporary_path.exists():
            temporary_path.unlink()

        # --------------------------------------------------------------
        # Build target once
        # --------------------------------------------------------------

        y_all = _build_target(
            frame=frame,
            cfg=cfg,
        )

        x_all = _build_feature_frame(
                frame=frame,
                cfg=cfg,
            )
        
        original_target_all = pd.to_numeric(
            frame[target_col],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        # --------------------------------------------------------------
        # Cross-validation
        # --------------------------------------------------------------

        n_written_rows = 0

        for held_out_year in years:

            fold_start = perf_counter()

            validation_mask = (
                frame[
                    validation_year_col
                ].astype(int)
                == held_out_year
            ).to_numpy()

            train_mask = (
                ~validation_mask
            )

            n_train = int(
                train_mask.sum()
            )

            n_validation = int(
                validation_mask.sum()
            )

            if n_train == 0 or n_validation == 0:
                raise RuntimeError(
                    "Invalid leave-one-year-out fold for "
                    f"{held_out_year}: "
                    f"train={n_train:,}, "
                    f"validation={n_validation:,}."
                )

            logger.info(
                "  Fold %s — train=%s, validation=%s",
                held_out_year,
                f"{n_train:,}",
                f"{n_validation:,}",
            )

            # ----------------------------------------------------------
            # Build fold predictors
            # ----------------------------------------------------------

            x_train = x_all.loc[
                train_mask
            ].reset_index(
                drop=True
            )

            x_validation = x_all.loc[
                validation_mask
            ].reset_index(
                drop=True
            )

            y_train = y_all[
                train_mask
            ]

            y_validation = y_all[
                validation_mask
            ]

            original_validation_target = (
                original_target_all[
                    validation_mask
                ]
            )

            # ----------------------------------------------------------
            # Supervised categorical encoding
            # ----------------------------------------------------------

            (
                x_train_encoded,
                x_validation_encoded,
            ) = _encode_features(
                x_train=x_train,
                x_validation=x_validation,
                y_train=y_train,
                cfg=cfg,
            )

            # ----------------------------------------------------------
            # Candidate models
            # ----------------------------------------------------------

            validation_keys = (
                frame.loc[
                    validation_mask,
                    key_columns,
                ]
                .reset_index(
                    drop=True
                )
                .copy()
            )

            for (
                model_name,
                model_cfg,
            ) in enabled_models.items():

                model_start = perf_counter()

                logger.info(
                    "    Training model: %s (%s)",
                    model_name,
                    model_cfg.type,
                )

                estimator = _build_estimator(
                    model_type=model_cfg.type,
                    parameters=model_cfg.parameters,
                    cfg=cfg,
                )

                estimator.fit(
                    x_train_encoded,
                    y_train,
                )

                prediction_transformed = np.asarray(
                    estimator.predict(
                        x_validation_encoded
                    ),
                    dtype=float,
                )

                prediction_quantity = _inverse_target(
                    values=prediction_transformed,
                    cfg=cfg,
                )

                # ------------------------------------------------------
                # Write fold/model predictions
                # ------------------------------------------------------

                prediction_frame = (
                    validation_keys.copy()
                )

                prediction_frame[
                    "run_id"
                ] = run_id

                prediction_frame[
                    "model_name"
                ] = model_name

                prediction_frame[
                    "model_type"
                ] = model_cfg.type

                prediction_frame[
                    "held_out_year"
                ] = int(
                    held_out_year
                )

                prediction_frame[
                    "n_training_rows"
                ] = n_train

                prediction_frame[
                    "n_validation_rows"
                ] = n_validation

                prediction_frame[
                    "target_transformed"
                ] = y_validation

                prediction_frame[
                    "prediction_transformed"
                ] = prediction_transformed

                prediction_frame[
                    "target_quantity_kg"
                ] = original_validation_target

                prediction_frame[
                    "prediction_quantity_kg"
                ] = prediction_quantity

                part_path = (
                    parts_dir
                    / (
                        f"{model_name}"
                        f"_holdout_{held_out_year}.parquet"
                    )
                )

                part_sql = sql_literal(
                    part_path
                )

                con.register(
                    "prediction_frame",
                    prediction_frame,
                )

                try:
                    con.execute(
                        f"""
                        COPY prediction_frame
                        TO {part_sql}
                        (
                            FORMAT PARQUET,
                            COMPRESSION ZSTD
                        )
                        """
                    )

                finally:
                    con.unregister(
                        "prediction_frame"
                    )

                n_written_rows += len(
                    prediction_frame
                )

                logger.info(
                    "    %s model complete in %.1f s",
                    model_name,
                    perf_counter() - model_start,
                )

                del estimator
                del prediction_frame

            logger.info(
                "  Fold %s complete in %.1f s",
                held_out_year,
                perf_counter() - fold_start,
            )

        # --------------------------------------------------------------
        # Combine parts
        # --------------------------------------------------------------

        parts_glob = sql_literal(
            parts_dir / "*.parquet"
        )

        temporary_sql = sql_literal(
            temporary_path
        )

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet(
                    {parts_glob},
                    union_by_name = TRUE
                )
            )
            TO {temporary_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        temporary_path.replace(
            output_path
        )

        output_sql = sql_literal(
            output_path
        )

        summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_prediction_rows,
                COUNT(DISTINCT model_name) AS n_models,
                COUNT(DISTINCT held_out_year) AS n_years
            FROM read_parquet({output_sql})
            """
        ).fetchone()

    finally:

        con.close()

        if parts_dir.exists():
            rmtree(
                parts_dir
            )

        if temporary_path.exists():
            temporary_path.unlink()

    elapsed = (
        perf_counter()
        - start
    )

    n_prediction_rows = int(
        summary[0]
    )

    n_models = int(
        summary[1]
    )

    n_validation_years = int(
        summary[2]
    )

    logger.info(
        "  Model prediction rows: %s",
        f"{n_prediction_rows:,}",
    )

    logger.info(
        "  Candidate models evaluated: %s",
        f"{n_models:,}",
    )

    logger.info(
        "  Validation years: %s",
        f"{n_validation_years:,}",
    )

    logger.info(
        "  Output: %s",
        output_path,
    )

    logger.info(
        "%s CHECK — Candidate-model cross-validation complete.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,

        "output_path": output_path,
        "model_predictions_path": output_path,

        "n_prediction_rows": n_prediction_rows,
        "n_models": n_models,
        "n_validation_years": n_validation_years,

        "duration_seconds": elapsed,
        "reused": False,
    }