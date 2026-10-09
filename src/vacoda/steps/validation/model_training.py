"""
vacoda.steps.validation.model_training

Cross-validate candidate quantity models.

The step:

1. reads the Validation training-sample artifact;
2. uses only rows marked ``training_selected`` for model fitting;
3. performs leave-one-year-out validation;
4. fits every enabled candidate model on the remaining years;
5. predicts the held-out year;
6. stores one reusable prediction checkpoint per candidate model;
7. combines all candidate-model out-of-fold predictions for Diagnostics.

This step does not select a production model.

Model comparison and methodological interpretation belong to Diagnostics
and Reporting.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)
from sklearn.neural_network import MLPRegressor
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
from vacoda.steps.validation.training_sample import training_definition_name
from vacoda.config.steps.validation import ValidationConfig

from vacoda.pipeline.artifact_manifest import (
    json_safe,
    manifest_path_for,
    validate_manifest,
    write_manifest,
)

from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal


MODEL_CACHE_VERSION = 2

MODEL_ROW_KEY = [
    "year",
    "exporter_iso",
    "importer_iso",
    "target_cmd",
]


# ======================================================================
# Generic helpers
# ======================================================================


def _ordered_unique(
    values: list[str],
) -> list[str]:
    """Return values without duplicates while preserving order."""

    return list(
        dict.fromkeys(values)
    )


def _format_checkpoint_value(
    value: float | int,
) -> str:
    """Return a filesystem-safe compact parameter value."""

    if isinstance(value, int):
        return str(value)

    return (
        f"{float(value):g}"
        .replace(".", "p")
        .replace("-", "m")
    )

def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    required_columns: set[str],
) -> None:
    """Require all columns needed for model development."""

    input_sql = sql_literal(
        input_path
    )

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


def _validate_model_row_keys(
    frame: pd.DataFrame,
) -> None:
    """Require a complete and unique canonical model-row identity."""

    missing_key_values = (
        frame[
            MODEL_ROW_KEY
        ]
        .isna()
        .any(axis=1)
    )

    if missing_key_values.any():

        examples = (
            frame.loc[
                missing_key_values,
                MODEL_ROW_KEY,
            ]
            .head(5)
            .to_dict("records")
        )

        raise RuntimeError(
            "Validation model-training data contains missing "
            "canonical row-key values. "
            f"Examples: {examples}"
        )

    duplicate_keys = frame.duplicated(
        MODEL_ROW_KEY,
        keep=False,
    )

    if duplicate_keys.any():

        examples = (
            frame.loc[
                duplicate_keys,
                MODEL_ROW_KEY,
            ]
            .head(5)
            .to_dict("records")
        )

        raise RuntimeError(
            "Validation model-training data is not unique at "
            f"canonical model-row grain {MODEL_ROW_KEY}. "
            f"Examples: {examples}"
        )


# ======================================================================
# Model-data fingerprint
# ======================================================================


def _model_data_fingerprint(
    *,
    frame: pd.DataFrame,
    cfg: ValidationConfig,
) -> str:
    """
    Fingerprint the logical training data relevant to model fitting.

    The fingerprint is independent of physical Parquet representation
    and input row order.
    """

    columns = list(
        dict.fromkeys(
            [
                *MODEL_ROW_KEY,
                cfg.model.target.column,
                *cfg.model.features.feature_columns,
            ]
        )
    )

    missing = sorted(
        set(columns)
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Cannot fingerprint Validation model data; "
            "missing column(s): "
            + ", ".join(missing)
        )

    canonical = (
        frame[
            columns
        ]
        .sort_values(
            MODEL_ROW_KEY,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    row_hashes = (
        pd.util.hash_pandas_object(
            canonical,
            index=False,
            categorize=True,
        )
    )

    digest = sha256()

    digest.update(
        json.dumps(
            columns,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )

    digest.update(
        row_hashes.to_numpy(
            dtype="uint64",
            copy=False,
        ).tobytes()
    )

    return digest.hexdigest()


# ======================================================================
# Model checkpoint helpers
# ======================================================================

def _model_checkpoint_dir(
    *,
    cfg: ValidationConfig,
    model_name: str,
) -> Path:

    checkpoint_group = (
        training_definition_name(
            cfg=cfg
        )
    )

    return (
        cfg.output_dir()
        / "model"
        / checkpoint_group
        / "checkpoints"
        / "models"
        / model_name
    )

def _model_fold_checkpoint_paths(
    *,
    cfg: ValidationConfig,
    model_name: str,
    held_out_year: int,
) -> dict[str, Path]:

    directory = (
        _model_checkpoint_dir(
            cfg=cfg,
            model_name=model_name,
        )
        / "folds"
        / str(int(held_out_year))
    )

    predictions_path = (
        directory
        / "predictions.parquet"
    )

    return {
        "predictions": predictions_path,
        "manifest": manifest_path_for(
            predictions_path
        ),
    }


def _model_fold_semantic_payload(
    *,
    cfg: ValidationConfig,
    model_name: str,
    model_cfg: Any,
    held_out_year: int,
    training_years: list[int],
    n_training_rows: int,
    n_validation_rows: int,
    fold_data_sha256: str,
) -> dict[str, Any]:

    return {
        "artifact": (
            "validation_model_cross_validation_fold"
        ),

        "cache_version": (
            MODEL_CACHE_VERSION
        ),

        "scope": {
            "target_hs": (
                cfg.scope.target_hs
            ),
        },

        "data": {
            "semantic_sha256": (
                fold_data_sha256
            ),
            "training_rows": int(
                n_training_rows
            ),
            "validation_rows": int(
                n_validation_rows
            ),
        },

        "target": {
            "column": (
                cfg.model.target.column
            ),
            "transform": (
                cfg.model.target.transform
            ),
        },

        "features": {
            "numeric": {
                name: asdict(feature_cfg)
                for name, feature_cfg
                in cfg.model.features.numeric.items()
            },
            "categorical": list(
                cfg.model.features.categorical
            ),
        },

        "validation": {
            "strategy": (
                cfg.model.validation.strategy
            ),
            "year_column": (
                cfg.model.validation.year_column
            ),
            "held_out_year": int(
                held_out_year
            ),
            "training_years": [
                int(year)
                for year in training_years
            ],
        },

        "model": {
            "name": (
                model_name
            ),
            "type": (
                model_cfg.type
            ),
            "parameters": json_safe(
                model_cfg.parameters
            ),
        },

        "random_seed": int(
            cfg.model.random_seed
        ),
    }

def _model_checkpoint_paths(
    *,
    cfg: ValidationConfig,
    model_name: str,
) -> dict[str, Path]:
    """Return checkpoint artifact paths for one candidate model."""

    directory = _model_checkpoint_dir(
        cfg=cfg,
        model_name=model_name,
    )

    predictions_path = (
        directory
        / "predictions.parquet"
    )

    return {
        "predictions": predictions_path,
        "manifest": manifest_path_for(
            predictions_path
        ),
    }


def _model_semantic_payload(
    *,
    cfg: ValidationConfig,
    model_name: str,
    model_cfg: Any,
    training_rows: int,
    training_data_sha256: str,
    validation_years: list[int],
) -> dict[str, Any]:
    """
    Return the semantic identity of one candidate-model CV run.

    Other enabled candidate models are deliberately excluded so that
    adding, removing, or changing another model does not invalidate an
    otherwise unchanged checkpoint.
    """

    return {
        "artifact": (
            "validation_model_cross_validation"
        ),

        "cache_version": (
            MODEL_CACHE_VERSION
        ),

        "scope": {
            "target_hs": (
                cfg.scope.target_hs
            ),
        },

        "data": {
            "rows": int(
                training_rows
            ),
            "semantic_sha256": (
                training_data_sha256
            ),
        },

        "target": {
            "column": (
                cfg.model.target.column
            ),
            "transform": (
                cfg.model.target.transform
            ),
        },

        "features": {
            "numeric": {
                name: asdict(
                    feature_cfg
                )
                for (
                    name,
                    feature_cfg,
                )
                in cfg.model.features.numeric.items()
            },

            "categorical": list(
                cfg.model.features.categorical
            ),
        },

        "validation": {
            "strategy": (
                cfg.model.validation.strategy
            ),
            "year_column": (
                cfg.model.validation.year_column
            ),
            "minimum_training_years": int(
                cfg.model.validation.minimum_training_years
            ),
            "validation_years": [
                int(year)
                for year in validation_years
            ],
        },

        "model": {
            "name": (
                model_name
            ),
            "type": (
                model_cfg.type
            ),
            "parameters": json_safe(
                model_cfg.parameters
            ),
        },

        "random_seed": int(
            cfg.model.random_seed
        ),
    }


def _checkpoint_prediction_columns_valid(
    *,
    prediction_path: Path,
) -> tuple[bool, str]:
    """Validate the row-level prediction checkpoint contract."""

    con = duckdb.connect()

    try:

        rows = con.execute(
            """
            DESCRIBE
            SELECT *
            FROM read_parquet(?)
            """,
            [
                str(
                    prediction_path
                )
            ],
        ).fetchall()

        columns = {
            str(row[0])
            for row in rows
        }

    except Exception as exc:
        return (
            False,
            "prediction checkpoint cannot be read: "
            f"{exc}",
        )

    finally:
        con.close()

    required = {
        *MODEL_ROW_KEY,

        "model_name",
        "model_type",
        "held_out_year",

        "n_training_rows",
        "n_validation_rows",

        "target_transformed",
        "prediction_transformed",

        "target_quantity_kg",
        "prediction_quantity_kg",
    }

    missing = sorted(
        required - columns
    )

    if missing:
        return (
            False,
            "prediction checkpoint format changed; "
            "missing column(s): "
            + ", ".join(missing),
        )

    return (
        True,
        "prediction checkpoint structure is compatible",
    )

def _model_fold_checkpoint_reusable(
    *,
    cfg: ValidationConfig,
    model_name: str,
    held_out_year: int,
    semantic_payload: dict[str, Any],
) -> tuple[bool, str]:

    paths = _model_fold_checkpoint_paths(
        cfg=cfg,
        model_name=model_name,
        held_out_year=held_out_year,
    )

    reusable, reason = validate_manifest(
        artifact_path=paths["predictions"],
        manifest_path=paths["manifest"],
        expected_payload=semantic_payload,
    )

    if not reusable:
        return False, reason

    (
        structure_valid,
        structure_reason,
    ) = _checkpoint_prediction_columns_valid(
        prediction_path=paths[
            "predictions"
        ],
    )

    if not structure_valid:
        return (
            False,
            structure_reason,
        )

    return (
        True,
        "manifest and prediction structure match",
    )

def _write_model_fold_checkpoint(
    *,
    cfg: ValidationConfig,
    model_name: str,
    held_out_year: int,
    predictions: pd.DataFrame,
    semantic_payload: dict[str, Any],
) -> Path:

    paths = _model_fold_checkpoint_paths(
        cfg=cfg,
        model_name=model_name,
        held_out_year=held_out_year,
    )

    predictions_path = paths[
        "predictions"
    ]

    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        predictions_path.with_suffix(
            predictions_path.suffix
            + ".tmp"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    con = duckdb.connect()

    try:

        con.register(
            "fold_predictions",
            predictions,
        )

        temporary_sql = sql_literal(
            temporary_path
        )

        con.execute(
            f"""
            COPY fold_predictions
            TO {temporary_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        con.unregister(
            "fold_predictions"
        )

    finally:
        con.close()

    temporary_path.replace(
        predictions_path
    )

    write_manifest(
        manifest_path=paths[
            "manifest"
        ],
        artifact_name=(
            f"validation_model_"
            f"{model_name}_"
            f"{held_out_year}_predictions"
        ),
        semantic_payload=(
            semantic_payload
        ),
        artifact_path=(
            predictions_path
        ),
        extra={
            "model_name": model_name,
            "held_out_year": int(
                held_out_year
            ),
            "prediction_rows": int(
                len(predictions)
            ),
        },
    )

    return predictions_path

def _model_checkpoint_reusable(
    *,
    cfg: ValidationConfig,
    model_name: str,
    semantic_payload: dict[str, Any],
) -> tuple[bool, str]:
    """
    Return whether one candidate-model checkpoint is reusable.

    Reuse requires:

    1. prediction artifact exists and is non-empty;
    2. manifest exists and matches current model semantics;
    3. artifact size/mtime matches its manifest;
    4. prediction schema remains compatible.
    """

    paths = _model_checkpoint_paths(
        cfg=cfg,
        model_name=model_name,
    )

    reusable, reason = validate_manifest(
        artifact_path=paths[
            "predictions"
        ],
        manifest_path=paths[
            "manifest"
        ],
        expected_payload=semantic_payload,
    )

    if not reusable:
        return (
            False,
            reason,
        )

    (
        structure_valid,
        structure_reason,
    ) = _checkpoint_prediction_columns_valid(
        prediction_path=paths[
            "predictions"
        ],
    )

    if not structure_valid:
        return (
            False,
            structure_reason,
        )

    return (
        True,
        "manifest and prediction structure match",
    )


def _write_model_checkpoint(
    *,
    cfg: ValidationConfig,
    model_name: str,
    predictions: pd.DataFrame,
    semantic_payload: dict[str, Any],
) -> Path:
    """Write one completed candidate-model checkpoint atomically."""

    paths = _model_checkpoint_paths(
        cfg=cfg,
        model_name=model_name,
    )

    predictions_path = (
        paths["predictions"]
    )

    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        predictions_path.with_suffix(
            predictions_path.suffix
            + ".tmp"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    con = duckdb.connect()

    try:

        con.register(
            "model_checkpoint_predictions",
            predictions,
        )

        temporary_sql = sql_literal(
            temporary_path
        )

        con.execute(
            f"""
            COPY model_checkpoint_predictions
            TO {temporary_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        con.unregister(
            "model_checkpoint_predictions"
        )

    finally:
        con.close()

    temporary_path.replace(
        predictions_path
    )

    write_manifest(
        manifest_path=paths[
            "manifest"
        ],
        artifact_name=(
            f"validation_model_"
            f"{model_name}_predictions"
        ),
        semantic_payload=semantic_payload,
        artifact_path=predictions_path,
        extra={
            "model_name": (
                model_name
            ),
            "prediction_rows": int(
                len(predictions)
            ),
        },
    )

    return predictions_path


# ======================================================================
# Feature preparation
# ======================================================================

def _fit_predict_embedding_mlp(
    *,
    x_train_numeric: np.ndarray,
    x_train_categorical: np.ndarray,
    y_train: np.ndarray,
    x_validation_numeric: np.ndarray,
    x_validation_categorical: np.ndarray,
    category_sizes: list[int],
    parameters: dict[str, Any],
    cfg: ValidationConfig,
    logger,
) -> np.ndarray:

    params = dict(parameters)

    batch_size = int(
        params["batch_size"]
    )

    learning_rate = float(
        params["learning_rate"]
    )

    max_epochs = int(
        params["max_epochs"]
    )

    patience = int(
        params["patience"]
    )

    early_stopping = bool(
        params["early_stopping"]
    )

    min_delta = float(
        params["min_delta"]
    )

    internal_validation_fraction = float(
        params["internal_validation_fraction"]
    )

    embedding_dims = [
        int(params["exporter_embedding_dim"]),
        int(params["importer_embedding_dim"]),
        int(params["product_embedding_dim"]),
    ]

    hidden_layer_1 = int(
        params["hidden_layer_1"]
    )

    hidden_layer_2 = int(
        params["hidden_layer_2"]
    )

    random_seed = int(
        cfg.model.random_seed
    )

    torch.manual_seed(random_seed)

    torch.set_num_threads(
        int(cfg.engine.threads)
    )

    logger.info(
        "  PyTorch CPU threads — intraop=%d, interop=%d",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        "  Embedding MLP — device=%s, "
        "batch_size=%d, learning_rate=%g, "
        "max_epochs=%d, early_stopping=%s, "
        "patience=%d, min_delta=%g, "
        "internal_validation_fraction=%.2f",
        device,
        batch_size,
        learning_rate,
        max_epochs,
        early_stopping,
        patience,
        min_delta,
        internal_validation_fraction,
    )

    logger.info(
        "  Embedding MLP architecture — "
        "embedding_dims=%s, hidden_layers=(%d, %d)",
        embedding_dims,
        hidden_layer_1,
        hidden_layer_2,
    )


    if len(category_sizes) != len(
        embedding_dims
    ):
        raise RuntimeError(
            "Embedding dimensions do not match "
            "configured categorical features."
        )

    model = EmbeddingMLP(
        category_sizes=category_sizes,
        embedding_dims=embedding_dims,
        numeric_dim=(
            x_train_numeric.shape[1]
        ),
        hidden_layer_1=hidden_layer_1,
        hidden_layer_2=hidden_layer_2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_function = nn.MSELoss()

    # --------------------------------------------------
    # Internal training/early-stopping split
    # --------------------------------------------------

    rng = np.random.default_rng(
        random_seed
    )

    n_rows = len(y_train)

    permutation = rng.permutation(
        n_rows
    )

    n_internal_validation = max(
        1,
        int(
            internal_validation_fraction
            * n_rows
        ),
    )

    validation_indices = (
        permutation[
            :n_internal_validation
        ]
    )

    fitting_indices = (
        permutation[
            n_internal_validation:
        ]
    )

    train_dataset = TensorDataset(
        torch.from_numpy(
            x_train_numeric[
                fitting_indices
            ]
        ).float(),
        torch.from_numpy(
            x_train_categorical[
                fitting_indices
            ]
        ).long(),
        torch.from_numpy(
            y_train[
                fitting_indices
            ].astype(np.float32)
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    internal_numeric = torch.from_numpy(
        x_train_numeric[
            validation_indices
        ]
    ).float().to(device)

    internal_categorical = torch.from_numpy(
        x_train_categorical[
            validation_indices
        ]
    ).long().to(device)

    internal_target = torch.from_numpy(
        y_train[
            validation_indices
        ].astype(np.float32)
    ).to(device)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    best_validation_loss = np.inf
    best_state = None
    best_epoch = 0

    early_stopping_reference_loss = np.inf
    epochs_without_meaningful_improvement = 0

    for epoch in range(1,
        max_epochs + 1,
    ):

        model.train()

        training_loss_sum = 0.0
        training_rows = 0

        for (
            batch_numeric,
            batch_categorical,
            batch_target,
        ) in train_loader:

            batch_numeric = (
                batch_numeric.to(device)
            )

            batch_categorical = (
                batch_categorical.to(
                    device
                )
            )

            batch_target = (
                batch_target.to(device)
            )

            optimizer.zero_grad()

            prediction = model(
                batch_numeric,
                batch_categorical,
            )

            loss = loss_function(
                prediction,
                batch_target,
            )

            loss.backward()

            optimizer.step()

            batch_n = (
                batch_target.shape[0]
            )

            training_loss_sum += (
                float(loss.item())
                * batch_n
            )

            training_rows += batch_n

        training_mse = (
            training_loss_sum
            / training_rows
        )

        # ----------------------------------------------
        # Internal validation
        # ----------------------------------------------

        model.eval()

        with torch.no_grad():

            internal_prediction = model(
                internal_numeric,
                internal_categorical,
            )

            internal_loss = float(
                loss_function(
                    internal_prediction,
                    internal_target,
                ).item()
            )

        logger.info(
            "  Embedding MLP epoch %d — "
            "train_MSE_log=%.6f, "
            "internal_validation_MSE_log=%.6f",
            epoch,
            training_mse,
            internal_loss,
        )

        # --------------------------------------------------
        # Preserve actual best model state
        # --------------------------------------------------

        if internal_loss < best_validation_loss:

            best_validation_loss = (
                internal_loss
            )

            best_epoch = epoch

            best_state = {
                name: tensor.detach()
                .cpu()
                .clone()
                for name, tensor
                in model.state_dict().items()
            }


        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if early_stopping:

            if (
                internal_loss
                < early_stopping_reference_loss
                - min_delta
            ):

                early_stopping_reference_loss = (
                    internal_loss
                )

                epochs_without_meaningful_improvement = 0

            else:

                epochs_without_meaningful_improvement += 1

                if (
                    epochs_without_meaningful_improvement
                    >= patience
                ):

                    logger.info(
                        "  Embedding MLP early stopping "
                        "at epoch %d; best epoch=%d, "
                        "best internal MSE_log=%.6f",
                        epoch,
                        best_epoch,
                        best_validation_loss,
                    )

                    break

    if best_state is None:
        raise RuntimeError(
            "Embedding MLP produced no valid "
            "training state."
        )

    logger.info(
        "  Embedding MLP fit complete — "
        "best_epoch=%d, best_internal_MSE_log=%.6f",
        best_epoch,
        best_validation_loss,
    )

    model.load_state_dict(best_state)
    model.to(device)

    # --------------------------------------------------
    # Held-out prediction
    # --------------------------------------------------

    model.eval()

    validation_dataset = TensorDataset(
        torch.from_numpy(
            x_validation_numeric
        ).float(),
        torch.from_numpy(
            x_validation_categorical
        ).long(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    predictions = []

    with torch.no_grad():

        for (
            batch_numeric,
            batch_categorical,
        ) in validation_loader:

            batch_prediction = model(
                batch_numeric.to(device),
                batch_categorical.to(
                    device
                ),
            )

            predictions.append(
                batch_prediction
                .cpu()
                .numpy()
            )

    return np.concatenate(
        predictions
    ).astype(float)

def _build_feature_frame(
    *,
    frame: pd.DataFrame,
    cfg: ValidationConfig,
) -> pd.DataFrame:
    """
    Build configured model predictors.

    Numeric transformations are applied here.

    Categorical variables remain strings and are encoded independently
    inside every leave-one-year-out training fold.
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

        numeric_values = values.astype(
            float
        )

        if not np.isfinite(
            numeric_values
        ).all():
            raise RuntimeError(
                f"Numeric model feature {column!r} contains "
                "non-finite values."
            )

        if feature_cfg.transform == "log":

            if (
                numeric_values <= 0
            ).any():
                raise RuntimeError(
                    f"Numeric model feature {column!r} requires "
                    "strictly positive values for log transform."
                )

            result[column] = np.log(
                numeric_values
            )

        elif feature_cfg.transform == "none":

            result[column] = (
                numeric_values
            )

        else:
            raise RuntimeError(
                "Unsupported numeric feature transform: "
                f"{feature_cfg.transform!r}"
            )

    # ------------------------------------------------------------------
    # Categorical features
    # ------------------------------------------------------------------

    for column in (
        cfg.model.features.categorical
    ):

        result[column] = (
            frame[column]
            .astype("string")
            .fillna(
                "__MISSING__"
            )
        )

    return result


def _build_target(
    *,
    frame: pd.DataFrame,
    cfg: ValidationConfig,
) -> np.ndarray:
    """Build the configured prediction target."""

    target_col = (
        cfg.model.target.column
    )

    values = pd.to_numeric(
        frame[target_col],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            f"Model target {target_col!r} contains "
            "non-finite values."
        )

    if (
        cfg.model.target.transform
        == "log"
    ):

        if np.any(
            values <= 0
        ):
            raise RuntimeError(
                f"Model target {target_col!r} must be "
                "strictly positive for log transform."
            )

        return np.log(
            values
        )

    if (
        cfg.model.target.transform
        == "none"
    ):
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

    if (
        cfg.model.target.transform
        == "log"
    ):
        return np.exp(
            values
        )

    return values


# ======================================================================
# Fold-local feature encoding
# ======================================================================

def _encode_embedding_features(
    *,
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    cfg: ValidationConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
]:
    numeric_columns = list(
        cfg.model.features.numeric.keys()
    )

    categorical_columns = list(
        cfg.model.features.categorical
    )

    # Numeric features
    x_train_numeric = (
        x_train[numeric_columns]
        .to_numpy(dtype=np.float32)
    )

    x_validation_numeric = (
        x_validation[numeric_columns]
        .to_numpy(dtype=np.float32)
    )

    # Fold-local standardization
    mean = x_train_numeric.mean(
        axis=0,
        keepdims=True,
    )

    std = x_train_numeric.std(
        axis=0,
        keepdims=True,
    )

    std[std == 0] = 1.0

    x_train_numeric = (
        x_train_numeric - mean
    ) / std

    x_validation_numeric = (
        x_validation_numeric - mean
    ) / std

    train_categorical_parts = []
    validation_categorical_parts = []
    category_sizes = []

    for column in categorical_columns:

        categories = (
            x_train[column]
            .astype("string")
            .fillna("__MISSING__")
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        mapping = {
            value: index + 1
            for index, value
            in enumerate(categories)
        }

        train_ids = (
            x_train[column]
            .astype("string")
            .fillna("__MISSING__")
            .map(mapping)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

        validation_ids = (
            x_validation[column]
            .astype("string")
            .fillna("__MISSING__")
            .map(mapping)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

        train_categorical_parts.append(
            train_ids
        )

        validation_categorical_parts.append(
            validation_ids
        )

        category_sizes.append(
            len(mapping) + 1
        )

    x_train_categorical = np.column_stack(
        train_categorical_parts
    )

    x_validation_categorical = np.column_stack(
        validation_categorical_parts
    )

    return (
        x_train_numeric,
        x_validation_numeric,
        x_train_categorical,
        x_validation_categorical,
        category_sizes,
    )


def _encode_features(
    *,
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    y_train: np.ndarray,
    cfg: ValidationConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Encode categorical features using training-fold evidence only.

    TargetEncoder.fit_transform internally cross-fits the training
    observations. The held-out year is transformed only after the
    encoder has been fitted on the remaining years.
    """

    numeric_columns = list(
        cfg.model.features.numeric.keys()
    )

    categorical_columns = list(
        cfg.model.features.categorical
    )

    train_parts: list[
        np.ndarray
    ] = []

    validation_parts: list[
        np.ndarray
    ] = []

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
            random_state=(
                cfg.model.random_seed
            ),
        )

        encoded_train = (
            encoder.fit_transform(
                x_train[
                    categorical_columns
                ],
                y_train,
            )
        )

        encoded_validation = (
            encoder.transform(
                x_validation[
                    categorical_columns
                ]
            )
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


# ======================================================================
# Estimator factory
# ======================================================================


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

    # ------------------------------------------------------------------
    # Ridge
    # ------------------------------------------------------------------

    if model_type == "ridge":

        estimator = Ridge(
            **params
        )

        return make_pipeline(
            StandardScaler(),
            estimator,
        )

    # ------------------------------------------------------------------
    # Histogram gradient boosting
    # ------------------------------------------------------------------

    if (
        model_type
        == "hist_gradient_boosting"
    ):

        params.setdefault(
            "random_state",
            cfg.model.random_seed,
        )

        return (
            HistGradientBoostingRegressor(
                **params
            )
        )

    # ------------------------------------------------------------------
    # Extra Trees
    # ------------------------------------------------------------------

    if model_type == "extra_trees":

        params.setdefault(
            "random_state",
            cfg.model.random_seed,
        )

        params.setdefault(
            "n_jobs",
            int(
                cfg.engine.threads
            ),
        )

        return ExtraTreesRegressor(
            **params
        )

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------

    if model_type == "random_forest":

        params.setdefault(
            "random_state",
            cfg.model.random_seed,
        )

        params.setdefault(
            "n_jobs",
            int(
                cfg.engine.threads
            ),
        )

        return RandomForestRegressor(
            **params
        )

    if model_type == "mlp":

        params.setdefault(
            "random_state",
            cfg.model.random_seed,
        )

        params.setdefault(
            "hidden_layer_sizes",
            (128, 64),
        )

        estimator = MLPRegressor(
            **params
        )

        return make_pipeline(
            StandardScaler(),
            estimator,
        )
    
    raise RuntimeError(
        "Unsupported Validation model type: "
        f"{model_type!r}"
    )


# ======================================================================
# MLP Embedding
# =====================================================================
class EmbeddingMLP(nn.Module):

    def __init__(
        self,
        *,
        category_sizes: list[int],
        embedding_dims: list[int],
        numeric_dim: int,
        hidden_layer_1: int,
        hidden_layer_2: int,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=size,
                    embedding_dim=dim,
                    padding_idx=0,
                )
                for size, dim
                in zip(
                    category_sizes,
                    embedding_dims,
                )
            ]
        )

        total_embedding_dim = sum(
            embedding_dims
        )

        self.network = nn.Sequential(
            nn.Linear(
                numeric_dim
                + total_embedding_dim,
                hidden_layer_1,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_layer_1,
                hidden_layer_2,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_layer_2,
                1,
            ),
        )

    def forward(
        self,
        numeric,
        categorical,
    ):
        embedded = [
            embedding(
                categorical[:, index]
            )
            for index, embedding
            in enumerate(
                self.embeddings
            )
        ]

        x = torch.cat(
            [
                numeric,
                *embedded,
            ],
            dim=1,
        )

        return (
            self.network(x)
            .squeeze(1)
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

    Only rows with ``training_selected = TRUE`` are used for fitting
    and validation.

    Under leave-one-year-out validation, every available year becomes
    one held-out fold. The held-out year is excluded from both feature
    encoding and estimator fitting.

    One reusable checkpoint is written per completed candidate model.
    """

    start = perf_counter()

    input_path = Path(
        input_path
    )

    base_output_path = cfg.product_path(
        "model_predictions"
    )

    checkpoint_group = (
        training_definition_name(
            cfg=cfg
        )
    )

    output_path = (
        base_output_path.parent
        / checkpoint_group
        / base_output_path.name
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

    logger.info(
        "%s Validation candidate-model cross-validation",
        get_symbols("build"),
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

    feature_columns = (
        _ordered_unique(
            numeric_columns
            + categorical_columns
        )
    )

    selected_columns = (
        _ordered_unique(
            [
                *MODEL_ROW_KEY,
                validation_year_col,
                *feature_columns,
                target_col,
                "training_selected",
            ]
        )
    )

    enabled_models = (
        cfg.model.models.enabled()
    )

    if not enabled_models:
        raise RuntimeError(
            "No Validation candidate models are enabled."
        )

    logger.info(
        "  Enabled candidate models: %s",
        ", ".join(
            enabled_models.keys()
        ),
    )

    # ------------------------------------------------------------------
    # Read selected training population
    # ------------------------------------------------------------------

    con = duckdb.connect()

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    try:

        con.execute(
            f"PRAGMA threads="
            f"{int(cfg.engine.threads)}"
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

            FROM read_parquet(
                {input_sql}
            )

            WHERE training_selected
            """
        ).df()

        if frame.empty:
            raise RuntimeError(
                "Validation model training contains no rows "
                "with training_selected = TRUE."
            )

        _validate_model_row_keys(
            frame
        )

        logger.info(
            "  Selected model-training rows: %s",
            f"{len(frame):,}",
        )

        # --------------------------------------------------------------
        # Resolve expanding-window validation folds
        # --------------------------------------------------------------

        years = sorted(
            int(value)
            for value
            in frame[
                validation_year_col
            ]
            .dropna()
            .unique()
        )

        if not years:
            raise RuntimeError(
                "No validation years are available in the "
                "selected model-training population."
            )

        minimum_training_years = int(
            cfg.model.validation.minimum_training_years
        )

        if len(years) <= minimum_training_years:
            raise RuntimeError(
                "Expanding-window validation does not provide "
                "enough years. "
                f"Available years: {len(years)}; "
                "minimum training years required before the "
                "first validation year: "
                f"{minimum_training_years}."
            )

        validation_years = years[
            minimum_training_years:
        ]

        logger.info(
            "  Expanding-window validation folds: %s",
            ", ".join(
                str(year)
                for year in validation_years
            ),
        )

        # --------------------------------------------------------------
        # Build target / predictors once
        # --------------------------------------------------------------

        y_all = _build_target(
            frame=frame,
            cfg=cfg,
        )

        x_all = _build_feature_frame(
            frame=frame,
            cfg=cfg,
        )

        original_target_all = (
            pd.to_numeric(
                frame[
                    target_col
                ],
                errors="raise",
            )
            .to_numpy(
                dtype=float
            )
        )

        # --------------------------------------------------------------
        # Fingerprint shared model evidence once
        # --------------------------------------------------------------

        logger.info(
            "  Building semantic fingerprint of selected "
            "model-training evidence..."
        )

        training_data_sha256 = (
            _model_data_fingerprint(
                frame=frame,
                cfg=cfg,
            )
        )

        logger.info(
            "  Training-evidence fingerprint: %s",
            training_data_sha256,
        )

        # --------------------------------------------------------------
        # Run or reuse one complete checkpoint per candidate model
        # --------------------------------------------------------------

        checkpoint_prediction_paths: list[
            Path
        ] = []

        n_reused_models = 0
        n_rebuilt_models = 0

        for (
            model_name,
            model_cfg,
        ) in enabled_models.items():

            model_start = perf_counter()

            semantic_payload = (
                _model_semantic_payload(
                    cfg=cfg,
                    model_name=model_name,
                    model_cfg=model_cfg,
                    training_rows=len(
                        frame
                    ),
                    training_data_sha256=(
                        training_data_sha256
                    ),
                    validation_years=validation_years,
                )
            )

            reusable = False
            reuse_reason = (
                "force=True"
            )

            if not force:
                (
                    reusable,
                    reuse_reason,
                ) = (
                    _model_checkpoint_reusable(
                        cfg=cfg,
                        model_name=model_name,
                        semantic_payload=(
                            semantic_payload
                        ),
                    )
                )

            checkpoint_paths = (
                _model_checkpoint_paths(
                    cfg=cfg,
                    model_name=model_name,
                )
            )

            # ==========================================================
            # Reuse completed model checkpoint
            # ==========================================================

            if reusable:

                logger.info(
                    "%s REUSE — Model checkpoint: %s (%s)",
                    get_symbols("reuse"),
                    model_name,
                    reuse_reason,
                )

                checkpoint_prediction_paths.append(
                    checkpoint_paths[
                        "predictions"
                    ]
                )

                n_reused_models += 1

                continue

            # ==========================================================
            # Build candidate model
            # ==========================================================

            logger.info(
                "%s REBUILD — Model checkpoint: %s (%s)",
                get_symbols("rebuilt"),
                model_name,
                reuse_reason,
            )

            model_fold_prediction_paths: list[Path] = []

            for held_out_year in validation_years:

                fold_start = perf_counter()

                fold_years = (
                    frame[
                        validation_year_col
                    ]
                    .astype(int)
                    .to_numpy()
                )

                validation_mask = (
                    fold_years == held_out_year
                )

                train_mask = (
                    fold_years < held_out_year
                )

                n_train = int(
                    train_mask.sum()
                )

                n_validation = int(
                    validation_mask.sum()
                )

                training_years = sorted(
                    set(
                        fold_years[
                            train_mask
                        ].tolist()
                    )
                )

                if (
                    len(training_years)
                    < minimum_training_years
                    or n_train == 0
                    or n_validation == 0
                ):
                    raise RuntimeError(
                        "Invalid expanding-window fold for "
                        f"{held_out_year}: "
                        f"training_years={training_years}, "
                        f"train={n_train:,}, "
                        f"validation={n_validation:,}."
                    )

                # ------------------------------------------------------
                # Fold checkpoint identity
                # ------------------------------------------------------

                fold_data_mask = (
                    train_mask
                    | validation_mask
                )

                fold_data_sha256 = (
                    _model_data_fingerprint(
                        frame=(
                            frame.loc[
                                fold_data_mask
                            ]
                            .reset_index(
                                drop=True
                            )
                        ),
                        cfg=cfg,
                    )
                )

                fold_semantic_payload = (
                    _model_fold_semantic_payload(
                        cfg=cfg,
                        model_name=model_name,
                        model_cfg=model_cfg,
                        held_out_year=held_out_year,
                        training_years=training_years,
                        n_training_rows=n_train,
                        n_validation_rows=n_validation,
                        fold_data_sha256=(
                            fold_data_sha256
                        ),
                    )
                )

                fold_reusable = False
                fold_reuse_reason = (
                    "force=True"
                )

                if not force:
                    (
                        fold_reusable,
                        fold_reuse_reason,
                    ) = (
                        _model_fold_checkpoint_reusable(
                            cfg=cfg,
                            model_name=model_name,
                            held_out_year=held_out_year,
                            semantic_payload=(
                                fold_semantic_payload
                            ),
                        )
                    )

                fold_checkpoint_paths = (
                    _model_fold_checkpoint_paths(
                        cfg=cfg,
                        model_name=model_name,
                        held_out_year=held_out_year,
                    )
                )

                # ------------------------------------------------------
                # Reuse completed fold checkpoint
                # ------------------------------------------------------

                if fold_reusable:

                    logger.info(
                        "%s REUSE — %s fold %s (%s)",
                        get_symbols("reuse"),
                        model_name,
                        held_out_year,
                        fold_reuse_reason,
                    )

                    model_fold_prediction_paths.append(
                        fold_checkpoint_paths[
                            "predictions"
                        ]
                    )

                    continue

                # ------------------------------------------------------
                # Rebuild fold
                # ------------------------------------------------------

                logger.info(
                    "  %s — validate=%s, "
                    "train_years=%s–%s, "
                    "train=%s, validation=%s",
                    model_name,
                    held_out_year,
                    training_years[0],
                    training_years[-1],
                    f"{n_train:,}",
                    f"{n_validation:,}",
                )

                # ------------------------------------------------------
                # Fold predictors / target
                # ------------------------------------------------------

                x_train = (
                    x_all.loc[
                        train_mask
                    ]
                    .reset_index(
                        drop=True
                    )
                )

                x_validation = (
                    x_all.loc[
                        validation_mask
                    ]
                    .reset_index(
                        drop=True
                    )
                )

                y_train = (
                    y_all[
                        train_mask
                    ]
                )

                y_validation = (
                    y_all[
                        validation_mask
                    ]
                )

                original_validation_target = (
                    original_target_all[
                        validation_mask
                    ]
                )

                # ------------------------------------------------------
                # Fold-local feature encoding
                # ------------------------------------------------------

                if model_cfg.type == "embedding_mlp":

                    (
                        x_train_numeric,
                        x_validation_numeric,
                        x_train_categorical,
                        x_validation_categorical,
                        category_sizes,
                    ) = _encode_embedding_features(
                        x_train=x_train,
                        x_validation=x_validation,
                        cfg=cfg,
                    )

                else:

                    (
                        x_train_encoded,
                        x_validation_encoded,
                    ) = _encode_features(
                        x_train=x_train,
                        x_validation=x_validation,
                        y_train=y_train,
                        cfg=cfg,
                    )

                # ------------------------------------------------------
                # Fit candidate and predict held-out year
                # ------------------------------------------------------

                if model_cfg.type == "embedding_mlp":

                    prediction_transformed = (
                        _fit_predict_embedding_mlp(
                            x_train_numeric=(
                                x_train_numeric
                            ),
                            x_train_categorical=(
                                x_train_categorical
                            ),
                            y_train=y_train,
                            x_validation_numeric=(
                                x_validation_numeric
                            ),
                            x_validation_categorical=(
                                x_validation_categorical
                            ),
                            category_sizes=(
                                category_sizes
                            ),
                            parameters=(
                                model_cfg.parameters
                            ),
                            cfg=cfg,
                            logger=logger,
                        )
                    )

                else:

                    estimator = (
                        _build_estimator(
                            model_type=(
                                model_cfg.type
                            ),
                            parameters=(
                                model_cfg.parameters
                            ),
                            cfg=cfg,
                        )
                    )

                    estimator.fit(
                        x_train_encoded,
                        y_train,
                    )

                    prediction_transformed = (
                        np.asarray(
                            estimator.predict(
                                x_validation_encoded
                            ),
                            dtype=float,
                        )
                    )

                # ------------------------------------------------------
                # Validate transformed prediction
                # ------------------------------------------------------

                if not np.isfinite(
                    prediction_transformed
                ).all():
                    raise RuntimeError(
                        f"Model {model_name!r} produced "
                        "non-finite transformed predictions "
                        f"for held-out year {held_out_year}."
                    )

                heldout_mae_log = float(
                    np.mean(
                        np.abs(
                            y_validation
                            - prediction_transformed
                        )
                    )
                )

                logger.info(
                    "  Held-out %s — MAE_log=%.6f",
                    held_out_year,
                    heldout_mae_log,
                )

                prediction_quantity = (
                    _inverse_target(
                        values=(
                            prediction_transformed
                        ),
                        cfg=cfg,
                    )
                )

                if not np.isfinite(
                    prediction_quantity
                ).all():
                    raise RuntimeError(
                        f"Model {model_name!r} produced "
                        "non-finite quantity predictions "
                        f"for held-out year {held_out_year}."
                    )

                # ------------------------------------------------------
                # Row-level out-of-fold evidence
                # ------------------------------------------------------

                fold_predictions = (
                    frame.loc[
                        validation_mask,
                        MODEL_ROW_KEY,
                    ]
                    .reset_index(
                        drop=True
                    )
                    .copy()
                )

                fold_predictions[
                    "model_name"
                ] = model_name

                fold_predictions[
                    "model_type"
                ] = model_cfg.type

                fold_predictions[
                    "held_out_year"
                ] = int(
                    held_out_year
                )

                fold_predictions[
                    "n_training_rows"
                ] = int(
                    n_train
                )

                fold_predictions[
                    "n_validation_rows"
                ] = int(
                    n_validation
                )

                fold_predictions[
                    "target_transformed"
                ] = y_validation

                fold_predictions[
                    "prediction_transformed"
                ] = (
                    prediction_transformed
                )

                fold_predictions[
                    "target_quantity_kg"
                ] = (
                    original_validation_target
                )

                fold_predictions[
                    "prediction_quantity_kg"
                ] = (
                    prediction_quantity
                )

                # ------------------------------------------------------
                # Persist fold immediately
                # ------------------------------------------------------

                fold_checkpoint_path = (
                    _write_model_fold_checkpoint(
                        cfg=cfg,
                        model_name=model_name,
                        held_out_year=held_out_year,
                        predictions=(
                            fold_predictions
                        ),
                        semantic_payload=(
                            fold_semantic_payload
                        ),
                    )
                )

                model_fold_prediction_paths.append(
                    fold_checkpoint_path
                )

                logger.info(
                    "  Fold %s checkpoint saved: %s",
                    held_out_year,
                    fold_checkpoint_path,
                )

                logger.info(
                    "  Fold %s complete in %.1f s",
                    held_out_year,
                    (
                        perf_counter()
                        - fold_start
                    ),
                )

                # ------------------------------------------------------
                # Fold cleanup
                # ------------------------------------------------------

                del fold_predictions
                del x_train
                del x_validation
                del y_train
                del y_validation
                del prediction_transformed
                del prediction_quantity

                if model_cfg.type == "embedding_mlp":

                    del x_train_numeric
                    del x_validation_numeric
                    del x_train_categorical
                    del x_validation_categorical
                    del category_sizes

                else:

                    del estimator
                    del x_train_encoded
                    del x_validation_encoded


            # ----------------------------------------------------------
            # Completed model checkpoint
            # ----------------------------------------------------------

            if not model_fold_prediction_paths:
                raise RuntimeError(
                    f"Model {model_name!r} produced no "
                    "cross-validation fold checkpoints."
                )

            fold_paths_sql = (
                ", ".join(
                    sql_literal(
                        path
                    )
                    for path
                    in model_fold_prediction_paths
                )
            )

            model_predictions = (
                con.execute(
                    f"""
                    SELECT *
                    FROM read_parquet(
                        [
                            {fold_paths_sql}
                        ],
                        union_by_name = TRUE
                    )
                    """
                )
                .df()
            )

            expected_rows = int(
                np.isin(
                    fold_years,
                    validation_years,
                ).sum()
            )

            if (
                len(model_predictions)
                != expected_rows
            ):
                raise RuntimeError(
                    f"Model {model_name!r} produced "
                    f"{len(model_predictions):,} "
                    "out-of-fold predictions, but "
                    f"{expected_rows:,} were expected."
                )

            checkpoint_path = (
                _write_model_checkpoint(
                    cfg=cfg,
                    model_name=model_name,
                    predictions=(
                        model_predictions
                    ),
                    semantic_payload=(
                        semantic_payload
                    ),
                )
            )

            checkpoint_prediction_paths.append(
                checkpoint_path
            )

            n_rebuilt_models += 1

            logger.info(
                "  Model %s complete in %.1f s",
                model_name,
                (
                    perf_counter()
                    - model_start
                ),
            )

            del model_predictions
            del model_fold_prediction_paths

        # --------------------------------------------------------------
        # Combine reusable + newly built checkpoints
        # --------------------------------------------------------------

        if not checkpoint_prediction_paths:
            raise RuntimeError(
                "No Validation model prediction checkpoints "
                "are available."
            )

        if temporary_path.exists():
            temporary_path.unlink()

        checkpoint_paths_sql = (
            ", ".join(
                sql_literal(
                    path
                )
                for path
                in checkpoint_prediction_paths
            )
        )

        temporary_sql = sql_literal(
            temporary_path
        )

        run_id_sql = sql_literal(
            run_id
        )

        logger.info(
            "  Combining %s model prediction checkpoint(s)...",
            len(
                checkpoint_prediction_paths
            ),
        )

        con.execute(
            f"""
            COPY (
                SELECT
                    {run_id_sql}
                        AS run_id,
                    *

                FROM read_parquet(
                    [
                        {checkpoint_paths_sql}
                    ],
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

        # --------------------------------------------------------------
        # Final output accounting
        # --------------------------------------------------------------

        output_sql = sql_literal(
            output_path
        )

        summary = con.execute(
            f"""
            SELECT
                COUNT(*)
                    AS n_prediction_rows,

                COUNT(
                    DISTINCT model_name
                )
                    AS n_models,

                COUNT(
                    DISTINCT held_out_year
                )
                    AS n_validation_years

            FROM read_parquet(
                {output_sql}
            )
            """
        ).fetchone()

        # --------------------------------------------------------------
        # Sanity check:
        # one OOF prediction per selected row per enabled model
        # --------------------------------------------------------------

        n_validation_rows_total = int(
            frame[
                validation_year_col
            ]
            .astype(int)
            .isin(
                validation_years
            )
            .sum()
        )

        expected_prediction_rows = (
            n_validation_rows_total
            * len(enabled_models)
        )

        actual_prediction_rows = int(
            summary[0]
        )

        if (
            actual_prediction_rows
            != expected_prediction_rows
        ):
            raise RuntimeError(
                "Combined Validation model-prediction row count "
                "does not match the expected expanding-window validation coverage."
                f"Expected {expected_prediction_rows:,}; "
                f"found {actual_prediction_rows:,}."
            )

    finally:

        con.close()

        if temporary_path.exists():
            temporary_path.unlink()

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

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
        "  Reused model checkpoints: %s",
        f"{n_reused_models:,}",
    )

    logger.info(
        "  Rebuilt model checkpoints: %s",
        f"{n_rebuilt_models:,}",
    )

    logger.info(
        "  Output: %s",
        output_path,
    )

    logger.info(
        "%s CHECK — Candidate-model "
        "cross-validation complete.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,

        "output_path": (
            output_path
        ),
        "model_predictions_path": (
            output_path
        ),

        "n_prediction_rows": (
            n_prediction_rows
        ),
        "n_models": (
            n_models
        ),
        "n_validation_years": (
            n_validation_years
        ),

        "n_reused_models": (
            n_reused_models
        ),
        "n_rebuilt_models": (
            n_rebuilt_models
        ),

        "duration_seconds": (
            elapsed
        ),

        "reused": (
            n_rebuilt_models == 0
        ),
    }