"""
vacoda.steps.validation.model_fitting

Fit the model selected by Validation on the complete model-training sample.

The step:

1. reads the selected-model artifact;
2. resolves the selected candidate against the current Validation config;
3. reads all rows marked ``training_selected``;
4. fits the selected model on the complete selected population;
5. stores the fitted estimator together with all preprocessing state
   required for later inference;
6. writes a reusable fitted-model metadata artifact.

For embedding MLP models, final fitting uses two phases:

    Phase A
        Use an internal validation split to determine the best training epoch.

    Phase B
        Reinitialize the network and train on 100% of the selected
        training population for exactly that many epochs.

This step does not perform inference on Validation rows.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import pickle
from time import perf_counter
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import TargetEncoder
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)
from vacoda.utils.display import (
    get_symbols,
    progress_bar,
)

from vacoda.config.steps.validation import (
    ValidationConfig,
)

from vacoda.pipeline.artifact_manifest import (
    json_safe,
    manifest_path_for,
    validate_manifest,
    write_manifest,
)

from vacoda.steps.validation.model_training import (
    MODEL_ROW_KEY,
    EmbeddingMLP,
    _build_estimator,
    _build_feature_frame,
    _build_target,
    _model_data_fingerprint,
    _ordered_unique,
    _require_columns,
    _validate_model_row_keys,
)

from vacoda.steps.validation.training_sample import (
    training_definition_name,
)

from vacoda.utils.sql import sql_literal


FITTED_MODEL_CACHE_VERSION = 1


# ======================================================================
# Selected-model artifact
# ======================================================================


def _read_selected_model(
    *,
    selected_model_path: Path,
) -> dict[str, Any]:
    """Read and validate the one-row selected-model artifact."""

    con = duckdb.connect()

    try:

        rows = con.execute(
            """
            SELECT
                model_name,
                model_type,
                selection_metric,
                selection_direction,
                selection_value

            FROM read_parquet(?)
            """,
            [
                str(
                    selected_model_path
                )
            ],
        ).fetchall()

    finally:
        con.close()

    if len(rows) != 1:
        raise RuntimeError(
            "Selected Validation model artifact must contain "
            f"exactly one row; found {len(rows):,}."
        )

    (
        model_name,
        model_type,
        selection_metric,
        selection_direction,
        selection_value,
    ) = rows[0]

    if (
        model_name is None
        or model_type is None
        or selection_metric is None
        or selection_value is None
    ):
        raise RuntimeError(
            "Selected Validation model artifact contains "
            "missing model-selection fields."
        )

    return {
        "model_name": str(
            model_name
        ),
        "model_type": str(
            model_type
        ),
        "selection_metric": str(
            selection_metric
        ),
        "selection_direction": str(
            selection_direction
        ),
        "selection_value": float(
            selection_value
        ),
    }


# ======================================================================
# Production-artifact paths
# ======================================================================


def _production_bundle_path(
    *,
    output_path: Path,
    model_name: str,
    model_type: str,
) -> Path:
    """Return the serialized production-model bundle path."""

    suffix = (
        ".pt"
        if model_type == "embedding_mlp"
        else ".pkl"
    )

    return (
        output_path.parent
        / "production"
        / model_name
        / f"model{suffix}"
    )


# ======================================================================
# Semantic identity
# ======================================================================


def _fitted_model_semantic_payload(
    *,
    cfg: ValidationConfig,
    selected_model: dict[str, Any],
    model_cfg: Any,
    training_rows: int,
    training_data_sha256: str,
    min_training_year: int,
    max_training_year: int,
) -> dict[str, Any]:
    """Return semantic identity of the final fitted model."""

    return {
        "artifact": (
            "validation_fitted_model"
        ),

        "cache_version": (
            FITTED_MODEL_CACHE_VERSION
        ),

        "scope": {
            "target_hs": (
                cfg.scope.target_hs
            ),
        },

        "training_definition": (
            training_definition_name(
                cfg=cfg
            )
        ),

        "selection": {
            "model_name": (
                selected_model[
                    "model_name"
                ]
            ),
            "model_type": (
                selected_model[
                    "model_type"
                ]
            ),
            "metric": (
                selected_model[
                    "selection_metric"
                ]
            ),
            "direction": (
                selected_model[
                    "selection_direction"
                ]
            ),
            "value": float(
                selected_model[
                    "selection_value"
                ]
            ),
        },

        "training_data": {
            "rows": int(
                training_rows
            ),
            "semantic_sha256": (
                training_data_sha256
            ),
            "min_year": int(
                min_training_year
            ),
            "max_year": int(
                max_training_year
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

        "model": {
            "name": (
                selected_model[
                    "model_name"
                ]
            ),
            "type": (
                model_cfg.type
            ),
            "parameters": json_safe(
                model_cfg.parameters
            ),
        },

        "fitting": {
            "procedure": (
                "full_selected_population"
            ),
            "embedding_epoch_rule": (
                "internal_validation_then_full_refit"
            ),
        },

        "random_seed": int(
            cfg.model.random_seed
        ),
    }


# ======================================================================
# Embedding preprocessing
# ======================================================================


def _fit_embedding_preprocessor(
    *,
    frame: pd.DataFrame,
    cfg: ValidationConfig,
) -> dict[str, Any]:
    """
    Fit preprocessing required by the embedding network.

    Numeric statistics are calculated after configured numeric
    transformations have already been applied by _build_feature_frame().

    Categorical ID 0 is reserved for unseen categories.
    """

    numeric_columns = list(
        cfg.model.features.numeric.keys()
    )

    categorical_columns = list(
        cfg.model.features.categorical
    )

    numeric = (
        frame[
            numeric_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    numeric_mean = numeric.mean(
        axis=0
    )

    numeric_std = numeric.std(
        axis=0
    )

    numeric_std[
        numeric_std == 0
    ] = 1.0

    categorical_mappings: dict[
        str,
        dict[str, int],
    ] = {}

    category_sizes: list[int] = []

    for column in categorical_columns:

        categories = (
            frame[
                column
            ]
            .astype("string")
            .fillna(
                "__MISSING__"
            )
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        mapping = {
            str(value): index + 1
            for index, value
            in enumerate(
                categories
            )
        }

        categorical_mappings[
            column
        ] = mapping

        category_sizes.append(
            len(mapping) + 1
        )

    return {
        "numeric_columns": (
            numeric_columns
        ),

        "categorical_columns": (
            categorical_columns
        ),

        "numeric_mean": (
            numeric_mean.astype(
                np.float32
            )
        ),

        "numeric_std": (
            numeric_std.astype(
                np.float32
            )
        ),

        "categorical_mappings": (
            categorical_mappings
        ),

        "category_sizes": (
            category_sizes
        ),

        "unknown_category_id": 0,
    }


def _transform_embedding_features(
    *,
    frame: pd.DataFrame,
    preprocessing: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Apply fitted embedding preprocessing."""

    numeric_columns = (
        preprocessing[
            "numeric_columns"
        ]
    )

    categorical_columns = (
        preprocessing[
            "categorical_columns"
        ]
    )

    numeric = (
        frame[
            numeric_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    numeric = (
        numeric
        - preprocessing[
            "numeric_mean"
        ]
    ) / preprocessing[
        "numeric_std"
    ]

    categorical_parts: list[
        np.ndarray
    ] = []

    for column in categorical_columns:

        mapping = (
            preprocessing[
                "categorical_mappings"
            ][
                column
            ]
        )

        ids = (
            frame[
                column
            ]
            .astype("string")
            .fillna(
                "__MISSING__"
            )
            .map(
                mapping
            )
            .fillna(
                preprocessing[
                    "unknown_category_id"
                ]
            )
            .to_numpy(
                dtype=np.int64
            )
        )

        categorical_parts.append(
            ids
        )

    categorical = np.column_stack(
        categorical_parts
    )

    return (
        numeric.astype(
            np.float32,
            copy=False,
        ),
        categorical.astype(
            np.int64,
            copy=False,
        ),
    )


# ======================================================================
# Embedding architecture
# ======================================================================


def _embedding_dimensions(
    *,
    parameters: dict[str, Any],
) -> list[int]:
    """Return embedding dimensions in configured categorical order."""

    return [
        int(
            parameters[
                "exporter_embedding_dim"
            ]
        ),
        int(
            parameters[
                "importer_embedding_dim"
            ]
        ),
        int(
            parameters[
                "product_embedding_dim"
            ]
        ),
    ]


def _build_embedding_model(
    *,
    category_sizes: list[int],
    numeric_dim: int,
    parameters: dict[str, Any],
    device: torch.device,
) -> EmbeddingMLP:
    """Construct the configured embedding network."""

    return EmbeddingMLP(
        category_sizes=(
            category_sizes
        ),
        embedding_dims=(
            _embedding_dimensions(
                parameters=parameters
            )
        ),
        numeric_dim=int(
            numeric_dim
        ),
        hidden_layer_1=int(
            parameters[
                "hidden_layer_1"
            ]
        ),
        hidden_layer_2=int(
            parameters[
                "hidden_layer_2"
            ]
        ),
    ).to(
        device
    )


# ======================================================================
# Phase A — determine final epoch
# ======================================================================


def _determine_embedding_best_epoch(
    *,
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y: np.ndarray,
    fitting_indices: np.ndarray,
    validation_indices: np.ndarray,
    category_sizes: list[int],
    parameters: dict[str, Any],
    cfg: ValidationConfig,
    logger,
) -> tuple[int, float, float,]:
    """
    Determine the preferred training epoch using internal validation.

    This phase is used only to choose training duration. Its fitted
    network is discarded afterwards.
    """

    params = dict(
        parameters
    )

    batch_size = int(
        params[
            "batch_size"
        ]
    )

    learning_rate = float(
        params[
            "learning_rate"
        ]
    )

    max_epochs = int(
        params[
            "max_epochs"
        ]
    )

    patience = int(
        params[
            "patience"
        ]
    )

    early_stopping = bool(
        params[
            "early_stopping"
        ]
    )

    min_delta = float(
        params[
            "min_delta"
        ]
    )

    random_seed = int(cfg.model.random_seed)

    n_rows = len(y)

    if n_rows < 2:
        raise RuntimeError(
            "Final embedding-model fitting requires "
            "at least two training rows."
        )

    fitting_indices = np.asarray(
        fitting_indices,
        dtype=np.int64,
    )

    validation_indices = np.asarray(
        validation_indices,
        dtype=np.int64,
    )

    if (
        len(fitting_indices) == 0
        or len(validation_indices) == 0
    ):
        raise RuntimeError(
            "Embedding epoch selection requires non-empty "
            "fitting and internal-validation populations."
        )

    if (
        fitting_indices.min() < 0
        or validation_indices.min() < 0
        or fitting_indices.max() >= n_rows
        or validation_indices.max() >= n_rows
    ):
        raise RuntimeError(
            "Embedding epoch-selection indices fall outside "
            "the available training population."
        )

    if np.intersect1d(
        fitting_indices,
        validation_indices,
    ).size:
        raise RuntimeError(
            "Embedding epoch-selection fitting and validation "
            "populations overlap."
        )

    if (
        len(fitting_indices)
        + len(validation_indices)
        != n_rows
    ):
        raise RuntimeError(
            "Embedding epoch-selection split does not cover "
            "the complete selected training population."
        )

    logger.info(
        "  Phase A — determine final training epoch"
    )

    logger.info(
        "    Epoch-selection fit rows: %s",
        f"{len(fitting_indices):,}",
    )

    logger.info(
        "    Internal validation rows: %s",
        f"{len(validation_indices):,}",
    )

    torch.manual_seed(
        random_seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = _build_embedding_model(
        category_sizes=(
            category_sizes
        ),
        numeric_dim=(
            x_numeric.shape[1]
        ),
        parameters=params,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_function = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.from_numpy(
            x_numeric[
                fitting_indices
            ]
        ).float(),
        torch.from_numpy(
            x_categorical[
                fitting_indices
            ]
        ).long(),
        torch.from_numpy(
            y[
                fitting_indices
            ].astype(
                np.float32
            )
        ),
    )

    loader_generator = (
        torch.Generator()
    )

    loader_generator.manual_seed(
        random_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    internal_numeric = (
        torch.from_numpy(
            x_numeric[
                validation_indices
            ]
        )
        .float()
        .to(
            device
        )
    )

    internal_categorical = (
        torch.from_numpy(
            x_categorical[
                validation_indices
            ]
        )
        .long()
        .to(
            device
        )
    )

    internal_target = (
        torch.from_numpy(
            y[
                validation_indices
            ].astype(
                np.float32
            )
        )
        .to(
            device
        )
    )

    best_validation_loss = np.inf
    best_epoch = 0

    early_stopping_reference_loss = (
        np.inf
    )

    epochs_without_meaningful_improvement = 0
    phase_start = perf_counter()

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        epoch_start = perf_counter()
        model.train()

        training_loss_sum = 0.0
        training_rows = 0

        for (
            batch_numeric,
            batch_categorical,
            batch_target,
        ) in train_loader:

            batch_numeric = (
                batch_numeric.to(
                    device
                )
            )

            batch_categorical = (
                batch_categorical.to(
                    device
                )
            )

            batch_target = (
                batch_target.to(
                    device
                )
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

            batch_n = int(
                batch_target.shape[0]
            )

            training_loss_sum += (
                float(
                    loss.item()
                )
                * batch_n
            )

            training_rows += (
                batch_n
            )

        training_mse = (
            training_loss_sum
            / training_rows
        )

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

        epoch_elapsed = (
            perf_counter()
            - epoch_start
        )

        phase_elapsed = (
            perf_counter()
            - phase_start
        )

        logger.info(
            "    Epoch %d — "
            "train_MSE_log=%.6f, "
            "internal_validation_MSE_log=%.6f, "
            "epoch_time=%.1fs, "
            "phase_time=%.1fmin",
            epoch,
            training_mse,
            internal_loss,
            epoch_elapsed,
            phase_elapsed / 60.0,
        )

        if (
            internal_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                internal_loss
            )

            best_epoch = (
                epoch
            )

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
                        "    Early stopping at epoch %d; "
                        "best epoch=%d, "
                        "best internal MSE_log=%.6f",
                        epoch,
                        best_epoch,
                        best_validation_loss,
                    )

                    break

    if best_epoch <= 0:
        raise RuntimeError(
            "Embedding epoch-selection phase did not "
            "produce a valid best epoch."
        )

    logger.info(
        "  Phase A complete — duration=%.2f min",
        (
            perf_counter()
            - phase_start
        ) / 60.0,
    )

    return (
        int(best_epoch),
        float(best_validation_loss),
        float(perf_counter() - phase_start),
    )


# ======================================================================
# Phase B — fit on all selected rows
# ======================================================================


def _fit_embedding_full_population(
    *,
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y: np.ndarray,
    category_sizes: list[int],
    parameters: dict[str, Any],
    final_epochs: int,
    cfg: ValidationConfig,
    logger,
) -> tuple[EmbeddingMLP, float,]:
    """Fit a fresh embedding network on all selected rows."""

    params = dict(
        parameters
    )

    batch_size = int(
        params[
            "batch_size"
        ]
    )

    learning_rate = float(
        params[
            "learning_rate"
        ]
    )

    random_seed = int(
        cfg.model.random_seed
    )

    torch.manual_seed(
        random_seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        "  Phase B — final fit on complete selected population"
    )

    logger.info(
        "    Device: %s",
        device,
    )

    logger.info(
        "    Training rows: %s",
        f"{len(y):,}",
    )

    logger.info(
        "    Final epochs: %s",
        f"{final_epochs:,}",
    )

    model = _build_embedding_model(
        category_sizes=(
            category_sizes
        ),
        numeric_dim=(
            x_numeric.shape[1]
        ),
        parameters=params,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_function = nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(
            x_numeric
        ).float(),
        torch.from_numpy(
            x_categorical
        ).long(),
        torch.from_numpy(
            y.astype(
                np.float32
            )
        ),
    )

    loader_generator = (
        torch.Generator()
    )

    loader_generator.manual_seed(
        random_seed
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    phase_start = perf_counter()

    for epoch in range(
        1,
        final_epochs + 1,
    ):
        epoch_start = perf_counter()
        model.train()

        training_loss_sum = 0.0
        training_rows = 0

        for (
            batch_numeric,
            batch_categorical,
            batch_target,
        ) in loader:

            batch_numeric = (
                batch_numeric.to(
                    device
                )
            )

            batch_categorical = (
                batch_categorical.to(
                    device
                )
            )

            batch_target = (
                batch_target.to(
                    device
                )
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

            batch_n = int(
                batch_target.shape[0]
            )

            training_loss_sum += (
                float(
                    loss.item()
                )
                * batch_n
            )

            training_rows += (
                batch_n
            )

        training_mse = (
            training_loss_sum
            / training_rows
        )

        epoch_elapsed = (
            perf_counter()
            - epoch_start
        )

        phase_elapsed = (
            perf_counter()
            - phase_start
        )

        average_epoch_seconds = (
            phase_elapsed / epoch
        )

        remaining_seconds = (
            average_epoch_seconds
            * (
                final_epochs - epoch
            )
        )

        logger.info(
            "    Final fit epoch %d/%d — "
            "train_MSE_log=%.6f, "
            "epoch_time=%.1fs, "
            "phase_time=%.1fmin, "
            "eta=%.1fmin",
            epoch,
            final_epochs,
            training_mse,
            epoch_elapsed,
            phase_elapsed / 60.0,
            remaining_seconds / 60.0,
        )


    phase_b_duration_seconds = (
        perf_counter()
        - phase_start
    )
    logger.info(
        "  Phase B complete — duration=%.2f min",
            phase_b_duration_seconds / 60.0,
    )

    model.eval()
    model.to(
        "cpu"
    )

    return (
        model,
        phase_b_duration_seconds,
    )


# ======================================================================
# sklearn production fitting
# ======================================================================


def _fit_sklearn_production_model(
    *,
    x_all: pd.DataFrame,
    y_all: np.ndarray,
    model_type: str,
    parameters: dict[str, Any],
    cfg: ValidationConfig,
    logger,
) -> dict[str, Any]:
    """
    Fit a non-embedding candidate and preserve its categorical encoder.

    This keeps the TargetEncoder available for later inference.
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

    if numeric_columns:

        train_parts.append(
            x_all[
                numeric_columns
            ].to_numpy(
                dtype=float
            )
        )

    target_encoder = None

    if categorical_columns:

        target_encoder = (
            TargetEncoder(
                target_type="continuous",
                smooth="auto",
                cv=5,
                random_state=(
                    cfg.model.random_seed
                ),
            )
        )

        encoded = (
            target_encoder.fit_transform(
                x_all[
                    categorical_columns
                ],
                y_all,
            )
        )

        train_parts.append(
            np.asarray(
                encoded,
                dtype=float,
            )
        )

    if not train_parts:
        raise RuntimeError(
            "No configured model features are available "
            "for final model fitting."
        )

    x_encoded = np.column_stack(
        train_parts
    )

    params = dict(
        parameters
    )

    # --------------------------------------------------------------
    # sklearn MLP:
    # use its internal early stopping only to determine n_iter_,
    # then refit a fresh estimator on all rows for that duration.
    # --------------------------------------------------------------

    final_iterations = None

    if model_type == "mlp":

        logger.info(
            "  Determining final sklearn MLP training duration..."
        )

        calibration_estimator = (
            _build_estimator(
                model_type=model_type,
                parameters=params,
                cfg=cfg,
            )
        )

        calibration_estimator.fit(
            x_encoded,
            y_all,
        )

        mlp_estimator = (
            calibration_estimator[
                -1
            ]
        )

        final_iterations = int(
            mlp_estimator.n_iter_
        )

        logger.info(
            "  sklearn MLP selected iterations: %s",
            f"{final_iterations:,}",
        )

        final_params = dict(
            params
        )

        final_params[
            "early_stopping"
        ] = False

        final_params[
            "max_iter"
        ] = (
            final_iterations
        )

        estimator = (
            _build_estimator(
                model_type=model_type,
                parameters=(
                    final_params
                ),
                cfg=cfg,
            )
        )

    else:

        estimator = (
            _build_estimator(
                model_type=model_type,
                parameters=params,
                cfg=cfg,
            )
        )

    logger.info(
        "  Fitting %s on %s selected rows...",
        model_type,
        f"{len(y_all):,}",
    )

    estimator.fit(
        x_encoded,
        y_all,
    )

    return {
        "estimator": estimator,
        "target_encoder": (
            target_encoder
        ),
        "numeric_columns": (
            numeric_columns
        ),
        "categorical_columns": (
            categorical_columns
        ),
        "final_iterations": (
            final_iterations
        ),
    }


# ======================================================================
# Bundle validation
# ======================================================================


def _validate_embedding_bundle(
    *,
    bundle_path: Path,
) -> None:
    """Require a reloadable embedding production bundle."""

    bundle = torch.load(
        bundle_path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "model_name",
        "model_type",
        "state_dict",
        "parameters",
        "preprocessing",
        "target_transform",
        "final_epochs",
    }

    missing = sorted(
        required
        - set(
            bundle
        )
    )

    if missing:
        raise RuntimeError(
            "Fitted embedding-model bundle is missing "
            "required field(s): "
            + ", ".join(
                missing
            )
        )

    preprocessing = (
        bundle[
            "preprocessing"
        ]
    )

    parameters = dict(
        bundle[
            "parameters"
        ]
    )

    model = EmbeddingMLP(
        category_sizes=[
            int(value)
            for value
            in preprocessing[
                "category_sizes"
            ]
        ],
        embedding_dims=(
            _embedding_dimensions(
                parameters=parameters
            )
        ),
        numeric_dim=len(
            preprocessing[
                "numeric_columns"
            ]
        ),
        hidden_layer_1=int(
            parameters[
                "hidden_layer_1"
            ]
        ),
        hidden_layer_2=int(
            parameters[
                "hidden_layer_2"
            ]
        ),
    )

    model.load_state_dict(
        bundle[
            "state_dict"
        ]
    )

    model.eval()


def _validate_sklearn_bundle(
    *,
    bundle_path: Path,
) -> None:
    """Require a reloadable sklearn production bundle."""

    with bundle_path.open(
        "rb"
    ) as handle:

        bundle = pickle.load(
            handle
        )

    required = {
        "model_name",
        "model_type",
        "estimator",
        "numeric_columns",
        "categorical_columns",
        "target_transform",
    }

    missing = sorted(
        required
        - set(
            bundle
        )
    )

    if missing:
        raise RuntimeError(
            "Fitted sklearn model bundle is missing "
            "required field(s): "
            + ", ".join(
                missing
            )
        )

    if not hasattr(
        bundle[
            "estimator"
        ],
        "predict",
    ):
        raise RuntimeError(
            "Reloaded sklearn production estimator "
            "does not expose predict()."
        )


# ======================================================================
# Public step
# ======================================================================


def fit_selected_validation_model(
    *,
    cfg: ValidationConfig,
    training_sample_path: Path,
    selected_model_path: Path,
    run_id: str,
    output_path: Path | None = None,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Fit the selected Validation model on all training-selected rows.

    The resulting production bundle contains both the fitted estimator
    and the preprocessing state required by Build validated trade.
    """

    start = perf_counter()

    training_sample_path = Path(
        training_sample_path
    )

    selected_model_path = Path(
        selected_model_path
    )

    if not training_sample_path.is_file():
        raise FileNotFoundError(
            "Validation training sample not found: "
            f"{training_sample_path}"
        )

    if not selected_model_path.is_file():
        raise FileNotFoundError(
            "Selected Validation model artifact not found: "
            f"{selected_model_path}"
        )

    # ------------------------------------------------------------------
    # Read selected model
    # ------------------------------------------------------------------

    selected_model = (
        _read_selected_model(
            selected_model_path=(
                selected_model_path
            )
        )
    )

    selected_model_name = (
        selected_model[
            "model_name"
        ]
    )

    selected_model_type = (
        selected_model[
            "model_type"
        ]
    )

    enabled_models = (
        cfg.model.models.enabled()
    )

    if (
        selected_model_name
        not in enabled_models
    ):
        raise RuntimeError(
            "Selected Validation model is not enabled "
            "in the current configuration: "
            f"{selected_model_name!r}."
        )

    model_cfg = (
        enabled_models[
            selected_model_name
        ]
    )

    if (
        str(
            model_cfg.type
        )
        != selected_model_type
    ):
        raise RuntimeError(
            "Selected Validation model type does not match "
            "the current configuration. "
            f"Selection artifact: {selected_model_type!r}; "
            f"configuration: {model_cfg.type!r}."
        )

    # ------------------------------------------------------------------
    # Resolve output
    # ------------------------------------------------------------------

    base_output_path = (
        Path(
            output_path
        )
        if output_path is not None
        else cfg.product_path(
            "fitted_model"
        )
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

    bundle_path = (
        _production_bundle_path(
            output_path=output_path,
            model_name=(
                selected_model_name
            ),
            model_type=(
                selected_model_type
            ),
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "%s Fit selected Validation model",
        get_symbols("build"),
    )

    logger.info(
        "  Training sample: %s",
        training_sample_path,
    )

    logger.info(
        "  Selected-model artifact: %s",
        selected_model_path,
    )

    logger.info(
        "  Selected model: %s",
        selected_model_name,
    )

    logger.info(
        "  Model type: %s",
        selected_model_type,
    )

    logger.info(
        "  Selection %s: %.6f",
        selected_model[
            "selection_metric"
        ],
        selected_model[
            "selection_value"
        ],
    )

    logger.info(
        "  Fitted-model metadata: %s",
        output_path,
    )

    logger.info(
        "  Production bundle: %s",
        bundle_path,
    )

    # ------------------------------------------------------------------
    # Load full selected training population
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

    con = duckdb.connect()

    try:

        con.execute(
            f"PRAGMA threads="
            f"{int(cfg.engine.threads)}"
        )

        _require_columns(
            con=con,
            input_path=(
                training_sample_path
            ),
            required_columns=set(
                selected_columns
            ),
        )

        training_sample_sql = (
            sql_literal(
                training_sample_path
            )
        )

        select_sql = ",\n".join(
            f'"{column}"'
            for column
            in selected_columns
        )

        frame = con.execute(
            f"""
            SELECT
                {select_sql}

            FROM read_parquet(
                {training_sample_sql}
            )

            WHERE training_selected
            """
        ).df()

    finally:
        con.close()

    if frame.empty:
        raise RuntimeError(
            "Final Validation model fitting found no rows "
            "with training_selected = TRUE."
        )

    _validate_model_row_keys(
        frame
    )

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
            "Final Validation model fitting contains "
            "no training years."
        )

    min_training_year = int(
        years[0]
    )

    max_training_year = int(
        years[-1]
    )

    training_rows = int(
        len(
            frame
        )
    )

    logger.info(
        "  Selected training rows: %s",
        f"{training_rows:,}",
    )

    logger.info(
        "  Training years: %s-%s",
        min_training_year,
        max_training_year,
    )

    # ------------------------------------------------------------------
    # Build semantic evidence fingerprint
    # ------------------------------------------------------------------

    logger.info(
        "  Building semantic fingerprint of final training evidence..."
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

    semantic_payload = (
        _fitted_model_semantic_payload(
            cfg=cfg,
            selected_model=(
                selected_model
            ),
            model_cfg=model_cfg,
            training_rows=(
                training_rows
            ),
            training_data_sha256=(
                training_data_sha256
            ),
            min_training_year=(
                min_training_year
            ),
            max_training_year=(
                max_training_year
            ),
        )
    )

    metadata_manifest_path = (
        manifest_path_for(
            output_path
        )
    )

    bundle_manifest_path = (
        manifest_path_for(
            bundle_path
        )
    )

    bundle_semantic_payload = {
        **semantic_payload,
        "component": (
            "production_model_bundle"
        ),
    }

    # ------------------------------------------------------------------
    # Reuse
    # ------------------------------------------------------------------

    metadata_reusable = False
    metadata_reason = (
        "force=True"
    )

    bundle_reusable = False
    bundle_reason = (
        "force=True"
    )

    if not force:

        (
            metadata_reusable,
            metadata_reason,
        ) = validate_manifest(
            artifact_path=output_path,
            manifest_path=(
                metadata_manifest_path
            ),
            expected_payload=(
                semantic_payload
            ),
        )

        (
            bundle_reusable,
            bundle_reason,
        ) = validate_manifest(
            artifact_path=bundle_path,
            manifest_path=(
                bundle_manifest_path
            ),
            expected_payload=(
                bundle_semantic_payload
            ),
        )

    if (
        metadata_reusable
        and bundle_reusable
    ):

        logger.info(
            "%s REUSE — Fitted Validation model "
            "(metadata: %s; bundle: %s)",
            get_symbols("reuse"),
            metadata_reason,
            bundle_reason,
        )

        if (
            selected_model_type
            == "embedding_mlp"
        ):
            _validate_embedding_bundle(
                bundle_path=(
                    bundle_path
                )
            )

        else:
            _validate_sklearn_bundle(
                bundle_path=(
                    bundle_path
                )
            )

        metadata_con = (
            duckdb.connect()
        )

        try:

            metadata_row = (
                metadata_con.execute(
                    """
                    SELECT
                        final_epochs,
                        epoch_selection_validation_loss

                    FROM read_parquet(?)
                    """,
                    [
                        str(
                            output_path
                        )
                    ],
                ).fetchone()
            )

        finally:
            metadata_con.close()

        final_epochs = (
            None
            if metadata_row is None
            or metadata_row[0] is None
            else int(
                metadata_row[0]
            )
        )

        epoch_selection_validation_loss = (
            None
            if metadata_row is None
            or metadata_row[1] is None
            else float(
                metadata_row[1]
            )
        )

        elapsed = (
            perf_counter()
            - start
        )

        return {
            "status": "completed",
            "run_id": run_id,

            "fitted_model_path": (
                output_path
            ),

            "model_bundle_path": (
                bundle_path
            ),

            "model_name": (
                selected_model_name
            ),

            "model_type": (
                selected_model_type
            ),

            "training_rows": (
                training_rows
            ),

            "min_training_year": (
                min_training_year
            ),

            "max_training_year": (
                max_training_year
            ),

            "final_epochs": (
                final_epochs
            ),

            "epoch_selection_validation_loss": (
                epoch_selection_validation_loss
            ),

            "duration_seconds": (
                elapsed
            ),

            "reused": True,
        }

    logger.info(
        "%s REBUILD — Fitted Validation model "
        "(metadata: %s; bundle: %s)",
        get_symbols("rebuilt"),
        metadata_reason,
        bundle_reason,
    )

    # ------------------------------------------------------------------
    # Build configured feature / target representation
    # ------------------------------------------------------------------

    x_all = (
        _build_feature_frame(
            frame=frame,
            cfg=cfg,
        )
    )

    y_all = (
        _build_target(
            frame=frame,
            cfg=cfg,
        )
    )

    parameters = dict(
        model_cfg.parameters
    )

    final_epochs = None

    epoch_selection_validation_loss = (
        None
    )

    # ==================================================================
    # Embedding MLP
    # ==================================================================

    if (
        selected_model_type
        == "embedding_mlp"
    ):

        torch.set_num_threads(
            int(
                cfg.engine.threads
            )
        )

        logger.info(
            "  PyTorch CPU threads — intraop=%d, interop=%d",
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
        )

        # --------------------------------------------------------------
        # Phase A preprocessing
        #
        # The preprocessing itself must not learn from the internal
        # validation rows used to select the training duration.
        # --------------------------------------------------------------

        random_seed = int(
            cfg.model.random_seed
        )

        internal_fraction = float(
            parameters[
                "internal_validation_fraction"
            ]
        )

        n_rows = len(
            x_all
        )

        rng = np.random.default_rng(
            random_seed
        )

        permutation = rng.permutation(
            n_rows
        )

        n_internal = max(
            1,
            int(
                internal_fraction
                * n_rows
            ),
        )

        n_internal = min(
            n_rows - 1,
            n_internal,
        )

        phase_a_validation_indices = (
            permutation[
                :n_internal
            ]
        )

        phase_a_fit_indices = (
            permutation[
                n_internal:
            ]
        )

        phase_a_fit_frame = (
            x_all.iloc[
                phase_a_fit_indices
            ]
            .reset_index(
                drop=True
            )
        )

        phase_a_validation_frame = (
            x_all.iloc[
                phase_a_validation_indices
            ]
            .reset_index(
                drop=True
            )
        )

        phase_a_preprocessing = (
            _fit_embedding_preprocessor(
                frame=(
                    phase_a_fit_frame
                ),
                cfg=cfg,
            )
        )

        # Reassemble Phase A arrays because the epoch helper expects
        # one population and performs the deterministic split itself.
        #
        # Put fit rows first and validation rows second, then explicitly
        # run a dedicated Phase A helper below using the same population.
        #
        # To avoid performing two different random splits, construct
        # the transformed full Phase A population in the original
        # permutation order and use the normal helper on it.
        #
        # Instead, the simpler and exact route is to fit preprocessing
        # on the Phase-A fitting population, then transform all rows
        # with that state before the deterministic split.
        # --------------------------------------------------------------

        phase_a_all_numeric, (
            phase_a_all_categorical
        ) = (
            _transform_embedding_features(
                frame=x_all,
                preprocessing=(
                    phase_a_preprocessing
                ),
            )
        )

        (
            final_epochs,
            epoch_selection_validation_loss,
            phase_a_duration_seconds,
        ) = _determine_embedding_best_epoch(
                x_numeric=(
                    phase_a_all_numeric
                ),
                x_categorical=(
                    phase_a_all_categorical
                ),
                y=y_all,
                fitting_indices=(
                    phase_a_fit_indices
                ),
                validation_indices=(
                    phase_a_validation_indices
                ),
                category_sizes=(
                    phase_a_preprocessing[
                        "category_sizes"
                    ]
                ),
                parameters=(
                    parameters
                ),
                cfg=cfg,
                logger=logger,
            )

        logger.info(
            "  Selected final epoch: %s",
            f"{final_epochs:,}",
        )

        logger.info(
            "  Best epoch-selection internal MSE_log: %.6f",
            epoch_selection_validation_loss,
        )

        # --------------------------------------------------------------
        # Phase B preprocessing on 100% of selected evidence
        # --------------------------------------------------------------

        final_preprocessing = (
            _fit_embedding_preprocessor(
                frame=x_all,
                cfg=cfg,
            )
        )

        (
            final_numeric,
            final_categorical,
        ) = (
            _transform_embedding_features(
                frame=x_all,
                preprocessing=(
                    final_preprocessing
                ),
            )
        )

        (
            final_model,
            phase_b_duration_seconds,
        ) = _fit_embedding_full_population(
                x_numeric=(
                    final_numeric
                ),
                x_categorical=(
                    final_categorical
                ),
                y=y_all,
                category_sizes=(
                    final_preprocessing[
                        "category_sizes"
                    ]
                ),
                parameters=(
                    parameters
                ),
                final_epochs=(
                    final_epochs
                ),
                cfg=cfg,
                logger=logger,
            )
        )

        # --------------------------------------------------------------
        # Serialize embedding bundle atomically
        # --------------------------------------------------------------

        temporary_bundle_path = (
            bundle_path.with_suffix(
                bundle_path.suffix
                + ".tmp"
            )
        )

        if (
            temporary_bundle_path.exists()
        ):
            temporary_bundle_path.unlink()

        bundle = {
            "model_name": (
                selected_model_name
            ),

            "model_type": (
                selected_model_type
            ),

            "state_dict": {
                name: tensor.detach()
                .cpu()
                .clone()
                for name, tensor
                in final_model.state_dict().items()
            },

            "parameters": json_safe(
                parameters
            ),

            "preprocessing": {
                "numeric_columns": (
                    final_preprocessing[
                        "numeric_columns"
                    ]
                ),

                "categorical_columns": (
                    final_preprocessing[
                        "categorical_columns"
                    ]
                ),

                "numeric_mean": (
                    final_preprocessing[
                        "numeric_mean"
                    ].tolist()
                ),

                "numeric_std": (
                    final_preprocessing[
                        "numeric_std"
                    ].tolist()
                ),

                "categorical_mappings": (
                    final_preprocessing[
                        "categorical_mappings"
                    ]
                ),

                "category_sizes": (
                    final_preprocessing[
                        "category_sizes"
                    ]
                ),

                "unknown_category_id": (
                    final_preprocessing[
                        "unknown_category_id"
                    ]
                ),

                "numeric_transforms": {
                    name: (
                        feature_cfg.transform
                    )
                    for (
                        name,
                        feature_cfg,
                    )
                    in cfg.model.features.numeric.items()
                },
            },

            "target_column": (
                cfg.model.target.column
            ),

            "target_transform": (
                cfg.model.target.transform
            ),

            "training_rows": (
                training_rows
            ),

            "training_data_sha256": (
                training_data_sha256
            ),

            "final_epochs": int(
                final_epochs
            ),

            "epoch_selection_validation_loss": float(
                epoch_selection_validation_loss
            ),

            "random_seed": int(
                cfg.model.random_seed
            ),
        }

        torch.save(
            bundle,
            temporary_bundle_path,
        )

        temporary_bundle_path.replace(
            bundle_path
        )

        _validate_embedding_bundle(
            bundle_path=(
                bundle_path
            )
        )

    # ==================================================================
    # sklearn models
    # ==================================================================

    else:

        fitted = (
            _fit_sklearn_production_model(
                x_all=x_all,
                y_all=y_all,
                model_type=(
                    selected_model_type
                ),
                parameters=(
                    parameters
                ),
                cfg=cfg,
                logger=logger,
            )
        )

        final_epochs = (
            fitted[
                "final_iterations"
            ]
        )

        temporary_bundle_path = (
            bundle_path.with_suffix(
                bundle_path.suffix
                + ".tmp"
            )
        )

        if (
            temporary_bundle_path.exists()
        ):
            temporary_bundle_path.unlink()

        bundle = {
            "model_name": (
                selected_model_name
            ),

            "model_type": (
                selected_model_type
            ),

            "estimator": (
                fitted[
                    "estimator"
                ]
            ),

            "target_encoder": (
                fitted[
                    "target_encoder"
                ]
            ),

            "numeric_columns": (
                fitted[
                    "numeric_columns"
                ]
            ),

            "categorical_columns": (
                fitted[
                    "categorical_columns"
                ]
            ),

            "numeric_transforms": {
                name: (
                    feature_cfg.transform
                )
                for (
                    name,
                    feature_cfg,
                )
                in cfg.model.features.numeric.items()
            },

            "target_column": (
                cfg.model.target.column
            ),

            "target_transform": (
                cfg.model.target.transform
            ),

            "parameters": (
                json_safe(
                    parameters
                )
            ),

            "training_rows": (
                training_rows
            ),

            "training_data_sha256": (
                training_data_sha256
            ),

            "random_seed": int(
                cfg.model.random_seed
            ),
        }

        with temporary_bundle_path.open(
            "wb"
        ) as handle:

            pickle.dump(
                bundle,
                handle,
                protocol=(
                    pickle.HIGHEST_PROTOCOL
                ),
            )

        temporary_bundle_path.replace(
            bundle_path
        )

        _validate_sklearn_bundle(
            bundle_path=(
                bundle_path
            )
        )

    # ------------------------------------------------------------------
    # Bundle manifest
    # ------------------------------------------------------------------

    write_manifest(
        manifest_path=(
            bundle_manifest_path
        ),
        artifact_name=(
            "validation_production_model_bundle"
        ),
        semantic_payload=(
            bundle_semantic_payload
        ),
        artifact_path=(
            bundle_path
        ),
        extra={
            "model_name": (
                selected_model_name
            ),
            "model_type": (
                selected_model_type
            ),
            "training_rows": (
                training_rows
            ),
            "final_epochs": (
                final_epochs
            ),
        },
    )

    # ------------------------------------------------------------------
    # Write fitted-model metadata artifact
    # ------------------------------------------------------------------

    metadata = pd.DataFrame(
        [
            {
                "model_name": (
                    selected_model_name
                ),

                "model_type": (
                    selected_model_type
                ),

                "selection_metric": (
                    selected_model[
                        "selection_metric"
                    ]
                ),

                "selection_direction": (
                    selected_model[
                        "selection_direction"
                    ]
                ),

                "selection_value": (
                    selected_model[
                        "selection_value"
                    ]
                ),

                "training_rows": (
                    training_rows
                ),

                "min_training_year": (
                    min_training_year
                ),

                "max_training_year": (
                    max_training_year
                ),

                "training_data_sha256": (
                    training_data_sha256
                ),

                "final_epochs": (
                    final_epochs
                ),

                "epoch_selection_validation_loss": (
                    epoch_selection_validation_loss
                ),

                "model_bundle_path": str(
                    bundle_path
                ),

                "run_id": (
                    run_id
                ),
            }
        ]
    )

    temporary_output_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    if temporary_output_path.exists():
        temporary_output_path.unlink()

    metadata_con = (
        duckdb.connect()
    )

    try:

        metadata_con.register(
            "fitted_model_metadata",
            metadata,
        )

        temporary_output_sql = (
            sql_literal(
                temporary_output_path
            )
        )

        metadata_con.execute(
            f"""
            COPY fitted_model_metadata
            TO {temporary_output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        metadata_con.unregister(
            "fitted_model_metadata"
        )

    finally:
        metadata_con.close()

    temporary_output_path.replace(
        output_path
    )

    write_manifest(
        manifest_path=(
            metadata_manifest_path
        ),
        artifact_name=(
            "validation_fitted_model"
        ),
        semantic_payload=(
            semantic_payload
        ),
        artifact_path=(
            output_path
        ),
        extra={
            "model_name": (
                selected_model_name
            ),
            "model_type": (
                selected_model_type
            ),
            "training_rows": (
                training_rows
            ),
            "model_bundle_path": str(
                bundle_path
            ),
            "final_epochs": (
                final_epochs
            ),
        },
    )

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    elapsed = (
        perf_counter()
        - start
    )

    logger.info(
        "  Final model: %s",
        selected_model_name,
    )

    logger.info(
        "  Model type: %s",
        selected_model_type,
    )

    logger.info(
        "  Final training rows: %s",
        f"{training_rows:,}",
    )

    logger.info(
        "  Training years: %s-%s",
        min_training_year,
        max_training_year,
    )

    if final_epochs is not None:

        logger.info(
            "  Final training epochs/iterations: %s",
            f"{int(final_epochs):,}",
        )

    logger.info(
        "  Production bundle: %s",
        bundle_path,
    )

    logger.info(
        "  Metadata artifact: %s",
        output_path,
    )

    logger.info(
        "%s CHECK — Selected Validation model fitted.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,

        "fitted_model_path": (
            output_path
        ),

        "model_bundle_path": (
            bundle_path
        ),

        "model_name": (
            selected_model_name
        ),

        "model_type": (
            selected_model_type
        ),

        "training_rows": (
            training_rows
        ),

        "min_training_year": (
            min_training_year
        ),

        "max_training_year": (
            max_training_year
        ),

        "final_epochs": (
            final_epochs
        ),

        "epoch_selection_validation_loss": (
            epoch_selection_validation_loss
        ),

        "duration_seconds": (
            elapsed
        ),

        "reused": False,
    }