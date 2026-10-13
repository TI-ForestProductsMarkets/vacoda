"""
vacoda.steps.validation.inference

Run the fitted Validation production model over the complete Validation
population.

This step produces model evidence only. It does not decide whether an
observed quantity should be preserved, replaced, imputed, or left unresolved.

For every canonical Validation row it records:

- fitted-model quantity prediction;
- fitted-model log-quantity prediction;
- observed-vs-model log residual where an observed quantity exists;
- categorical support flags derived from the fitted preprocessing state.

These outputs are consumed later by Build validated trade.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
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

from vacoda.steps.validation.model_fitting import (
    _embedding_dimensions,
    _transform_embedding_features,
)

from vacoda.steps.validation.model_training import (
    MODEL_ROW_KEY,
    EmbeddingMLP,
    _build_feature_frame,
    _ordered_unique,
    _require_columns,
    _validate_model_row_keys,
)

from vacoda.steps.validation.training_sample import (
    training_definition_name,
)

from vacoda.utils.display import (
    get_symbols,
)


INFERENCE_CACHE_VERSION = 1


# ======================================================================
# Helpers
# ======================================================================


def _read_fitted_model_metadata(
    *,
    fitted_model_path: Path,
) -> dict[str, Any]:
    """Read the one-row fitted-model metadata artifact."""

    con = duckdb.connect()

    try:

        row = con.execute(
            """
            SELECT
                model_name,
                model_type,
                training_rows,
                final_epochs

            FROM read_parquet(?)
            """,
            [
                str(
                    fitted_model_path
                )
            ],
        ).fetchone()

    finally:
        con.close()

    if row is None:
        raise RuntimeError(
            "Fitted Validation model metadata is empty."
        )

    (
        model_name,
        model_type,
        training_rows,
        final_epochs,
    ) = row

    if (
        model_name is None
        or model_type is None
    ):
        raise RuntimeError(
            "Fitted Validation model metadata is missing "
            "model identity."
        )

    return {
        "model_name": str(
            model_name
        ),
        "model_type": str(
            model_type
        ),
        "training_rows": (
            None
            if training_rows is None
            else int(
                training_rows
            )
        ),
        "final_epochs": (
            None
            if final_epochs is None
            else int(
                final_epochs
            )
        ),
    }


def _inference_output_path(
    *,
    cfg: ValidationConfig,
) -> Path:
    """
    Return the Validation inference artifact path.

    This intentionally does not require a new YAML product yet.
    It lives under the current training-definition model directory.
    """

    checkpoint_group = (
        training_definition_name(
            cfg=cfg
        )
    )

    return (
        cfg.output_dir()
        / "model"
        / checkpoint_group
        / (
            f"target_"
            f"{cfg.scope.target_hs.lower()}"
            "_inference.parquet"
        )
    )


def _load_bundle(
    *,
    model_bundle_path: Path,
    model_type: str,
) -> dict[str, Any]:
    """Load the serialized fitted production bundle."""

    if model_type == "embedding_mlp":

        bundle = torch.load(
            model_bundle_path,
            map_location="cpu",
            weights_only=False,
        )

    else:

        with model_bundle_path.open(
            "rb"
        ) as handle:

            bundle = pickle.load(
                handle
            )

    if not isinstance(
        bundle,
        dict,
    ):
        raise RuntimeError(
            "Fitted Validation model bundle does not "
            "contain a dictionary payload."
        )

    return bundle


def _validate_bundle_identity(
    *,
    bundle: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Require fitted metadata and serialized bundle to agree."""

    bundle_model_name = bundle.get(
        "model_name"
    )

    bundle_model_type = bundle.get(
        "model_type"
    )

    if (
        str(bundle_model_name)
        != metadata["model_name"]
    ):
        raise RuntimeError(
            "Fitted-model bundle model name does not match "
            "the fitted-model metadata."
        )

    if (
        str(bundle_model_type)
        != metadata["model_type"]
    ):
        raise RuntimeError(
            "Fitted-model bundle model type does not match "
            "the fitted-model metadata."
        )


def _validate_bundle_feature_contract(
    *,
    bundle: dict[str, Any],
    cfg: ValidationConfig,
    model_type: str,
) -> None:
    """
    Require inference-time feature configuration to match the fitted bundle.
    """

    configured_numeric = list(
        cfg.model.features.numeric.keys()
    )

    configured_categorical = list(
        cfg.model.features.categorical
    )

    if model_type == "embedding_mlp":

        preprocessing = bundle.get(
            "preprocessing"
        )

        if not isinstance(
            preprocessing,
            dict,
        ):
            raise RuntimeError(
                "Embedding production bundle is missing "
                "preprocessing state."
            )

        bundle_numeric = list(
            preprocessing.get(
                "numeric_columns",
                [],
            )
        )

        bundle_categorical = list(
            preprocessing.get(
                "categorical_columns",
                [],
            )
        )

    else:

        bundle_numeric = list(
            bundle.get(
                "numeric_columns",
                [],
            )
        )

        bundle_categorical = list(
            bundle.get(
                "categorical_columns",
                [],
            )
        )

    if (
        bundle_numeric
        != configured_numeric
    ):
        raise RuntimeError(
            "Configured numeric Validation features do not match "
            "the fitted production bundle. "
            f"Configured={configured_numeric!r}; "
            f"bundle={bundle_numeric!r}."
        )

    if (
        bundle_categorical
        != configured_categorical
    ):
        raise RuntimeError(
            "Configured categorical Validation features do not match "
            "the fitted production bundle. "
            f"Configured={configured_categorical!r}; "
            f"bundle={bundle_categorical!r}."
        )


def _support_flags_from_mappings(
    *,
    feature_frame: pd.DataFrame,
    categorical_columns: list[str],
    categorical_mappings: dict[
        str,
        dict[str, int],
    ],
) -> pd.DataFrame:
    """
    Return seen/unseen flags relative to fitted categorical mappings.

    Missing categorical values use the same ``__MISSING__`` representation
    as model training.
    """

    flags = pd.DataFrame(
        index=feature_frame.index
    )

    output_names = {
        "exporter_iso": (
            "exporter_seen"
        ),
        "importer_iso": (
            "importer_seen"
        ),
        "target_cmd": (
            "hs6_seen"
        ),
        "target_hs4": (
            "hs4_seen"
        ),
    }

    for column in categorical_columns:

        if column not in categorical_mappings:
            raise RuntimeError(
                "Fitted categorical mapping is missing "
                f"feature {column!r}."
            )

        values = (
            feature_frame[
                column
            ]
            .astype("string")
            .fillna(
                "__MISSING__"
            )
        )

        known_values = set(
            categorical_mappings[
                column
            ].keys()
        )

        output_column = output_names.get(
            column,
            f"{column}_seen",
        )

        flags[
            output_column
        ] = values.isin(
            known_values
        )

    return flags

def _model_input_valid_mask(
    *,
    frame: pd.DataFrame,
    cfg: ValidationConfig,
) -> pd.Series:
    """
    Return rows for which all configured numeric model predictors
    are usable at inference time.

    Log-transformed predictors must be finite and strictly positive.
    Untransformed predictors must be finite.
    """

    valid = pd.Series(
        True,
        index=frame.index,
        dtype=bool,
    )

    for (
        column,
        feature_cfg,
    ) in cfg.model.features.numeric.items():

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        column_valid = (
            values.notna()
            & np.isfinite(
                values.astype(float)
            )
        )

        if feature_cfg.transform == "log":

            column_valid &= (
                values > 0
            )

        elif feature_cfg.transform != "none":

            raise RuntimeError(
                "Unsupported numeric feature transform: "
                f"{feature_cfg.transform!r}"
            )

        valid &= column_valid

    return valid


def _predict_embedding_model(
    *,
    feature_frame: pd.DataFrame,
    bundle: dict[str, Any],
    cfg: ValidationConfig,
    logger,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """Run the fitted embedding MLP over the complete inference population."""

    required = {
        "state_dict",
        "parameters",
        "preprocessing",
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
            "Embedding production bundle is missing "
            "required field(s): "
            + ", ".join(
                missing
            )
        )

    parameters = dict(
        bundle[
            "parameters"
        ]
    )

    preprocessing = dict(
        bundle[
            "preprocessing"
        ]
    )

    numeric_columns = list(
        preprocessing[
            "numeric_columns"
        ]
    )

    categorical_columns = list(
        preprocessing[
            "categorical_columns"
        ]
    )

    categorical_mappings = (
        preprocessing[
            "categorical_mappings"
        ]
    )

    category_sizes = [
        int(
            value
        )
        for value
        in preprocessing[
            "category_sizes"
        ]
    ]

    (
        x_numeric,
        x_categorical,
    ) = _transform_embedding_features(
        frame=feature_frame,
        preprocessing=preprocessing,
    )

    support_flags = (
        _support_flags_from_mappings(
            feature_frame=(
                feature_frame
            ),
            categorical_columns=(
                categorical_columns
            ),
            categorical_mappings=(
                categorical_mappings
            ),
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.set_num_threads(
        int(
            cfg.engine.threads
        )
    )

    logger.info(
        "  Inference device: %s",
        device,
    )

    logger.info(
        "  PyTorch CPU threads — intraop=%d, interop=%d",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )

    model = EmbeddingMLP(
        category_sizes=(
            category_sizes
        ),
        embedding_dims=(
            _embedding_dimensions(
                parameters=parameters
            )
        ),
        numeric_dim=len(
            numeric_columns
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

    model.to(
        device
    )

    model.eval()

    batch_size = int(
        parameters[
            "batch_size"
        ]
    )

    dataset = TensorDataset(
        torch.from_numpy(
            x_numeric
        ).float(),
        torch.from_numpy(
            x_categorical
        ).long(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    predictions: list[
        np.ndarray
    ] = []

    with torch.no_grad():

        for (
            batch_numeric,
            batch_categorical,
        ) in loader:

            batch_prediction = model(
                batch_numeric.to(
                    device
                ),
                batch_categorical.to(
                    device
                ),
            )

            predictions.append(
                batch_prediction
                .detach()
                .cpu()
                .numpy()
            )

    if not predictions:
        raise RuntimeError(
            "Embedding inference produced no prediction batches."
        )

    prediction = np.concatenate(
        predictions
    ).astype(
        float,
        copy=False,
    )

    if len(
        prediction
    ) != len(
        feature_frame
    ):
        raise RuntimeError(
            "Embedding inference prediction-row count does not "
            "match the input population."
        )

    if not np.isfinite(
        prediction
    ).all():
        raise RuntimeError(
            "Embedding inference produced non-finite "
            "log-quantity predictions."
        )

    return (
        prediction,
        support_flags,
    )


def _sklearn_support_flags(
    *,
    feature_frame: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Derive categorical seen/unseen flags from fitted TargetEncoder state."""

    categorical_columns = list(
        bundle.get(
            "categorical_columns",
            [],
        )
    )

    target_encoder = bundle.get(
        "target_encoder"
    )

    flags = pd.DataFrame(
        index=feature_frame.index
    )

    output_names = {
        "exporter_iso": (
            "exporter_seen"
        ),
        "importer_iso": (
            "importer_seen"
        ),
        "target_cmd": (
            "hs6_seen"
        ),
        "target_hs4": (
            "hs4_seen"
        ),
    }

    if not categorical_columns:
        return flags

    if target_encoder is None:
        raise RuntimeError(
            "Fitted sklearn model has categorical features "
            "but no TargetEncoder state."
        )

    categories = getattr(
        target_encoder,
        "categories_",
        None,
    )

    if (
        categories is None
        or len(
            categories
        )
        != len(
            categorical_columns
        )
    ):
        raise RuntimeError(
            "Fitted sklearn TargetEncoder does not expose "
            "the expected category levels."
        )

    for (
        column,
        learned_categories,
    ) in zip(
        categorical_columns,
        categories,
    ):

        values = (
            feature_frame[
                column
            ]
            .astype("string")
            .fillna(
                "__MISSING__"
            )
        )

        known_values = {
            str(
                value
            )
            for value
            in learned_categories
        }

        output_column = output_names.get(
            column,
            f"{column}_seen",
        )

        flags[
            output_column
        ] = values.isin(
            known_values
        )

    return flags


def _predict_sklearn_model(
    *,
    feature_frame: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """Run a fitted sklearn production model."""

    estimator = bundle.get(
        "estimator"
    )

    if estimator is None:
        raise RuntimeError(
            "Fitted sklearn production bundle is missing estimator."
        )

    numeric_columns = list(
        bundle.get(
            "numeric_columns",
            [],
        )
    )

    categorical_columns = list(
        bundle.get(
            "categorical_columns",
            [],
        )
    )

    parts: list[
        np.ndarray
    ] = []

    if numeric_columns:

        parts.append(
            feature_frame[
                numeric_columns
            ].to_numpy(
                dtype=float
            )
        )

    target_encoder = bundle.get(
        "target_encoder"
    )

    if categorical_columns:

        if target_encoder is None:
            raise RuntimeError(
                "Fitted sklearn production bundle is missing "
                "TargetEncoder state."
            )

        encoded = (
            target_encoder.transform(
                feature_frame[
                    categorical_columns
                ]
            )
        )

        parts.append(
            np.asarray(
                encoded,
                dtype=float,
            )
        )

    if not parts:
        raise RuntimeError(
            "No model features are available for sklearn inference."
        )

    x_encoded = np.column_stack(
        parts
    )

    prediction = np.asarray(
        estimator.predict(
            x_encoded
        ),
        dtype=float,
    )

    if not np.isfinite(
        prediction
    ).all():
        raise RuntimeError(
            "sklearn inference produced non-finite predictions."
        )

    support_flags = (
        _sklearn_support_flags(
            feature_frame=(
                feature_frame
            ),
            bundle=bundle,
        )
    )

    return (
        prediction,
        support_flags,
    )


def _inverse_target_transform(
    *,
    prediction: np.ndarray,
    target_transform: str,
) -> np.ndarray:
    """Convert model-space prediction back to quantity space."""

    if target_transform == "log":

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):

            quantity = np.exp(
                prediction
            )

    elif target_transform == "none":

        quantity = prediction.copy()

    else:

        raise RuntimeError(
            "Unsupported fitted target transform: "
            f"{target_transform!r}."
        )

    quantity = np.asarray(
        quantity,
        dtype=float,
    )

    if (
        not np.isfinite(
            quantity
        ).all()
        or (
            quantity <= 0
        ).any()
    ):
        raise RuntimeError(
            "Inverse target transformation produced "
            "non-positive or non-finite quantity predictions."
        )

    return quantity


def _build_residual_evidence(
    *,
    observed_quantity: pd.Series,
    prediction_log_quantity: np.ndarray,
) -> pd.DataFrame:
    """
    Compare observed quantity with model quantity where log comparison
    is mathematically valid.
    """

    observed = pd.to_numeric(
        observed_quantity,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    available = np.isfinite(
        observed
    )

    prediction_available = (
        np.isfinite(
            prediction_log_quantity
        )
    )

    valid_for_log = (
        available
        & (
            observed > 0
        )
        & prediction_available
    )

    residual = np.full(
        len(
            observed
        ),
        np.nan,
        dtype=float,
    )

    residual[
        valid_for_log
    ] = (
        np.log(
            observed[
                valid_for_log
            ]
        )
        - prediction_log_quantity[
            valid_for_log
        ]
    )

    return pd.DataFrame(
        {
            "model_prediction_available": (
                prediction_available
            ),

            "observed_quantity_available": (
                available
            ),

            "observed_quantity_valid_for_log": (
                valid_for_log
            ),

            "quantity_log_residual": (
                residual
            ),

            "quantity_abs_log_residual": (
                np.abs(
                    residual
                )
            ),
        }
    )


def _inference_semantic_payload(
    *,
    cfg: ValidationConfig,
    metadata: dict[str, Any],
    bundle: dict[str, Any],
    input_path: Path,
    input_rows: int,
) -> dict[str, Any]:
    """Return semantic identity of the inference artifact."""

    return {
        "artifact": (
            "validation_inference"
        ),

        "cache_version": (
            INFERENCE_CACHE_VERSION
        ),

        "scope": {
            "target_hs": (
                cfg.scope.target_hs
            ),
            "min_year": int(
                cfg.scope.min_year
            ),
            "max_year": int(
                cfg.scope.max_year
            ),
        },

        "training_definition": (
            training_definition_name(
                cfg=cfg
            )
        ),

        "population": {
            "rows": int(
                input_rows
            ),
            "scope": (
                "complete_validation_population"
            ),
            "input_path": str(
                input_path.resolve()
            ),
            "input_size_bytes": int(
                input_path.stat().st_size
            ),
            "input_mtime_ns": int(
                input_path.stat().st_mtime_ns
            ),
        },

        "model": {
            "name": (
                metadata[
                    "model_name"
                ]
            ),
            "type": (
                metadata[
                    "model_type"
                ]
            ),
            "training_rows": (
                metadata[
                    "training_rows"
                ]
            ),
            "final_epochs": (
                metadata[
                    "final_epochs"
                ]
            ),
            "training_data_sha256": (
                bundle.get(
                    "training_data_sha256"
                )
            ),
            "parameters": json_safe(
                bundle.get(
                    "parameters",
                    {},
                )
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

        "decision_policy": (
            "none_inference_evidence_only"
        ),
    }


# ======================================================================
# Public step
# ======================================================================


def run_validation_inference(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    fitted_model_path: Path,
    model_bundle_path: Path,
    run_id: str,
    output_path: Path | None = None,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Run fitted-model inference over all Validation rows.

    No preserve/replace/impute decision is made here.
    """

    start = perf_counter()

    input_path = Path(
        input_path
    )

    fitted_model_path = Path(
        fitted_model_path
    )

    model_bundle_path = Path(
        model_bundle_path
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            "Validation inference input not found: "
            f"{input_path}"
        )

    if not fitted_model_path.is_file():
        raise FileNotFoundError(
            "Fitted Validation model metadata not found: "
            f"{fitted_model_path}"
        )

    if not model_bundle_path.is_file():
        raise FileNotFoundError(
            "Fitted Validation model bundle not found: "
            f"{model_bundle_path}"
        )

    output_path = (
        Path(
            output_path
        )
        if output_path is not None
        else _inference_output_path(
            cfg=cfg
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "%s Run Validation inference",
        get_symbols("build"),
    )

    logger.info(
        "  Input: %s",
        input_path,
    )

    logger.info(
        "  Fitted-model metadata: %s",
        fitted_model_path,
    )

    logger.info(
        "  Production bundle: %s",
        model_bundle_path,
    )

    logger.info(
        "  Output: %s",
        output_path,
    )

    # ------------------------------------------------------------------
    # Fitted model identity
    # ------------------------------------------------------------------

    metadata = (
        _read_fitted_model_metadata(
            fitted_model_path=(
                fitted_model_path
            )
        )
    )

    bundle = _load_bundle(
        model_bundle_path=(
            model_bundle_path
        ),
        model_type=(
            metadata[
                "model_type"
            ]
        ),
    )

    _validate_bundle_identity(
        bundle=bundle,
        metadata=metadata,
    )

    _validate_bundle_feature_contract(
        bundle=bundle,
        cfg=cfg,
        model_type=(
            metadata[
                "model_type"
            ]
        ),
    )

    # ------------------------------------------------------------------
    # Resolve inference population size
    #
    # Do this before loading the complete population into pandas so that
    # an existing reusable inference artifact can return immediately.
    # ------------------------------------------------------------------

    count_con = duckdb.connect()

    try:

        count_con.execute(
            f"PRAGMA threads="
            f"{int(cfg.engine.threads)}"
        )

        input_rows = int(
            count_con.execute(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                """,
                [
                    str(
                        input_path
                    )
                ],
            ).fetchone()[0]
        )

    finally:
        count_con.close()

    if input_rows <= 0:
        raise RuntimeError(
            "Validation inference input contains no rows."
        )

    logger.info(
        "  Inference population rows: %s",
        f"{input_rows:,}",
    )

    logger.info(
        "  Model: %s",
        metadata[
            "model_name"
        ],
    )

    logger.info(
        "  Model type: %s",
        metadata[
            "model_type"
        ],
    )


    # ------------------------------------------------------------------
    # Semantic checkpoint — before loading/predicting full population
    # ------------------------------------------------------------------

    semantic_payload = (
        _inference_semantic_payload(
            cfg=cfg,
            metadata=metadata,
            bundle=bundle,
            input_path=input_path,
            input_rows=input_rows,
        )
    )

    manifest_path = (
        manifest_path_for(
            output_path
        )
    )

    reusable = False
    reuse_reason = "force=True"

    if not force:

        (
            reusable,
            reuse_reason,
        ) = validate_manifest(
            artifact_path=output_path,
            manifest_path=manifest_path,
            expected_payload=semantic_payload,
        )

    if reusable:

        logger.info(
            "%s REUSE — Validation inference (%s)",
            get_symbols("reuse"),
            reuse_reason,
        )

        elapsed = (
            perf_counter()
            - start
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "inference_path": output_path,
            "model_name": metadata["model_name"],
            "model_type": metadata["model_type"],
            "rows": input_rows,
            "duration_seconds": elapsed,
            "reused": True,
        }

    logger.info(
        "%s REBUILD — Validation inference (%s)",
        get_symbols("rebuilt"),
        reuse_reason,
    )
    # ------------------------------------------------------------------
    # Load complete inference population
    # ------------------------------------------------------------------

    target_column = (
        cfg.model.target.column
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
                *feature_columns,
                target_column,
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
            input_path=input_path,
            required_columns=set(
                selected_columns
            ),
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

            FROM read_parquet(?)
            """,
            [
                str(
                    input_path
                )
            ],
        ).df()

    finally:
        con.close()

    if frame.empty:
        raise RuntimeError(
            "Validation inference input contains no rows."
        )

    _validate_model_row_keys(
        frame
    )

    loaded_rows = int(
        len(
            frame
        )
    )

    if loaded_rows != input_rows:
        raise RuntimeError(
            "Validation inference row count changed between "
            "checkpoint inspection and population loading. "
            f"Expected {input_rows:,}; loaded {loaded_rows:,}."
        )

    logger.info(
        "  Inference rows: %s",
        f"{input_rows:,}",
    )

    logger.info(
        "  Rows used in final training sample: %s",
        f"{int(frame['training_selected'].sum()):,}",
    )

    # ------------------------------------------------------------------
    # Determine which rows can actually be evaluated by the model
    # ------------------------------------------------------------------

    model_input_valid = (
        _model_input_valid_mask(
            frame=frame,
            cfg=cfg,
        )
    )

    n_model_input_valid = int(
        model_input_valid.sum()
    )

    n_model_input_invalid = (
        input_rows
        - n_model_input_valid
    )

    logger.info(
        "  Rows with usable model inputs: %s",
        f"{n_model_input_valid:,}",
    )

    logger.info(
        "  Rows without usable model inputs: %s",
        f"{n_model_input_invalid:,}",
    )

    if n_model_input_valid <= 0:
        raise RuntimeError(
            "Validation inference contains no rows "
            "with usable model inputs."
        )

    # ------------------------------------------------------------------
    # Build predictors only for rows where all required numeric inputs
    # are valid.
    # ------------------------------------------------------------------

    predictable_frame = (
        frame.loc[
            model_input_valid
        ]
        .reset_index(
            drop=True
        )
    )

    feature_frame = (
        _build_feature_frame(
            frame=predictable_frame,
            cfg=cfg,
        )
    )

    # ------------------------------------------------------------------
    # Model inference on usable rows only
    # ------------------------------------------------------------------

    model_type = (
        metadata[
            "model_type"
        ]
    )

    if model_type == "embedding_mlp":

        (
            prediction_valid_rows,
            support_flags_valid_rows,
        ) = _predict_embedding_model(
            feature_frame=feature_frame,
            bundle=bundle,
            cfg=cfg,
            logger=logger,
        )

    else:

        (
            prediction_valid_rows,
            support_flags_valid_rows,
        ) = _predict_sklearn_model(
            feature_frame=feature_frame,
            bundle=bundle,
        )

    target_transform = str(
        bundle.get(
            "target_transform",
            cfg.model.target.transform,
        )
    )

    prediction_quantity_valid_rows = (
        _inverse_target_transform(
            prediction=prediction_valid_rows,
            target_transform=target_transform,
        )
    )

    # ------------------------------------------------------------------
    # Expand prediction arrays back to the complete Validation population
    # ------------------------------------------------------------------

    prediction_model_space = np.full(
        input_rows,
        np.nan,
        dtype=float,
    )

    prediction_quantity = np.full(
        input_rows,
        np.nan,
        dtype=float,
    )

    valid_positions = (
        model_input_valid
        .to_numpy(
            dtype=bool
        )
    )

    prediction_model_space[
        valid_positions
    ] = prediction_valid_rows

    prediction_quantity[
        valid_positions
    ] = prediction_quantity_valid_rows

    support_flags = pd.DataFrame(
        pd.NA,
        index=frame.index,
        columns=(
            support_flags_valid_rows.columns
        ),
        dtype="boolean",
    )

    support_flags.loc[
        model_input_valid,
        :
    ] = (
        support_flags_valid_rows
        .to_numpy()
    )
    # ------------------------------------------------------------------
    # Observed-vs-model evidence
    # ------------------------------------------------------------------

    if target_transform != "log":
        raise RuntimeError(
            "Current Validation residual evidence expects "
            "a log-transformed quantity target."
        )

    residual_evidence = (
        _build_residual_evidence(
            observed_quantity=(
                frame[
                    target_column
                ]
            ),
            prediction_log_quantity=(
                prediction_model_space
            ),
        )
    )

    # ------------------------------------------------------------------
    # Assemble inference artifact
    # ------------------------------------------------------------------

    output_columns = (
        _ordered_unique(
            [
                *MODEL_ROW_KEY,
                "target_hs4",
                target_column,
                "training_selected",
            ]
        )
    )

    inference = frame[
        output_columns
    ].copy()

    inference[
        "model_input_valid"
    ] = (
        model_input_valid
        .to_numpy(
            dtype=bool
        )
    )

    inference[
        "model_prediction_log_quantity"
    ] = prediction_model_space

    inference[
        "model_prediction_quantity_kg"
    ] = prediction_quantity

    for column in (
        residual_evidence.columns
    ):

        inference[
            column
        ] = residual_evidence[
            column
        ].to_numpy()

    for column in (
        support_flags.columns
    ):

        inference[
            column
        ] = support_flags[
            column
        ].to_numpy()

    inference[
        "validation_model_name"
    ] = metadata[
        "model_name"
    ]

    inference[
        "validation_model_type"
    ] = metadata[
        "model_type"
    ]

    # ------------------------------------------------------------------
    # Basic inference diagnostics
    # ------------------------------------------------------------------

    observed_rows = int(
        inference[
            "observed_quantity_available"
        ].sum()
    )

    missing_observed_rows = (
        input_rows
        - observed_rows
    )

    logger.info(
        "  Rows with observed canonical quantity: %s",
        f"{observed_rows:,}",
    )

    logger.info(
        "  Rows without observed canonical quantity: %s",
        f"{missing_observed_rows:,}",
    )

    for column in (
        "exporter_seen",
        "importer_seen",
        "hs6_seen",
        "hs4_seen",
    ):

        if column not in inference.columns:
            continue

        support = inference[
            column
        ]

        evaluated_rows = int(
            support.notna().sum()
        )

        seen_rows = int(
            support.fillna(False).sum()
        )

        unseen_rows = (
            evaluated_rows
            - seen_rows
        )

        not_evaluated_rows = (
            input_rows
            - evaluated_rows
        )

        logger.info(
            "  %s — seen=%s, unseen=%s, not_evaluated=%s",
            column,
            f"{seen_rows:,}",
            f"{unseen_rows:,}",
            f"{not_evaluated_rows:,}",
        )


    # ------------------------------------------------------------------
    # Write atomically
    # ------------------------------------------------------------------

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    write_con = duckdb.connect()

    try:

        write_con.register(
            "validation_inference",
            inference,
        )

        write_con.execute(
            """
            COPY validation_inference
            TO ?
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """,
            [
                str(
                    temporary_path
                )
            ],
        )

    finally:
        write_con.close()

    temporary_path.replace(
        output_path
    )

    write_manifest(
        manifest_path=(
            manifest_path
        ),
        artifact_name=(
            "validation_inference"
        ),
        semantic_payload=(
            semantic_payload
        ),
        artifact_path=(
            output_path
        ),
        extra={
            "model_name": (
                metadata[
                    "model_name"
                ]
            ),
            "model_type": (
                metadata[
                    "model_type"
                ]
            ),
            "rows": (
                input_rows
            ),
            "observed_quantity_rows": (
                observed_rows
            ),
            "missing_quantity_rows": (
                missing_observed_rows
            ),
        },
    )

    elapsed = (
        perf_counter()
        - start
    )

    logger.info(
        "%s CHECK — Validation inference complete.",
        get_symbols("check"),
    )

    logger.info(
        "  Rows: %s",
        f"{input_rows:,}",
    )

    logger.info(
        "  Output: %s",
        output_path,
    )

    logger.info(
        "  Duration: %.2f min",
        elapsed / 60.0,
    )

    return {
        "status": "completed",
        "run_id": run_id,
        "inference_path": (
            output_path
        ),
        "model_name": (
            metadata[
                "model_name"
            ]
        ),
        "model_type": (
            metadata[
                "model_type"
            ]
        ),
        "rows": (
            input_rows
        ),
        "observed_quantity_rows": (
            observed_rows
        ),
        "missing_quantity_rows": (
            missing_observed_rows
        ),
        "duration_seconds": (
            elapsed
        ),
        "reused": False,
    }