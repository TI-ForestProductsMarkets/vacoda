"""
vacoda.steps.validation.model_selection

Select the Validation model that proceeds to production fitting.

The step:

1. reads combined out-of-fold candidate-model predictions;
2. verifies that candidate models were evaluated on identical OOF coverage;
3. computes overall MAE on the configured transformed target scale;
4. selects the candidate with the lowest MAE;
5. writes one reusable selected-model artifact.

This step does not fit the production model.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb

from vacoda.config.steps.validation import ValidationConfig
from vacoda.pipeline.artifact_manifest import (
    manifest_path_for,
    validate_manifest,
    write_manifest,
)
from vacoda.steps.validation.training_sample import (
    training_definition_name,
)
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal


MODEL_SELECTION_CACHE_VERSION = 1


# ======================================================================
# Helpers
# ======================================================================


def _selected_model_semantic_payload(
    *,
    cfg: ValidationConfig,
    predictions_sha256: str,
) -> dict[str, Any]:
    """Return semantic identity of the selected-model artifact."""

    return {
        "artifact": (
            "validation_selected_model"
        ),

        "cache_version": (
            MODEL_SELECTION_CACHE_VERSION
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

        "source": {
            "model_predictions_sha256": (
                predictions_sha256
            ),
        },

        "selection": {
            "metric": (
                "mae_log"
            ),
            "direction": (
                "minimize"
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
    }


def _file_sha256(
    path: Path,
) -> str:
    """Return SHA256 of one artifact file."""

    digest = sha256()

    with path.open("rb") as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def _require_prediction_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
) -> None:
    """Require the prediction columns needed for model selection."""

    rows = con.execute(
        """
        DESCRIBE
        SELECT *
        FROM read_parquet(?)
        """,
        [
            str(
                input_path
            )
        ],
    ).fetchall()

    available = {
        str(row[0])
        for row in rows
    }

    required = {
        "year",
        "exporter_iso",
        "importer_iso",
        "target_cmd",
        "model_name",
        "model_type",
        "held_out_year",
        "target_transformed",
        "prediction_transformed",
    }

    missing = sorted(
        required
        - available
    )

    if missing:
        raise RuntimeError(
            "Validation model-selection input is missing "
            "required column(s): "
            + ", ".join(
                missing
            )
        )


# ======================================================================
# Public step
# ======================================================================


def select_validation_model(
    *,
    cfg: ValidationConfig,
    model_predictions_path: Path,
    run_id: str,
    output_path: Path | None = None,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Select the strongest Validation candidate model.

    Selection criterion
    -------------------
    The enabled candidate with the lowest overall out-of-fold
    mean absolute error on the transformed target scale is selected.

    All candidates must have identical expanding-window OOF coverage.
    """

    start = perf_counter()

    model_predictions_path = Path(
        model_predictions_path
    )

    if not model_predictions_path.is_file():
        raise FileNotFoundError(
            "Validation model predictions not found: "
            f"{model_predictions_path}"
        )

    base_output_path = (
        Path(output_path)
        if output_path is not None
        else cfg.product_path(
            "selected_model"
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "%s Select Validation production-model candidate",
        get_symbols("build"),
    )

    logger.info(
        "  Model predictions: %s",
        model_predictions_path,
    )

    logger.info(
        "  Output: %s",
        output_path,
    )

    predictions_sha256 = (
        _file_sha256(
            model_predictions_path
        )
    )

    semantic_payload = (
        _selected_model_semantic_payload(
            cfg=cfg,
            predictions_sha256=(
                predictions_sha256
            ),
        )
    )

    manifest_path = (
        manifest_path_for(
            output_path
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
        ) = validate_manifest(
            artifact_path=output_path,
            manifest_path=manifest_path,
            expected_payload=(
                semantic_payload
            ),
        )

    if reusable:

        logger.info(
            "%s REUSE — Selected Validation model (%s)",
            get_symbols("reuse"),
            reuse_reason,
        )

        con = duckdb.connect()

        try:

            row = con.execute(
                """
                SELECT
                    model_name,
                    model_type,
                    selection_value

                FROM read_parquet(?)
                """,
                [
                    str(
                        output_path
                    )
                ],
            ).fetchone()

        finally:
            con.close()

        if row is None:
            raise RuntimeError(
                "Reusable selected-model artifact is empty."
            )

        selected_model_name = str(
            row[0]
        )

        selected_model_type = str(
            row[1]
        )

        selection_value = float(
            row[2]
        )

        elapsed = (
            perf_counter()
            - start
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "selected_model_path": (
                output_path
            ),
            "selected_model_name": (
                selected_model_name
            ),
            "selected_model_type": (
                selected_model_type
            ),
            "selection_metric": (
                "mae_log"
            ),
            "selection_value": (
                selection_value
            ),
            "duration_seconds": (
                elapsed
            ),
            "reused": True,
        }

    logger.info(
        "%s REBUILD — Selected Validation model (%s)",
        get_symbols("rebuilt"),
        reuse_reason,
    )

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

        _require_prediction_columns(
            con=con,
            input_path=(
                model_predictions_path
            ),
        )

        predictions_sql = sql_literal(
            model_predictions_path
        )

        # --------------------------------------------------------------
        # Candidate evaluation summary
        # --------------------------------------------------------------

        candidate_summary = con.execute(
            f"""
            SELECT
                model_name,
                model_type,

                COUNT(*)
                    AS n_predictions,

                COUNT(
                    DISTINCT held_out_year
                )
                    AS n_validation_years,

                MIN(held_out_year)
                    AS min_validation_year,

                MAX(held_out_year)
                    AS max_validation_year,

                AVG(
                    ABS(
                        prediction_transformed
                        - target_transformed
                    )
                )
                    AS mae_log

            FROM read_parquet(
                {predictions_sql}
            )

            GROUP BY
                model_name,
                model_type

            ORDER BY
                mae_log ASC,
                model_name ASC
            """
        ).df()

        if candidate_summary.empty:
            raise RuntimeError(
                "Validation model-selection input contains "
                "no candidate predictions."
            )

        # --------------------------------------------------------------
        # Require identical OOF coverage counts / years
        # --------------------------------------------------------------

        coverage_columns = [
            "n_predictions",
            "n_validation_years",
            "min_validation_year",
            "max_validation_year",
        ]

        for column in coverage_columns:

            if (
                candidate_summary[
                    column
                ].nunique(
                    dropna=False
                )
                != 1
            ):

                raise RuntimeError(
                    "Validation candidate models do not have "
                    "identical OOF evaluation coverage. "
                    f"Column {column!r} differs between models."
                )

        # --------------------------------------------------------------
        # Stronger coverage check:
        # require exact same canonical OOF row keys for every candidate.
        # --------------------------------------------------------------

        coverage_check = con.execute(
            f"""
            WITH row_coverage AS (

                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    held_out_year,

                    COUNT(
                        DISTINCT model_name
                    )
                        AS n_models

                FROM read_parquet(
                    {predictions_sql}
                )

                GROUP BY
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    held_out_year
            ),

            model_count AS (

                SELECT
                    COUNT(
                        DISTINCT model_name
                    )
                        AS n_models

                FROM read_parquet(
                    {predictions_sql}
                )
            )

            SELECT
                COUNT(*)

            FROM row_coverage
            CROSS JOIN model_count

            WHERE
                row_coverage.n_models
                != model_count.n_models
            """
        ).fetchone()

        n_incomplete_rows = int(
            coverage_check[0]
        )

        if n_incomplete_rows != 0:
            raise RuntimeError(
                "Validation candidate models were not evaluated "
                "on identical canonical OOF rows. "
                f"{n_incomplete_rows:,} row keys have incomplete "
                "candidate-model coverage."
            )

        # --------------------------------------------------------------
        # Select minimum overall OOF MAE
        # --------------------------------------------------------------

        selected = (
            candidate_summary.iloc[0]
        )

        selected_model_name = str(
            selected[
                "model_name"
            ]
        )

        selected_model_type = str(
            selected[
                "model_type"
            ]
        )

        selection_value = float(
            selected[
                "mae_log"
            ]
        )

        n_predictions = int(
            selected[
                "n_predictions"
            ]
        )

        n_validation_years = int(
            selected[
                "n_validation_years"
            ]
        )

        min_validation_year = int(
            selected[
                "min_validation_year"
            ]
        )

        max_validation_year = int(
            selected[
                "max_validation_year"
            ]
        )

        # --------------------------------------------------------------
        # Require winner to still exist in configured enabled models
        # --------------------------------------------------------------

        enabled_models = (
            cfg.model.models.enabled()
        )

        if (
            selected_model_name
            not in enabled_models
        ):
            raise RuntimeError(
                "Selected Validation model is not present among "
                "the currently enabled candidate models: "
                f"{selected_model_name!r}."
            )

        configured_type = str(
            enabled_models[
                selected_model_name
            ].type
        )

        if (
            configured_type
            != selected_model_type
        ):
            raise RuntimeError(
                "Selected Validation model type does not match "
                "the current configuration. "
                f"Predictions: {selected_model_type!r}; "
                f"configuration: {configured_type!r}."
            )

        # --------------------------------------------------------------
        # Write one-row selection artifact
        # --------------------------------------------------------------

        temporary_sql = sql_literal(
            temporary_path
        )

        selected_name_sql = sql_literal(
            selected_model_name
        )

        selected_type_sql = sql_literal(
            selected_model_type
        )

        run_id_sql = sql_literal(
            run_id
        )

        con.execute(
            f"""
            COPY (
                SELECT
                    {selected_name_sql}
                        AS model_name,

                    {selected_type_sql}
                        AS model_type,

                    'mae_log'
                        AS selection_metric,

                    'minimize'
                        AS selection_direction,

                    {selection_value}
                        ::DOUBLE
                        AS selection_value,

                    {n_predictions}
                        ::BIGINT
                        AS n_predictions,

                    {n_validation_years}
                        ::INTEGER
                        AS n_validation_years,

                    {min_validation_year}
                        ::INTEGER
                        AS min_validation_year,

                    {max_validation_year}
                        ::INTEGER
                        AS max_validation_year,

                    {run_id_sql}
                        AS run_id
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

    finally:

        con.close()

        if temporary_path.exists():
            temporary_path.unlink()

    write_manifest(
        manifest_path=manifest_path,
        artifact_name=(
            "validation_selected_model"
        ),
        semantic_payload=(
            semantic_payload
        ),
        artifact_path=(
            output_path
        ),
        extra={
            "selected_model_name": (
                selected_model_name
            ),
            "selected_model_type": (
                selected_model_type
            ),
            "selection_metric": (
                "mae_log"
            ),
            "selection_value": (
                selection_value
            ),
        },
    )

    elapsed = (
        perf_counter()
        - start
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
        "  Selection metric: MAE_log",
    )

    logger.info(
        "  Selection value: %.6f",
        selection_value,
    )

    logger.info(
        "  OOF rows: %s",
        f"{n_predictions:,}",
    )

    logger.info(
        "  Validation years: %s-%s",
        min_validation_year,
        max_validation_year,
    )

    logger.info(
        "  Output: %s",
        output_path,
    )

    logger.info(
        "%s CHECK — Validation model selection complete.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,

        "selected_model_path": (
            output_path
        ),

        "selected_model_name": (
            selected_model_name
        ),

        "selected_model_type": (
            selected_model_type
        ),

        "selection_metric": (
            "mae_log"
        ),

        "selection_value": (
            selection_value
        ),

        "n_predictions": (
            n_predictions
        ),

        "n_validation_years": (
            n_validation_years
        ),

        "duration_seconds": (
            elapsed
        ),

        "reused": False,
    }