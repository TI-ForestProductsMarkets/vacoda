"""
vacoda.steps.release.prepare_input
-----------------------------
Resolve the two validated inputs required by VACODA Release.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from vacoda.config.steps.release import ReleaseConfig
from vacoda.config.steps.validation import ValidationConfig
from vacoda.steps.validation.training_sample import training_definition_name


def resolve_release_inputs(
    *,
    cfg: ReleaseConfig,
    validation_cfg: ValidationConfig,
) -> dict[str, Any]:
    """Resolve Release inputs using YAML and Validation's training identity."""
    if cfg.scope != validation_cfg.scope:
        raise ValueError("Release and Validation scopes do not match.")

    training_definition = training_definition_name(cfg=validation_cfg)
    prepared_path = cfg.prepared_input_path()
    inference_path = cfg.inference_input_path(
        training_definition=training_definition
    )
    return {
        "prepared_input_path": prepared_path,
        "inference_input_path": inference_path,
        "training_definition": training_definition,
    }


def _require_parquet(label: str, path: Path) -> None:
    """Check file presence and Parquet magic bytes without reading all rows."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} parquet not found: {path}")
    if path.stat().st_size < 12:
        raise ValueError(f"{label} parquet is empty or truncated: {path}")
    with path.open("rb") as stream:
        header = stream.read(4)
        stream.seek(-4, 2)
        footer = stream.read(4)
    if header != b"PAR1" or footer != b"PAR1":
        raise ValueError(f"{label} is not a valid Parquet container: {path}")


def prepare_release_input(
    *,
    cfg: ReleaseConfig,
    validation_cfg: ValidationConfig,
    run_id: str,
    logger,
) -> dict[str, Any]:
    """Resolve and verify the two Validation artifacts consumed by Release."""
    start = perf_counter()
    resolved = resolve_release_inputs(cfg=cfg, validation_cfg=validation_cfg)
    prepared_path = resolved["prepared_input_path"]
    inference_path = resolved["inference_input_path"]

    _require_parquet("Validation Prepared", prepared_path)
    _require_parquet("Validation Inference", inference_path)

    logger.info("Release input preparation complete.")
    logger.info("  Training definition: %s", resolved["training_definition"])
    logger.info("  Validation Prepared: %s", prepared_path)
    logger.info("  Validation Inference: %s", inference_path)

    return {
        "status": "completed",
        "run_id": run_id,
        "prepared_input_path": prepared_path,
        "inference_input_path": inference_path,
        "training_definition": resolved["training_definition"],
        "duration_seconds": perf_counter() - start,
    }
