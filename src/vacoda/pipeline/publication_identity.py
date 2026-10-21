
"""Build the readable VACODA Release publication identity."""

from pathlib import Path
import re

import duckdb

from vacoda.steps.validation.training_sample import training_definition_name


def _component(value: str) -> str:
    value = str(value).strip().lower()

    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value):
        raise ValueError(f"Invalid publication component: {value!r}")

    return value


def release_identity(
    *,
    cif_fob_cfg,
    trade_sync_cfg,
    validation_cfg,
) -> str:

    estimation = cif_fob_cfg.method.estimation

    sources = estimation.training_factor_sources
    approach = "_".join(_component(source) for source in sources)

    cif_model = _component(estimation.models.primary)

    # Current Trade Sync quantity selection.
    # Replace with a resolved config property when configurable.
    quantity_method = "value_weighted"

    training_name = training_definition_name(cfg=validation_cfg)

    selected_base = validation_cfg.product_path("selected_model")

    selected_path = (
        selected_base.parent
        / training_name
        / selected_base.name
    )

    if not selected_path.is_file():
        raise FileNotFoundError(
            f"Validation selected-model artifact missing: {selected_path}"
        )

    with duckdb.connect() as con:
        rows = con.execute(
            """
            SELECT model_name
            FROM read_parquet(?)
            """,
            [str(selected_path)],
        ).fetchall()

    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one selected Validation model."
        )

    model_name = _component(rows[0][0])

    return str(
        Path(f"cif_fob__{approach}__{cif_model}")
        / f"trade_sync__{quantity_method}"
        / f"validation__{training_name}"
        / f"model__{model_name}"
    )
