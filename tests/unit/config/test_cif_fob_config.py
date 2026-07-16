import copy

import pytest

from vacoda.config.cif_fob import CifFobConfig, load_cif_fob_config
from vacoda.config.pipeline import PipelineConfig


def _pipeline_context():
    pipeline_cfg = PipelineConfig.from_yaml()
    return pipeline_cfg.scope, pipeline_cfg.engine


@pytest.mark.parametrize(
    "model_name,estimator_type",
    [
        ("hist_gradient_boosting", "anything_goes"),
        ("xgboost_log_squared", "custom_xgb_variant"),
        ("log_linear_regression", "log_lin"),
    ],
)
def test_cif_fob_config_keeps_estimator_type_as_configured(
    model_name: str,
    estimator_type: str,
) -> None:
    raw = copy.deepcopy(load_cif_fob_config())

    raw["cif_fob"]["method"]["estimation"]["models"]["candidates"][
        model_name
    ]["estimator_type"] = estimator_type

    scope, engine = _pipeline_context()

    cfg = CifFobConfig.from_dict(
        raw,
        scope=scope,
        engine=engine,
    )

    assert (
        cfg.method.estimation.models.candidates[model_name].estimator_type
        == estimator_type
    )


def test_cif_fob_config_parses_model_output_prefixes() -> None:
    raw = copy.deepcopy(load_cif_fob_config())
    scope, engine = _pipeline_context()

    cfg = CifFobConfig.from_dict(
        raw,
        scope=scope,
        engine=engine,
    )

    candidates = cfg.method.estimation.models.candidates

    assert candidates["hist_gradient_boosting"].output_prefix == "hgb"
    assert candidates["xgboost_log_squared"].output_prefix == "xgb_log_sq"
    assert candidates["log_linear_regression"].output_prefix == "ols"


def test_cif_fob_config_keeps_output_prefix_as_configured() -> None:
    raw = copy.deepcopy(load_cif_fob_config())

    raw["cif_fob"]["method"]["estimation"]["models"]["candidates"][
        "hist_gradient_boosting"
    ]["output_prefix"] = "hgb-prefix"

    scope, engine = _pipeline_context()

    cfg = CifFobConfig.from_dict(
        raw,
        scope=scope,
        engine=engine,
    )

    assert (
        cfg.method.estimation.models.candidates[
            "hist_gradient_boosting"
        ].output_prefix
        == "hgb-prefix"
    )


def test_cif_fob_config_parses_model_checkpoints_subdir() -> None:
    raw = copy.deepcopy(load_cif_fob_config())

    raw["cif_fob"]["method"]["estimation"]["models"][
        "model_checkpoints"
    ] = {"subdir": "model_checkpoints"}

    scope, engine = _pipeline_context()

    cfg = CifFobConfig.from_dict(
        raw,
        scope=scope,
        engine=engine,
    )

    assert (
        cfg.method.estimation.models.model_checkpoints.subdir
        == "model_checkpoints"
    )
