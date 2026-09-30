import pytest

from vacoda.config.pipeline import (
    PipelineEngineConfig,
    PipelineScopeConfig,
)
from vacoda.config.steps.validation import (
    ValidationConfig,
    load_validation_config,
)


@pytest.fixture
def validation_scope() -> PipelineScopeConfig:
    return PipelineScopeConfig(
        target_hs="4401",
        min_year=2022,
        max_year=2025,
    )


@pytest.fixture
def validation_engine() -> PipelineEngineConfig:
    return PipelineEngineConfig(
        name="duckdb",
        threads=4,
    )


def test_validation_config_from_yaml_parses_target_hs_rendering(
    validation_scope: PipelineScopeConfig,
    validation_engine: PipelineEngineConfig,
):
    raw = load_validation_config()

    cfg = ValidationConfig.from_dict(
        raw,
        scope=validation_scope,
        engine=validation_engine,
    )

    assert cfg.input.filename == "comtrade_target_4401_value_deflated.parquet"
    assert cfg.products.training_sample.filename == "training/comtrade_target_4401_training_sample.parquet"
    assert cfg.products.release.filename == "vacoda_target_4401_trade.parquet"
    assert cfg.model.target.column == "quantity_selected_kg"


def test_validation_config_rejects_non_parquet_input_format(
    validation_scope: PipelineScopeConfig,
    validation_engine: PipelineEngineConfig,
):
    raw = load_validation_config()
    raw["validation"]["input"]["format"] = "csv"

    with pytest.raises(ValueError, match="currently must be 'parquet'"):
        ValidationConfig.from_dict(
            raw,
            scope=validation_scope,
            engine=validation_engine,
        )
