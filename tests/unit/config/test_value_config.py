from pathlib import Path

import pytest

from vacoda.config.pipeline import (
    PipelineEngineConfig,
    PipelineScopeConfig,
)
from vacoda.config.steps.value import ValueConfig


@pytest.fixture
def scope() -> PipelineScopeConfig:
    return PipelineScopeConfig(
        target_hs="H5",
        min_year=2017,
        max_year=2025,
    )


@pytest.fixture
def engine() -> PipelineEngineConfig:
    return PipelineEngineConfig(
        name="local",
        threads=16,
    )


def test_value_config_loads(
    scope: PipelineScopeConfig,
    engine: PipelineEngineConfig,
) -> None:
    cfg = ValueConfig.from_yaml(
        scope=scope,
        engine=engine,
    )

    assert cfg.method.reference_year == 2025
    assert cfg.method.preserve_nominal_values is True
    assert cfg.input.columns.year == "year"
    assert (
        cfg.input.columns.nominal_value
        == "trade_value_reconciled"
    )

    assert (
        cfg.reference_data.us_gdp_deflator.country_code
        == "USA"
    )

    assert (
        cfg.reference_data.us_gdp_deflator.indicator_code
        == "NY.GDP.DEFL.ZS"
    )


def test_value_config_renders_target_hs(
    scope: PipelineScopeConfig,
    engine: PipelineEngineConfig,
) -> None:
    cfg = ValueConfig.from_yaml(
        scope=scope,
        engine=engine,
    )

    assert cfg.input.filename == (
        "comtrade_target_h5_trade_synced.parquet"
    )

    assert cfg.output.subdir == "value/target_h5"

    assert cfg.products.deflated_trade.filename == (
        "comtrade_target_h5_value_deflated.parquet"
    )


def test_value_reference_paths(
    scope: PipelineScopeConfig,
    engine: PipelineEngineConfig,
) -> None:
    cfg = ValueConfig.from_yaml(
        scope=scope,
        engine=engine,
    )

    assert cfg.deflator_source_path().name == (
        "API_NY.GDP.DEFL.ZS_DS2_en_csv_v2_38103.csv"
    )

    assert cfg.deflator_prepared_path().name == (
        "us_gdp_deflator_annual.parquet"
    )

    assert (
        cfg.deflator_source_path().as_posix().endswith(
            "references/macroeconomic/us_gdp_deflator/source/"
            "API_NY.GDP.DEFL.ZS_DS2_en_csv_v2_38103.csv"
        )
    )

    assert (
        cfg.deflator_prepared_path().as_posix().endswith(
            "references/macroeconomic/us_gdp_deflator/prepared/"
            "us_gdp_deflator_annual.parquet"
        )
    )


def test_value_product_paths(
    scope: PipelineScopeConfig,
    engine: PipelineEngineConfig,
) -> None:
    cfg = ValueConfig.from_yaml(
        scope=scope,
        engine=engine,
    )

    assert cfg.product_path("deflated_trade").name == (
        "comtrade_target_h5_value_deflated.parquet"
    )

    assert cfg.product_path("audit").name == (
        "comtrade_target_h5_value_deflated.audit.csv"
    )


def test_value_report_paths(
    scope: PipelineScopeConfig,
    engine: PipelineEngineConfig,
) -> None:
    cfg = ValueConfig.from_yaml(
        scope=scope,
        engine=engine,
    )

    assert cfg.artifacts_path("TEST").name == (
        "target_h5_value_decision_report_TEST.md"
    )

    assert cfg.artifacts_latest_path().name == (
        "target_h5_value_decision_report_latest.md"
    )


def test_unknown_value_product_raises(
    scope: PipelineScopeConfig,
    engine: PipelineEngineConfig,
) -> None:
    cfg = ValueConfig.from_yaml(
        scope=scope,
        engine=engine,
    )

    with pytest.raises(KeyError):
        cfg.product_path("does_not_exist")