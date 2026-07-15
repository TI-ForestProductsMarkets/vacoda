"""
vacoda.config.cif_fob

Config reader for configs/steps/cif_fob.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vacoda.config.base import (
    get_step_config_path,
    load_yaml_config,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_str,
)
from vacoda.config.paths import get_data_dir, get_output_dir, get_repo_references_dir
from vacoda.config.pipeline import PipelineEngineConfig, PipelineScopeConfig


CIF_FOB_CONFIG_PATH = get_step_config_path("cif_fob")


def load_cif_fob_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_yaml_config(config_path or CIF_FOB_CONFIG_PATH)

def _render_value(value: Any, *, target_hs: str) -> Any:
    if isinstance(value, dict):
        return {k: _render_value(v, target_hs=target_hs) for k, v in value.items()}

    if isinstance(value, list):
        return [_render_value(item, target_hs=target_hs) for item in value]

    if isinstance(value, str):
        return value.format(
            target_hs=target_hs.upper(),
            target_hs_upper=target_hs.upper(),
            target_hs_lower=target_hs.lower(),
        )

    return value


@dataclass(frozen=True)
class CifFobInputConfig:
    stage: str
    subdir: str
    filename: str
    format: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobInputConfig":
        return cls(
            stage=require_str(cfg.get("stage"), key="cif_fob.input.stage").strip().lower(),
            subdir=require_str(cfg.get("subdir"), key="cif_fob.input.subdir"),
            filename=require_str(cfg.get("filename"), key="cif_fob.input.filename"),
            format=require_str(cfg.get("format"), key="cif_fob.input.format").strip().lower(),
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

    def resolve_path(self) -> Path:
        return self.resolve_dir() / self.filename


@dataclass(frozen=True)
class CifFobOutputConfig:
    stage: str
    subdir: str
    format: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobOutputConfig":
        return cls(
            stage=require_str(cfg.get("stage"), key="cif_fob.output.stage").strip().lower(),
            subdir=require_str(cfg.get("subdir"), key="cif_fob.output.subdir"),
            format=require_str(cfg.get("format"), key="cif_fob.output.format").strip().lower(),
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base


@dataclass(frozen=True)
class CifFobLoggingConfig:
    level: str
    subdir: str
    file: str
    timestamp: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobLoggingConfig":
        return cls(
            level=require_str(
                cfg.get("level"),
                key="cif_fob.logging.level",
            ).strip().upper(),
            subdir=require_str(
                cfg.get("subdir"),
                key="cif_fob.logging.subdir",
                allow_empty=True,
            ),
            file=require_str(
                cfg.get("file"),
                key="cif_fob.logging.file",
            ),
            timestamp=require_bool(
                cfg.get("timestamp"),
                key="cif_fob.logging.timestamp",
            ),
        )

    def resolve_log_dir(self) -> Path:
        base = get_output_dir("logs")
        if not self.subdir:
            return base

        subdir_path = Path(self.subdir)
        parts = list(subdir_path.parts)

        # Treat "logs/..." as already anchored at output.logs root.
        if parts and parts[0].lower() == "logs":
            parts = parts[1:]

        return base / Path(*parts) if parts else base

    def resolve_log_file(self) -> Path:
        return self.resolve_log_dir() / self.file


@dataclass(frozen=True)
class CifFobSupportedFlowsConfig:
    import_code: str
    export_code: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobSupportedFlowsConfig":
        return cls(
            import_code=require_str(
                cfg.get("import"),
                key="cif_fob.source_data.supported_flows.import",
            ),
            export_code=require_str(
                cfg.get("export"),
                key="cif_fob.source_data.supported_flows.export",
            ),
        )


@dataclass(frozen=True)
class CifFobSourceColumnsConfig:
    year: str
    reporter: str
    partner: str
    flow: str
    commodity: str
    is_reported: str
    is_aggregate: str
    canonical_value: str
    source_value: str
    direct_cif_value: str
    direct_fob_value: str
    allocation_weight: str
    quantity: str
    quantity_unit: str
    quantity_estimated: str

    alt_quantity: str
    alt_quantity_unit: str
    alt_quantity_estimated: str

    net_weight: str
    net_weight_estimated: str

    gross_weight: str
    gross_weight_estimated: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobSourceColumnsConfig":
        return cls(
            year=require_str(cfg.get("year"), key="cif_fob.source_data.columns.year"),
            reporter=require_str(cfg.get("reporter"), key="cif_fob.source_data.columns.reporter"),
            partner=require_str(cfg.get("partner"), key="cif_fob.source_data.columns.partner"),
            flow=require_str(cfg.get("flow"), key="cif_fob.source_data.columns.flow"),
            commodity=require_str(cfg.get("commodity"), key="cif_fob.source_data.columns.commodity"),
            is_reported=require_str(cfg.get("is_reported"), key="cif_fob.source_data.columns.is_reported"),
            is_aggregate=require_str(cfg.get("is_aggregate"), key="cif_fob.source_data.columns.is_aggregate"),
            canonical_value=require_str(
                cfg.get("canonical_value"),
                key="cif_fob.source_data.columns.canonical_value",
            ),
            source_value=require_str(
                cfg.get("source_value"),
                key="cif_fob.source_data.columns.source_value",
            ),
            direct_cif_value=require_str(
                cfg.get("direct_cif_value"),
                key="cif_fob.source_data.columns.direct_cif_value",
            ),
            direct_fob_value=require_str(
                cfg.get("direct_fob_value"),
                key="cif_fob.source_data.columns.direct_fob_value",
            ),
            allocation_weight=require_str(
                cfg.get("allocation_weight"),
                key="cif_fob.source_data.columns.allocation_weight",
            ),
            quantity=require_str(cfg.get("quantity"), key="cif_fob.source_data.columns.quantity"),
            quantity_unit=require_str(cfg.get("quantity_unit"), key="cif_fob.source_data.columns.quantity_unit"),
            quantity_estimated=require_str(cfg.get("quantity_estimated"), key="cif_fob.source_data.columns.quantity_estimated"),

            alt_quantity=require_str(cfg.get("alt_quantity"), key="cif_fob.source_data.columns.alt_quantity"),
            alt_quantity_unit=require_str(cfg.get("alt_quantity_unit"), key="cif_fob.source_data.columns.alt_quantity_unit"),
            alt_quantity_estimated=require_str(cfg.get("alt_quantity_estimated"), key="cif_fob.source_data.columns.alt_quantity_estimated"),

            net_weight=require_str(cfg.get("net_weight"), key="cif_fob.source_data.columns.net_weight"),
            net_weight_estimated=require_str(cfg.get("net_weight_estimated"), key="cif_fob.source_data.columns.net_weight_estimated"),

            gross_weight=require_str(cfg.get("gross_weight"), key="cif_fob.source_data.columns.gross_weight"),
            gross_weight_estimated=require_str(cfg.get("gross_weight_estimated"), key="cif_fob.source_data.columns.gross_weight_estimated"),
        )


@dataclass(frozen=True)
class CifFobSourceDataConfig:
    supported_flows: CifFobSupportedFlowsConfig
    columns: CifFobSourceColumnsConfig
    allocate_supplementary_measures: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobSourceDataConfig":
        return cls(
            supported_flows=CifFobSupportedFlowsConfig.from_dict(
                require_dict(
                    cfg.get("supported_flows"),
                    key="cif_fob.source_data.supported_flows",
                )
            ),
            columns=CifFobSourceColumnsConfig.from_dict(
                require_dict(cfg.get("columns"), key="cif_fob.source_data.columns")
            ),
            allocate_supplementary_measures=require_bool(
                cfg.get("allocate_supplementary_measures"),
                key="cif_fob.source_data.allocate_supplementary_measures",
            ),
        )


@dataclass(frozen=True)
class CifFobProductConfig:
    enable: bool
    filename: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, key: str) -> "CifFobProductConfig":
        return cls(
            enable=require_bool(cfg.get("enable"), key=f"{key}.enable"),
            filename=require_str(cfg.get("filename"), key=f"{key}.filename"),
        )


@dataclass(frozen=True)
class CifFobProductsConfig:
    prepared_input: CifFobProductConfig
    mirror_pairs: CifFobProductConfig
    training_sample: CifFobProductConfig
    training_features: CifFobProductConfig
    training_evidence_audit: CifFobProductConfig
    estimated_rates: CifFobProductConfig
    temporal_validation_summary: CifFobProductConfig
    temporal_validation_folds: CifFobProductConfig
    upper_tail_audit_summary: CifFobProductConfig
    upper_tail_route_history: CifFobProductConfig
    upper_tail_observations: CifFobProductConfig
    diagnostics: CifFobProductConfig
    corrected_trade: CifFobProductConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobProductsConfig":
        return cls(
            prepared_input=CifFobProductConfig.from_dict(
                require_dict(cfg.get("prepared_input"), key="cif_fob.products.prepared_input"),
                key="cif_fob.products.prepared_input",
            ),
            mirror_pairs=CifFobProductConfig.from_dict(
                require_dict(cfg.get("mirror_pairs"), key="cif_fob.products.mirror_pairs"),
                key="cif_fob.products.mirror_pairs",
            ),
            training_sample=CifFobProductConfig.from_dict(
                require_dict(cfg.get("training_sample"), key="cif_fob.products.training_sample"),
                key="cif_fob.products.training_sample",
            ),
            training_features=CifFobProductConfig.from_dict(
                require_dict(cfg.get("training_features"), key="cif_fob.products.training_features"),
                key="cif_fob.products.training_features",
            ),
            training_evidence_audit=CifFobProductConfig.from_dict(
                require_dict(cfg.get("training_evidence_audit"), key="cif_fob.products.training_evidence_audit"),
                key="cif_fob.products.training_evidence_audit",
            ),
            estimated_rates=CifFobProductConfig.from_dict(
                require_dict(cfg.get("estimated_rates"), key="cif_fob.products.estimated_rates"),
                key="cif_fob.products.estimated_rates",
            ),
            corrected_trade=CifFobProductConfig.from_dict(
                require_dict(cfg.get("corrected_trade"), key="cif_fob.products.corrected_trade"),
                key="cif_fob.products.corrected_trade",
            ),
            temporal_validation_summary=CifFobProductConfig.from_dict(
                require_dict(cfg.get("temporal_validation_summary"), key="cif_fob.products.temporal_validation_summary"),
                key="cif_fob.products.temporal_validation_summary",
            ),
            temporal_validation_folds=CifFobProductConfig.from_dict(
                require_dict(cfg.get("temporal_validation_folds"), key="cif_fob.products.temporal_validation_folds"),
                key="cif_fob.products.temporal_validation_folds",
            ),
            upper_tail_audit_summary=CifFobProductConfig.from_dict(
                require_dict(cfg.get("upper_tail_audit_summary"), key="cif_fob.products.upper_tail_audit_summary"),
                key="cif_fob.products.upper_tail_audit_summary",
            ),
            upper_tail_route_history=CifFobProductConfig.from_dict(
                require_dict(cfg.get("upper_tail_route_history"), key="cif_fob.products.upper_tail_route_history"),
                key="cif_fob.products.upper_tail_route_history",
            ),
            upper_tail_observations=CifFobProductConfig.from_dict(
                require_dict(cfg.get("upper_tail_observations"), key="cif_fob.products.upper_tail_observations"),
                key="cif_fob.products.upper_tail_observations",
            ),
            diagnostics=CifFobProductConfig.from_dict(
                require_dict(cfg.get("diagnostics"), key="cif_fob.products.diagnostics"),
                key="cif_fob.products.diagnostics",
            ),
        )


@dataclass(frozen=True)
class CifFobTrainingConfig:
    minimum_quantity_concordance: float
    minimum_direct_value_coverage: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobTrainingConfig":
        return cls(
            minimum_quantity_concordance=require_float(
                cfg.get("minimum_quantity_concordance"),
                key="cif_fob.method.training.minimum_quantity_concordance",
                min_value=0.0,
                max_value=1.0,
            ),
            minimum_direct_value_coverage=require_float(
                cfg.get("minimum_direct_value_coverage"),
                key="cif_fob.method.training.minimum_direct_value_coverage",
                min_value=0.0,
                max_value=1.0,
            ),
        )


def _require_str_list(value: Any, *, key: str) -> list[str]:
    if value is None:
        raise TypeError(f"{key} must be a list")

    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return [require_str(item, key=key) for item in value]


def _require_output_prefix(*, value: Any, key: str) -> str:
    """Return a normalized, SQL-safe model output prefix."""
    prefix = require_str(value, key=key).strip().lower()
    return prefix


@dataclass(frozen=True)
class CifFobFeatureInputConfig:
    stage: str | None
    subdir: str
    source: str | None
    path: str | None
    filename: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, key: str) -> "CifFobFeatureInputConfig":
        filename = require_str(cfg.get("filename"), key=f"{key}.filename")

        if cfg.get("stage") is not None:
            return cls(
                stage=require_str(cfg.get("stage"), key=f"{key}.stage").strip().lower(),
                subdir=require_str(cfg.get("subdir"), key=f"{key}.subdir"),
                source=None,
                path=None,
                filename=filename,
            )

        if cfg.get("source") is not None:
            return cls(
                stage=None,
                subdir="",
                source=require_str(cfg.get("source"), key=f"{key}.source").strip().lower(),
                path=require_str(cfg.get("path"), key=f"{key}.path"),
                filename=filename,
            )

        raise TypeError(
            f"{key} must define either stage/subdir or source/path."
        )

    def resolve_dir(self) -> Path:
        if self.stage is not None:
            base = get_data_dir(self.stage)
            return base / self.subdir if self.subdir else base

        source = (self.source or "").strip().lower()
        if source == "repo":
            base = get_repo_references_dir()
        elif source == "data":
            base = get_data_dir("references")
        else:
            raise ValueError(
                "cif_fob feature source must be 'repo' or 'data', "
                f"got {self.source!r}."
            )

        path = Path(self.path or "")
        if path.parts and path.parts[0] == "references":
            path = Path(*path.parts[1:])

        return base / path if path.parts else base

    def resolve_path(self) -> Path:
        return self.resolve_dir() / self.filename


@dataclass(frozen=True)
class CifFobGeographyCrosswalkConfig:
    input: CifFobFeatureInputConfig
    allow_proxy_mappings: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "CifFobGeographyCrosswalkConfig":
        return cls(
            input=CifFobFeatureInputConfig.from_dict(
                cfg,
                key=key,
            ),
            allow_proxy_mappings=require_bool(
                cfg.get("allow_proxy_mappings"),
                key=f"{key}.allow_proxy_mappings",
            ),
        )

    def resolve_path(self) -> Path:
        return self.input.resolve_path()


@dataclass(frozen=True)
class CifFobProductMedianUnitValueConfig:
    method: str
    minimum_observations: int
    fallback: list[str]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobProductMedianUnitValueConfig":
        return cls(
            method=require_str(
                cfg.get("method"),
                key="cif_fob.features.product.median_unit_value.method",
            ),
            minimum_observations=require_int(
                cfg.get("minimum_observations"),
                key="cif_fob.features.product.median_unit_value.minimum_observations",
                min_value=1,
            ),
            fallback=_require_str_list(
                cfg.get("fallback"),
                key="cif_fob.features.product.median_unit_value.fallback",
            ),
        )


@dataclass(frozen=True)
class CifFobProductFeaturesConfig:
    group_digits: int
    median_unit_value: CifFobProductMedianUnitValueConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobProductFeaturesConfig":
        return cls(
            group_digits=require_int(
                cfg.get("group_digits"),
                key="cif_fob.features.product.group_digits",
                min_value=1,
            ),
            median_unit_value=CifFobProductMedianUnitValueConfig.from_dict(
                require_dict(
                    cfg.get("median_unit_value"),
                    key="cif_fob.features.product.median_unit_value",
                )
            ),
        )


@dataclass(frozen=True)
class CifFobFeaturesConfig:
    geography: CifFobFeatureInputConfig
    pair_geography: CifFobFeatureInputConfig
    geography_crosswalk: CifFobGeographyCrosswalkConfig
    pair_geography_crosswalk: CifFobGeographyCrosswalkConfig
    oil_price: CifFobFeatureInputConfig
    product: CifFobProductFeaturesConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobFeaturesConfig":
        return cls(
            geography=CifFobFeatureInputConfig.from_dict(
                require_dict(
                    cfg.get("geography"),
                    key="cif_fob.features.geography",
                ),
                key="cif_fob.features.geography",
            ),
            pair_geography=CifFobFeatureInputConfig.from_dict(
                require_dict(
                    cfg.get("pair_geography"),
                    key="cif_fob.features.pair_geography",
                ),
                key="cif_fob.features.pair_geography",
            ),
            geography_crosswalk=CifFobGeographyCrosswalkConfig.from_dict(
                require_dict(
                    cfg.get("geography_crosswalk"),
                    key="cif_fob.features.geography_crosswalk",
                ),
                key="cif_fob.features.geography_crosswalk",
            ),
            pair_geography_crosswalk=CifFobGeographyCrosswalkConfig.from_dict(
                require_dict(
                    cfg.get("pair_geography_crosswalk"),
                    key="cif_fob.features.pair_geography_crosswalk",
                ),
                key="cif_fob.features.pair_geography_crosswalk",
            ),
            oil_price=CifFobFeatureInputConfig.from_dict(
                require_dict(
                    cfg.get("oil_price"),
                    key="cif_fob.features.oil_price",
                ),
                key="cif_fob.features.oil_price",
            ),
            product=CifFobProductFeaturesConfig.from_dict(
                require_dict(
                    cfg.get("product"),
                    key="cif_fob.features.product",
                )
            ),
        )



@dataclass(frozen=True)
class CifFobEstimationSourceWeightsConfig:
    direct_reported_weight: float
    direct_estimated_weight: float
    mirror_reported_weight: float
    mirror_estimated_weight: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobEstimationSourceWeightsConfig":
        return cls(
            direct_reported_weight=require_float(
                cfg.get("direct_reported_weight"),
                key=(
                    "cif_fob.method.estimation.source_weights."
                    "direct_reported_weight"
                ),
                min_value=0.0,
            ),
            direct_estimated_weight=require_float(
                cfg.get("direct_estimated_weight"),
                key=(
                    "cif_fob.method.estimation.source_weights."
                    "direct_estimated_weight"
                ),
                min_value=0.0,
            ),
            mirror_reported_weight=require_float(
                cfg.get("mirror_reported_weight"),
                key=(
                    "cif_fob.method.estimation.source_weights."
                    "mirror_reported_weight"
                ),
                min_value=0.0,
            ),
            mirror_estimated_weight=require_float(
                cfg.get("mirror_estimated_weight"),
                key=(
                    "cif_fob.method.estimation.source_weights."
                    "mirror_estimated_weight"
                ),
                min_value=0.0,
            ),
        )


@dataclass(frozen=True)
class CifFobRepresentationValueBandConfig:
    enabled: bool
    boundaries_usd: tuple[int, ...]
    minimum_weight: float
    maximum_weight: float
    normalize_mean_to_one: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobRepresentationValueBandConfig":
        boundaries_raw = cfg.get("boundaries_usd")

        if not isinstance(boundaries_raw, list):
            raise TypeError(
                "cif_fob.method.estimation.representation_weighting."
                "value_band.boundaries_usd must be a list."
            )

        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key=(
                    "cif_fob.method.estimation.representation_weighting."
                    "value_band.enabled"
                ),
            ),
            boundaries_usd=tuple(
                require_int(
                    value,
                    key=(
                        "cif_fob.method.estimation.representation_weighting."
                        "value_band.boundaries_usd"
                    ),
                    min_value=0,
                )
                for value in boundaries_raw
            ),
            minimum_weight=require_float(
                cfg.get("minimum_weight"),
                key=(
                    "cif_fob.method.estimation.representation_weighting."
                    "value_band.minimum_weight"
                ),
                min_value=0.0,
            ),
            maximum_weight=require_float(
                cfg.get("maximum_weight"),
                key=(
                    "cif_fob.method.estimation.representation_weighting."
                    "value_band.maximum_weight"
                ),
                min_value=0.0,
            ),
            normalize_mean_to_one=require_bool(
                cfg.get("normalize_mean_to_one"),
                key=(
                    "cif_fob.method.estimation.representation_weighting."
                    "value_band.normalize_mean_to_one"
                ),
            ),
        )


@dataclass(frozen=True)
class CifFobRepresentationWeightingConfig:
    value_band: CifFobRepresentationValueBandConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobRepresentationWeightingConfig":
        return cls(
            value_band=CifFobRepresentationValueBandConfig.from_dict(
                require_dict(
                    cfg.get("value_band"),
                    key=(
                        "cif_fob.method.estimation.representation_weighting."
                        "value_band"
                    ),
                )
            )
        )


@dataclass(frozen=True)
class CifFobMaximumFactorRuleConfig:
    method: str
    iqr_multiplier: float
    percentile: float
    domain_cap: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        minimum_factor: float,
    ) -> "CifFobMaximumFactorRuleConfig":
        method = require_str(
            cfg.get("method"),
            key="cif_fob.method.estimation.maximum_factor_rule.method",
        ).strip().lower()

        if method != "robust_adaptive_cap":
            raise ValueError(
                "Unsupported cif_fob maximum_factor_rule.method: "
                f"{method!r}."
            )

        return cls(
            method=method,
            iqr_multiplier=require_float(
                cfg.get("iqr_multiplier"),
                key=(
                    "cif_fob.method.estimation."
                    "maximum_factor_rule.iqr_multiplier"
                ),
                min_value=0.0,
            ),
            percentile=require_float(
                cfg.get("percentile"),
                key=(
                    "cif_fob.method.estimation."
                    "maximum_factor_rule.percentile"
                ),
                min_value=0.0,
                max_value=1.0,
            ),
            domain_cap=require_float(
                cfg.get("domain_cap"),
                key=(
                    "cif_fob.method.estimation."
                    "maximum_factor_rule.domain_cap"
                ),
                min_value=minimum_factor,
            ),
        )


@dataclass(frozen=True)
class CifFobTemporalBacktestOutputConfig:
    summary_product: str
    fold_product: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobTemporalBacktestOutputConfig":
        return cls(
            summary_product=require_str(
                cfg.get("summary_product"),
                key=(
                    "method.estimation.validation."
                    "temporal_backtest.output.summary_product"
                ),
            ).strip(),
            fold_product=require_str(
                cfg.get("fold_product"),
                key=(
                    "method.estimation.validation."
                    "temporal_backtest.output.fold_product"
                ),
            ).strip(),
        )
    
@dataclass(frozen=True)
class CifFobTemporalBacktestConfig:
    enabled: bool
    strategy: str
    validation_years: tuple[int, ...]
    minimum_training_years: int
    upper_tail_threshold: float
    metrics: tuple[str, ...]
    output: CifFobTemporalBacktestOutputConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobTemporalBacktestConfig":
        strategy = require_str(
            cfg.get("strategy"),
            key=(
                "method.estimation.validation."
                "temporal_backtest.strategy"
            ),
        ).strip().lower()

        if strategy != "expanding_window":
            raise ValueError(
                "Only 'expanding_window' is currently supported for "
                "method.estimation.validation.temporal_backtest.strategy."
            )

        validation_years_raw = cfg.get(
            "validation_years",
        )

        if not isinstance(validation_years_raw, list):
            raise TypeError(
                "method.estimation.validation.temporal_backtest."
                "validation_years must be a list."
            )

        validation_years = tuple(
            int(year)
            for year in validation_years_raw
        )

        if len(set(validation_years)) != len(validation_years):
            raise ValueError(
                "Temporal backtest validation_years contains duplicates."
            )

        minimum_training_years = require_int(
            cfg.get("minimum_training_years"),
            key="cif_fob.method.estimation.validation.temporal_backtest.minimum_training_years",
            min_value=1,
        )

        upper_tail_threshold = require_float(
            cfg.get("upper_tail_threshold"),
            key="cif_fob.method.estimation.validation.temporal_backtest.upper_tail_threshold",
            min_value=0.0,
        )

        if upper_tail_threshold == 0:
            raise ValueError(
                "upper_tail_threshold must be positive."
            )

        metrics_raw = cfg.get(
            "metrics",
        )

        if not isinstance(metrics_raw, list):
            raise TypeError(
                "method.estimation.validation.temporal_backtest."
                "metrics must be a list."
            )

        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key="cif_fob.method.estimation.validation.temporal_backtest.enabled",
            ),
            strategy=strategy,
            validation_years=validation_years,
            minimum_training_years=minimum_training_years,
            upper_tail_threshold=upper_tail_threshold,
            metrics=tuple(
                require_str(
                    metric,
                    key=(
                        "method.estimation.validation."
                        "temporal_backtest.metrics"
                    ),
                ).strip()
                for metric in metrics_raw
            ),
            output=CifFobTemporalBacktestOutputConfig.from_dict(
                require_dict(
                    cfg.get("output"),
                    key="cif_fob.method.estimation.validation.temporal_backtest.output",
                )
            ),
        )


@dataclass(frozen=True)
class CifFobUpperTailAuditProductsConfig:
    summary: str
    route_history: str
    observations: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobUpperTailAuditProductsConfig":
        return cls(
            summary=require_str(
                cfg.get("summary"),
                key="cif_fob.method.estimation.diagnostics.upper_tail_audit.products.summary",
            ).strip(),
            route_history=require_str(
                cfg.get("route_history"),
                key="cif_fob.method.estimation.diagnostics.upper_tail_audit.products.route_history",
            ).strip(),
            observations=require_str(
                cfg.get("observations"),
                key="cif_fob.method.estimation.diagnostics.upper_tail_audit.products.observations",
            ).strip(),
        )


@dataclass(frozen=True)
class CifFobUpperTailAuditPersistenceConfig:
    minimum_prior_years: int
    historical_high_share_threshold: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobUpperTailAuditPersistenceConfig":
        return cls(
            minimum_prior_years=require_int(
                cfg.get("minimum_prior_years"),
                key="cif_fob.method.estimation.diagnostics.upper_tail_audit.persistence.minimum_prior_years",
                min_value=1,
            ),
            historical_high_share_threshold=require_float(
                cfg.get("historical_high_share_threshold"),
                key="cif_fob.method.estimation.diagnostics.upper_tail_audit.persistence.historical_high_share_threshold",
                min_value=0.0,
                max_value=1.0,
            ),
        )


@dataclass(frozen=True)
class CifFobUpperTailAuditConfig:
    enabled: bool
    threshold: float
    extreme_threshold: float
    persistence: CifFobUpperTailAuditPersistenceConfig
    value_bands_usd: tuple[int, ...]
    quantity_agreement_bands: tuple[float, ...]
    products: CifFobUpperTailAuditProductsConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobUpperTailAuditConfig":
        value_bands_raw = cfg.get("value_bands_usd")
        quantity_bands_raw = cfg.get("quantity_agreement_bands")

        if not isinstance(value_bands_raw, list):
            raise TypeError(
                "cif_fob.method.estimation.diagnostics.upper_tail_audit.value_bands_usd must be a list."
            )
        if not isinstance(quantity_bands_raw, list):
            raise TypeError(
                "cif_fob.method.estimation.diagnostics.upper_tail_audit.quantity_agreement_bands must be a list."
            )

        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key="cif_fob.method.estimation.diagnostics.upper_tail_audit.enabled",
            ),
            threshold=require_float(
                cfg.get("threshold"),
                key="cif_fob.method.estimation.diagnostics.upper_tail_audit.threshold",
                min_value=0.0,
            ),
            extreme_threshold=require_float(
                cfg.get("extreme_threshold"),
                key="cif_fob.method.estimation.diagnostics.upper_tail_audit.extreme_threshold",
                min_value=0.0,
            ),
            persistence=CifFobUpperTailAuditPersistenceConfig.from_dict(
                require_dict(
                    cfg.get("persistence"),
                    key="cif_fob.method.estimation.diagnostics.upper_tail_audit.persistence",
                )
            ),
            value_bands_usd=tuple(
                require_int(
                    value,
                    key="cif_fob.method.estimation.diagnostics.upper_tail_audit.value_bands_usd",
                    min_value=0,
                )
                for value in value_bands_raw
            ),
            quantity_agreement_bands=tuple(
                require_float(
                    value,
                    key="cif_fob.method.estimation.diagnostics.upper_tail_audit.quantity_agreement_bands",
                    min_value=0.0,
                    max_value=1.0,
                )
                for value in quantity_bands_raw
            ),
            products=CifFobUpperTailAuditProductsConfig.from_dict(
                require_dict(cfg.get("products"), key="cif_fob.method.estimation.diagnostics.upper_tail_audit.products")
            ),
        )

    @property
    def minimum_prior_years(self) -> int:
        return self.persistence.minimum_prior_years

    @property
    def historical_high_share_threshold(self) -> float:
        return self.persistence.historical_high_share_threshold


@dataclass(frozen=True)
class CifFobEstimationDiagnosticsConfig:
    upper_tail_audit: CifFobUpperTailAuditConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobEstimationDiagnosticsConfig":
        return cls(
            upper_tail_audit=CifFobUpperTailAuditConfig.from_dict(
                require_dict(
                    cfg.get("upper_tail_audit"),
                    key="cif_fob.method.estimation.diagnostics.upper_tail_audit",
                )
            ),
        )

@dataclass(frozen=True)
class CifFobEstimationValidationConfig:
    strategy: str
    holdout_year: int | None
    temporal_backtest: CifFobTemporalBacktestConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobEstimationValidationConfig":
        strategy = require_str(
            cfg.get("strategy"),
            key="cif_fob.method.estimation.validation.strategy",
        ).strip().lower()

        holdout_year_raw = cfg.get("holdout_year")

        return cls(
            strategy=strategy,
            holdout_year=(
                require_int(
                    holdout_year_raw,
                    key=(
                        "cif_fob.method.estimation.validation."
                        "holdout_year"
                    ),
                )
                if holdout_year_raw is not None
                else None
            ),
            temporal_backtest=CifFobTemporalBacktestConfig.from_dict(
                require_dict(
                    cfg.get("temporal_backtest"),
                    key="cif_fob.method.estimation.validation.temporal_backtest",
                )
            ),
        )


@dataclass(frozen=True)
class CifFobModelBaseConfig:
    name: str
    enabled: bool
    type: str
    estimator_type: str
    output_prefix: str
    categorical_predictors: tuple[str, ...]

@dataclass(frozen=True)
class CifFobSampleWeightingConfig:
    method: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "CifFobSampleWeightingConfig":
        method = require_str(
            cfg.get("method"),
            key=f"{key}.method",
        ).strip().lower()

        allowed_methods = {
            "source_only",
            "value_band_weighting",
        }

        if method not in allowed_methods:
            raise ValueError(
                f"{key}.method must be one of "
                f"{sorted(allowed_methods)}, got {method!r}."
            )

        return cls(method=method)
    

@dataclass(frozen=True)
class CifFobHistGradientBoostingConfig:
    name: str
    enabled: bool
    type: Literal["machine_learning"]
    estimator_type: str
    output_prefix: str
    categorical_predictors: tuple[str, ...]

    loss: str
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_bins: int
    random_state: int
    target_transformation: str
    sample_weighting: CifFobSampleWeightingConfig

    @classmethod
    def from_dict(
        cls,
        name: str,
        cfg: dict[str, Any],
    ) -> "CifFobHistGradientBoostingConfig":
        estimator_type = require_str(
            cfg.get("estimator_type"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.estimator_type"
            ),
        ).strip().lower()

        max_bins = require_int(
            cfg.get("max_bins"),
            key=f"cif_fob.method.estimation.models.candidates.{name}.max_bins",
            min_value=2,
        )

        if max_bins > 255:
            raise ValueError(
                f"{name}.max_bins must be <= 255, got {max_bins}."
            )

        target_transformation = require_str(
            cfg.get("target_transformation"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.target_transformation"
            ),
        ).strip().lower()
        
        if target_transformation not in {"log", "none"}:
            raise ValueError(
        f"{name}.target_transformation must be 'log' or 'none', "
        f"got {target_transformation!r}."
            )
        
        sample_weighting = CifFobSampleWeightingConfig.from_dict(
            require_dict(
                cfg.get("sample_weighting", {"method": "source_only"}),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.sample_weighting"
                ),
            ),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.sample_weighting"
            ),
        )

        return cls(
            name=name,
            enabled=require_bool(
                cfg.get("enabled"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.enabled",
            ),
            type="machine_learning",
            estimator_type=estimator_type,
            output_prefix=_require_output_prefix(
                value=cfg.get("output_prefix"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.output_prefix"
                ),
            ),
            categorical_predictors=tuple(
                _require_str_list(
                    cfg.get("categorical_predictors", []),
                    key=(
                        "cif_fob.method.estimation.models."
                        f"candidates.{name}.categorical_predictors"
                    ),
                )
            ),
            loss=require_str(
                cfg.get("loss"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.loss",
            ).strip().lower(),
            learning_rate=require_float(
                cfg.get("learning_rate"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.learning_rate"
                ),
                min_value=0.0,
            ),
            max_iter=require_int(
                cfg.get("max_iter"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.max_iter",
                min_value=1,
            ),
            max_leaf_nodes=require_int(
                cfg.get("max_leaf_nodes"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.max_leaf_nodes"
                ),
                min_value=2,
            ),
            min_samples_leaf=require_int(
                cfg.get("min_samples_leaf"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.min_samples_leaf"
                ),
                min_value=1,
            ),
            l2_regularization=require_float(
                cfg.get("l2_regularization"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.l2_regularization"
                ),
                min_value=0.0,
            ),
            max_bins=max_bins,
            random_state=require_int(
                cfg.get("random_state"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.random_state"
                ),
            ),
            target_transformation=target_transformation,
            sample_weighting=sample_weighting,

        )
    
@dataclass(frozen=True)
class CifFobLogLinearRegressionConfig:
    name: str
    enabled: bool
    type: Literal["statistical"]
    estimator_type: str
    output_prefix: str
    categorical_predictors: tuple[str, ...]

    target_transformation: str
    estimator_method: str
    fit_intercept: bool
    covariance_method: str
    inverse_transform: str

    @classmethod
    def from_dict(
        cls,
        name: str,
        cfg: dict[str, Any],
    ) -> "CifFobLogLinearRegressionConfig":
        estimator_type = require_str(
            cfg.get("estimator_type"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.estimator_type"
            ),
        ).strip().lower()

        target_cfg = require_dict(
            cfg.get("target"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.target"
            ),
        )
        estimator_cfg = require_dict(
            cfg.get("estimator"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.estimator"
            ),
        )
        covariance_cfg = require_dict(
            cfg.get("covariance"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.covariance"
            ),
        )
        prediction_cfg = require_dict(
            cfg.get("prediction"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.prediction"
            ),
        )

        target_transformation = require_str(
            target_cfg.get("transformation"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.target.transformation"
            ),
        ).strip().lower()

        if target_transformation not in {"none", "log"}:
            raise ValueError(
                "Statistical target transformation must be 'none' or 'log'."
            )

        covariance_method = require_str(
            covariance_cfg.get("method"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.covariance.method"
            ),
        ).strip().lower()

        if covariance_method not in {"nonrobust", "hc0", "hc1", "hc2", "hc3"}:
            raise ValueError(
                "Unsupported statistical covariance method: "
                f"{covariance_method!r}"
            )

        return cls(
            name=name,
            enabled=require_bool(
                cfg.get("enabled"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.enabled",
            ),
            type="statistical",
            estimator_type=estimator_type,
            output_prefix=_require_output_prefix(
                value=cfg.get("output_prefix"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.output_prefix"
                ),
            ),
            categorical_predictors=tuple(
                _require_str_list(
                    cfg.get("categorical_predictors"),
                    key=(
                        "cif_fob.method.estimation.models."
                        f"candidates.{name}.categorical_predictors"
                    ),
                )
            ),
            target_transformation=target_transformation,
            estimator_method=require_str(
                estimator_cfg.get("method"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.estimator.method"
                ),
            ).strip().lower(),
            fit_intercept=require_bool(
                estimator_cfg.get("fit_intercept"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.estimator.fit_intercept"
                ),
            ),
            covariance_method=covariance_method,
            inverse_transform=require_str(
                prediction_cfg.get("inverse_transform"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.prediction.inverse_transform"
                ),
            ).strip().lower(),
        )


@dataclass(frozen=True)
class CifFobXGBoostConfig:
    name: str
    enabled: bool
    type: Literal["machine_learning"]
    estimator_type: str
    output_prefix: str
    categorical_predictors: tuple[str, ...]

    target_transformation: str
    objective: str
    eval_metric: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    gamma: float
    random_state: int
    max_bins: int
    tree_method: str
    enable_categorical: bool
    max_cat_to_onehot: int
    early_stopping_rounds: int
    n_jobs: int

    @classmethod
    def from_dict(
        cls,
        name: str,
        cfg: dict[str, Any],
    ) -> "CifFobXGBoostConfig":
        estimator_type = require_str(
            cfg.get("estimator_type"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.estimator_type"
            ),
        ).strip().lower()

        target_transformation = require_str(
            cfg.get("target_transformation"),
            key=(
                "cif_fob.method.estimation.models."
                f"candidates.{name}.target_transformation"
            ),
        ).strip().lower()

        if target_transformation not in {"log", "none"}:
            raise ValueError(
                f"{name}.target_transformation must be 'log' or 'none', "
                f"got {target_transformation!r}."
            )

        return cls(
            name=name,
            enabled=require_bool(
                cfg.get("enabled"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.enabled",
            ),
            type="machine_learning",
            estimator_type=estimator_type,
            output_prefix=_require_output_prefix(
                value=cfg.get("output_prefix"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.output_prefix"
                ),
            ),
            categorical_predictors=tuple(
                _require_str_list(
                    cfg.get("categorical_predictors"),
                    key=(
                        "cif_fob.method.estimation.models."
                        f"candidates.{name}.categorical_predictors"
                    ),
                )
            ),
            target_transformation=target_transformation,
            objective=require_str(
                cfg.get("objective"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.objective",
            ).strip(),
            eval_metric=require_str(
                cfg.get("eval_metric"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.eval_metric",
            ).strip(),
            n_estimators=require_int(
                cfg.get("n_estimators"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.n_estimators"
                ),
                min_value=1,
            ),
            learning_rate=require_float(
                cfg.get("learning_rate"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.learning_rate"
                ),
                min_value=0.0,
            ),
            max_depth=require_int(
                cfg.get("max_depth"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.max_depth",
                min_value=1,
            ),
            min_child_weight=require_float(
                cfg.get("min_child_weight"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.min_child_weight"
                ),
                min_value=0.0,
            ),
            subsample=require_float(
                cfg.get("subsample"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.subsample",
                min_value=0.0,
                max_value=1.0,
            ),
            colsample_bytree=require_float(
                cfg.get("colsample_bytree"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.colsample_bytree"
                ),
                min_value=0.0,
                max_value=1.0,
            ),
            reg_alpha=require_float(
                cfg.get("reg_alpha"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.reg_alpha",
                min_value=0.0,
            ),
            reg_lambda=require_float(
                cfg.get("reg_lambda"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.reg_lambda",
                min_value=0.0,
            ),
            gamma=require_float(
                cfg.get("gamma"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.gamma",
                min_value=0.0,
            ),
            random_state=require_int(
                cfg.get("random_state"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.random_state"
                ),
            ),
            max_bins=require_int(
                cfg.get("max_bins"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.max_bins",
                min_value=2,
            ),
            tree_method=require_str(
                cfg.get("tree_method"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.tree_method",
            ).strip().lower(),
            enable_categorical=require_bool(
                cfg.get("enable_categorical"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.enable_categorical"
                ),
            ),
            max_cat_to_onehot=require_int(
                cfg.get("max_cat_to_onehot"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.max_cat_to_onehot"
                ),
                min_value=1,
            ),
            early_stopping_rounds=require_int(
                cfg.get("early_stopping_rounds"),
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}.early_stopping_rounds"
                ),
                min_value=1,
            ),
            n_jobs=require_int(
                cfg.get("n_jobs"),
                key=f"cif_fob.method.estimation.models.candidates.{name}.n_jobs",
                min_value=1,
            ),
        )


CifFobCandidateModelConfig = (
    CifFobHistGradientBoostingConfig
    | CifFobXGBoostConfig
    | CifFobLogLinearRegressionConfig
)

@dataclass(frozen=True)
class CifFobModelsConfig:
    """Registry of configured CIF/FOB estimation model candidates."""

    primary: str
    candidates: dict[str, CifFobCandidateModelConfig]

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobModelsConfig":
        candidates_raw = require_dict(
            cfg.get("candidates"),
            key="cif_fob.method.estimation.models.candidates",
        )

        candidates: dict[str, CifFobCandidateModelConfig] = {}

        hgb_names = {
            "hist_gradient_boosting",
            "hist_gradient_boosting_squared",
            "hist_gradient_boosting_weighted",
            "hist_gradient_boosting_squared_level",
            "hist_gradient_boosting_squared_regularized",
        }
        log_linear_names = {
            "log_linear_regression",
            "log_linear_regression_smearing",
        }

        for name, candidate_cfg in candidates_raw.items():
            candidate_cfg = require_dict(
                candidate_cfg,
                key=(
                    "cif_fob.method.estimation.models."
                    f"candidates.{name}"
                ),
            )

            if name in hgb_names:
                candidates[name] = (
                    CifFobHistGradientBoostingConfig.from_dict(
                        name=name,
                        cfg=candidate_cfg,
                    )
                )
            elif name == "xgboost_log_squared":
                candidates[name] = CifFobXGBoostConfig.from_dict(
                    name=name,
                    cfg=candidate_cfg,
                )
            elif name in log_linear_names:
                candidates[name] = (
                    CifFobLogLinearRegressionConfig.from_dict(
                        name=name,
                        cfg=candidate_cfg,
                    )
                )
            else:
                raise ValueError(
                    "Unsupported CIF/FOB model candidate "
                    f"{name!r}."
                )

        primary = require_str(
            cfg.get("primary"),
            key="cif_fob.method.estimation.models.primary",
        ).strip()

        if primary not in candidates:
            raise ValueError(
                f"Configured primary CIF/FOB model {primary!r} is not "
                "present in models.candidates. Available candidates: "
                + ", ".join(sorted(candidates))
            )

        return cls(
            primary=primary,
            candidates=candidates,
        )

    def enabled(self) -> dict[str, CifFobCandidateModelConfig]:
        """Return enabled model candidates, preserving YAML order."""
        return {
            name: candidate
            for name, candidate in self.candidates.items()
            if candidate.enabled
        }

    @property
    def enabled_candidates(self) -> dict[str, CifFobCandidateModelConfig]:
        """Alias for callers that prefer property-style access."""
        return self.enabled()

    def get_candidate(
        self,
        name: str | None = None,
    ) -> CifFobCandidateModelConfig:
        """Return a named candidate, or the configured primary."""
        candidate_name = name or self.primary

        if candidate_name not in self.candidates:
            raise KeyError(
                f"Unknown CIF/FOB model candidate {candidate_name!r}. "
                f"Available candidates: {sorted(self.candidates)}"
            )

        return self.candidates[candidate_name]


@dataclass(frozen=True)
class CifFobPredictorSetConfig:
    """One named predictor set used for model-comparison runs."""

    name: str
    enabled: bool
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        name: str,
        cfg: dict[str, Any],
    ) -> "CifFobPredictorSetConfig":
        numeric = tuple(
            _require_str_list(
                cfg.get("numeric"),
                key=(
                    "cif_fob.method.estimation."
                    f"predictor_sets.{name}.numeric"
                ),
            )
        )
        categorical = tuple(
            _require_str_list(
                cfg.get("categorical"),
                key=(
                    "cif_fob.method.estimation."
                    f"predictor_sets.{name}.categorical"
                ),
            )
        )

        predictors = list(numeric + categorical)
        duplicates = sorted(
            {
                predictor
                for predictor in predictors
                if predictors.count(predictor) > 1
            }
        )
        if duplicates:
            raise ValueError(
                f"Predictor set {name!r} contains duplicates: "
                + ", ".join(duplicates)
            )

        if not predictors:
            raise ValueError(
                f"Predictor set {name!r} must contain at least one "
                "numeric or categorical predictor."
            )

        return cls(
            name=name,
            enabled=require_bool(
                cfg.get("enabled"),
                key=f"cif_fob.method.estimation.predictor_sets.{name}.enabled",
            ),
            numeric=numeric,
            categorical=categorical,
        )

    @property
    def all_predictors(self) -> list[str]:
        """Return numeric + categorical predictors in stable order."""
        return [
            *self.numeric,
            *self.categorical,
        ]


def _parse_predictor_sets(
    raw: dict[str, Any] | None,
) -> dict[str, CifFobPredictorSetConfig]:
    """Parse optional model-comparison predictor sets."""
    if raw is None:
        return {}

    raw = require_dict(
        raw,
        key="cif_fob.method.estimation.predictor_sets",
    )

    return {
        name: CifFobPredictorSetConfig.from_dict(
            name=name,
            cfg=require_dict(
                predictor_set_cfg,
                key=(
                    "cif_fob.method.estimation."
                    f"predictor_sets.{name}"
                ),
            ),
        )
        for name, predictor_set_cfg in raw.items()
    }


@dataclass(frozen=True)
class CifFobEstimationModelConfig:
    name: str
    loss: str
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_bins: int
    random_state: int

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobEstimationModelConfig":
        max_bins = require_int(
            cfg.get("max_bins"),
            key="cif_fob.method.estimation.model.max_bins",
            min_value=2,
        )
        if max_bins > 255:
            raise ValueError(
                "'cif_fob.method.estimation.model.max_bins' must be <= 255, "
                f"got {max_bins}."
            )
        return cls(
            name=require_str(
                cfg.get("name"),
                key="cif_fob.method.estimation.model.name",
            ).strip().lower(),
            loss=require_str(
                cfg.get("loss"),
                key="cif_fob.method.estimation.model.loss",
            ).strip().lower(),
            learning_rate=require_float(
                cfg.get("learning_rate"),
                key="cif_fob.method.estimation.model.learning_rate",
                min_value=0.0,
            ),
            max_iter=require_int(
                cfg.get("max_iter"),
                key="cif_fob.method.estimation.model.max_iter",
                min_value=1,
            ),
            max_leaf_nodes=require_int(
                cfg.get("max_leaf_nodes"),
                key=(
                    "cif_fob.method.estimation.model."
                    "max_leaf_nodes"
                ),
                min_value=2,
            ),
            min_samples_leaf=require_int(
                cfg.get("min_samples_leaf"),
                key=(
                    "cif_fob.method.estimation.model."
                    "min_samples_leaf"
                ),
                min_value=1,
            ),
            l2_regularization=require_float(
                cfg.get("l2_regularization"),
                key=(
                    "cif_fob.method.estimation.model."
                    "l2_regularization"
                ),
                min_value=0.0,
            ),
            max_bins=max_bins,
            random_state=require_int(
                cfg.get("random_state"),
                key="cif_fob.method.estimation.model.random_state",
            ),
        )


@dataclass(frozen=True)
class CifFobEstimationConfig:
    minimum_evidence_value_usd: float
    minimum_factor: float
    maximum_factor_rule: CifFobMaximumFactorRuleConfig
    exclude_proxy_weights: bool
    representation_weighting: CifFobRepresentationWeightingConfig

    diagnostics: Any
    training_factor_sources: list[str]
    source_weights: Any
    validation: Any
    models: CifFobModelsConfig
    primary_predictor_set: str | None = None

    # New model-comparison predictor sets.
    predictor_sets: dict[str, CifFobPredictorSetConfig] = field(
        default_factory=dict
    )

    @property
    def maximum_factor(self) -> float:
        """Return the effective upper cap implied by maximum_factor_rule."""
        return self.maximum_factor_rule.domain_cap

    def get_predictor_set(
        self,
        name: str,
    ) -> CifFobPredictorSetConfig:
        """Return one named predictor set."""
        if name not in self.predictor_sets:
            raise KeyError(
                f"Unknown CIF/FOB predictor set {name!r}. "
                f"Available predictor sets: {sorted(self.predictor_sets)}"
            )

        return self.predictor_sets[name]

    def primary_predictor_set_name(self) -> str | None:
        """Return configured primary predictor set, or a default."""
        if not self.predictor_sets:
            return None

        if self.primary_predictor_set:
            if self.primary_predictor_set not in self.predictor_sets:
                raise KeyError(
                    "Configured primary predictor set "
                    f"{self.primary_predictor_set!r} is not present. "
                    f"Available predictor sets: {sorted(self.predictor_sets)}"
                )
            return self.primary_predictor_set

        # Default to the last configured set, which is usually the full
        # model in a staged comparison ladder.
        return next(reversed(self.predictor_sets))


    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobEstimationConfig":
        minimum_factor = require_float(
            cfg.get("minimum_factor"),
            key="cif_fob.method.estimation.minimum_factor",
            min_value=0.0,
        )

        maximum_factor_rule = CifFobMaximumFactorRuleConfig.from_dict(
            require_dict(
                cfg.get("maximum_factor_rule"),
                key="cif_fob.method.estimation.maximum_factor_rule",
            ),
            minimum_factor=minimum_factor,
        )


        training_factor_sources = _require_str_list(
                                    cfg.get(
                                        "training_factor_sources"
                                        ), 
                                    key=("cif_fob.method.estimation."
                                         "training_factor_sources"
                                         ),
                )
        supported_factor_sources = {"direct", "mirror"} 
        unsupported_factor_sources = sorted( 
                                set(training_factor_sources) - 
                                supported_factor_sources ) 
        if unsupported_factor_sources: 
            raise ValueError( "Unsupported CIF/FOB training factor sources: " + ", ".join(unsupported_factor_sources) ) 
        if not training_factor_sources: 
            raise ValueError( "cif_fob.method.estimation.training_factor_sources " "must contain at least one source." )

        return cls(
            minimum_evidence_value_usd=require_float(
                cfg.get("minimum_evidence_value_usd"),
                key=(
                    "cif_fob.method.estimation."
                    "minimum_evidence_value_usd"
                ),
                min_value=0.0,
            ),
            minimum_factor=minimum_factor,
            maximum_factor_rule=maximum_factor_rule,
            exclude_proxy_weights=require_bool(
                cfg.get("exclude_proxy_weights"),
                key=(
                    "cif_fob.method.estimation."
                    "exclude_proxy_weights"
                ),
            ),
            representation_weighting=(
                CifFobRepresentationWeightingConfig.from_dict(
                    require_dict(
                        cfg.get("representation_weighting"),
                        key=(
                            "cif_fob.method.estimation."
                            "representation_weighting"
                        ),
                    )
                )
            ),
            training_factor_sources=training_factor_sources,
            source_weights=(
                CifFobEstimationSourceWeightsConfig.from_dict(
                    require_dict(
                        cfg.get("source_weights"),
                        key=(
                            "cif_fob.method.estimation."
                            "source_weights"
                        ),
                    )
                )
            ),
            diagnostics=CifFobEstimationDiagnosticsConfig.from_dict(
                require_dict(
                    cfg.get("diagnostics"),
                    key="cif_fob.method.estimation.diagnostics",
                )
            ),
            validation=(
                CifFobEstimationValidationConfig.from_dict(
                    require_dict(
                        cfg.get("validation"),
                        key=(
                            "cif_fob.method.estimation."
                            "validation"
                        ),
                    )
                )
            ),
            models=CifFobModelsConfig.from_dict(
                require_dict(
                    cfg.get("models"),
                    key="cif_fob.method.estimation.models",
                )
            ),
            primary_predictor_set=(
                require_str(
                    cfg.get("primary_predictor_set"),
                    key=(
                        "cif_fob.method.estimation."
                        "primary_predictor_set"
                    ),
                ).strip()
                if cfg.get("primary_predictor_set") is not None
                else None
            ),
            predictor_sets=_parse_predictor_sets(
                cfg.get("predictor_sets")
            ),
        )


@dataclass(frozen=True)
class CifFobPredictorsConfig:
    bilateral: list[str]
    fuel_and_time: list[str]
    product: list[str]
    route: list[str]
    optional_connectivity: list[str]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobPredictorsConfig":
        return cls(
            bilateral=_require_str_list(
                cfg.get("bilateral"),
                key="cif_fob.method.predictors.bilateral",
            ),
            fuel_and_time=_require_str_list(
                cfg.get("fuel_and_time"),
                key="cif_fob.method.predictors.fuel_and_time",
            ),
            product=_require_str_list(
                cfg.get("product"),
                key="cif_fob.method.predictors.product",
            ),
            route=_require_str_list(
                cfg.get("route"),
                key="cif_fob.method.predictors.route",
            ),
            optional_connectivity=_require_str_list(
                cfg.get("optional_connectivity"),
                key="cif_fob.method.predictors.optional_connectivity",
            ),
        )

    def all(self) -> list[str]:
        return (
            self.bilateral
            + self.fuel_and_time
            + self.product
            + self.route
            + self.optional_connectivity
        )


@dataclass(frozen=True)
class CifFobOutliersConfig:
    method: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobOutliersConfig":
        return cls(
            method=require_str(cfg.get("method"), key="cif_fob.method.outliers.method"),
        )


@dataclass(frozen=True)
class CifFobMethodConfig:
    flow_keys: list[str]
    training: CifFobTrainingConfig
    estimation: CifFobEstimationConfig
    predictors: CifFobPredictorsConfig
    outliers: CifFobOutliersConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobMethodConfig":
        flow_keys_raw = cfg.get("flow_keys")
        predictors_raw = cfg.get("predictors")

        if not isinstance(flow_keys_raw, list):
            raise TypeError("cif_fob.method.flow_keys must be a list")
        if not isinstance(predictors_raw, dict):
            raise TypeError("cif_fob.method.predictors must be a dictionary")

        return cls(
            flow_keys=[require_str(item, key="cif_fob.method.flow_keys") for item in flow_keys_raw],
            training=CifFobTrainingConfig.from_dict(
                require_dict(cfg.get("training"), key="cif_fob.method.training")
            ),
            estimation=CifFobEstimationConfig.from_dict(
                require_dict(cfg.get("estimation"), key="cif_fob.method.estimation")
            ),
            predictors=CifFobPredictorsConfig.from_dict(
                require_dict(cfg.get("predictors"), key="cif_fob.method.predictors")
            ),
            outliers=CifFobOutliersConfig.from_dict(
                require_dict(cfg.get("outliers"), key="cif_fob.method.outliers")
            ),
        )



@dataclass(frozen=True)
class CifFobArtifactsConfig:
    enable: bool
    kind: str
    subdir: str
    filename: str
    timestamp: bool
    write_latest: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobArtifactsConfig":
        kind = require_str(
            cfg.get("kind"),
            key="cif_fob.artifacts.kind",
        ).strip().lower()

        if not kind:
            raise ValueError(
                "cif_fob.artifacts.kind must not be empty."
            )

        filename = require_str(
            cfg.get("filename"),
            key="cif_fob.artifacts.filename",
        ).strip()

        if not filename:
            raise ValueError(
                "cif_fob.artifacts.filename must not be empty."
            )

        return cls(
            enable=require_bool(
                cfg.get("enable"),
                key="cif_fob.artifacts.enable",
            ),
            kind=kind,
            subdir=require_str(
                cfg.get("subdir"),
                key="cif_fob.artifacts.subdir",
                allow_empty=True,
            ).strip(),
            filename=filename,
            timestamp=require_bool(
                cfg.get("timestamp"),
                key="cif_fob.artifacts.timestamp",
            ),
            write_latest=require_bool(
                cfg.get("write_latest"),
                key="cif_fob.artifacts.write_latest",
            ),
        )

    def resolve_dir(self) -> Path:
        base = get_output_dir(self.kind)
        subdir_path = Path(self.subdir)

        parts = list(subdir_path.parts)

        # Allow subdir values such as "reports/cif_fob/..." without
        # duplicating the artifact kind in the resolved path.
        if parts and parts[0].lower() == self.kind:
            parts = parts[1:]

        return (
            base / Path(*parts)
            if parts
            else base
        )

    def timestamped_path(self, run_id: str) -> Path:
        filename = self.filename

        if filename.lower().endswith(".md"):
            filename = filename[:-3]

        if self.timestamp:
            filename = f"{filename}_{run_id}"

        return self.resolve_dir() / f"{filename}.md"

    def latest_path(self) -> Path:
        filename = self.filename

        if filename.lower().endswith(".md"):
            filename = filename[:-3]

        return self.resolve_dir() / f"{filename}_latest.md"

@dataclass(frozen=True)
class CifFobConfig:
    input: CifFobInputConfig
    output: CifFobOutputConfig
    source_data: CifFobSourceDataConfig
    products: CifFobProductsConfig
    logging: CifFobLoggingConfig
    artifacts: CifFobArtifactsConfig
    features: CifFobFeaturesConfig
    method: CifFobMethodConfig
    scope: PipelineScopeConfig
    engine: PipelineEngineConfig

    @classmethod
    def from_yaml(
        cls,
        config_path: Path | None = None,
        *,
        scope: PipelineScopeConfig | None = None,
        engine: PipelineEngineConfig | None = None,
    ) -> "CifFobConfig":
        return cls.from_dict(
            load_cif_fob_config(config_path),
            scope=scope,
            engine=engine,
        )

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        scope: PipelineScopeConfig | None = None,
        engine: PipelineEngineConfig | None = None,
    ) -> "CifFobConfig":
        if scope is None:
            raise ValueError("CifFobConfig requires pipeline scope.")
        if engine is None:
            raise ValueError("CifFobConfig requires pipeline engine.")

        cif_fob_raw = require_dict(raw.get("cif_fob"), key="cif_fob")
        rendered = require_dict(
            _render_value(cif_fob_raw, target_hs=scope.target_hs),
            key="cif_fob",
        )

        return cls(
            input=CifFobInputConfig.from_dict(
                require_dict(rendered.get("input"), key="cif_fob.input")
            ),
            output=CifFobOutputConfig.from_dict(
                require_dict(rendered.get("output"), key="cif_fob.output")
            ),
            source_data=CifFobSourceDataConfig.from_dict(
                require_dict(rendered.get("source_data"), key="cif_fob.source_data")
            ),
            products=CifFobProductsConfig.from_dict(
                require_dict(rendered.get("products"), key="cif_fob.products")
            ),
            logging=CifFobLoggingConfig.from_dict(
                require_dict(rendered.get("logging"), key="cif_fob.logging")
            ),
            artifacts=CifFobArtifactsConfig.from_dict(
                require_dict(
                    rendered.get("artifacts"),
                    key="cif_fob.artifacts",
                )
            ),
            features=CifFobFeaturesConfig.from_dict(
                require_dict(rendered.get("features"), key="cif_fob.features")
            ),
            method=CifFobMethodConfig.from_dict(
                require_dict(rendered.get("method"), key="cif_fob.method")
            ),
            scope=scope,
            engine=engine,
        )

    def input_path(self) -> Path:
        return self.input.resolve_path()

    def output_dir(self) -> Path:
        return self.output.resolve_dir()

    def artifact_dir(self) -> Path:
        return self.artifacts.resolve_dir()

    def artifact_path(self, run_id: str) -> Path:
        return self.artifacts.timestamped_path(run_id)

    def artifact_latest_path(self) -> Path:
        return self.artifacts.latest_path()

    def product_path(self, product_name: str) -> Path:
        products = {
            "prepared_input": self.products.prepared_input,
            "mirror_pairs": self.products.mirror_pairs,
            "training_sample": self.products.training_sample,
            "training_features": self.products.training_features,
            "training_evidence_audit": self.products.training_evidence_audit,
            "estimated_rates": self.products.estimated_rates,
            "temporal_validation_summary": self.products.temporal_validation_summary,
            "temporal_validation_folds": self.products.temporal_validation_folds,
            "upper_tail_audit_summary": self.products.upper_tail_audit_summary,
            "upper_tail_route_history": self.products.upper_tail_route_history,
            "upper_tail_observations": self.products.upper_tail_observations,
            "diagnostics": self.products.diagnostics,
            "corrected_trade": self.products.corrected_trade,
        }

        if product_name not in products:
            raise KeyError(f"Unknown CIF/FOB product: {product_name!r}")

        return self.output_dir() / products[product_name].filename

    def resolved_product_paths(self) -> dict[str, Path]:
        return {
            "prepared_input": self.product_path("prepared_input"),
            "mirror_pairs": self.product_path("mirror_pairs"),
            "training_sample": self.product_path("training_sample"),
            "training_features": self.product_path("training_features"),
            "training_evidence_audit": self.product_path("training_evidence_audit"),
            "estimated_rates": self.product_path("estimated_rates"),
            "temporal_validation_summary": self.product_path("temporal_validation_summary"),
            "temporal_validation_folds": self.product_path("temporal_validation_folds"),
            "upper_tail_audit_summary": self.product_path("upper_tail_audit_summary"),
            "upper_tail_route_history": self.product_path("upper_tail_route_history"),
            "upper_tail_observations": self.product_path("upper_tail_observations"),
            "diagnostics": self.product_path("diagnostics"),
            "corrected_trade": self.product_path("corrected_trade"),
        }

    def resolved_enabled_product_paths(self) -> dict[str, Path]:
        products = {
            "prepared_input": self.products.prepared_input,
            "mirror_pairs": self.products.mirror_pairs,
            "training_sample": self.products.training_sample,
            "training_features": self.products.training_features,
            "training_evidence_audit": self.products.training_evidence_audit,
            "estimated_rates": self.products.estimated_rates,
            "temporal_validation_summary": self.products.temporal_validation_summary,
            "temporal_validation_folds": self.products.temporal_validation_folds,
            "upper_tail_audit_summary": self.products.upper_tail_audit_summary,
            "upper_tail_route_history": self.products.upper_tail_route_history,
            "upper_tail_observations": self.products.upper_tail_observations,
            "diagnostics": self.products.diagnostics,
            "corrected_trade": self.products.corrected_trade,
        }

        return {
            name: self.output_dir() / product.filename
            for name, product in products.items()
            if product.enable
        }
