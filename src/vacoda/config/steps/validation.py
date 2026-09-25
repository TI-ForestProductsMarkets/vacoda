"""
vacoda.config.validation

Config reader for configs/steps/validation.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from vacoda.config.paths import get_data_dir, get_output_dir
from vacoda.config.pipeline import (
    PipelineEngineConfig,
    PipelineScopeConfig,
)


VALIDATION_CONFIG_PATH = get_step_config_path("validation")


def load_validation_config(
    config_path: Path | None = None,
) -> dict[str, Any]:
    return load_yaml_config(config_path or VALIDATION_CONFIG_PATH)


def _render_value(value: Any, *, target_hs: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _render_value(item, target_hs=target_hs)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _render_value(item, target_hs=target_hs)
            for item in value
        ]

    if isinstance(value, str):
        return value.format(
            target_hs=target_hs.upper(),
            target_hs_upper=target_hs.upper(),
            target_hs_lower=target_hs.lower(),
        )

    return value


def _require_str_list(
    value: Any,
    *,
    key: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list.")

    result = tuple(
        require_str(item, key=key).strip()
        for item in value
    )

    if any(not item for item in result):
        raise ValueError(f"{key} must not contain empty values.")

    if len(set(result)) != len(result):
        raise ValueError(f"{key} must not contain duplicate values.")

    return result


def _require_float_list(
    value: Any,
    *,
    key: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list.")

    result = tuple(
        require_float(
            item,
            key=key,
            min_value=min_value,
            max_value=max_value,
        )
        for item in value
    )

    if tuple(sorted(result)) != result:
        raise ValueError(f"{key} must be sorted in ascending order.")

    if len(set(result)) != len(result):
        raise ValueError(f"{key} must not contain duplicate values.")

    return result


def _require_int_list(
    value: Any,
    *,
    key: str,
    min_value: int | None = None,
    max_value: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list.")

    result = tuple(
        require_int(
            item,
            key=key,
            min_value=min_value,
            max_value=max_value,
        )
        for item in value
    )

    if tuple(sorted(result)) != result:
        raise ValueError(f"{key} must be sorted in ascending order.")

    if len(set(result)) != len(result):
        raise ValueError(f"{key} must not contain duplicate values.")

    return result


# ---------------------------------------------------------------------
# Input / output
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationInputConfig:
    stage: str
    subdir: str
    filename: str
    format: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationInputConfig":
        format_name = require_str(
            cfg.get("format"),
            key="validation.input.format",
        ).strip().lower()

        if format_name != "parquet":
            raise ValueError(
                "validation.input.format currently must be 'parquet', "
                f"got {format_name!r}."
            )

        return cls(
            stage=require_str(
                cfg.get("stage"),
                key="validation.input.stage",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="validation.input.subdir",
                allow_empty=True,
            ),
            filename=require_str(
                cfg.get("filename"),
                key="validation.input.filename",
            ),
            format=format_name,
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

    def resolve_path(self) -> Path:
        return self.resolve_dir() / self.filename


@dataclass(frozen=True)
class ValidationOutputConfig:
    stage: str
    subdir: str
    format: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationOutputConfig":
        format_name = require_str(
            cfg.get("format"),
            key="validation.output.format",
        ).strip().lower()

        if format_name != "parquet":
            raise ValueError(
                "validation.output.format currently must be 'parquet', "
                f"got {format_name!r}."
            )

        return cls(
            stage=require_str(
                cfg.get("stage"),
                key="validation.output.stage",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="validation.output.subdir",
                allow_empty=True,
            ),
            format=format_name,
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base


# ---------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationProductConfig:
    enable: bool
    filename: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "ValidationProductConfig":
        return cls(
            enable=require_bool(
                cfg.get("enable"),
                key=f"{key}.enable",
            ),
            filename=require_str(
                cfg.get("filename"),
                key=f"{key}.filename",
            ),
        )


@dataclass(frozen=True)
class ValidationProductsConfig:
    prepared_input: ValidationProductConfig
    structural_audit: ValidationProductConfig
    quantity_selection: ValidationProductConfig
    quantity_selection_audit: ValidationProductConfig
    diagnostics: ValidationProductConfig
    validated_trade: ValidationProductConfig
    consistency_audit: ValidationProductConfig
    release: ValidationProductConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationProductsConfig":
        def product(name: str) -> ValidationProductConfig:
            key = f"validation.products.{name}"

            value = cfg.get(name)

            if name == "diagnostics" and value is None:
                value = cfg.get("model_diagnostics")
                key = (
                    "validation.products.diagnostics "
                    "(or validation.products.model_diagnostics)"
                )

            return ValidationProductConfig.from_dict(
                require_dict(
                    value,
                    key=key,
                ),
                key=key,
            )

        return cls(
            prepared_input=product("prepared_input"),
            structural_audit=product("structural_audit"),
            quantity_selection=product("quantity_selection"),
            quantity_selection_audit=product("quantity_selection_audit"),
            diagnostics=product("diagnostics"),
            validated_trade=product("validated_trade"),
            consistency_audit=product("consistency_audit"),
            release=product("release"),
        )


# ---------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationDuplicateKeysConfig:
    fail_on_duplicates: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationDuplicateKeysConfig":
        return cls(
            fail_on_duplicates=require_bool(
                cfg.get("fail_on_duplicates"),
                key=(
                    "validation.method.structural."
                    "duplicate_keys.fail_on_duplicates"
                ),
            )
        )


@dataclass(frozen=True)
class ValidationStructuralConfig:
    required_fields: tuple[str, ...]
    canonical_key: tuple[str, ...]
    duplicate_keys: ValidationDuplicateKeysConfig
    require_positive_value: bool
    require_nonnegative_quantity: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationStructuralConfig":

        return cls(
            required_fields=_require_str_list(
                cfg.get("required_fields"),
                key=(
                    "validation.method.structural."
                    "required_fields"
                ),
            ),
            canonical_key=_require_str_list(
                cfg.get("canonical_key"),
                key=(
                    "validation.method.structural."
                    "canonical_key"
                ),
            ),
            duplicate_keys=ValidationDuplicateKeysConfig.from_dict(
                require_dict(
                    cfg.get("duplicate_keys"),
                    key=(
                        "validation.method.structural."
                        "duplicate_keys"
                    ),
                )
            ),
            require_positive_value=require_bool(
                cfg.get("require_positive_value"),
                key=(
                    "validation.method.structural."
                    "require_positive_value"
                ),
            ),
            require_nonnegative_quantity=require_bool(
                cfg.get("require_nonnegative_quantity"),
                key=(
                    "validation.method.structural."
                    "require_nonnegative_quantity"
                ),
            ),
        )



# ---------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationMethodConfig:
    structural: ValidationStructuralConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationMethodConfig":
        return cls(
            structural=ValidationStructuralConfig.from_dict(
                require_dict(
                    cfg.get("structural"),
                    key="validation.method.structural",
                )
            ),
        )


# ---------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReleaseColumnsConfig:
    year: str
    reporter: str
    partner: str
    flow: str
    hs_code: str
    quantity_kg: str
    trade_value_usd: str
    trade_value_constant_usd: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationReleaseColumnsConfig":
        return cls(
            year=require_str(
                cfg.get("year"),
                key="validation.release.columns.year",
            ),
            reporter=require_str(
                cfg.get("reporter"),
                key="validation.release.columns.reporter",
            ),
            partner=require_str(
                cfg.get("partner"),
                key="validation.release.columns.partner",
            ),
            flow=require_str(
                cfg.get("flow"),
                key="validation.release.columns.flow",
            ),
            hs_code=require_str(
                cfg.get("hs_code"),
                key="validation.release.columns.hs_code",
            ),
            quantity_kg=require_str(
                cfg.get("quantity_kg"),
                key="validation.release.columns.quantity_kg",
            ),
            trade_value_usd=require_str(
                cfg.get("trade_value_usd"),
                key=(
                    "validation.release.columns."
                    "trade_value_usd"
                ),
            ),
            trade_value_constant_usd=require_str(
                cfg.get("trade_value_constant_usd"),
                key=(
                    "validation.release.columns."
                    "trade_value_constant_usd"
                ),
            ),
        )

    def mapping(self) -> dict[str, str]:
        return {
            "year": self.year,
            "reporter": self.reporter,
            "partner": self.partner,
            "flow": self.flow,
            "hs_code": self.hs_code,
            "quantity_kg": self.quantity_kg,
            "trade_value_usd": self.trade_value_usd,
            "trade_value_constant_usd": (
                self.trade_value_constant_usd
            ),
        }


@dataclass(frozen=True)
class ValidationReleaseConfig:
    columns: ValidationReleaseColumnsConfig
    sort_by: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationReleaseConfig":
        return cls(
            columns=ValidationReleaseColumnsConfig.from_dict(
                require_dict(
                    cfg.get("columns"),
                    key="validation.release.columns",
                )
            ),
            sort_by=_require_str_list(
                cfg.get("sort_by"),
                key="validation.release.sort_by",
            ),
        )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationDiagnosticsColumnsConfig:
    year: str
    product: str
    quantity: str
    nominal_value: str
    constant_value: str
    export_quantity: str
    import_quantity: str
    export_estimated: str
    import_estimated: str
    quantity_relative_difference: str
    unit_value_reference: str
    unit_value_reference_observations: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationDiagnosticsColumnsConfig":
        return cls(
            year=require_str(
                cfg.get("year"),
                key="validation.diagnostics.columns.year",
            ),
            product=require_str(
                cfg.get("product"),
                key="validation.diagnostics.columns.product",
            ),
            quantity=require_str(
                cfg.get("quantity"),
                key="validation.diagnostics.columns.quantity",
            ),
            nominal_value=require_str(
                cfg.get("nominal_value"),
                key="validation.diagnostics.columns.nominal_value",
            ),
            constant_value=require_str(
                cfg.get("constant_value"),
                key="validation.diagnostics.columns.constant_value",
            ),
            export_quantity=require_str(
                cfg.get("export_quantity"),
                key="validation.diagnostics.columns.export_quantity",
            ),
            import_quantity=require_str(
                cfg.get("import_quantity"),
                key="validation.diagnostics.columns.import_quantity",
            ),
            export_estimated=require_str(
                cfg.get("export_estimated"),
                key="validation.diagnostics.columns.export_estimated",
            ),
            import_estimated=require_str(
                cfg.get("import_estimated"),
                key="validation.diagnostics.columns.import_estimated",
            ),
            quantity_relative_difference=require_str(
                cfg.get("quantity_relative_difference"),
                key=(
                    "validation.diagnostics.columns."
                    "quantity_relative_difference"
                ),
            ),
            unit_value_reference=require_str(
                cfg.get("unit_value_reference"),
                key="validation.diagnostics.columns.unit_value_reference",
            ),
            unit_value_reference_observations=require_str(
                cfg.get("unit_value_reference_observations"),
                key=(
                    "validation.diagnostics.columns."
                    "unit_value_reference_observations"
                ),
            ),
        )


@dataclass(frozen=True)
class ValidationDiagnosticsTrainingPopulationConfig:
    require_bilateral_quantities: bool
    require_non_estimated_quantities: bool
    disagreement_thresholds: tuple[float, ...]
    unit_value_reference_min_observations: tuple[int, ...]
    unit_value_log_distance_thresholds: tuple[float, ...]

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationDiagnosticsTrainingPopulationConfig":
        return cls(
            require_bilateral_quantities=require_bool(
                cfg.get("require_bilateral_quantities"),
                key=(
                    "validation.diagnostics.training_population."
                    "require_bilateral_quantities"
                ),
            ),
            require_non_estimated_quantities=require_bool(
                cfg.get("require_non_estimated_quantities"),
                key=(
                    "validation.diagnostics.training_population."
                    "require_non_estimated_quantities"
                ),
            ),
            disagreement_thresholds=_require_float_list(
                cfg.get("disagreement_thresholds"),
                key=(
                    "validation.diagnostics.training_population."
                    "disagreement_thresholds"
                ),
                min_value=0.0,
            ),
            unit_value_reference_min_observations=_require_int_list(
                cfg.get("unit_value_reference_min_observations"),
                key=(
                    "validation.diagnostics.training_population."
                    "unit_value_reference_min_observations"
                ),
                min_value=1,
            ),
            unit_value_log_distance_thresholds=_require_float_list(
                cfg.get("unit_value_log_distance_thresholds"),
                key=(
                    "validation.diagnostics.training_population."
                    "unit_value_log_distance_thresholds"
                ),
                min_value=0.0,
            ),
        )


@dataclass(frozen=True)
class ValidationDiagnosticsTargetConfig:
    transform: Literal["none", "log"]

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationDiagnosticsTargetConfig":
        transform = require_str(
            cfg.get("transform"),
            key="validation.diagnostics.target.transform",
        ).strip().lower()

        allowed = {"none", "log"}
        if transform not in allowed:
            raise ValueError(
                "validation.diagnostics.target.transform must be one of "
                f"{sorted(allowed)}, got {transform!r}."
            )

        return cls(transform=transform)


@dataclass(frozen=True)
class ValidationDiagnosticsSelectionConfig:
    enabled: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationDiagnosticsSelectionConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key="validation.diagnostics.selection.enabled",
            )
        )


@dataclass(frozen=True)
class ValidationDiagnosticsConfig:
    columns: ValidationDiagnosticsColumnsConfig
    training_population: ValidationDiagnosticsTrainingPopulationConfig
    target: ValidationDiagnosticsTargetConfig
    selection: ValidationDiagnosticsSelectionConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationDiagnosticsConfig":
        return cls(
            columns=ValidationDiagnosticsColumnsConfig.from_dict(
                require_dict(
                    cfg.get("columns"),
                    key="validation.diagnostics.columns",
                )
            ),
            training_population=ValidationDiagnosticsTrainingPopulationConfig.from_dict(
                require_dict(
                    cfg.get("training_population"),
                    key="validation.diagnostics.training_population",
                )
            ),
            target=ValidationDiagnosticsTargetConfig.from_dict(
                require_dict(
                    cfg.get("target"),
                    key="validation.diagnostics.target",
                )
            ),
            selection=ValidationDiagnosticsSelectionConfig.from_dict(
                require_dict(
                    cfg.get("selection"),
                    key="validation.diagnostics.selection",
                )
            ),
        )


# ---------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationFailOnConfig:
    structural_failure: bool
    duplicate_keys: bool
    invalid_hs_codes: bool
    invalid_years: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationFailOnConfig":
        return cls(
            structural_failure=require_bool(
                cfg.get("structural_failure"),
                key=(
                    "validation.decision.fail_on."
                    "structural_failure"
                ),
            ),
            duplicate_keys=require_bool(
                cfg.get("duplicate_keys"),
                key=(
                    "validation.decision.fail_on."
                    "duplicate_keys"
                ),
            ),
            invalid_hs_codes=require_bool(
                cfg.get("invalid_hs_codes"),
                key=(
                    "validation.decision.fail_on."
                    "invalid_hs_codes"
                ),
            ),
            invalid_years=require_bool(
                cfg.get("invalid_years"),
                key=(
                    "validation.decision.fail_on."
                    "invalid_years"
                ),
            ),
        )


@dataclass(frozen=True)
class ValidationDecisionConfig:
    fail_on: ValidationFailOnConfig
    allow_flagged_outliers: bool
    allow_imputed_rows: bool
    ready_flag: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationDecisionConfig":
        return cls(
            fail_on=ValidationFailOnConfig.from_dict(
                require_dict(
                    cfg.get("fail_on"),
                    key="validation.decision.fail_on",
                )
            ),
            allow_flagged_outliers=require_bool(
                cfg.get("allow_flagged_outliers"),
                key=(
                    "validation.decision."
                    "allow_flagged_outliers"
                ),
            ),
            allow_imputed_rows=require_bool(
                cfg.get("allow_imputed_rows"),
                key=(
                    "validation.decision."
                    "allow_imputed_rows"
                ),
            ),
            ready_flag=require_str(
                cfg.get("ready_flag"),
                key="validation.decision.ready_flag",
            ),
        )


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationLoggingConfig:
    level: str
    subdir: str
    file: str
    timestamp: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationLoggingConfig":
        return cls(
            level=require_str(
                cfg.get("level"),
                key="validation.logging.level",
            ).strip().upper(),
            subdir=require_str(
                cfg.get("subdir"),
                key="validation.logging.subdir",
                allow_empty=True,
            ),
            file=require_str(
                cfg.get("file"),
                key="validation.logging.file",
            ),
            timestamp=require_bool(
                cfg.get("timestamp"),
                key="validation.logging.timestamp",
            ),
        )

    def resolve_log_dir(self) -> Path:
        base = get_output_dir("logs")

        if not self.subdir:
            return base

        parts = list(Path(self.subdir).parts)

        if parts and parts[0].lower() == "logs":
            parts = parts[1:]

        return base / Path(*parts) if parts else base

    def resolve_log_file(self) -> Path:
        return self.resolve_log_dir() / self.file


# ---------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationArtifactsConfig:
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
    ) -> "ValidationArtifactsConfig":
        return cls(
            enable=require_bool(
                cfg.get("enable"),
                key="validation.artifacts.enable",
            ),
            kind=require_str(
                cfg.get("kind"),
                key="validation.artifacts.kind",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="validation.artifacts.subdir",
                allow_empty=True,
            ),
            filename=require_str(
                cfg.get("filename"),
                key="validation.artifacts.filename",
            ),
            timestamp=require_bool(
                cfg.get("timestamp"),
                key="validation.artifacts.timestamp",
            ),
            write_latest=require_bool(
                cfg.get("write_latest"),
                key="validation.artifacts.write_latest",
            ),
        )

    def resolve_dir(self) -> Path:
        base = get_output_dir(self.kind)
        return base / self.subdir if self.subdir else base

    def timestamped_path(self, run_id: str) -> Path:
        filename = (
            f"{self.filename}_{run_id}.md"
            if self.timestamp
            else f"{self.filename}.md"
        )

        return self.resolve_dir() / filename

    def latest_path(self) -> Path:
        return (
            self.resolve_dir()
            / f"{self.filename}_latest.md"
        )


# ---------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationConfig:
    input: ValidationInputConfig
    output: ValidationOutputConfig
    products: ValidationProductsConfig
    method: ValidationMethodConfig
    diagnostics: ValidationDiagnosticsConfig
    release: ValidationReleaseConfig
    decision: ValidationDecisionConfig
    logging: ValidationLoggingConfig
    artifacts: ValidationArtifactsConfig
    scope: PipelineScopeConfig
    engine: PipelineEngineConfig

    @classmethod
    def from_yaml(
        cls,
        config_path: Path | None = None,
        *,
        scope: PipelineScopeConfig | None = None,
        engine: PipelineEngineConfig | None = None,
    ) -> "ValidationConfig":
        return cls.from_dict(
            load_validation_config(config_path),
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
    ) -> "ValidationConfig":
        if scope is None:
            raise ValueError(
                "ValidationConfig requires pipeline scope."
            )

        if engine is None:
            raise ValueError(
                "ValidationConfig requires pipeline engine."
            )

        root = require_dict(
            raw.get("validation"),
            key="validation",
        )

        rendered = require_dict(
            _render_value(
                root,
                target_hs=scope.target_hs,
            ),
            key="validation",
        )

        return cls(
            input=ValidationInputConfig.from_dict(
                require_dict(
                    rendered.get("input"),
                    key="validation.input",
                )
            ),
            output=ValidationOutputConfig.from_dict(
                require_dict(
                    rendered.get("output"),
                    key="validation.output",
                )
            ),
            products=ValidationProductsConfig.from_dict(
                require_dict(
                    rendered.get("products"),
                    key="validation.products",
                )
            ),
            method=ValidationMethodConfig.from_dict(
                require_dict(
                    rendered.get("method"),
                    key="validation.method",
                )
            ),
            diagnostics=ValidationDiagnosticsConfig.from_dict(
                require_dict(
                    rendered.get("diagnostics"),
                    key="validation.diagnostics",
                )
            ),
            release=ValidationReleaseConfig.from_dict(
                require_dict(
                    rendered.get("release"),
                    key="validation.release",
                )
            ),
            decision=ValidationDecisionConfig.from_dict(
                require_dict(
                    rendered.get("decision"),
                    key="validation.decision",
                )
            ),
            logging=ValidationLoggingConfig.from_dict(
                require_dict(
                    rendered.get("logging"),
                    key="validation.logging",
                )
            ),
            artifacts=ValidationArtifactsConfig.from_dict(
                require_dict(
                    rendered.get("artifacts"),
                    key="validation.artifacts",
                )
            ),
            scope=scope,
            engine=engine,
        )

    def input_path(self) -> Path:
        return self.input.resolve_path()

    def output_dir(self) -> Path:
        return self.output.resolve_dir()

    def product_path(self, product_name: str) -> Path:
        products = {
            "prepared_input": self.products.prepared_input,
            "structural_audit": self.products.structural_audit,
            "quantity_selection": self.products.quantity_selection,
            "quantity_selection_audit": self.products.quantity_selection_audit,
            "model_diagnostics": self.products.diagnostics,
            "diagnostics": self.products.diagnostics,
            "validated_trade": self.products.validated_trade,
            "consistency_audit": self.products.consistency_audit,
            "release": self.products.release,
        }

        if product_name not in products:
            raise KeyError(
                f"Unknown Validation product: {product_name!r}"
            )

        return (
            self.output_dir()
            / products[product_name].filename
        )

    def resolved_product_paths(self) -> dict[str, Path]:
        return {
            name: self.product_path(name)
            for name in (
                "prepared_input",
                "structural_audit",
                "quantity_selection",
                "quantity_selection_audit",
                "model_diagnostics",
                "diagnostics",
                "validated_trade",
                "consistency_audit",
                "release",
            )
        }

    def enabled_products(self) -> dict[str, bool]:
        return {
            "prepared_input": self.products.prepared_input.enable,
            "structural_audit": self.products.structural_audit.enable,
            "quantity_selection": self.products.quantity_selection.enable,
            "quantity_selection_audit": self.products.quantity_selection_audit.enable,
            "model_diagnostics": self.products.diagnostics.enable,
            "diagnostics": self.products.diagnostics.enable,
            "validated_trade": self.products.validated_trade.enable,
            "consistency_audit": self.products.consistency_audit.enable,
            "release": self.products.release.enable,
        }

    def log_path(self) -> Path:
        return self.logging.resolve_log_file()

    def artifacts_path(self, run_id: str) -> Path:
        return self.artifacts.timestamped_path(run_id)

    def artifacts_latest_path(self) -> Path:
        return self.artifacts.latest_path()