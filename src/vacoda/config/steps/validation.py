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
    missingness: ValidationProductConfig
    unit_value_diagnostics: ValidationProductConfig
    outliers: ValidationProductConfig
    consistency_audit: ValidationProductConfig
    provenance: ValidationProductConfig
    validated_trade: ValidationProductConfig
    release: ValidationProductConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationProductsConfig":
        def product(name: str) -> ValidationProductConfig:
            key = f"validation.products.{name}"

            return ValidationProductConfig.from_dict(
                require_dict(
                    cfg.get(name),
                    key=key,
                ),
                key=key,
            )

        return cls(
            prepared_input=product("prepared_input"),
            structural_audit=product("structural_audit"),
            missingness=product("missingness"),
            unit_value_diagnostics=product(
                "unit_value_diagnostics"
            ),
            outliers=product("outliers"),
            consistency_audit=product("consistency_audit"),
            provenance=product("provenance"),
            validated_trade=product("validated_trade"),
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
    valid_flows: tuple[str, ...]
    require_positive_value: bool
    require_nonnegative_quantity: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationStructuralConfig":
        valid_flows = _require_str_list(
            cfg.get("valid_flows"),
            key="validation.method.structural.valid_flows",
        )

        allowed_flows = {"M", "X"}
        unknown = sorted(set(valid_flows) - allowed_flows)

        if unknown:
            raise ValueError(
                "Unknown validation flows: "
                + ", ".join(unknown)
            )

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
            valid_flows=valid_flows,
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
# Missingness
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationMissingnessConfig:
    fields: tuple[str, ...]
    summarize_by: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationMissingnessConfig":
        return cls(
            fields=_require_str_list(
                cfg.get("fields"),
                key="validation.method.missingness.fields",
            ),
            summarize_by=_require_str_list(
                cfg.get("summarize_by"),
                key=(
                    "validation.method.missingness."
                    "summarize_by"
                ),
            ),
        )


# ---------------------------------------------------------------------
# Unit value
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationUnitValueConfig:
    value_column: str
    quantity_column: str
    output_column: str
    require_positive_value: bool
    require_positive_quantity: bool
    transform: Literal["none", "log"]
    peer_group: tuple[str, ...]
    fallback_peer_group: tuple[str, ...]
    minimum_group_rows: int

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationUnitValueConfig":
        transform = require_str(
            cfg.get("transform"),
            key="validation.method.unit_value.transform",
        ).strip().lower()

        allowed = {"none", "log"}

        if transform not in allowed:
            raise ValueError(
                "validation.method.unit_value.transform must be one "
                f"of {sorted(allowed)}, got {transform!r}."
            )

        return cls(
            value_column=require_str(
                cfg.get("value_column"),
                key=(
                    "validation.method.unit_value."
                    "value_column"
                ),
            ),
            quantity_column=require_str(
                cfg.get("quantity_column"),
                key=(
                    "validation.method.unit_value."
                    "quantity_column"
                ),
            ),
            output_column=require_str(
                cfg.get("output_column"),
                key=(
                    "validation.method.unit_value."
                    "output_column"
                ),
            ),
            require_positive_value=require_bool(
                cfg.get("require_positive_value"),
                key=(
                    "validation.method.unit_value."
                    "require_positive_value"
                ),
            ),
            require_positive_quantity=require_bool(
                cfg.get("require_positive_quantity"),
                key=(
                    "validation.method.unit_value."
                    "require_positive_quantity"
                ),
            ),
            transform=transform,
            peer_group=_require_str_list(
                cfg.get("peer_group"),
                key="validation.method.unit_value.peer_group",
            ),
            fallback_peer_group=_require_str_list(
                cfg.get("fallback_peer_group"),
                key=(
                    "validation.method.unit_value."
                    "fallback_peer_group"
                ),
            ),
            minimum_group_rows=require_int(
                cfg.get("minimum_group_rows"),
                key=(
                    "validation.method.unit_value."
                    "minimum_group_rows"
                ),
                min_value=1,
            ),
        )


# ---------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationOutlierThresholdsConfig:
    warning_score: float
    severe_score: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationOutlierThresholdsConfig":
        result = cls(
            warning_score=require_float(
                cfg.get("warning_score"),
                key=(
                    "validation.method.outliers."
                    "thresholds.warning_score"
                ),
                min_value=0.0,
            ),
            severe_score=require_float(
                cfg.get("severe_score"),
                key=(
                    "validation.method.outliers."
                    "thresholds.severe_score"
                ),
                min_value=0.0,
            ),
        )

        if result.severe_score < result.warning_score:
            raise ValueError(
                "validation.method.outliers.thresholds."
                "severe_score must be >= warning_score."
            )

        return result


@dataclass(frozen=True)
class ValidationOutliersConfig:
    method: Literal["robust_mad"]
    column: str
    group_by: tuple[str, ...]
    thresholds: ValidationOutlierThresholdsConfig
    action: Literal["flag"]
    remove: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationOutliersConfig":
        method = require_str(
            cfg.get("method"),
            key="validation.method.outliers.method",
        ).strip().lower()

        if method != "robust_mad":
            raise ValueError(
                "validation.method.outliers.method currently "
                f"must be 'robust_mad', got {method!r}."
            )

        action = require_str(
            cfg.get("action"),
            key="validation.method.outliers.action",
        ).strip().lower()

        if action != "flag":
            raise ValueError(
                "validation.method.outliers.action currently "
                f"must be 'flag', got {action!r}."
            )

        remove = require_bool(
            cfg.get("remove"),
            key="validation.method.outliers.remove",
        )

        if remove:
            raise ValueError(
                "validation.method.outliers.remove currently "
                "must be false. Validation flags outliers "
                "rather than deleting them."
            )

        return cls(
            method=method,
            column=require_str(
                cfg.get("column"),
                key="validation.method.outliers.column",
            ),
            group_by=_require_str_list(
                cfg.get("group_by"),
                key="validation.method.outliers.group_by",
            ),
            thresholds=ValidationOutlierThresholdsConfig.from_dict(
                require_dict(
                    cfg.get("thresholds"),
                    key=(
                        "validation.method.outliers."
                        "thresholds"
                    ),
                )
            ),
            action=action,
            remove=remove,
        )


# ---------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationEnabledCheckConfig:
    enabled: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "ValidationEnabledCheckConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key=f"{key}.enabled",
            )
        )


@dataclass(frozen=True)
class ValidationToleranceCheckConfig:
    enabled: bool
    relative_tolerance: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "ValidationToleranceCheckConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key=f"{key}.enabled",
            ),
            relative_tolerance=require_float(
                cfg.get("relative_tolerance"),
                key=f"{key}.relative_tolerance",
                min_value=0.0,
            ),
        )


@dataclass(frozen=True)
class ValidationConsistencyConfig:
    positive_value_requires_quantity: ValidationEnabledCheckConfig
    positive_quantity_requires_value: ValidationEnabledCheckConfig
    constant_value_consistency: ValidationToleranceCheckConfig
    target_hs_validity: ValidationEnabledCheckConfig
    year_scope_validity: ValidationEnabledCheckConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationConsistencyConfig":
        return cls(
            positive_value_requires_quantity=(
                ValidationEnabledCheckConfig.from_dict(
                    require_dict(
                        cfg.get(
                            "positive_value_requires_quantity"
                        ),
                        key=(
                            "validation.method.consistency."
                            "positive_value_requires_quantity"
                        ),
                    ),
                    key=(
                        "validation.method.consistency."
                        "positive_value_requires_quantity"
                    ),
                )
            ),
            positive_quantity_requires_value=(
                ValidationEnabledCheckConfig.from_dict(
                    require_dict(
                        cfg.get(
                            "positive_quantity_requires_value"
                        ),
                        key=(
                            "validation.method.consistency."
                            "positive_quantity_requires_value"
                        ),
                    ),
                    key=(
                        "validation.method.consistency."
                        "positive_quantity_requires_value"
                    ),
                )
            ),
            constant_value_consistency=(
                ValidationToleranceCheckConfig.from_dict(
                    require_dict(
                        cfg.get("constant_value_consistency"),
                        key=(
                            "validation.method.consistency."
                            "constant_value_consistency"
                        ),
                    ),
                    key=(
                        "validation.method.consistency."
                        "constant_value_consistency"
                    ),
                )
            ),
            target_hs_validity=(
                ValidationEnabledCheckConfig.from_dict(
                    require_dict(
                        cfg.get("target_hs_validity"),
                        key=(
                            "validation.method.consistency."
                            "target_hs_validity"
                        ),
                    ),
                    key=(
                        "validation.method.consistency."
                        "target_hs_validity"
                    ),
                )
            ),
            year_scope_validity=(
                ValidationEnabledCheckConfig.from_dict(
                    require_dict(
                        cfg.get("year_scope_validity"),
                        key=(
                            "validation.method.consistency."
                            "year_scope_validity"
                        ),
                    ),
                    key=(
                        "validation.method.consistency."
                        "year_scope_validity"
                    ),
                )
            ),
        )


# ---------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationImputationTargetConfig:
    enabled: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "ValidationImputationTargetConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key=f"{key}.enabled",
            )
        )


@dataclass(frozen=True)
class ValidationImputationConfig:
    enabled: bool
    preserve_original_values: bool
    quantity: ValidationImputationTargetConfig
    value: ValidationImputationTargetConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationImputationConfig":
        result = cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key="validation.method.imputation.enabled",
            ),
            preserve_original_values=require_bool(
                cfg.get("preserve_original_values"),
                key=(
                    "validation.method.imputation."
                    "preserve_original_values"
                ),
            ),
            quantity=ValidationImputationTargetConfig.from_dict(
                require_dict(
                    cfg.get("quantity"),
                    key=(
                        "validation.method.imputation."
                        "quantity"
                    ),
                ),
                key="validation.method.imputation.quantity",
            ),
            value=ValidationImputationTargetConfig.from_dict(
                require_dict(
                    cfg.get("value"),
                    key="validation.method.imputation.value",
                ),
                key="validation.method.imputation.value",
            ),
        )

        if not result.enabled and (
            result.quantity.enabled
            or result.value.enabled
        ):
            raise ValueError(
                "validation.method.imputation is disabled, "
                "but an imputation target is enabled."
            )

        return result


# ---------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationMethodConfig:
    structural: ValidationStructuralConfig
    missingness: ValidationMissingnessConfig
    unit_value: ValidationUnitValueConfig
    outliers: ValidationOutliersConfig
    consistency: ValidationConsistencyConfig
    imputation: ValidationImputationConfig

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
            missingness=ValidationMissingnessConfig.from_dict(
                require_dict(
                    cfg.get("missingness"),
                    key="validation.method.missingness",
                )
            ),
            unit_value=ValidationUnitValueConfig.from_dict(
                require_dict(
                    cfg.get("unit_value"),
                    key="validation.method.unit_value",
                )
            ),
            outliers=ValidationOutliersConfig.from_dict(
                require_dict(
                    cfg.get("outliers"),
                    key="validation.method.outliers",
                )
            ),
            consistency=ValidationConsistencyConfig.from_dict(
                require_dict(
                    cfg.get("consistency"),
                    key="validation.method.consistency",
                )
            ),
            imputation=ValidationImputationConfig.from_dict(
                require_dict(
                    cfg.get("imputation"),
                    key="validation.method.imputation",
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
# Diagnostics
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationDiagnosticsConfig:
    unit_value_percentiles: tuple[float, ...]
    missingness_dimensions: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValidationDiagnosticsConfig":
        return cls(
            unit_value_percentiles=_require_float_list(
                cfg.get("unit_value_percentiles"),
                key=(
                    "validation.diagnostics."
                    "unit_value_percentiles"
                ),
                min_value=0.0,
                max_value=1.0,
            ),
            missingness_dimensions=_require_str_list(
                cfg.get("missingness_dimensions"),
                key=(
                    "validation.diagnostics."
                    "missingness_dimensions"
                ),
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
    release: ValidationReleaseConfig
    decision: ValidationDecisionConfig
    diagnostics: ValidationDiagnosticsConfig
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
            diagnostics=ValidationDiagnosticsConfig.from_dict(
                require_dict(
                    rendered.get("diagnostics"),
                    key="validation.diagnostics",
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
            "missingness": self.products.missingness,
            "unit_value_diagnostics": (
                self.products.unit_value_diagnostics
            ),
            "outliers": self.products.outliers,
            "consistency_audit": (
                self.products.consistency_audit
            ),
            "provenance": self.products.provenance,
            "validated_trade": self.products.validated_trade,
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
                "missingness",
                "unit_value_diagnostics",
                "outliers",
                "consistency_audit",
                "provenance",
                "validated_trade",
                "release",
            )
        }

    def enabled_products(self) -> dict[str, bool]:
        return {
            "prepared_input": self.products.prepared_input.enable,
            "structural_audit": (
                self.products.structural_audit.enable
            ),
            "missingness": self.products.missingness.enable,
            "unit_value_diagnostics": (
                self.products.unit_value_diagnostics.enable
            ),
            "outliers": self.products.outliers.enable,
            "consistency_audit": (
                self.products.consistency_audit.enable
            ),
            "provenance": self.products.provenance.enable,
            "validated_trade": self.products.validated_trade.enable,
            "release": self.products.release.enable,
        }

    def log_path(self) -> Path:
        return self.logging.resolve_log_file()

    def artifacts_path(self, run_id: str) -> Path:
        return self.artifacts.timestamped_path(run_id)

    def artifacts_latest_path(self) -> Path:
        return self.artifacts.latest_path()