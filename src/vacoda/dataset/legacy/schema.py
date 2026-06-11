"""
vacoda.dataset.schema

Schema utilities for validating and standardizing dataset DataFrames.

This module does not load YAML directly. It receives schema config dictionaries
from vacoda.config.dataset.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from vacoda.config.dataset import (
    DatasetSchemaConfig,
    RawSchemaConfig,
    StandardizedSchemaConfig,
)


def find_missing_fields(
    df: pd.DataFrame,
    fields: list[str],) -> list[str]:
    """Return configured fields that are missing from the DataFrame."""
    return [field for field in fields if field not in df.columns]


def find_extra_fields(
    df: pd.DataFrame,
    allowed_fields: list[str],) -> list[str]:
    """Return DataFrame fields not listed in allowed fields."""
    allowed = set(allowed_fields)
    return [field for field in df.columns if field not in allowed]


def validate_required_fields(
    df: pd.DataFrame,
    required_fields: list[str],
    *,
    context: str = "dataset",) -> None:
    """Raise if required fields are missing."""
    missing = find_missing_fields(df, required_fields)

    if missing:
        raise KeyError(
            f"Missing required field(s) in {context}: {missing}"
        )


def validate_raw_schema(
    df: pd.DataFrame,
    schema: DatasetSchemaConfig,
    *,
    context: str = "dataset.raw",) -> None:
    validate_required_fields(
        df,
        schema.raw.required_fields,
        context=context,
    )


def rename_to_standardized(
    df: pd.DataFrame,
    schema: DatasetSchemaConfig,) -> pd.DataFrame:
    """Rename raw source fields to standardized vacoda names."""
    field_mapping = schema.standardized.field_mapping
    return df.rename(columns=field_mapping)


def validate_standardized_schema(
    df: pd.DataFrame,
    schema: DatasetSchemaConfig,
    *,
    context: str = "dataset.standardized",) -> None:
    """Validate standardized fields after renaming."""
    required_fields = schema.standardized.required_fields

    validate_required_fields(
        df,
        required_fields,
        context=context,
    )

    if schema.strict:
        field_mapping = schema.standardized.field_mapping
        allowed_fields = list(field_mapping.values())

        extra_fields = find_extra_fields(df, allowed_fields)

        if extra_fields:
            raise KeyError(
                f"Unexpected extra field(s) in {context}: {extra_fields}"
            )


def apply_standardized_dtypes(
    df: pd.DataFrame,
    schema: DatasetSchemaConfig,
    *,
    strict: bool = False,
    context: str = "dataset.standardized",) -> pd.DataFrame:
    """
    Apply standardized dtype mapping to fields that exist.

    If strict=True, raise when a dtype-configured field is missing.
    """
    dtype_mapping = schema.standardized.dtypes

    existing_mapping = {
        field: dtype
        for field, dtype in dtype_mapping.items()
        if field in df.columns
    }

    missing_dtype_fields = [
        field
        for field in dtype_mapping
        if field not in df.columns
    ]

    if strict and missing_dtype_fields:
        raise KeyError(
            f"Configured dtype field(s) missing in {context}: "
            f"{missing_dtype_fields}"
        )

    if not existing_mapping:
        return df

    try:
        return df.astype(existing_mapping)
    except Exception as exc:
        raise TypeError(
            f"Failed to apply dtype mapping in {context}. "
            f"Mapping attempted: {existing_mapping}"
        ) from exc


def standardize_dataset_schema(
    df: pd.DataFrame,
    schema: DatasetSchemaConfig,
    *,
    context: str = "dataset",) -> pd.DataFrame:
    """
    Validate raw schema, rename to standardized fields, validate again,
    and apply standardized dtypes.
    """
    validate_raw_schema(
        df,
        schema,
        context=f"{context}.raw",
    )

    df = rename_to_standardized(df, schema)

    validate_standardized_schema(
        df,
        schema,
        context=f"{context}.standardized",
    )

    return apply_standardized_dtypes(
        df,
        schema,
        strict=False,
        context=f"{context}.standardized",
    )