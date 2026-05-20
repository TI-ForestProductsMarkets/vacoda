import pandas as pd
import pytest

from vacoda.dataset.schema import (
    standardize_dataset_schema,
    validate_raw_schema,
    rename_to_standardized,
)


def make_schema_cfg():
    return {
        "name": "comtrade_annual_hs_to_vacoda_base",
        "strict": False,
        "raw": {
            "required_columns": [
                "RefYear",
                "ReporterCode",
                "ReporterISO",
                "PartnerCode",
                "PartnerISO",
                "FlowCode",
                "CmdCode",
                "ClassificationCode",
                "PrimaryValue",
                "NetWgt",
            ],
        },
        "standardized": {
            "name": "vacoda_base_trade",
            "column_mapping": {
                "RefYear": "year",
                "ReporterCode": "reporter_code",
                "ReporterISO": "reporter_iso",
                "PartnerCode": "partner_code",
                "PartnerISO": "partner_iso",
                "FlowCode": "flow_code",
                "CmdCode": "commodity_code",
                "ClassificationCode": "hs_version",
                "PrimaryValue": "value_usd",
                "NetWgt": "net_weight_kg",
            },
            "required_columns": [
                "year",
                "reporter_code",
                "reporter_iso",
                "partner_code",
                "partner_iso",
                "flow_code",
                "commodity_code",
                "hs_version",
                "value_usd",
                "net_weight_kg",
            ],
            "dtypes": {
                "year": "int16",
                "reporter_code": "int32",
                "reporter_iso": "string",
                "partner_code": "int32",
                "partner_iso": "string",
                "flow_code": "string",
                "commodity_code": "string",
                "hs_version": "string",
                "value_usd": "float64",
                "net_weight_kg": "float64",
            },
        },
    }


def make_raw_df():
    return pd.DataFrame(
        {
            "RefYear": [2024],
            "ReporterCode": [56],
            "ReporterISO": ["BEL"],
            "PartnerCode": [0],
            "PartnerISO": ["W00"],
            "FlowCode": ["X"],
            "CmdCode": ["440112"],
            "ClassificationCode": ["H6"],
            "PrimaryValue": [4876902.599],
            "NetWgt": [36984672.29],
            "ExtraRawColumn": ["allowed_when_strict_false"],
        }
    )


def test_validate_raw_schema_raises_for_missing_required_column():
    schema_cfg = make_schema_cfg()
    df = make_raw_df().drop(columns=["PrimaryValue"])

    with pytest.raises(KeyError, match="Missing required column"):
        validate_raw_schema(df, schema_cfg)


def test_rename_to_standardized_renames_columns():
    schema_cfg = make_schema_cfg()
    df = make_raw_df()

    out = rename_to_standardized(df, schema_cfg)

    assert "PrimaryValue" not in out.columns
    assert "value_usd" in out.columns
    assert "NetWgt" not in out.columns
    assert "net_weight_kg" in out.columns


def test_standardize_dataset_schema_returns_standardized_columns():
    schema_cfg = make_schema_cfg()
    df = make_raw_df()

    out = standardize_dataset_schema(df, schema_cfg)

    assert "year" in out.columns
    assert "reporter_code" in out.columns
    assert "commodity_code" in out.columns
    assert "value_usd" in out.columns
    assert "net_weight_kg" in out.columns


def test_standardize_dataset_schema_applies_dtypes():
    schema_cfg = make_schema_cfg()
    df = make_raw_df()

    out = standardize_dataset_schema(df, schema_cfg)

    assert str(out["year"].dtype) == "int16"
    assert str(out["reporter_code"].dtype) == "int32"
    assert str(out["value_usd"].dtype) == "float64"


def test_strict_false_allows_extra_unmapped_columns():
    schema_cfg = make_schema_cfg()
    schema_cfg["strict"] = False

    df = make_raw_df()
    out = standardize_dataset_schema(df, schema_cfg)

    assert "ExtraRawColumn" in out.columns