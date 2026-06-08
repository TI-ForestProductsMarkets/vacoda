"""
vacoda.hs_sync.comtrade_mapping_audit

Audit normalized UNSD/Comtrade HS correlation mappings and resulting
target-HS coverage outputs.

This audit is intended to run after the Comtrade-only coverage flow.

It checks:

1. Normalized mapping file schema.
2. Direction metadata.
3. Relationship reversal correctness.
4. Code-format validity.
5. Duplicate/null mapping rows.
6. Relationship-label vs actual cardinality consistency.
7. Coverage-output invariants:
   observed pairs = easy + needs + missing
   rows/value are preserved across result classes
   missing mappings are zero, when expected.
8. Needs-weights readiness for equal-split allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from vacoda.io.reports import write_text_report
from vacoda.utils.logger import get_logger

if TYPE_CHECKING:
    from vacoda.config.hs_sync import HsSyncConfig


logger = get_logger("HS_SYNC.COMTRADE_MAPPING_AUDIT")


VALID_RELATIONSHIPS = {"1:1", "n:1", "1:n", "n:n"}

REVERSED_RELATIONSHIP = {
    "1:1": "1:1",
    "n:1": "1:n",
    "1:n": "n:1",
    "n:n": "n:n",
}

REQUIRED_MAPPING_COLUMNS = {
    "code.source",
    "code.target",
    "Relationship",
    "adjustment",
    "group.id",
    "source_hs",
    "target_hs",
    "source_system",
    "target_system",
    "relationship_original",
    "mapping_source",
    "mapping_file",
    "direction_reversed",
}

REQUIRED_COVERAGE_COLUMNS = {
    "source_hs",
    "source_cmd",
    "conversion_need",
    "n_rows",
    "value_total",
}


@dataclass(frozen=True)
class MappingExpectation:
    mapping_key: str
    source_hs: str
    target_hs: str
    source_system: str
    target_system: str
    direction_reversed: bool


EXPECTED_MAPPINGS: dict[str, MappingExpectation] = {
    "H2_to_H5": MappingExpectation(
        mapping_key="H2_to_H5",
        source_hs="H2",
        target_hs="H5",
        source_system="HS2002",
        target_system="HS2017",
        direction_reversed=True,
    ),
    "H3_to_H5": MappingExpectation(
        mapping_key="H3_to_H5",
        source_hs="H3",
        target_hs="H5",
        source_system="HS2007",
        target_system="HS2017",
        direction_reversed=True,
    ),
    "H4_to_H5": MappingExpectation(
        mapping_key="H4_to_H5",
        source_hs="H4",
        target_hs="H5",
        source_system="HS2012",
        target_system="HS2017",
        direction_reversed=True,
    ),
    "H6_to_H5": MappingExpectation(
        mapping_key="H6_to_H5",
        source_hs="H6",
        target_hs="H5",
        source_system="HS2022",
        target_system="HS2017",
        direction_reversed=False,
    ),
}


def _as_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert booleans that may have been read from CSV as bool/string.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
    )


def _status(condition: bool) -> str:
    return "pass" if condition else "fail"


def _format_number(value: float | int) -> str:
    if pd.isna(value):
        return ""

    value = float(value)

    if abs(value) >= 1e12:
        return f"{value / 1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.2f}K"

    return f"{value:.2f}"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected audit input does not exist: {path}")

    return pd.read_csv(path, dtype=str)


def _read_coverage_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected coverage product does not exist: {path}")

    df = pd.read_csv(path, dtype=str)

    for col in ["n_rows", "value_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def _relationship_counts(df: pd.DataFrame, *, file_key: str) -> pd.DataFrame:
    counts = (
        df["Relationship"]
        .value_counts(dropna=False)
        .rename_axis("relationship")
        .reset_index(name="n_rows")
        .sort_values("relationship")
    )
    counts.insert(0, "mapping_key", file_key)
    return counts


def _audit_mapping_schema(
    *,
    mapping_key: str,
    path: Path,
    df: pd.DataFrame,
) -> pd.DataFrame:
    missing_cols = sorted(REQUIRED_MAPPING_COLUMNS - set(df.columns))

    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "mapping_key": mapping_key,
            "check": "file_exists",
            "status": _status(path.exists()),
            "detail": str(path),
        }
    )

    rows.append(
        {
            "mapping_key": mapping_key,
            "check": "required_columns_present",
            "status": _status(len(missing_cols) == 0),
            "detail": "" if not missing_cols else ", ".join(missing_cols),
        }
    )

    rows.append(
        {
            "mapping_key": mapping_key,
            "check": "non_empty_file",
            "status": _status(len(df) > 0),
            "detail": f"{len(df):,} rows",
        }
    )

    if "code.source" in df.columns:
        null_source = int(df["code.source"].isna().sum())
        source_6_digit = df["code.source"].astype(str).str.fullmatch(r"\d{6}").fillna(False)
        rows.append(
            {
                "mapping_key": mapping_key,
                "check": "source_codes_are_six_digits",
                "status": _status(bool(source_6_digit.all())),
                "detail": f"invalid={int((~source_6_digit).sum()):,}, null={null_source:,}",
            }
        )

    if "code.target" in df.columns:
        null_target = int(df["code.target"].isna().sum())
        target_6_digit = df["code.target"].astype(str).str.fullmatch(r"\d{6}").fillna(False)
        rows.append(
            {
                "mapping_key": mapping_key,
                "check": "target_codes_are_six_digits",
                "status": _status(bool(target_6_digit.all())),
                "detail": f"invalid={int((~target_6_digit).sum()):,}, null={null_target:,}",
            }
        )

    if "Relationship" in df.columns:
        invalid_rel = sorted(set(df["Relationship"].dropna()) - VALID_RELATIONSHIPS)
        rows.append(
            {
                "mapping_key": mapping_key,
                "check": "valid_relationship_labels",
                "status": _status(len(invalid_rel) == 0),
                "detail": "" if not invalid_rel else ", ".join(invalid_rel),
            }
        )

    duplicate_count = int(
        df.duplicated(
            subset=[
                col
                for col in ["code.source", "code.target", "Relationship"]
                if col in df.columns
            ]
        ).sum()
    )
    rows.append(
        {
            "mapping_key": mapping_key,
            "check": "no_duplicate_mapping_rows",
            "status": _status(duplicate_count == 0),
            "detail": f"duplicates={duplicate_count:,}",
        }
    )

    return pd.DataFrame(rows)


def _audit_mapping_metadata(
    *,
    mapping_key: str,
    df: pd.DataFrame,
    expectation: MappingExpectation | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if expectation is None:
        return pd.DataFrame(
            [
                {
                    "mapping_key": mapping_key,
                    "check": "known_mapping_expectation",
                    "status": "warning",
                    "detail": "No expectation configured for mapping key.",
                }
            ]
        )

    checks = {
        "source_hs": expectation.source_hs,
        "target_hs": expectation.target_hs,
        "source_system": expectation.source_system,
        "target_system": expectation.target_system,
    }

    for col, expected in checks.items():
        if col not in df.columns:
            rows.append(
                {
                    "mapping_key": mapping_key,
                    "check": f"{col}_metadata",
                    "status": "fail",
                    "detail": f"missing column; expected={expected}",
                }
            )
            continue

        values = sorted(set(df[col].dropna().astype(str)))
        ok = values == [expected]
        rows.append(
            {
                "mapping_key": mapping_key,
                "check": f"{col}_metadata",
                "status": _status(ok),
                "detail": f"expected={expected}; observed={values}",
            }
        )

    if "direction_reversed" not in df.columns:
        rows.append(
            {
                "mapping_key": mapping_key,
                "check": "direction_reversed_metadata",
                "status": "fail",
                "detail": f"missing column; expected={expectation.direction_reversed}",
            }
        )
    else:
        observed = sorted(set(_as_bool_series(df["direction_reversed"]).dropna()))
        ok = observed == [expectation.direction_reversed]
        rows.append(
            {
                "mapping_key": mapping_key,
                "check": "direction_reversed_metadata",
                "status": _status(ok),
                "detail": (
                    f"expected={expectation.direction_reversed}; "
                    f"observed={observed}"
                ),
            }
        )

    return pd.DataFrame(rows)


def _audit_relationship_reversal(
    *,
    mapping_key: str,
    df: pd.DataFrame,
    expectation: MappingExpectation | None,
) -> pd.DataFrame:
    if expectation is None:
        return pd.DataFrame()

    if "relationship_original" not in df.columns or "Relationship" not in df.columns:
        return pd.DataFrame(
            [
                {
                    "mapping_key": mapping_key,
                    "relationship_original": "",
                    "relationship_normalized": "",
                    "expected_relationship": "",
                    "n_rows": 0,
                    "status": "fail",
                    "detail": "Missing relationship_original or Relationship column.",
                }
            ]
        )

    tmp = df[["relationship_original", "Relationship"]].copy()
    tmp["relationship_original"] = tmp["relationship_original"].astype(str).str.strip()
    tmp["Relationship"] = tmp["Relationship"].astype(str).str.strip()

    if expectation.direction_reversed:
        tmp["expected_relationship"] = tmp["relationship_original"].map(
            REVERSED_RELATIONSHIP
        )
    else:
        tmp["expected_relationship"] = tmp["relationship_original"]

    tmp["status"] = tmp["Relationship"].eq(tmp["expected_relationship"]).map(
        {True: "pass", False: "fail"}
    )

    out = (
        tmp.groupby(
            [
                "relationship_original",
                "Relationship",
                "expected_relationship",
                "status",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_rows")
        .rename(columns={"Relationship": "relationship_normalized"})
        .sort_values(
            [
                "status",
                "relationship_original",
                "relationship_normalized",
            ]
        )
    )

    out.insert(0, "mapping_key", mapping_key)
    return out


def _infer_actual_mapping_category(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["code.source", "code.target", "Relationship"]].copy()

    source_to_targets = (
        tmp.groupby("code.source")["code.target"]
        .nunique()
        .reset_index(name="n_targets_for_source")
    )

    target_to_sources = (
        tmp.groupby("code.target")["code.source"]
        .nunique()
        .reset_index(name="n_sources_for_target")
    )

    tmp = tmp.merge(source_to_targets, on="code.source", how="left")
    tmp = tmp.merge(target_to_sources, on="code.target", how="left")

    def classify(row: pd.Series) -> str:
        n_targets = int(row["n_targets_for_source"])
        n_sources = int(row["n_sources_for_target"])

        if n_sources == 1 and n_targets == 1:
            return "actual_1_to_1"
        if n_sources == 1 and n_targets > 1:
            return "actual_1_to_n"
        if n_sources > 1 and n_targets == 1:
            return "actual_n_to_1"
        return "actual_n_to_n"

    tmp["actual_mapping_category"] = tmp.apply(classify, axis=1)

    return tmp


def _audit_cardinality(
    *,
    mapping_key: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    actual = _infer_actual_mapping_category(df)

    out = (
        actual.groupby(["Relationship", "actual_mapping_category"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["Relationship", "actual_mapping_category"])
    )
    out.insert(0, "mapping_key", mapping_key)
    return out


def _construct_product_path(
    *,
    cfg: "HsSyncConfig",
    product_name: str,
    suffix: str = ".csv",
) -> Path:
    out_dir = cfg.coverage_output_dir()
    filename = cfg.product_filename(product_name)
    product_out_dir = cfg.product_output_dir(product_name, default_dir=out_dir)
    return product_out_dir / f"{filename}{suffix}"


def _load_coverage_products(
    *,
    cfg: "HsSyncConfig",
) -> dict[str, pd.DataFrame]:
    products = {
        "source_code_coverage": _read_coverage_csv(
            _construct_product_path(cfg=cfg, product_name="source_code_coverage")
        ),
        "mapping_coverage": _read_coverage_csv(
            _construct_product_path(cfg=cfg, product_name="mapping_coverage")
        ),
        "missing_mappings": _read_coverage_csv(
            _construct_product_path(cfg=cfg, product_name="missing_mappings")
        ),
        "needs_weights": _read_coverage_csv(
            _construct_product_path(cfg=cfg, product_name="needs_weights")
        ),
        "easy_without_weights": _read_coverage_csv(
            _construct_product_path(cfg=cfg, product_name="easy_without_weights")
        ),
    }
    return products

def _audit_coverage_invariants(
    *,
    products: dict[str, pd.DataFrame],
    expect_no_missing: bool,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """
    Audit coverage invariants at the observed source-code-pair grain.

    Important:
    - source_code_coverage is the canonical one-row-per-observed-source-pair table.
    - mapping_coverage / needs_weights / easy_without_weights may be expanded
      by target code, so they must not be used directly for row/value
      preservation checks.
    """

    source = products["source_code_coverage"]

    rows: list[dict[str, Any]] = []

    required_cols = {
        "source_hs",
        "source_cmd",
        "conversion_need",
        "n_rows",
        "value_total",
    }

    missing_required = sorted(required_cols - set(source.columns))
    rows.append(
        {
            "check": "source_code_coverage_required_columns",
            "expected": ", ".join(sorted(required_cols)),
            "actual": ", ".join(sorted(source.columns)),
            "status": _status(len(missing_required) == 0),
            "detail": "" if not missing_required else f"missing={missing_required}",
        }
    )

    if missing_required:
        return pd.DataFrame(rows)

    # Ensure numeric columns are numeric.
    source = source.copy()
    source["n_rows"] = pd.to_numeric(source["n_rows"], errors="coerce").fillna(0)
    source["value_total"] = pd.to_numeric(
        source["value_total"],
        errors="coerce",
    ).fillna(0)

    # One row per observed source-HS/code pair.
    duplicated_source_pairs = int(
        source.duplicated(subset=["source_hs", "source_cmd"]).sum()
    )

    rows.append(
        {
            "check": "source_code_coverage_unique_source_pairs",
            "expected": 0,
            "actual": duplicated_source_pairs,
            "status": _status(duplicated_source_pairs == 0),
            "detail": "duplicates in source_hs/source_cmd",
        }
    )

    valid_classes = {
        "easy_without_weights",
        "needs_weights",
        "missing_mapping",
    }

    observed_classes = sorted(set(source["conversion_need"].dropna()))
    invalid_classes = sorted(set(observed_classes) - valid_classes)

    rows.append(
        {
            "check": "valid_conversion_need_classes",
            "expected": ", ".join(sorted(valid_classes)),
            "actual": ", ".join(observed_classes),
            "status": _status(len(invalid_classes) == 0),
            "detail": "" if not invalid_classes else f"invalid={invalid_classes}",
        }
    )

    missing_source = source[source["conversion_need"] == "missing_mapping"]
    missing_count = len(missing_source)

    rows.append(
        {
            "check": "missing_mapping_count",
            "expected": 0 if expect_no_missing else "not enforced",
            "actual": missing_count,
            "status": _status((missing_count == 0) if expect_no_missing else True),
            "detail": "",
        }
    )

    # Partition check should happen within source_code_coverage itself.
    n_total_pairs = len(source)
    n_classified_pairs = int(source["conversion_need"].isin(valid_classes).sum())

    rows.append(
        {
            "check": "source_pairs_classified_once",
            "expected": n_total_pairs,
            "actual": n_classified_pairs,
            "status": _status(n_total_pairs == n_classified_pairs),
            "detail": (
                source["conversion_need"]
                .value_counts(dropna=False)
                .to_dict()
                .__repr__()
            ),
        }
    )

    total_rows = float(source["n_rows"].sum())
    class_rows = (
        source.groupby("conversion_need", dropna=False)["n_rows"]
        .sum()
        .reset_index()
    )

    rows.append(
        {
            "check": "row_count_defined",
            "expected": "> 0",
            "actual": total_rows,
            "status": _status(total_rows > 0),
            "detail": class_rows.to_dict(orient="records").__repr__(),
        }
    )

    total_value = float(source["value_total"].sum())
    class_values = (
        source.groupby("conversion_need", dropna=False)["value_total"]
        .sum()
        .reset_index()
    )

    rows.append(
        {
            "check": "value_total_defined",
            "expected": "> 0",
            "actual": _format_number(total_value),
            "status": _status(total_value > 0),
            "detail": class_values.to_dict(orient="records").__repr__(),
        }
    )

    # Optional cross-check: missing_mappings product should agree with
    # source_code_coverage missing class, but compare unique source pairs only.
    missing_product = products["missing_mappings"]

    if not missing_product.empty and {"source_hs", "source_cmd"} <= set(
        missing_product.columns
    ):
        missing_product_pairs = (
            missing_product[["source_hs", "source_cmd"]]
            .drop_duplicates()
            .shape[0]
        )
    else:
        missing_product_pairs = len(missing_product)

    rows.append(
        {
            "check": "missing_product_matches_source_coverage",
            "expected": missing_count,
            "actual": missing_product_pairs,
            "status": _status(missing_count == missing_product_pairs),
            "detail": "",
        }
    )

    # Optional cross-checks for easy/needs products by unique source pairs.
    for product_name, expected_class in [
        ("easy_without_weights", "easy_without_weights"),
        ("needs_weights", "needs_weights"),
    ]:
        product = products[product_name]

        expected_pairs = int((source["conversion_need"] == expected_class).sum())

        if {"source_hs", "source_cmd"} <= set(product.columns):
            actual_pairs = (
                product[["source_hs", "source_cmd"]]
                .drop_duplicates()
                .shape[0]
            )
        else:
            actual_pairs = len(product)

        rows.append(
            {
                "check": f"{product_name}_unique_pairs_match_source_coverage",
                "expected": expected_pairs,
                "actual": actual_pairs,
                "status": _status(expected_pairs == actual_pairs),
                "detail": (
                    "Compared as unique source_hs/source_cmd pairs because "
                    "product tables may be expanded by target code."
                ),
            }
        )

    return pd.DataFrame(rows)

def _audit_coverage_invariants_deprecated(
    *,
    products: dict[str, pd.DataFrame],
    expect_no_missing: bool,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    source = products["source_code_coverage"]
    missing = products["missing_mappings"]
    needs = products["needs_weights"]
    easy = products["easy_without_weights"]

    rows: list[dict[str, Any]] = []

    total_source_pairs = len(source)
    classified_pairs = len(easy) + len(needs) + len(missing)

    rows.append(
        {
            "check": "source_pairs_partitioned",
            "expected": total_source_pairs,
            "actual": classified_pairs,
            "status": _status(total_source_pairs == classified_pairs),
            "detail": f"easy={len(easy):,}; needs={len(needs):,}; missing={len(missing):,}",
        }
    )

    missing_count = len(missing)
    rows.append(
        {
            "check": "missing_mapping_count",
            "expected": 0 if expect_no_missing else "not enforced",
            "actual": missing_count,
            "status": _status((missing_count == 0) if expect_no_missing else True),
            "detail": "",
        }
    )

    total_rows = float(source["n_rows"].sum())
    classified_rows = float(
        easy["n_rows"].sum() + needs["n_rows"].sum() + missing["n_rows"].sum()
    )

    rows.append(
        {
            "check": "row_count_preserved",
            "expected": total_rows,
            "actual": classified_rows,
            "status": _status(abs(total_rows - classified_rows) <= tolerance),
            "detail": f"diff={total_rows - classified_rows:.6f}",
        }
    )

    total_value = float(source["value_total"].sum())
    classified_value = float(
        easy["value_total"].sum()
        + needs["value_total"].sum()
        + missing["value_total"].sum()
    )

    rows.append(
        {
            "check": "value_total_preserved",
            "expected": _format_number(total_value),
            "actual": _format_number(classified_value),
            "status": _status(abs(total_value - classified_value) <= max(tolerance, abs(total_value) * 1e-12)),
            "detail": f"diff={total_value - classified_value:.6f}",
        }
    )

    if "conversion_need" in source.columns:
        invalid_classes = sorted(
            set(source["conversion_need"].dropna())
            - {"easy_without_weights", "needs_weights", "missing_mapping"}
        )
        rows.append(
            {
                "check": "valid_conversion_need_classes",
                "expected": "easy_without_weights, needs_weights, missing_mapping",
                "actual": ", ".join(sorted(set(source["conversion_need"].dropna()))),
                "status": _status(len(invalid_classes) == 0),
                "detail": "" if not invalid_classes else ", ".join(invalid_classes),
            }
        )

    return pd.DataFrame(rows)


def _audit_needs_weights_readiness(
    *,
    source_code_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check whether source-code-level needs-weights cases are ready for
    equal-split allocation.

    Uses source_code_coverage because it is one row per observed source
    HS-code pair and contains n_target_codes / target_codes.
    """

    rows: list[dict[str, Any]] = []

    if "conversion_need" not in source_code_coverage.columns:
        return pd.DataFrame(
            [
                {
                    "check": "needs_weights_readiness_input",
                    "status": "fail",
                    "detail": "source_code_coverage missing conversion_need column.",
                }
            ]
        )

    needs = source_code_coverage[
        source_code_coverage["conversion_need"] == "needs_weights"
    ].copy()

    if needs.empty:
        return pd.DataFrame(
            [
                {
                    "check": "needs_weights_non_empty",
                    "status": "warning",
                    "detail": "No source-code-level needs-weights rows found.",
                }
            ]
        )

    rows.append(
        {
            "check": "needs_weights_non_empty",
            "status": "pass",
            "detail": f"source_code_pairs={len(needs):,}",
        }
    )

    if "n_target_codes" in needs.columns:
        n_target = pd.to_numeric(
            needs["n_target_codes"],
            errors="coerce",
        )
        invalid = int((n_target < 2).sum())
        nulls = int(n_target.isna().sum())

        rows.append(
            {
                "check": "needs_weights_have_multiple_targets",
                "status": _status(invalid == 0 and nulls == 0),
                "detail": f"invalid_rows={invalid:,}; null_rows={nulls:,}",
            }
        )
    else:
        rows.append(
            {
                "check": "needs_weights_have_multiple_targets",
                "status": "fail",
                "detail": "n_target_codes column not found in source_code_coverage.",
            }
        )

    if "target_codes" in needs.columns:
        target_codes = needs["target_codes"].fillna("").astype(str).str.strip()
        empty_targets = int(target_codes.eq("").sum())

        rows.append(
            {
                "check": "needs_weights_have_target_codes",
                "status": _status(empty_targets == 0),
                "detail": f"empty_target_codes={empty_targets:,}",
            }
        )
    else:
        rows.append(
            {
                "check": "needs_weights_have_target_codes",
                "status": "fail",
                "detail": "target_codes column not found in source_code_coverage.",
            }
        )

    if "value_total" in needs.columns:
        values = pd.to_numeric(needs["value_total"], errors="coerce").fillna(0)
        top = needs.copy()
        top["value_total"] = values
        top = top.sort_values("value_total", ascending=False).head(10)

        share = top["value_total"].sum() / values.sum() if values.sum() else 0.0

        rows.append(
            {
                "check": "top_10_needs_weight_value_concentration",
                "status": "info",
                "detail": (
                    f"top_10_value={_format_number(top['value_total'].sum())}; "
                    f"all_needs_value={_format_number(values.sum())}; "
                    f"share={share:.2%}"
                ),
            }
        )

    return pd.DataFrame(rows)


def _audit_status_summary(
    *,
    mapping_schema_audit: pd.DataFrame,
    metadata_audit: pd.DataFrame,
    reversal_audit: pd.DataFrame,
    cardinality_audit: pd.DataFrame,
    coverage_invariants: pd.DataFrame,
    needs_readiness: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact end-of-report audit summary.

    Only tables with an explicit status column are summarized.
    Cardinality audit is reported as informational unless converted into
    explicit pass/fail checks.
    """

    sections = {
        "mapping_schema": mapping_schema_audit,
        "mapping_metadata": metadata_audit,
        "relationship_reversal": reversal_audit,
        "coverage_invariants": coverage_invariants,
        "needs_weights_readiness": needs_readiness,
    }

    rows: list[dict[str, Any]] = []

    for section, df in sections.items():
        if "status" not in df.columns:
            rows.append(
                {
                    "section": section,
                    "status": "info",
                    "pass": 0,
                    "warning": 0,
                    "fail": 0,
                    "info": len(df),
                    "detail": "No status column.",
                }
            )
            continue

        counts = df["status"].value_counts(dropna=False).to_dict()
        n_fail = int(counts.get("fail", 0))
        n_warning = int(counts.get("warning", 0))
        n_pass = int(counts.get("pass", 0))
        n_info = int(counts.get("info", 0))

        if n_fail > 0:
            section_status = "FAIL"
        elif n_warning > 0:
            section_status = "WARNING"
        else:
            section_status = "PASS"

        rows.append(
            {
                "section": section,
                "status": section_status,
                "pass": n_pass,
                "warning": n_warning,
                "fail": n_fail,
                "info": n_info,
                "detail": "",
            }
        )

    summary = pd.DataFrame(rows)

    total_fail = int((summary["status"] == "FAIL").sum())
    total_warning = int((summary["status"] == "WARNING").sum())

    if total_fail > 0:
        overall = "FAIL"
    elif total_warning > 0:
        overall = "WARNING"
    else:
        overall = "PASS"

    overall_row = pd.DataFrame(
        [
            {
                "section": "OVERALL",
                "status": overall,
                "pass": int(summary["pass"].sum()),
                "warning": int(summary["warning"].sum()),
                "fail": int(summary["fail"].sum()),
                "info": int(summary["info"].sum()),
                "detail": (
                    "All hard checks passed."
                    if overall == "PASS"
                    else "Inspect failed/warning sections above."
                ),
            }
        ]
    )

    return pd.concat([summary, overall_row], ignore_index=True)

def _audit_needs_weights_readiness_deprecated(
    *,
    needs_weights: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if needs_weights.empty:
        return pd.DataFrame(
            [
                {
                    "check": "needs_weights_non_empty",
                    "status": "warning",
                    "detail": "No needs-weights rows found.",
                }
            ]
        )

    if "n_target_codes" in needs_weights.columns:
        n_target = pd.to_numeric(
            needs_weights["n_target_codes"],
            errors="coerce",
        )
        invalid = int((n_target < 2).sum())
        rows.append(
            {
                "check": "needs_weights_have_multiple_targets",
                "status": _status(invalid == 0),
                "detail": f"invalid_rows={invalid:,}",
            }
        )

    if "target_codes" in needs_weights.columns:
        empty_targets = int(
            needs_weights["target_codes"].isna().sum()
            + needs_weights["target_codes"].astype(str).str.strip().eq("").sum()
        )
        rows.append(
            {
                "check": "needs_weights_have_target_codes",
                "status": _status(empty_targets == 0),
                "detail": f"empty_target_codes={empty_targets:,}",
            }
        )

    if "value_total" in needs_weights.columns:
        values = pd.to_numeric(needs_weights["value_total"], errors="coerce").fillna(0)
        top = needs_weights.copy()
        top["value_total"] = values
        top = top.sort_values("value_total", ascending=False).head(10)

        rows.append(
            {
                "check": "top_10_needs_weight_value_share_reported",
                "status": "info",
                "detail": (
                    f"top_10_value={_format_number(top['value_total'].sum())}; "
                    f"all_needs_value={_format_number(values.sum())}"
                ),
            }
        )

    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    return df.to_markdown(index=False)


def build_comtrade_mapping_audit_report(
    *,
    mapping_schema_audit: pd.DataFrame,
    metadata_audit: pd.DataFrame,
    reversal_audit: pd.DataFrame,
    relationship_counts: pd.DataFrame,
    cardinality_audit: pd.DataFrame,
    coverage_invariants: pd.DataFrame,
    needs_readiness: pd.DataFrame,
) -> str:
    summary = _audit_status_summary(
        mapping_schema_audit=mapping_schema_audit,
        metadata_audit=metadata_audit,
        reversal_audit=reversal_audit,
        cardinality_audit=cardinality_audit,
        coverage_invariants=coverage_invariants,
        needs_readiness=needs_readiness,
    )

    overall_status = summary.loc[
        summary["section"] == "OVERALL",
        "status",
    ].iloc[0]

    failed_schema = mapping_schema_audit[mapping_schema_audit["status"] == "fail"]
    failed_metadata = metadata_audit[metadata_audit["status"] == "fail"]
    failed_reversal = reversal_audit[reversal_audit["status"] == "fail"]
    failed_coverage = coverage_invariants[coverage_invariants["status"] == "fail"]
    failed_needs = needs_readiness[needs_readiness["status"] == "fail"]

    failed_blocks = [
        failed_schema,
        failed_metadata,
        failed_reversal,
        failed_coverage,
        failed_needs,
    ]

    n_failed = sum(len(df) for df in failed_blocks)

    parts: list[str] = []

    parts.append("# UNSD/Comtrade HS mapping normalization audit")
    parts.append("")
    parts.append(f"Overall status: **{overall_status}**")
    parts.append("")
    parts.append(
        "This audit checks that normalized UNSD/Comtrade HS correspondence "
        "files are structurally valid, that direction reversal was applied "
        "as expected, and that the resulting target-HS coverage products "
        "preserve all observed source HS-code pairs, rows, and values."
    )
    parts.append("")

    parts.append("## Audit status summary")
    parts.append("")
    parts.append(_df_to_markdown(summary))
    parts.append("")

    parts.append("## Mapping file schema checks")
    parts.append("")
    parts.append(_df_to_markdown(mapping_schema_audit))
    parts.append("")

    parts.append("## Mapping metadata checks")
    parts.append("")
    parts.append(_df_to_markdown(metadata_audit))
    parts.append("")

    parts.append("## Relationship counts after normalization")
    parts.append("")
    parts.append(_df_to_markdown(relationship_counts))
    parts.append("")

    parts.append("## Relationship reversal audit")
    parts.append("")
    parts.append(
        "For H2/H3/H4 → H5 mappings, the raw UNSD tables were reversed from "
        "HS2017 → older-HS into older-HS → HS2017. The relationship labels "
        "must therefore be reversed as: 1:1→1:1, n:1→1:n, 1:n→n:1, n:n→n:n. "
        "For H6→H5, no reversal is expected."
    )
    parts.append("")
    parts.append(_df_to_markdown(reversal_audit))
    parts.append("")

    parts.append("## Relationship label vs actual cardinality")
    parts.append("")
    parts.append(
        "This recomputes mapping cardinality from source/target codes and "
        "compares it with the normalized relationship labels. Differences "
        "should be inspected, especially before allocation."
    )
    parts.append("")
    parts.append(_df_to_markdown(cardinality_audit))
    parts.append("")

    parts.append("## Coverage invariants")
    parts.append("")
    parts.append(_df_to_markdown(coverage_invariants))
    parts.append("")

    parts.append("## Needs-weights readiness")
    parts.append("")
    parts.append(_df_to_markdown(needs_readiness))
    parts.append("")

    if n_failed > 0:
        parts.append("## Failed checks")
        parts.append("")

        failed_all = pd.concat(
            [df for df in failed_blocks if not df.empty],
            ignore_index=True,
            sort=False,
        )

        parts.append(_df_to_markdown(failed_all))
        parts.append("")

    parts.append("## Final audit verdict")
    parts.append("")
    parts.append(_df_to_markdown(summary.tail(1)))
    parts.append("")

    return "\n".join(parts)

def run_comtrade_mapping_audit(
    *,
    cfg: "HsSyncConfig",
    run_id: str | None = None,
    expect_no_missing: bool = True,
    fail_on_error: bool = True,
    report_dir: Path | None = None,
    filename: str = "audit_comtrade_hs_mapping_normalization.md",
) -> dict[str, Any]:
    """
    Run audit for normalized UNSD/Comtrade mapping files and coverage outputs.

    Parameters
    ----------
    cfg:
        Loaded HS-sync config.

    run_id:
        Optional run identifier used for timestamped report writing.

    expect_no_missing:
        If True, missing mappings are treated as audit failures.

    fail_on_error:
        If True, raise AssertionError if any hard check fails.

    report_dir:
        Optional report directory. If omitted, writes into the same directory
        as cfg.coverage.artifact.

    filename:
        Markdown report filename.
    """

    mapping_files = cfg.coverage_mapping_files()

    mapping_schema_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    reversal_parts: list[pd.DataFrame] = []
    relationship_count_parts: list[pd.DataFrame] = []
    cardinality_parts: list[pd.DataFrame] = []

    for mapping_key, path in mapping_files.items():
        df = _read_csv_if_exists(path)
        expectation = EXPECTED_MAPPINGS.get(mapping_key)

        mapping_schema_parts.append(
            _audit_mapping_schema(
                mapping_key=mapping_key,
                path=path,
                df=df,
            )
        )

        metadata_parts.append(
            _audit_mapping_metadata(
                mapping_key=mapping_key,
                df=df,
                expectation=expectation,
            )
        )

        reversal_parts.append(
            _audit_relationship_reversal(
                mapping_key=mapping_key,
                df=df,
                expectation=expectation,
            )
        )

        relationship_count_parts.append(
            _relationship_counts(
                df,
                file_key=mapping_key,
            )
        )

        cardinality_parts.append(
            _audit_cardinality(
                mapping_key=mapping_key,
                df=df,
            )
        )

    mapping_schema_audit = pd.concat(mapping_schema_parts, ignore_index=True)
    metadata_audit = pd.concat(metadata_parts, ignore_index=True)
    reversal_audit = pd.concat(reversal_parts, ignore_index=True)
    relationship_counts = pd.concat(relationship_count_parts, ignore_index=True)
    cardinality_audit = pd.concat(cardinality_parts, ignore_index=True)

    coverage_products = _load_coverage_products(cfg=cfg)

    coverage_invariants = _audit_coverage_invariants(
        products=coverage_products,
        expect_no_missing=expect_no_missing,
    )

    needs_readiness = _audit_needs_weights_readiness(source_code_coverage=coverage_products["source_code_coverage"],)

    report = build_comtrade_mapping_audit_report(
        mapping_schema_audit=mapping_schema_audit,
        metadata_audit=metadata_audit,
        reversal_audit=reversal_audit,
        relationship_counts=relationship_counts,
        cardinality_audit=cardinality_audit,
        coverage_invariants=coverage_invariants,
        needs_readiness=needs_readiness,
    )

    if report_dir is None:
        report_dir = cfg.coverage.artifact.resolve_output_dir()

    report_outputs = write_text_report(
        report,
        report_dir=report_dir,
        filename=filename,
        timestamp=True,
        write_latest=True,
        run_id=run_id,
    )

    hard_failures = []

    for df in [
        mapping_schema_audit,
        metadata_audit,
        reversal_audit,
        coverage_invariants,
        needs_readiness,
    ]:
        if "status" in df.columns:
            failed = df[df["status"] == "fail"]
            if not failed.empty:
                hard_failures.append(failed)

    if hard_failures:
        failed_all = pd.concat(hard_failures, ignore_index=True, sort=False)
        logger.error(
            "Comtrade mapping audit failed:\n%s",
            failed_all.to_string(index=False),
        )

        if fail_on_error:
            raise AssertionError(
                "Comtrade mapping audit failed. See audit report: "
                f"{report_outputs.get('latest') or report_outputs.get('report')}"
            )

    logger.info("Comtrade mapping audit complete.")
    logger.info("  Report: %s", report_outputs.get("report"))
    if report_outputs.get("latest") is not None:
        logger.info("  Report latest: %s", report_outputs.get("latest"))

    return {
        "status": "completed" if not hard_failures else "failed",
        "report": report_outputs.get("report"),
        "report_latest": report_outputs.get("latest"),
        "mapping_schema_audit": mapping_schema_audit,
        "metadata_audit": metadata_audit,
        "reversal_audit": reversal_audit,
        "relationship_counts": relationship_counts,
        "cardinality_audit": cardinality_audit,
        "coverage_invariants": coverage_invariants,
        "needs_readiness": needs_readiness,
    }