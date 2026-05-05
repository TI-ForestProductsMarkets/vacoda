"""
Module: vacoda.preprocessing.build_dataset
Description: This module contains functions for building a decoded dataset by merging trade data with various metadata tables.
"""

from __future__ import annotations

import pandas as pd
from vacoda.utils.logger import get_logger
logger = get_logger("BUILD_DATASET")

from vacoda.preprocessing.metadata import (
    prepare_commodity_metadata,
    prepare_estimation_metadata,
    prepare_flow_metadata,
    prepare_hs_version_metadata,
    prepare_reporter_metadata,
    prepare_unit_metadata,
)
from vacoda.preprocessing.trade import prepare_trade_table

from vacoda.preprocessing.derived_features import (
    add_hs_validity_flag,
    add_price_features,
)
from vacoda.preprocessing.geography import add_continent_columns
from vacoda.preprocessing.partner_source import (
    apply_partner_source_harmonization,
    add_partner_reliability,
)


def _normalize_merge_key(series: pd.Series) -> pd.Series:
    """Normalize join keys to stable string values while preserving missing values."""
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    # Common artifact from float parsing of code columns (e.g., 8.0 -> 8)
    s = s.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
    return s

def _validate_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_keys: list[str],
    right_keys: list[str],
    label: str,
) -> None:
    """Pre-merge sanity checks — warnings for problems, DEBUG detail otherwise."""
    logger.debug(f"[{label}] Lookup table shape: {right.shape[0]:,} rows × {right.shape[1]} cols")

    # 1. Null join keys in left (trade) table
    null_count = int(left[left_keys].isna().any(axis=1).sum())
    if null_count:
        logger.warning(
            f"[{label}] {null_count:,} rows in left table have null join key(s) {left_keys} "
            f"— they will produce null metadata after merge."
        )
    else:
        logger.debug(f"[{label}] No null join keys in left table.")

    # 2. Duplicate keys in right (lookup) table → would cause fanout
    dup_count = int(right[right_keys].duplicated(keep=False).sum())
    if dup_count:
        logger.warning(
            f"[{label}] Right table has {dup_count:,} duplicate rows on join key(s) {right_keys} "
            f"— merge may increase row count."
        )
    else:
        logger.debug(f"[{label}] Lookup keys are unique — no fanout risk.")


def _merge_and_check(
    left: pd.DataFrame,
    right: pd.DataFrame,
    label: str,
    on: str | list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
) -> pd.DataFrame:
    """Left-merge with pre-validation and detailed post-merge diagnostics."""
    def _to_list(k: str | list[str] | None) -> list[str]:
        if k is None:
            return []
        return [k] if isinstance(k, str) else list(k)

    lkeys = _to_list(on) or _to_list(left_on)
    rkeys = _to_list(on) or _to_list(right_on)

    # Normalize merge keys on both sides to avoid dtype-mismatch merge failures.
    left_m = left.copy()
    right_m = right.copy()
    for lk, rk in zip(lkeys, rkeys):
        if lk in left_m.columns:
            left_m[lk] = _normalize_merge_key(left_m[lk])
        if rk in right_m.columns:
            right_m[rk] = _normalize_merge_key(right_m[rk])

    _validate_merge(left_m, right_m, lkeys, rkeys, label)

    n_before = len(left_m)
    cols_before = set(left_m.columns)
    merged = left_m.merge(right_m, on=on, left_on=left_on, right_on=right_on, how="left")
    n_after = len(merged)

    # Columns added by this merge
    new_cols = [c for c in merged.columns if c not in cols_before]

    if n_after != n_before:
        logger.warning(
            f"[{label}] Row count changed: {n_before:,} → {n_after:,} "
            f"(+{n_after - n_before:,}). Check for duplicate keys in right table."
        )
    else:
        logger.info(f"[{label}] Merge OK — {n_after:,} rows.")

    # Unmatched rows (any new column is null → row did not match the lookup)
    if new_cols:
        unmatched = int(merged[new_cols[0]].isna().sum())
        matched = n_after - unmatched
        pct = 100 * matched / n_after if n_after else 0
        logger.debug(
            f"[{label}] Match rate: {matched:,}/{n_after:,} rows ({pct:.2f}%) — "
            f"{unmatched:,} unmatched (will have null metadata)."
        )
        logger.debug(f"[{label}] New columns added ({len(new_cols)}): {new_cols}")

    return merged


def check_flow_codes(df: pd.DataFrame, known: set[int] = set({1, 2, 3, 4})) -> None:
    """Warn if any non-null flow_code values fall outside the known set."""
    flow_num = pd.to_numeric(df["flow_code"], errors="coerce")
    unexpected = flow_num.dropna().astype(int).loc[lambda s: ~s.isin(known)].unique().tolist()
    if unexpected:
        row_count = df["flow_code"].astype(object).isin(unexpected).sum()
        logger.warning(
            f"Unexpected flow_code values outside {sorted(known)}: {sorted(unexpected)} "
            f"({row_count} rows)"
        )
    else:
        logger.info(f"All flow_code values are within the expected set {sorted(known)}.")


def build_decoded_dataset(bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trade      = prepare_trade_table(bundle["main"])
    reporter   = prepare_reporter_metadata(bundle["reporter"])
    hs_version = prepare_hs_version_metadata(bundle["hs_version"])
    commodity  = prepare_commodity_metadata(bundle["commodity"])
    flow       = prepare_flow_metadata(bundle["flow"])
    unit       = prepare_unit_metadata(bundle["unit"])
    estimation = prepare_estimation_metadata(bundle["estimation"])

    df = trade.copy()
    n_input = len(df)

    df = _merge_and_check(df, reporter.add_prefix("reporter_"),
                          label="reporter",
                          left_on="reporter_code", right_on="reporter_country_code")

    df = _merge_and_check(df, reporter.add_prefix("partner_"),
                          label="partner",
                          left_on="partner_code", right_on="partner_country_code")

    df = _merge_and_check(df, hs_version,
                          label="hs_version",
                          on="hs_code_short")

    df = _merge_and_check(df, commodity,
                          label="commodity",
                          on=["hs_code_short", "commodity_code"])

    df = _merge_and_check(df, flow,
                          label="flow",
                          on="flow_code")

    df = _merge_and_check(df, unit,
                          label="unit",
                          on="unit_code")


    # Add Flow-Code Diagnostic Check
    check_flow_codes(df)

    # Add diagnostic check
    unmatched = df[df['unit_short_name'].isna()]['unit_code'].unique()
    if len(unmatched) > 0:
        logger.warning(f"{len(unmatched)} unit codes have no metadata: {unmatched}")

    df = _merge_and_check(df, estimation,
                          label="estimation",
                          on="estimation_code")

    # Final reconciliation: output must have exactly as many rows as the raw trade table
    n_output = len(df)
    if n_output != n_input:
        logger.warning(
            f"Row count mismatch: input trade table had {n_input:,} rows, "
            f"decoded dataset has {n_output:,} rows (delta: {n_output - n_input:+,}). "
            f"Check for fanout in lookup merges."
        )
    else:
        logger.info(f"Row count reconciliation OK — {n_output:,} rows in, {n_output:,} rows out.")

    return df


def build_base_dataset(bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Builds a base analytic dataset by preparing the trade table and engineering additional features such as normalized values, 
    estimation flags, HS validity flags, and price features.
    """
    df = build_decoded_dataset(bundle)
    #df = add_estimation_flags(df)
    df = add_hs_validity_flag(df)
    df = add_price_features(df, weight_col="netweight_kg_raw", value_col="value_usd")
    df = add_continent_columns(df)
    df = apply_partner_source_harmonization(df)
    df = add_partner_reliability(df)
    return df


def build_decoded_api_dataset(bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Minimal decoded dataset builder for API-only data (no metadata merges).
    Expects bundle = {"main": DataFrame}
    """
    trade = prepare_trade_table(bundle["main"])
    df = trade.copy()
    # You can add API-specific cleaning or feature engineering here if needed
    return df

def build_base_api_dataset(bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Builds a base analytic dataset from API data only (no metadata).
    """
    df = build_decoded_api_dataset(bundle)
    if df is None:
        logger.error("build_decoded_api_dataset(bundle) returned None in build_base_api_dataset. Bundle keys: %s", list(bundle.keys()))
        raise RuntimeError("build_decoded_api_dataset(bundle) returned None in build_base_api_dataset.")

    #df = add_hs_validity_flag(df)
    df = add_price_features(df, weight_col="netweight_kg_raw", value_col="value_usd_raw")
    if df is None:
        logger.error("add_price_features(df) returned None in build_base_api_dataset.")
        raise RuntimeError("add_price_features(df) returned None in build_base_api_dataset.")
    #df = add_continent_columns(df, 
    #                           reporter_col_iso="reporter_iso", 
    #                           partner_col_iso="partner_iso") 
    #df = apply_partner_source_harmonization(df) if "partner_code" in df.columns else df
    return df
