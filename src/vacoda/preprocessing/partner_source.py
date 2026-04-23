"""
Module: vacoda.preprocessing.partner_source
Description: Harmonization and reliability classification of partner_source values.
"""
from __future__ import annotations

import pandas as pd

from vacoda.mappings.partner_source import PARTNER_SOURCE_EXACT_MAP, PARTNER_SOURCE_RULE_MAP


def normalize_text_basic(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.lower()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


def harmonize_partner_source(s: pd.Series) -> pd.Series:
    """Harmonize raw partner_source values into clean, consistent labels.

    Returns a Series of canonical labels. Unmapped or missing values are
    preserved as-is from the normalised form.
    """
    norm = normalize_text_basic(s)
    final = s.astype("string").copy()
    exact_mask = norm.isin(PARTNER_SOURCE_EXACT_MAP)
    rule_mask = norm.isin(PARTNER_SOURCE_RULE_MAP)
    final.loc[exact_mask] = norm.loc[exact_mask].map(PARTNER_SOURCE_EXACT_MAP)
    final.loc[rule_mask] = norm.loc[rule_mask].map(PARTNER_SOURCE_RULE_MAP)
    return final


def apply_partner_source_harmonization(df: pd.DataFrame,
                                       partner_col: str = "partner_source") -> pd.DataFrame:
    """Harmonize partner_source in df.

    Reads from ``{partner_col}_raw`` (the original column) and stores the
    harmonized canonical label under ``{partner_col}``.
    """
    df = df.copy()
    df[partner_col] = harmonize_partner_source(df[f"{partner_col}_raw"])
    return df


def classify_partner_source(ps) -> str:
    if pd.isna(ps):
        return "unknown"
    if ps == "Origin":
        return "high"
    elif ps in ["Consignment", "Origin/Consignment for Intra-EU"]:
        return "medium"
    elif ps == "Last Known Destination":
        return "low"
    else:
        return "unknown"


def add_partner_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Classify partner reliability from the harmonized partner_source column."""
    df = df.copy()
    df["partner_reliability"] = df["partner_source"].apply(classify_partner_source)
    return df
