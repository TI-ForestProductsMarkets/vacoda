"""
Module: vacoda.preprocessing.geography
Description: Geographic enrichment — continent lookups for reporter and partner countries.
"""
from __future__ import annotations

import pandas as pd
import pycountry_convert as pc


def country_to_continent(alpha3):
    try:
        alpha2 = pc.country_alpha3_to_country_alpha2(alpha3)
        continent_code = pc.country_alpha2_to_continent_code(alpha2)
        return pc.convert_continent_code_to_continent_name(continent_code)
    except Exception:
        return None


def add_continent_columns(df: pd.DataFrame, 
                          reporter_col_iso="reporter_iso", 
                          partner_col_iso="partner_iso") -> pd.DataFrame:
    df = df.copy()
    df["reporter_continent"] = df[reporter_col_iso].apply(country_to_continent)
    df["partner_continent"] = df[partner_col_iso].apply(country_to_continent)
    return df
