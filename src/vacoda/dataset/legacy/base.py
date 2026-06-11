from __future__ import annotations

import pandas as pd


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the column names of the given DataFrame by stripping whitespace and replacing spaces with underscores.
    Parameters:
    -----------
    - df: pd.DataFrame - The input DataFrame whose columns need to be standardized.
    
    Returns:
    --------
    - pd.DataFrame: A new DataFrame with standardized column names.
    """
    out = df.copy()
    out.columns = out.columns.str.strip().str.replace(" ", "_", regex=False)
    return out
