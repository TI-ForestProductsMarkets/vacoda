
import pandas as pd

from vacoda.utils.logger import get_logger

_logger = get_logger("RECOVERY")


def recover_feature_from_supfeature(
    df: pd.DataFrame,
    main_feature: str,
    sup_feature: str,
    unit_feature: str,
    kg_unit_code: int = 8,
    recovery_feature_name: str = "netweight_kg",
    recovery_feature_flag: str = "netweight_kg_recovery_flag",
    overwrite: bool = False,
    logger=None
) -> pd.DataFrame:
    """
    Recover weight from supplementary quantity when the supplementary unit is kg.

    Logic:
    - A row is eligible if:
        * weight is missing or <= 0
        * supplementary quantity is > 0
        * unit_code == kg_unit_code (default: 8 for kg)
    - If overwrite=False:
        * create/use `recovery_feature_name`
        * preserve original weight column
    - If overwrite=True:
        * fill the original weight column directly

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    main_feature : str
        Name of the net weight column.
    sup_feature : str
        Name of the supplementary quantity column.
    unit_feature : str
        Name of the supplementary unit code column.
    kg_unit_code : int, default 8
        Unit code representing kilograms.
    recovery_feature_name : str, default "netweight_kg"
        Output column for recovered weight when overwrite=False.
    recovery_feature_flag : str, default "netweight_kg_recovery_flag"
        Boolean flag indicating rows recovered from supplementary quantity.
    overwrite : bool, default False
        Whether to overwrite `weight_col` directly.

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    log = logger if logger is not None else _logger

    # Normalize dtypes for robust comparisons (unit codes may arrive as strings).
    weight_series = pd.to_numeric(out[main_feature], errors="coerce")
    sup_series    = pd.to_numeric(out[sup_feature],  errors="coerce")
    missing_or_invalid_weight = weight_series.isna() | (weight_series <= 0)
    
    has_valid_sup = sup_series.notna() & (sup_series > 0)

    # If unit_feature is None, skip all unit code logic
    if unit_feature is None:
        recoverable = missing_or_invalid_weight & has_valid_sup
    else:
        unit_series = pd.to_numeric(out[unit_feature], errors="coerce").astype("Int64")
        # If sup_feature name contains 'kg', skip the has_weight_unit condition
        if 'kg' in sup_feature.lower():
            recoverable = missing_or_invalid_weight & has_valid_sup
            log.info(f"Unit code check skipped for {main_feature} since its supplementary feature {sup_feature} contains 'kg'.")
        else:
            has_weight_unit = unit_series == kg_unit_code
            recoverable = missing_or_invalid_weight & has_valid_sup & has_weight_unit

    # The flag is True only where value was missing/invalid and is now filled
    out[recovery_feature_flag] = False
    out.loc[recoverable & (sup_series.notna() & (sup_series > 0)), recovery_feature_flag] = True

    if overwrite:
        out[main_feature] = weight_series
        out.loc[recoverable, main_feature] = sup_series.loc[recoverable]
        
        # Set flag only where value was missing and is now filled
        out[recovery_feature_flag] = False
        out.loc[recoverable, recovery_feature_flag] = True
        
    else:
        out[recovery_feature_name] = weight_series
        out.loc[recoverable, recovery_feature_name] = sup_series.loc[recoverable]
        out[recovery_feature_flag] = False
        out.loc[recoverable, recovery_feature_flag] = True

    return out


def log_recovery_summary(
    df: pd.DataFrame,
    recovery_feature_flag: str,
    recovery_feature_name: str,
    valid_before: int,
    logger=None
) -> None:
    """
    Log a concise summary of recovery results, including recovered rows, missing after, and coverage delta.
    """
    log = logger if logger is not None else _logger
    recovered_n = int(df[recovery_feature_flag].sum())
    total_n = len(df)
    missing_after = int((df[recovery_feature_name].isna() | (df[recovery_feature_name] <= 0)).sum())

    ## Lets find the percentage of missing_after
    missing_after_pct = missing_after / total_n * 100 if total_n > 0 else 0.0
    valid_after = total_n - missing_after
    coverage_after = valid_after / total_n * 100 if total_n > 0 else 0.0
    coverage_before = valid_before / total_n * 100 if total_n > 0 else 0.0
    delta = coverage_after - coverage_before
    log.info("  ┌───────────────────────────────────────────┐")
    log.info("  │ Recoverability Summary                    │")
    log.info("  ├──────────────────────┬────────────────────┤")
    log.info(f"  │ Recovered            │ {recovered_n:>8,}           │")
    log.info(f"  │ Missing after        │ {missing_after:8,} ({missing_after_pct:.2f}%)   │")
    log.info(f"  │ Coverage             │ {coverage_after:6.2f}%  ({delta:+.2f}%)  │ ") 
    log.info("  └──────────────────────┴────────────────────┘")



