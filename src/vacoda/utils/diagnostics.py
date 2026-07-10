"""
vacoda.utils.diagnostics
-----------------------
This module provides utility functions for computing and logging diagnostics related to feature coverage and recoverability in pandas
"""

import pandas as pd

from vacoda.utils.logger import get_logger

_logger = get_logger("DIAGNOSTICS")


def compute_feature_coverage(df: pd.DataFrame, feature: str) -> pd.Series:
    """
    Compute valid / missing count and percentage for a single column.
    Returns a Series with keys: total_rows, valid_n, valid_pct, missing_n, missing_pct
    """
    N = len(df)
    missing = df[feature].isna() | (df[feature] <= 0)
    missing_n = int(missing.sum())
    valid_n = N - missing_n
    return pd.Series({
        "feature_name": feature,
        "Total rows": N,
        "Valid rows": valid_n,
        "Valid percentage": valid_n / N * 100 if N > 0 else 0.0,
        "Missing rows": missing_n,
        "Missing percentage": missing_n / N * 100 if N > 0 else 0.0,
    })

def compute_features_coverage(df, features):
    """
    Compute coverage for multiple features and return a DataFrame.
    Logs diagnostics if input or output is unexpected.
    """
    _logger.info(f"Feature-Coverage: For features={features}")
    if not isinstance(df, pd.DataFrame):
        _logger.error(f"compute_features_coverage: df is not a DataFrame, got {type(df)}")
        raise ValueError(f"df must be a pandas DataFrame, got {type(df)}")
    if not isinstance(features, (list, tuple)):
        _logger.error(f"compute_features_coverage: features is not a list/tuple, got {type(features)}")
        raise ValueError(f"features must be a list or tuple, got {type(features)}")
    coverage_dict = {col: compute_feature_coverage(df, col) for col in features}
    if not coverage_dict:
        _logger.warning("compute_features_coverage: No features to compute coverage for.")
    try:
        df_cov = pd.DataFrame(coverage_dict).T
        _logger.debug(f"compute_features_coverage: Returning DataFrame with shape {df_cov.shape}")
        return df_cov
    except Exception as e:
        _logger.error(f"compute_features_coverage: Failed to convert coverage_dict to DataFrame: {e}")
        raise

def log_feature_coverage(stats: pd.Series) -> None:
    """
    Log the output of compute_feature_coverage in a human-readable format.
    """
    _logger.info(f"Coverage for {stats['feature_name']}:")
    _logger.info(f"  Total rows: {int(stats['Total rows']):,}")
    _logger.info(f"  Valid: {int(stats['Valid rows']):,} ({stats['Valid percentage']:.2f}%)")
    _logger.info(f"  Missing: {int(stats['Missing rows']):,} ({stats['Missing percentage']:.2f}%)")
    _logger.info("-" * 40)

def log_features_coverage(df_cov):
    """
    Log the output of compute_features_coverage for all features in a human-readable format.
    Handles only DataFrame input.
    """
    import pandas as pd
    if not isinstance(df_cov, pd.DataFrame):
        _logger.warning("log_features_coverage: Expected a DataFrame, got %s", type(df_cov))
        return
    for feature, stats in df_cov.iterrows():
        log_feature_coverage(stats)

def compute_data_coverage(df: pd.DataFrame, cols: list[str], indicator="absence") -> pd.Series:

    """
    Compute coverage statistics for specified columns in a DataFrame.
    Parameters:
    - df: Input DataFrame.
    - cols: List of column names to check for coverage.
    - indicator: "absence" (default) to treat NaN or non-positive values as absent,
                 "presence" to treat non-NaN and positive values as present.
    Returns:
    - A pandas Series containing coverage statistics.
    """
    N = len(df)

    if indicator == "absence":
        mask_by_col = {
            col: df[col].isna() | (df[col] <= 0)
            for col in cols
        }
    elif indicator == "presence":
        mask_by_col = {
            col: df[col].notna() & (df[col] > 0)
            for col in cols
        }
    else:
        raise ValueError("indicator must be 'absence' or 'presence'")

    masks = pd.concat(mask_by_col.values(), axis=1)
    masks.columns = cols

    all_indicator = masks.all(axis=1)   # AND
    none_indicator = ~masks.any(axis=1)
    at_least_one_indicator = masks.any(axis=1)  # OR

    stats = {"total_rows": float(N)}

    for col, mask in mask_by_col.items():
        stats[f"{col}_{indicator}_%"] = mask.mean() * 100

    stats[f"all_{indicator}_%"] = all_indicator.mean() * 100
    stats[f"none_{indicator}_%"] = none_indicator.mean() * 100
    stats[f"at_least_one_{indicator}_%"] = at_least_one_indicator.mean() * 100

    return pd.Series(stats)     


def assess_feature_recoverability(
    df: pd.DataFrame,
    main_feature: str,
    supporting_feature: str,
    unit_feature: str,
    kg_unit_code: int = 8,
) -> pd.Series:
    """
    Assess the recoverability of a main feature using a supporting feature and unit information.

    Parameters:
    - df: Input DataFrame.
    - main_feature: The main feature to assess recoverability for.
    - supporting_feature: The supporting feature used for recovery.
    - unit_feature: The feature representing the unit of measurement.
    - kg_unit_code: The code representing kilograms (default: 8).

    Returns:
    - A pandas Series containing recoverability statistics.
    """


    # If unit_feature is None, skip unit code logic
    if unit_feature is None and any(u in supporting_feature.lower() for u in ("kg", "usd")):
        has_weight = df[main_feature].notna() & (df[main_feature] > 0)
        has_sup = df[supporting_feature].notna() & (df[supporting_feature] > 0)
        missing_weight = ~has_weight
        recoverable_direct = missing_weight & has_sup
        potentially_recoverable_nonkg = pd.Series([False] * len(df), index=df.index)  # Not meaningful, keeping it for structure
    else:
        # Cast unit_code to Int64 for numeric operations, but keep original for merging
        unit_series = pd.to_numeric(df[unit_feature], errors="coerce").astype("Int64")
        has_weight = df[main_feature].notna() & (df[main_feature] > 0)
        has_sup = df[supporting_feature].notna() & (df[supporting_feature] > 0)
        has_weight_unit = unit_series == kg_unit_code
        has_nonkg_meaningful_unit = (
            unit_series.notna() & (unit_series > 1) & (unit_series != kg_unit_code)
        )
        missing_weight = ~has_weight
        recoverable_direct = missing_weight & has_sup & has_weight_unit
        potentially_recoverable_nonkg = missing_weight & has_sup & has_nonkg_meaningful_unit

    missing_weight_n = int(missing_weight.sum())

    return pd.Series(
        {
            "total_rows": len(df),
            "missing_feature_percentage_total": missing_weight.mean() * 100,
            "missing_feature_n": missing_weight_n,
            "recoverable_kg_percentage_total": recoverable_direct.mean() * 100,
            "recoverable_kg_n": int(recoverable_direct.sum()),
            "potentially_recoverable_nonkg_percentage_total": potentially_recoverable_nonkg.mean()
            * 100,
            "potentially_recoverable_nonkg_n": int(potentially_recoverable_nonkg.sum()),
            "recoverable_kg_percentage_of_missing": (
                recoverable_direct.sum() / missing_weight_n * 100
                if missing_weight_n > 0
                else 0.0
            ),
            "potentially_recoverable_nonkg_percentage_of_missing": (
                potentially_recoverable_nonkg.sum() / missing_weight_n * 100
                if missing_weight_n > 0
                else 0.0
            ),
            "unresolved_n": int(
                missing_weight_n
                - recoverable_direct.sum()
                - potentially_recoverable_nonkg.sum()
            ),
            "unresolved_percentage_of_missing": (
                (missing_weight_n - recoverable_direct.sum() - potentially_recoverable_nonkg.sum())
                / missing_weight_n * 100
                if missing_weight_n > 0
                else 0.0
            ),
        }
    )


def log_feature_recoverability(stats: pd.Series) -> None:
    """Log the output of assess_feature_recoverability in a human-readable format."""

    _logger.info(f"    Missing:                           {int(stats['missing_feature_n']):,} ({stats['missing_feature_percentage_total']:.2f}% of total)")
    _logger.info(
        f"    Recoverable (kg):                  {int(stats['recoverable_kg_n']):,}"
        f" ({stats['recoverable_kg_percentage_of_missing']:.2f}% of missing)"
    )
    _logger.info(
        f"    Potentially Recoverable (non-kg):  {int(stats['potentially_recoverable_nonkg_n']):,}"
        f" ({stats['potentially_recoverable_nonkg_percentage_of_missing']:.2f}% of missing)"
    )
    _logger.info(
        f"    Unresolved:                        {int(stats['unresolved_n']):,}"
        f" ({stats['unresolved_percentage_of_missing']:.2f}% of missing)"
    )
    _logger.info("")


def run_multi_recovery_cycles(
    df,
    main_feature,
    supporting_features,
    unit_features=None,
    recovery_feature_name=None,
    recovery_feature_flag=None,
    recovery_col_prefix=None,
    recovery_flag_prefix=None,
    kg_unit_code=8,
    logger=None
):
    """
    Attempts recovery of main_feature using each supporting_feature (and optional unit_feature).
    Parameters   
    ----------
    df : pd.DataFrame
        Input DataFrame containing the features to analyze and recover.
    main_feature : str
        The name of the main feature for which recoverability is being assessed and recovery is attempted.
    supporting_features : List[str]
        A list of supporting features to use for recovery attempts, in order of preference. 
    unit_features : List[str], optional
        A list of unit features corresponding to each supporting feature, used to determine recoverability based on
    recovery_feature_name : str, optional
        Final recovered feature column to create in the returned dataset.
    recovery_feature_flag : str, optional
        Final recovery flag column to create in the returned dataset.
    recovery_col_prefix : str, optional
        Backward-compatible alias for recovery_feature_name.
    recovery_flag_prefix : str, optional
        Backward-compatible alias for recovery_feature_flag.
    kg_unit_code : int, default=8
        The code representing kilograms in the unit feature, used to determine direct recoverability.

    Returns:
    -----------
    pd.DataFrame
        A DataFrame with additional columns for recovered features and recovery flags for each supporting feature.  
    """
    from vacoda.preprocessing.recovery import  recover_feature_from_supfeature, log_recovery_summary
    from vacoda.utils.display import get_recovery_symbol
    MAIN_FEATURE_NAME = main_feature
    final_feature_name = recovery_feature_name or recovery_col_prefix or main_feature
    final_flag_name = (
        recovery_feature_flag
        or recovery_flag_prefix
        or f"{final_feature_name}_recovery_flag"
    )

    if unit_features is None:
        unit_features = [None] * len(supporting_features)
    if len(unit_features) != len(supporting_features):
        raise ValueError(
            "unit_features must have the same length as supporting_features"
        )

    log = logger if logger is not None else _logger
    cycle_flag_names = []
    current_feature_name = MAIN_FEATURE_NAME
    
    # Capture initial coverage before any recovery
    initial_coverage = compute_feature_coverage(df, MAIN_FEATURE_NAME)
    initial_valid_n = int(initial_coverage["Valid rows"])
    initial_valid_pct = initial_coverage["Valid percentage"]

    if initial_valid_pct == 100.0:
        log.info(f"Initial coverage for {MAIN_FEATURE_NAME} is already {initial_valid_pct:.2f}%. No recovery needed. Existing!")
        log.info("")
        if final_feature_name != MAIN_FEATURE_NAME:
            df[final_feature_name] = df[MAIN_FEATURE_NAME]
        df[final_flag_name] = False
        return df

    for idx, (sup_col, unit_col) in enumerate(zip(supporting_features, unit_features), 1):
        cycle_flag_name = f"{final_flag_name}__cycle_{idx}"
        log.info("")
        log.info(f" [Recovery Cycle {idx}] {current_feature_name} ← {sup_col} ({unit_col}) ")
        log.info("┄"*60)

        # 1. Assess recoverability & display its coverage before recovery
        stats = assess_feature_recoverability(df, 
                                              main_feature       = current_feature_name, 
                                              supporting_feature = sup_col, 
                                              unit_feature       = unit_col, 
                                              kg_unit_code       = kg_unit_code)
        log_feature_recoverability(stats)

        coverage_before = compute_feature_coverage(df, current_feature_name)

        # 2.  Recovery
        df = recover_feature_from_supfeature(
                    df,
                    main_feature=current_feature_name,
                    sup_feature=sup_col,
                    unit_feature=unit_col,
                    recovery_feature_name=final_feature_name,
                    recovery_feature_flag=cycle_flag_name,
                    kg_unit_code=kg_unit_code,
                    logger=log,
                )
        
         # 3. Summarization of recovery results
        log_recovery_summary(
            df,
            cycle_flag_name,
            final_feature_name,
            int(coverage_before["Valid rows"]),
            logger=log
        )

       
        #Calculate percent of missing weights recovered in this cycle
        percent_recovered = 0.0
        check = ""
        if stats['missing_feature_n'] > 0:
            percent_recovered = stats['recoverable_kg_n'] / stats['missing_feature_n'] * 100
            check = get_recovery_symbol(percent_recovered, threshold=30)

        log.info(f"   {check} Cycle {idx} complete: {percent_recovered:.2f}% of missing features recovered")
        log.info("")

        # Update main_feature for next cycle
        current_feature_name = final_feature_name
        cycle_flag_names.append(cycle_flag_name)

    if cycle_flag_names:
        df[final_flag_name] = df[cycle_flag_names].any(axis=1)
        df = df.drop(columns=cycle_flag_names)

    # SUMMARY of all recovery cycles
    final_coverage = compute_feature_coverage(df, final_feature_name)
    final_valid_n = int(final_coverage["Valid rows"])
    final_valid_pct = final_coverage["Valid percentage"]
    final_missing_n = int(final_coverage["Missing rows"])
    final_missing_pct = final_coverage["Missing percentage"]
    gain_n = final_valid_n - initial_valid_n
    gain_pct = final_valid_pct - initial_valid_pct

    log.info("")
    log.info("═"*68)
    log.info(f"  SUMMARY — Recovered {gain_n:+,} rows ({gain_pct:+.2f}%)")
    log.info(f"          Coverage: {initial_valid_pct:.2f}% → {final_valid_pct:.2f}%")
    log.info(f"          Remaining missing: {final_missing_n:,} rows ({final_missing_pct:.2f}%)")
    log.info("═"*68)
    log.info("")
    return df
