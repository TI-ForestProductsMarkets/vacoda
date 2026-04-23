"""
This module implements a weight imputation strategy based on reference price per kg values computed from similar records. The main steps are:
1. Prepare weight-related columns and flags.
2. Compute reference price per kg values for different groupings and filter them based on count and coefficient of variation criteria.
3. Iteratively impute missing weights using reference prices from the most specific group down to more general groups, allowing for a year window to borrow information from adjacent years.
4. Evaluate the coverage of imputation and perform backtesting to assess accuracy.
"""
import numpy as np
import pandas as pd


import numpy as np
import pandas as pd


def build_group_levels(df: pd.DataFrame) -> list:
    commodity_col = "commodity_code" if "commodity_code" in df.columns else "hs_code_short"
    reporter_col = "reporter_code" if "reporter_code" in df.columns else "reporter_country_name"
    partner_col = "partner_code" if "partner_code" in df.columns else "partner_country_name"

    print("Using commodity column:", commodity_col)
    print("Using reporter column:", reporter_col)
    print("Using partner column:", partner_col)

    if df[commodity_col].nunique(dropna=False) <= 1:
        print(f"{commodity_col} is constant in this dataset; excluding it from group levels.")
        return [
            [reporter_col, partner_col, "year"],
            [reporter_col, "year"],
            ["year"],
        ]

    return [
        [commodity_col, reporter_col, partner_col, "year"],
        [commodity_col, reporter_col, "year"],
        [commodity_col, "year"],
        [commodity_col],
    ]


def prepare_weight_columns(df: pd.DataFrame, 
                           weight_col: str,
                           value_col: str) -> pd.DataFrame:
    out = df.copy()

    # Treat 0 as missing
    out.loc[out[weight_col] == 0, weight_col] = np.nan

    # Usability flag
    out["usable_for_kg"] = (
        out[weight_col].notna()
        & (out[weight_col] > 0)
        & out[value_col].notna()
        & (out[value_col] > 0)
    )

    # Price per kg (only where valid)
    out["price_per_kg"] = np.nan
    mask = out["usable_for_kg"]
    out.loc[mask, "price_per_kg"] = (
        out.loc[mask, value_col] / out.loc[mask, weight_col]
    )

    return out


def compute_reference_prices(
    df: pd.DataFrame,
    group_cols: list,
    lower_count: int = 5,
    max_cv: float = 1.0,
) -> pd.DataFrame:
    ref = (
        df[df["usable_for_kg"]]
        .groupby(group_cols)["price_per_kg"]
        .agg(["median", "std", "count"])
        .reset_index()
    )

    # Guard against zero/invalid medians before CV computation
    ref = ref[ref["median"] > 0].copy()
    ref["cv"] = ref["std"] / ref["median"]
    ref["cv"] = ref["cv"].replace([np.inf, -np.inf], np.nan)

    ref_valid = ref[
        (ref["count"] >= lower_count) & (ref["cv"] < max_cv)
    ].copy()

    return ref_valid.rename(columns={"median": "ref_price_per_kg"})


def impute_weight(
    df: pd.DataFrame,
    weight_col: str,
    value_col: str,
    group_levels: list,
    lower_count: int = 5,
    max_cv: float = 1.0,
    reference_df: pd.DataFrame | None = None,
    year_window: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    ref_source = out if reference_df is None else reference_df.copy()

    out["imputed_netweight_kg"] = np.nan
    out["weight_imputed"] = False
    out["imputation_level"] = None
    out["imputation_year_shift"] = np.nan

    missing_mask = (
        out[weight_col].isna()
        & out[value_col].notna()
        & (out[value_col] > 0)
    )

    for level_idx, group_cols in enumerate(group_levels):
        for shift in range(0, year_window + 1):
            for direction in ([shift] if shift == 0 else [shift, -shift]):
                tmp_ref = ref_source.copy()
                tmp_ref["year"] = tmp_ref["year"] + direction

                ref = compute_reference_prices(tmp_ref, group_cols, lower_count, max_cv)

                out = out.merge(
                    ref[group_cols + ["ref_price_per_kg"]],
                    on=group_cols,
                    how="left",
                )

                fill_mask = (
                    missing_mask
                    & out["ref_price_per_kg"].notna()
                    & out["imputed_netweight_kg"].isna()
                )

                out.loc[fill_mask, "imputed_netweight_kg"] = (
                    out.loc[fill_mask, "value_usd"] / out.loc[fill_mask, "ref_price_per_kg"]
                )
                out.loc[fill_mask, "weight_imputed"] = True
                out.loc[fill_mask, "imputation_level"] = level_idx
                out.loc[fill_mask, "imputation_year_shift"] = direction

                out.drop(columns=["ref_price_per_kg"], inplace=True)

    # Initialize flags/derived columns
    out["imputation_flag_extreme"] = False

    # Keep invalid (non-positive) imputations as missing
    out.loc[out["imputed_netweight_kg"] <= 0, "imputed_netweight_kg"] = np.nan

    # Compute implied price only for imputed rows
    out["imputed_price"] = np.nan
    mask_imp = out["imputed_netweight_kg"].notna()
    out.loc[mask_imp, "imputed_price"] = (
        out.loc[mask_imp, "value_usd"] / out.loc[mask_imp, "imputed_netweight_kg"]
    )

    # Flag extreme imputations using observed price distribution from reference source
    lower = ref_source["price_per_kg"].quantile(0.01)
    upper = ref_source["price_per_kg"].quantile(0.99)
    p95 = ref_source["price_per_kg"].quantile(0.95)

    out.loc[mask_imp, "imputation_flag_extreme"] = (
        (out.loc[mask_imp, "imputed_price"] < lower)
        | (out.loc[mask_imp, "imputed_price"] > upper)
    )

    # Additional guard: flag imputations where implied price exceeds 3× p95
    out.loc[out["imputed_price"] > p95 * 3, "imputation_flag_extreme"] = True

    # Confidence score
    out["imputation_confidence"] = np.where(out["weight_imputed"], 1.0, np.nan)

    # Penalise coarser levels
    out.loc[out["imputation_level"] == 1, "imputation_confidence"] *= 0.8
    out.loc[out["imputation_level"] == 2, "imputation_confidence"] *= 0.6
    out.loc[out["imputation_level"] == 3, "imputation_confidence"] *= 0.4

    # Penalise farther year shifts
    out.loc[out["imputation_year_shift"].abs() == 1, "imputation_confidence"] *= 0.9
    out.loc[out["imputation_year_shift"].abs() == 2, "imputation_confidence"] *= 0.75

    # Penalise extreme imputations
    out.loc[out["imputation_flag_extreme"], "imputation_confidence"] *= 0.3

    # Reset invalid rows fully
    invalid_imputed = out["weight_imputed"] & out["imputed_netweight_kg"].isna()
    out.loc[invalid_imputed, "weight_imputed"] = False
    out.loc[invalid_imputed, "imputation_level"] = None
    out.loc[invalid_imputed, "imputation_year_shift"] = np.nan
    out.loc[invalid_imputed, "imputation_confidence"] = np.nan
    out.loc[invalid_imputed, "imputation_flag_extreme"] = False
    out.loc[invalid_imputed, "imputed_price"] = np.nan

    return out


def evaluate_coverage(df: pd.DataFrame, weight_col: str) -> dict:
    original_missing = df[weight_col].isna()
    missing_before = original_missing.sum()
    imputed_now = (original_missing & df["imputed_netweight_kg"].notna()).sum()
    coverage = round(imputed_now / missing_before * 100, 2) if missing_before else None

    print(f"Missing before: {missing_before} | Imputed: {imputed_now} | Coverage: {coverage}%")
    return {
        "missing_before": missing_before,
        "imputed": imputed_now,
        "coverage_%": coverage,
    }


def evaluate_backtest(
    df_base: pd.DataFrame,
    weight_col: str,
    value_col: str,
    group_levels: list,
    lower_count: int = 5,
    max_cv: float = 1.0,
    n_samples: int = 2000,
    random_state: int = 42,
    year_window: int = 1,
) -> dict:
    known = df_base.loc[
        df_base[weight_col].notna()
        & (df_base[weight_col] > 0)
        & df_base[value_col].notna()
        & (df_base[value_col] > 0)
    ].copy()

    test_idx = known.sample(min(n_samples, len(known)), random_state=random_state).index
    reference_df = df_base.drop(index=test_idx).copy()
    test_df = df_base.loc[test_idx].copy()

    test_df["true_netweight_kg"] = test_df[weight_col]
    test_df[weight_col] = np.nan
    test_df["usable_for_kg"] = False
    test_df["price_per_kg"] = np.nan

    test_df = impute_weight(
        test_df,
        group_levels,
        lower_count=lower_count,
        max_cv=max_cv,
        reference_df=reference_df,
        year_window=year_window,
    )

    test_df["imputed_success"] = test_df["imputed_netweight_kg"].notna()
    test_df["abs_error"] = (
        test_df["imputed_netweight_kg"] - test_df["true_netweight_kg"]
    ).abs()
    test_df["rel_error"] = test_df["abs_error"] / test_df["true_netweight_kg"]

    success_rate = round(test_df["imputed_success"].mean() * 100, 2)
    successful_rel_error = test_df.loc[test_df["imputed_success"], "rel_error"]

    if successful_rel_error.empty:
        median_rel_error = None
        p90_rel_error = None
        mean_rel_error = None
        max_rel_error = None
    else:
        median_rel_error = round(successful_rel_error.median(), 4)
        p90_rel_error = round(successful_rel_error.quantile(0.90), 4)
        mean_rel_error = round(successful_rel_error.mean(), 4)
        max_rel_error = round(successful_rel_error.max(), 4)

    print(
        f"Backtest success rate: {success_rate}% | "
        f"Median: {median_rel_error} | "
        f"P90: {p90_rel_error} | "
        f"Mean: {mean_rel_error} | "
        f"Max: {max_rel_error}"
    )

    return {
        "success_rate_%": success_rate,
        "median_rel_error": median_rel_error,
        "p90_rel_error": p90_rel_error,
        "mean_rel_error": mean_rel_error,
        "max_rel_error": max_rel_error,
        "test_df": test_df,
    }
