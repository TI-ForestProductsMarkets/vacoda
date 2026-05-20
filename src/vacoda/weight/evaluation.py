from typing import Sequence

import numpy as np
import pandas as pd

from vacoda.utils.logger import get_logger

logger = get_logger("VALIDATION")


def build_evaluation_base(
    df,
    target_col="netweight_kg_raw",
):
    base = df.loc[
        df[target_col].notna()
        & (df[target_col] > 0)].copy()
    
    logger.debug(
        "Built evaluation base for %s with %d/%d eligible rows",
        target_col,
        len(base),
        len(df),
    )
    return base


def _validate_candidate_inputs(
    df: pd.DataFrame,
    candidate_cols: Sequence[str],
    unit_cols: Sequence[str | None],
    target_col: str,
    value_col: str,
    estimation_flag_cols: Sequence[str | None] | None = None,
) -> None:
    if not candidate_cols:
        logger.error("Weight candidate evaluation requires at least one candidate column")
        raise ValueError("candidate_cols must contain at least one candidate column")

    if len(candidate_cols) != len(unit_cols):
        logger.error(
            "Candidate/unit length mismatch: %d candidates vs %d unit columns",
            len(candidate_cols),
            len(unit_cols),
        )
        raise ValueError("candidate_cols and unit_cols must have the same length")

    if estimation_flag_cols is not None and len(candidate_cols) != len(estimation_flag_cols):
        logger.error(
            "Candidate/estimation-flag length mismatch: %d candidates vs %d estimation flags",
            len(candidate_cols),
            len(estimation_flag_cols),
        )
        raise ValueError("candidate_cols and estimation_flag_cols must have the same length")

    required_cols = {target_col, value_col, *candidate_cols}
    required_cols.update(unit_col for unit_col in unit_cols if unit_col is not None)

    if estimation_flag_cols is not None:
        required_cols.update(flag_col for flag_col in estimation_flag_cols if flag_col is not None)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("Missing required columns for weight candidate evaluation: %s", sorted(missing_cols))
        raise KeyError(f"Missing required columns: {sorted(missing_cols)}")

    logger.debug(
        "Validated weight candidates=%s with unit columns=%s estimation flags=%s against target=%s and value=%s",
        list(candidate_cols),
        list(unit_cols),
        list(estimation_flag_cols) if estimation_flag_cols is not None else None,
        target_col,
        value_col,
    )


def _evaluate_weight_candidate(
    df,
    candidate_col,
    unit_col=None,
    estimation_flag_col=None,
    target_col="netweight_kg_raw",
    value_col="value_usd_raw",
    kg_code=8,
):
    base_candidate_mask = (
        df[candidate_col].notna()
        & (df[candidate_col] > 0)
    )

    comparable_mask = (
        df[target_col].notna()
        & (df[target_col] > 0)
        & base_candidate_mask
    )

    mask = comparable_mask.copy()

    if unit_col is not None:
        mask &= df[unit_col].eq(kg_code)

    if mask.sum() == 0:
        logger.debug(
            "Candidate %s has no eligible rows for target=%s (unit_col=%s, kg_code=%s)",
            candidate_col,
            target_col,
            unit_col,
            kg_code,
        )
        return pd.Series({
            "coverage_pct": 0.0,
            "value_usd_pct": 0.0,
            "exact_match_pct": np.nan,
            "median_ape": np.nan,
            "p90_ape": np.nan,
            "match_1pct": np.nan,
            "match_5pct": np.nan,
            "match_10pct": np.nan,
            "median_log_price_gap": np.nan,
            "candidate_estimated_pct": np.nan,
            "comparable_estimated_pct": np.nan,
            "eligible_estimated_pct_kg": np.nan,
        })

    eligible = df.loc[mask].copy()
    cand = eligible[candidate_col].astype(float)
    true = eligible[target_col].astype(float)

    exact_match = cand.eq(true)
    ape = (cand - true).abs() / true

    price_mask = eligible[value_col].notna() & (eligible[value_col] > 0)
    if price_mask.any():
        implied_price = eligible.loc[price_mask, value_col].astype(float) / cand.loc[price_mask]
        true_price = eligible.loc[price_mask, value_col].astype(float) / true.loc[price_mask]
        log_price_gap = (np.log(implied_price) - np.log(true_price)).abs()
        median_log_price_gap = log_price_gap.median()
    else:
        median_log_price_gap = np.nan

    if estimation_flag_col is not None and estimation_flag_col in df.columns:
        candidate_estimated_pct = df.loc[base_candidate_mask, estimation_flag_col].fillna(False).mean() * 100
        comparable_estimated_pct = df.loc[comparable_mask, estimation_flag_col].fillna(False).mean() * 100
        eligible_estimated_pct_kg = df.loc[mask, estimation_flag_col].fillna(False).mean() * 100
    else:
        candidate_estimated_pct = np.nan
        comparable_estimated_pct = np.nan
        eligible_estimated_pct_kg = np.nan

    metrics = pd.Series({
        "coverage_pct": mask.mean() * 100,
        "value_usd_pct": price_mask.mean() * 100,
        "exact_match_pct": exact_match.mean() * 100,
        "median_ape": ape.median() * 100,
        "p90_ape": ape.quantile(0.90) * 100,
        "match_1pct": (ape <= 0.01).mean() * 100,
        "match_5pct": (ape <= 0.05).mean() * 100,
        "match_10pct": (ape <= 0.10).mean() * 100,
        "median_log_price_gap": median_log_price_gap,
        "candidate_estimated_pct": candidate_estimated_pct,
        "comparable_estimated_pct": comparable_estimated_pct,
        "eligible_estimated_pct_kg": eligible_estimated_pct_kg,
    })

    logger.debug(
        "Candidate %s metrics: coverage_pct=%.3f exact_match_pct=%.3f match_5pct=%.3f comparable_estimated_pct=%s eligible_estimated_pct_kg=%s value_usd_pct=%.3f median_log_price_gap=%s",
        candidate_col,
        metrics["coverage_pct"],
        metrics["exact_match_pct"],
        metrics["match_5pct"],
        f"{metrics['comparable_estimated_pct']:.3f}" if pd.notna(metrics["comparable_estimated_pct"]) else "nan",
        f"{metrics['eligible_estimated_pct_kg']:.3f}" if pd.notna(metrics["eligible_estimated_pct_kg"]) else "nan",
        metrics["value_usd_pct"],
        f"{metrics['median_log_price_gap']:.6f}" if pd.notna(metrics["median_log_price_gap"]) else "nan",
    )
    return metrics


def evaluate_weight_candidates(
    df,
    candidate_cols,
    unit_cols,
    estimation_flag_cols=None,
    target_col="netweight_kg_raw",
    value_col="value_usd_raw",
    kg_code=8):
    """
    Evaluate multiple candidate columns and return one diagnostics table.
    """
    logger.debug(
        "Evaluating %d weight candidates for target=%s",
        len(candidate_cols),
        target_col,
    )

    if estimation_flag_cols is None:
        estimation_flag_cols = [None] * len(candidate_cols)

    _validate_candidate_inputs(
        df=df,
        candidate_cols=candidate_cols,
        unit_cols=unit_cols,
        target_col=target_col,
        value_col=value_col,
        estimation_flag_cols=estimation_flag_cols,
    )

    diagnostics = pd.DataFrame({
        candidate_col: _evaluate_weight_candidate(
            df,
            candidate_col=candidate_col,
            unit_col=unit_col,
            estimation_flag_col=estimation_flag_col,
            target_col=target_col,
            value_col=value_col,
            kg_code=kg_code,
        )
        for candidate_col, unit_col, estimation_flag_col in zip(candidate_cols, unit_cols, estimation_flag_cols)
    }).T

    diagnostics = diagnostics.sort_values(
        by=["exact_match_pct", "match_5pct", "coverage_pct", "median_log_price_gap"],
        ascending=[False, False, False, True],
    )

    logger.debug(
        "Completed weight candidate evaluation. Best candidate=%s",
        diagnostics.index[0] if not diagnostics.empty else "none",
    )
    return diagnostics


def rank_weight_candidates(diagnostics_df, return_all_metrics=True):
    """
    Rank candidates for sequential recovery using a compact decision subset.

    The input diagnostics table may contain richer metrics for analysis;
    only a few are used here for the actual ordering.
    """
    ranked = diagnostics_df.copy()
    ranked["exact_match_pct"] = ranked["exact_match_pct"].fillna(-1.0)
    ranked["match_5pct"] = ranked["match_5pct"].fillna(-1.0)
    ranked["coverage_pct"] = ranked["coverage_pct"].fillna(0.0)
    ranked["value_usd_pct"] = ranked["value_usd_pct"].fillna(0.0)
    ranked["median_log_price_gap"] = ranked["median_log_price_gap"].fillna(np.inf)

    ranked = ranked.sort_values(
        by=["exact_match_pct", "match_5pct", "coverage_pct", "value_usd_pct", "median_log_price_gap"],
        ascending=[False, False, False, False, True],
    )

    ranked["selection_rank"] = range(1, len(ranked) + 1)
    logger.debug(
        "Ranked weight candidates in order: %s",
        ranked.index.tolist(),
    )

    ranking_view_cols = [
        "selection_rank",
        "coverage_pct",
        "value_usd_pct",
        "exact_match_pct",
        "match_5pct",
        "median_log_price_gap",
    ]

    optional_cols = [
        "candidate_estimated_pct",
        "comparable_estimated_pct",
        "eligible_estimated_pct_kg",
    ]
    ranking_view_cols.extend([col for col in optional_cols if col in ranked.columns])

    if return_all_metrics:
        return ranked

    return ranked[ranking_view_cols]


def get_ranked_weight_candidates(diagnostics_df: pd.DataFrame) -> list[str]:
    """Return only the ranked candidate names for pipeline consumption."""
    ranked = rank_weight_candidates(diagnostics_df, return_all_metrics=True)
    logger.debug("Returning ranked candidate names: %s", ranked.index.tolist())
    return ranked.index.tolist()


def format_weight_candidate_ranking(diagnostics_df: pd.DataFrame) -> str:
    """Render a compact ranking table for logs and notebook summaries."""
    ranked = rank_weight_candidates(diagnostics_df, return_all_metrics=True)

    display_cols = [
        "selection_rank",
        "coverage_pct",
        "exact_match_pct",
        "match_5pct",
        "value_usd_pct",
        "median_log_price_gap",
    ]
    optional_cols = [
        "candidate_estimated_pct",
        "comparable_estimated_pct",
        "eligible_estimated_pct_kg",
    ]
    display_cols.extend([col for col in optional_cols if col in ranked.columns])

    compact = ranked.loc[:, display_cols].copy()
    compact = compact.rename(
        columns={
            "selection_rank": "rk",
            "coverage_pct": "cov%",
            "exact_match_pct": "eq%",
            "match_5pct": "m5%",
            "value_usd_pct": "usd%",
            "median_log_price_gap": "gap",
            "candidate_estimated_pct": "est_cand%",
            "comparable_estimated_pct": "est_cmp%",
            "eligible_estimated_pct_kg": "est_kg%",
        }
    )
    compact.index.name = "candidate"

    round_map = {
        "cov%": 2,
        "eq%": 2,
        "m5%": 2,
        "usd%": 2,
        "gap": 4,
        "est_cand%": 2,
        "est_cmp%": 2,
        "est_kg%": 2,
    }
    round_map = {k: v for k, v in round_map.items() if k in compact.columns}

    return compact.round(round_map).to_string()