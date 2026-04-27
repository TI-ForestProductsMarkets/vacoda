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
        & (df[target_col] > 0)
    ].copy()
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

    required_cols = {target_col, value_col, *candidate_cols}
    required_cols.update(unit_col for unit_col in unit_cols if unit_col is not None)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("Missing required columns for weight candidate evaluation: %s", sorted(missing_cols))
        raise KeyError(f"Missing required columns: {sorted(missing_cols)}")

    logger.debug(
        "Validated weight candidates=%s with unit columns=%s against target=%s and value=%s",
        list(candidate_cols),
        list(unit_cols),
        target_col,
        value_col,
    )


def _evaluate_weight_candidate(
    df,
    candidate_col,
    unit_col=None,
    target_col="netweight_kg_raw",
    value_col="value_usd_raw",
    kg_code=8,
):
    mask = (
        df[target_col].notna()
        & (df[target_col] > 0)
        & df[candidate_col].notna()
        & (df[candidate_col] > 0)
    )


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
    })

    logger.debug(
        "Candidate %s metrics: coverage_pct=%.3f exact_match_pct=%.3f match_5pct=%.3f value_usd_pct=%.3f median_log_price_gap=%s",
        candidate_col,
        metrics["coverage_pct"],
        metrics["exact_match_pct"],
        metrics["match_5pct"],
        metrics["value_usd_pct"],
        f"{metrics['median_log_price_gap']:.6f}" if pd.notna(metrics["median_log_price_gap"]) else "nan",
    )
    return metrics

def evaluate_weight_candidates(
    df,
    candidate_cols,
    unit_cols,
    target_col="netweight_kg_raw",
    value_col="value_usd_raw",
    kg_code=8,
):
    """
    Evaluate multiple candidate columns and return one diagnostics table.

    This keeps the notebook call-site compact while preserving a small
    per-candidate helper for the actual metric computation.
    """
    logger.debug(
        "Evaluating %d weight candidates for target=%s",
        len(candidate_cols),
        target_col,
    )

    _validate_candidate_inputs(
        df=df,
        candidate_cols=candidate_cols,
        unit_cols=unit_cols,
        target_col=target_col,
        value_col=value_col,
    )

    diagnostics = pd.DataFrame({
        candidate_col: _evaluate_weight_candidate(
            df,
            candidate_col=candidate_col,
            unit_col=unit_col,
            target_col=target_col,
            value_col=value_col,
            kg_code=kg_code,
        )
        for candidate_col, unit_col in zip(candidate_cols, unit_cols)
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
    compact = ranked.loc[
        :,
        [
            "selection_rank",
            "coverage_pct",
            "exact_match_pct",
            "match_5pct",
            "value_usd_pct",
            "median_log_price_gap",
        ],
    ].copy()
    compact = compact.rename(
        columns={
            "selection_rank": "rk",
            "coverage_pct": "cov%",
            "exact_match_pct": "eq%",
            "match_5pct": "m5%",
            "value_usd_pct": "usd%",
            "median_log_price_gap": "gap",
        }
    )
    compact.index.name = "candidate"
    return compact.round({"cov%": 2, "eq%": 2, "m5%": 2, "usd%": 2, "gap": 4}).to_string()
