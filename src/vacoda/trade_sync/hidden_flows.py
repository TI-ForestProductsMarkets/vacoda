from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd



@dataclass
class HiddenFlowMatchConfig:
    year_window: int = 1

    # hard filters
    require_same_commodity: bool = True
    require_positive_value: bool = True
    require_opposite_flow: bool = True

    # candidate size control
    max_candidates: Optional[int] = 5000

    # scoring tolerances
    price_rel_tol: float = 0.30      # 30% relative gap ~= decent match
    value_rel_tol: float = 0.50      # more tolerant because CIF/FOB etc.
    weight_rel_tol: float = 0.50

    # score weights
    weight_price: float = 0.45
    weight_value: float = 0.20
    weight_weight: float = 0.15
    weight_year: float = 0.10
    weight_estimation: float = 0.10

    # estimation preference
    prefer_non_estimated: bool = True

    # minimum final score to call plausible
    plausible_score_threshold: float = 0.60

def safe_rel_gap(a: pd.Series | float, b: pd.Series | float) -> pd.Series:
    """
    Relative gap using max(|a|, |b|) in denominator.
    Result in [0, inf). Lower is better.
    """
    a_arr = np.asarray(a, dtype="float64")
    b_arr = np.asarray(b, dtype="float64")
    denom = np.maximum(np.maximum(np.abs(a_arr), np.abs(b_arr)), 1e-12)
    return pd.Series(np.abs(a_arr - b_arr) / denom)


def closeness_from_rel_gap(rel_gap: pd.Series, tol: float) -> pd.Series:
    """
    Convert relative gap to score in [0,1].
    gap = 0         -> 1
    gap = tol       -> 0
    gap > tol       -> 0
    """
    score = 1.0 - (rel_gap / tol)
    score = score.clip(lower=0.0, upper=1.0)
    return score


def year_score(target_year: int, candidate_year: pd.Series, year_window: int) -> pd.Series:
    """
    Linear year closeness score:
    same year -> 1
    at boundary year_window -> 0
    beyond -> 0
    """
    gap = (candidate_year - target_year).abs()
    if year_window <= 0:
        return (gap == 0).astype(float)
    score = 1.0 - (gap / year_window)
    return score.clip(lower=0.0, upper=1.0)


def estimation_quality_score(estimation_code: pd.Series) -> pd.Series:
    """
    Lower estimation_code is treated as better.
    Example UN Comtrade interpretation often used:
      0 = no estimation
      2 = quantity estimated
      4 = net weight estimated
      6 = both estimated
    Unknown / missing gets intermediate penalty.
    """
    s = estimation_code.copy()

    out = pd.Series(0.5, index=s.index, dtype="float64")
    out.loc[s == 0] = 1.0
    out.loc[s == 2] = 0.7
    out.loc[s == 4] = 0.5
    out.loc[s == 6] = 0.2
    return out


def add_price_if_missing(
    df: pd.DataFrame,
    value_col: str = "value_raw",
    weight_col: str = "netweight_kg",
    price_col: str = "calc_price_per_kg",
) -> pd.DataFrame:
    """
    Adds price column if not already present.
    """
    out = df.copy()
    if price_col not in out.columns:
        out[price_col] = np.nan
        valid = out[weight_col].notna() & (out[weight_col] > 0) & out[value_col].notna() & (out[value_col] > 0)
        out.loc[valid, price_col] = out.loc[valid, value_col] / out.loc[valid, weight_col]
    return out


def infer_opposite_flow_map(
    df: pd.DataFrame,
    flow_col: str = "flow_code",
) -> dict:
    """
    Default mapping for common simple cases.
    Adjust this for your project if your flow codes differ.
    """
    unique_flows = sorted(df[flow_col].dropna().unique().tolist())

    # Common placeholder assumption:
    # 1 = import, 2 = export
    default_map = {}
    if 1 in unique_flows and 2 in unique_flows:
        default_map[1] = [2]
        default_map[2] = [1]

    return default_map


def generate_candidates(
    target_row: pd.Series,
    df: pd.DataFrame,
    config: HiddenFlowMatchConfig,
    *,
    reporter_col: str = "reporter_code",
    partner_col: str = "partner_code",
    commodity_col: str = "commodity_code",
    year_col: str = "year",
    flow_col: str = "flow_code",
    value_col: str = "value_raw",
    weight_col: str = "netweight_kg",
    price_col: str = "calc_price_per_kg",
    estimation_col: str = "estimation_code",
    flow_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Generate plausible candidate rows for a target trade record.

    Generic logic:
    - same commodity
    - opposite flow (import -> export, export -> import), if requested
    - target partner appears as candidate partner OR candidate reporter
    - within year window
    - positive value if requested

    This is intentionally broad. Scoring will refine later.
    """
    data = add_price_if_missing(df, value_col=value_col, weight_col=weight_col, price_col=price_col)

    target_reporter = target_row[reporter_col]
    target_partner = target_row[partner_col]
    target_commodity = target_row[commodity_col]
    target_year = target_row[year_col]
    target_flow = target_row[flow_col]

    mask = pd.Series(True, index=data.index)

    if config.require_same_commodity:
        mask &= data[commodity_col] == target_commodity

    mask &= data[year_col].between(target_year - config.year_window, target_year + config.year_window)

    if config.require_positive_value:
        mask &= data[value_col].notna() & (data[value_col] > 0)

    if config.require_opposite_flow:
        if flow_map is None:
            flow_map = infer_opposite_flow_map(data, flow_col=flow_col)
        allowed_flows = flow_map.get(target_flow, None)
        if allowed_flows is not None:
            mask &= data[flow_col].isin(allowed_flows)

    # Generic "hidden chain" logic:
    # candidate should involve either the target partner or target reporter somehow.
    #
    # For example if target is Germany importing from NL,
    # good candidates may be exports from USA to NL.
    # So candidate.partner == target.partner is especially useful.
    #
    # We also allow candidate.reporter == target.partner or candidate.partner == target.reporter
    # to keep it generic.
    connection_mask = (
        (data[partner_col] == target_partner) |
        (data[reporter_col] == target_partner) |
        (data[partner_col] == target_reporter)
    )

    mask &= connection_mask

    # Remove exact same row if index matches
    if target_row.name in data.index:
        mask &= data.index != target_row.name

    candidates = data.loc[mask].copy()

    if config.max_candidates is not None and len(candidates) > config.max_candidates:
        # coarse pruning before scoring: keep nearest years first
        candidates["_tmp_year_gap"] = (candidates[year_col] - target_year).abs()
        candidates = candidates.sort_values("_tmp_year_gap", ascending=True).head(config.max_candidates).copy()
        candidates = candidates.drop(columns="_tmp_year_gap")

    return candidates


def score_candidates(
    target_row: pd.Series,
    candidates: pd.DataFrame,
    config: HiddenFlowMatchConfig,
    *,
    reporter_col: str = "reporter_code",
    partner_col: str = "partner_code",
    commodity_col: str = "commodity_code",
    year_col: str = "year",
    flow_col: str = "flow_code",
    value_col: str = "value_raw",
    weight_col: str = "netweight_kg",
    price_col: str = "calc_price_per_kg",
    estimation_col: str = "estimation_code",
) -> pd.DataFrame:
    """
    Score candidate rows against one target row.
    Returns candidates with detailed gap and score columns.
    """
    if candidates.empty:
        return candidates.copy()

    scored = candidates.copy()

    target_year = target_row[year_col]
    target_value = target_row.get(value_col, np.nan)
    target_weight = target_row.get(weight_col, np.nan)
    target_price = target_row.get(price_col, np.nan)

    # Relative gaps
    scored["value_gap_rel"] = np.nan
    scored["weight_gap_rel"] = np.nan
    scored["price_gap_rel"] = np.nan

    if pd.notna(target_value) and target_value > 0:
        scored["value_gap_rel"] = safe_rel_gap(scored[value_col], target_value)

    if pd.notna(target_weight) and target_weight > 0:
        scored["weight_gap_rel"] = safe_rel_gap(scored[weight_col], target_weight)

    if pd.notna(target_price) and target_price > 0:
        scored["price_gap_rel"] = safe_rel_gap(scored[price_col], target_price)

    # Component scores
    scored["score_price"] = 0.0
    scored["score_value"] = 0.0
    scored["score_weight"] = 0.0

    if pd.notna(target_price) and target_price > 0:
        scored["score_price"] = closeness_from_rel_gap(scored["price_gap_rel"], config.price_rel_tol)

    if pd.notna(target_value) and target_value > 0:
        scored["score_value"] = closeness_from_rel_gap(scored["value_gap_rel"], config.value_rel_tol)

    if pd.notna(target_weight) and target_weight > 0:
        valid_weight = scored[weight_col].notna() & (scored[weight_col] > 0)
        scored.loc[valid_weight, "score_weight"] = closeness_from_rel_gap(
            scored.loc[valid_weight, "weight_gap_rel"], config.weight_rel_tol
        )

    scored["score_year"] = year_score(target_year, scored[year_col], config.year_window)

    if estimation_col in scored.columns:
        scored["score_estimation"] = estimation_quality_score(scored[estimation_col])
    else:
        scored["score_estimation"] = 0.5

    # structural bonus/penalty

    target_reporter = target_row[reporter_col]
    target_partner = target_row[partner_col]

    scored["score_connection"] = 0.0

    # strongest hidden-flow pattern:
    # candidate.partner == target.partner
    # ex: target = DE import from NL, candidate = US export to NL
    scored.loc[scored[partner_col] == target_partner, "score_connection"] += 1.0

    # weaker but still useful:
    scored.loc[scored[reporter_col] == target_partner, "score_connection"] += 0.4
    scored.loc[scored[partner_col] == target_reporter, "score_connection"] += 0.2

    scored["score_connection"] = scored["score_connection"].clip(upper=1.0)

    # fold connection lightly into year score rather than adding a new config weight
    scored["score_year_connection"] = 0.7 * scored["score_year"] + 0.3 * scored["score_connection"]

    # --------------------------------------------------------
    # Final weighted score
    # --------------------------------------------------------
    scored["match_score"] = (
        config.weight_price * scored["score_price"] +
        config.weight_value * scored["score_value"] +
        config.weight_weight * scored["score_weight"] +
        config.weight_year * scored["score_year_connection"] +
        config.weight_estimation * scored["score_estimation"]
    )

    scored["is_plausible_hidden_counterpart"] = scored["match_score"] >= config.plausible_score_threshold

    scored = scored.sort_values("match_score", ascending=False).copy()
    scored["match_rank"] = np.arange(1, len(scored) + 1)

    return scored


def find_best_hidden_matches(
    target_row: pd.Series,
    df: pd.DataFrame,
    config: Optional[HiddenFlowMatchConfig] = None,
    *,
    top_k: int = 10,
    reporter_col: str = "reporter_code",
    partner_col: str = "partner_code",
    commodity_col: str = "commodity_code",
    year_col: str = "year",
    flow_col: str = "flow_code",
    value_col: str = "value_raw",
    weight_col: str = "netweight_kg",
    price_col: str = "calc_price_per_kg",
    estimation_col: str = "estimation_code",
    flow_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper:
    generate candidates -> score them -> return top_k
    """
    if config is None:
        config = HiddenFlowMatchConfig()

    candidates = generate_candidates(
        target_row=target_row,
        df=df,
        config=config,
        reporter_col=reporter_col,
        partner_col=partner_col,
        commodity_col=commodity_col,
        year_col=year_col,
        flow_col=flow_col,
        value_col=value_col,
        weight_col=weight_col,
        price_col=price_col,
        estimation_col=estimation_col,
        flow_map=flow_map,
    )

    scored = score_candidates(
        target_row=target_row,
        candidates=candidates,
        config=config,
        reporter_col=reporter_col,
        partner_col=partner_col,
        commodity_col=commodity_col,
        year_col=year_col,
        flow_col=flow_col,
        value_col=value_col,
        weight_col=weight_col,
        price_col=price_col,
        estimation_col=estimation_col,
    )

    return scored.head(top_k).copy()



def find_hidden_matches_for_subset(
    df_target: pd.DataFrame,
    df_all: pd.DataFrame,
    config: Optional[HiddenFlowMatchConfig] = None,
    *,
    top_k_per_row: int = 3,
    id_col: Optional[str] = None,
    reporter_col: str = "reporter_code",
    partner_col: str = "partner_code",
    commodity_col: str = "commodity_code",
    year_col: str = "year",
    flow_col: str = "flow_code",
    value_col: str = "value_raw",
    weight_col: str = "netweight_kg",
    price_col: str = "calc_price_per_kg",
    estimation_col: str = "estimation_code",
    flow_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Runs matching for many target rows and stacks the results.
    """
    if config is None:
        config = HiddenFlowMatchConfig()

    rows = []

    for idx, target_row in df_target.iterrows():
        matched = find_best_hidden_matches(
            target_row=target_row,
            df=df_all,
            config=config,
            top_k=top_k_per_row,
            reporter_col=reporter_col,
            partner_col=partner_col,
            commodity_col=commodity_col,
            year_col=year_col,
            flow_col=flow_col,
            value_col=value_col,
            weight_col=weight_col,
            price_col=price_col,
            estimation_col=estimation_col,
            flow_map=flow_map,
        )

        if matched.empty:
            continue

        matched = matched.copy()
        matched["target_index"] = idx

        if id_col is not None and id_col in df_target.columns:
            matched["target_id"] = target_row[id_col]

        rows.append(matched)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, axis=0, ignore_index=True)
    return out