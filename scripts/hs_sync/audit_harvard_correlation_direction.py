"""
Audit directionality of Harvard/Growth Lab correlation-group CSV files.

This script checks whether files such as:

    from_HS2022_to_HS2017.csv

are being interpreted correctly as:

    source_hs/source_cmd -> target_hs/target_cmd

It does not use weights. It only audits mapping structure and empirical
support from the trade source summary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacoda.config.hs_sync import (
    get_cmd_col,
    get_coverage_input_pattern,
    get_coverage_input_stage,
    get_coverage_input_subdir,
    get_coverage_mapping_files,
    get_coverage_mappings_stage,
    get_coverage_mappings_subdir,
    get_engine_name,
    get_engine_threads,
    get_hs_col,
    get_progress_bool,
    get_value_col,
    get_year_col,
    load_hs_sync_config,
)
from vacoda.config.paths import get_data_dir
from vacoda.hs_sync.diagnostics import build_trade_source_pair_summary


HS_TO_OFFICIAL = {
    "H0": "HS1988",
    "H1": "HS1996",
    "H2": "HS2002",
    "H3": "HS2007",
    "H4": "HS2012",
    "H5": "HS2017",
    "H6": "HS2022",
}


def _resolve_data_stage_dir(stage: str, subdir: str) -> Path:
    base = get_data_dir(stage)
    return base / subdir if subdir else base


def _normalize_code(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.mask(out.isin(["", "nan", "None", "<NA>"]))
    is_numeric = out.str.match(r"^[0-9]+$", na=False)
    out = out.where(~is_numeric, out.str.zfill(6))
    return out


def _parse_mapping_key(key: str) -> tuple[str, str]:
    parts = key.upper().split("_TO_")
    if len(parts) != 2:
        raise ValueError(
            f"Mapping key must look like H4_to_H5 or H6_to_H5. Got: {key}"
        )
    return parts[0], parts[1]


def _actual_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    targets_per_source = (
        df.groupby("source_cmd")["target_cmd"]
        .nunique()
        .rename("n_targets_per_source")
        .reset_index()
    )

    sources_per_target = (
        df.groupby("target_cmd")["source_cmd"]
        .nunique()
        .rename("n_sources_per_target")
        .reset_index()
    )

    out = (
        df.merge(targets_per_source, on="source_cmd", how="left")
        .merge(sources_per_target, on="target_cmd", how="left")
    )

    out["actual_mapping_category"] = "actual_n_to_n"

    out.loc[
        (out["n_targets_per_source"] == 1)
        & (out["n_sources_per_target"] == 1),
        "actual_mapping_category",
    ] = "actual_1_to_1"

    out.loc[
        (out["n_targets_per_source"] == 1)
        & (out["n_sources_per_target"] > 1),
        "actual_mapping_category",
    ] = "actual_n_to_1"

    out.loc[
        (out["n_targets_per_source"] > 1)
        & (out["n_sources_per_target"] == 1),
        "actual_mapping_category",
    ] = "actual_1_to_n"

    return out


def _support_rates(
    *,
    mapping: pd.DataFrame,
    observed_source_codes: set[str],
    observed_target_codes: set[str],
) -> dict[str, float]:
    source_codes = set(mapping["source_cmd"].dropna())
    target_codes = set(mapping["target_cmd"].dropna())

    source_hit = len(source_codes & observed_source_codes)
    target_hit = len(target_codes & observed_target_codes)

    source_rate = source_hit / len(source_codes) if source_codes else 0.0
    target_rate = target_hit / len(target_codes) if target_codes else 0.0

    return {
        "n_mapping_source_codes": len(source_codes),
        "n_mapping_target_codes": len(target_codes),
        "source_code_hit_rate": source_rate,
        "target_code_hit_rate": target_rate,
        "mean_hit_rate": (source_rate + target_rate) / 2,
    }


def main() -> None:
    cfg = load_hs_sync_config()

    input_dir = _resolve_data_stage_dir(
        get_coverage_input_stage(cfg),
        get_coverage_input_subdir(cfg),
    )

    mapping_dir = _resolve_data_stage_dir(
        get_coverage_mappings_stage(cfg),
        get_coverage_mappings_subdir(cfg),
    )

    print("=" * 80)
    print("Building observed trade source summary")
    print("=" * 80)

    source_summary = build_trade_source_pair_summary(
        data_dir=input_dir,
        pattern=get_coverage_input_pattern(cfg),
        engine=get_engine_name(cfg),
        year_col=get_year_col(cfg),
        hs_col=get_hs_col(cfg),
        cmd_col=get_cmd_col(cfg),
        value_col=get_value_col(cfg),
        min_year=2017,
        threads=get_engine_threads(cfg),
        show_progress=get_progress_bool(cfg),
    )

    source_summary = source_summary.copy()
    source_summary["source_hs"] = (
        source_summary["source_hs"].astype("string").str.strip().str.upper()
    )
    source_summary["source_cmd"] = _normalize_code(source_summary["source_cmd"])

    observed_by_hs = {
        hs: set(group["source_cmd"].dropna())
        for hs, group in source_summary.groupby("source_hs")
    }

    mapping_files = get_coverage_mapping_files(cfg)

    audit_rows: list[dict[str, object]] = []

    for mapping_key, filename in mapping_files.items():
        configured_source_hs, configured_target_hs = _parse_mapping_key(mapping_key)

        path = mapping_dir / filename

        raw = pd.read_csv(path, dtype=str)

        print()
        print("=" * 80)
        print(f"{mapping_key}: {path.name}")
        print("=" * 80)
        print("Columns:", list(raw.columns))
        print("Unique adjustment values:", raw.get("adjustment", pd.Series(dtype=str)).dropna().unique()[:10])

        if "code.source" not in raw.columns or "code.target" not in raw.columns:
            raise KeyError(
                f"{path} must contain code.source and code.target. "
                f"Available columns: {list(raw.columns)}"
            )

        mapping = raw.rename(
            columns={
                "code.source": "source_cmd",
                "code.target": "target_cmd",
                "Relationship": "relationship_original",
                "adjustment": "adjustment_original",
                "group.id": "group_id",
            }
        ).copy()

        mapping["source_cmd"] = _normalize_code(mapping["source_cmd"])
        mapping["target_cmd"] = _normalize_code(mapping["target_cmd"])

        mapping = _actual_cardinality(mapping)

        card_summary = (
            mapping.groupby(["relationship_original", "actual_mapping_category"], dropna=False)
            .agg(
                n_rows=("source_cmd", "size"),
                n_source_codes=("source_cmd", "nunique"),
                n_target_codes=("target_cmd", "nunique"),
            )
            .reset_index()
            .sort_values(["relationship_original", "actual_mapping_category"])
        )

        print()
        print("Relationship label vs actual source→target cardinality:")
        print(card_summary.to_string(index=False))

        observed_source_codes = observed_by_hs.get(configured_source_hs, set())
        observed_target_codes = observed_by_hs.get(configured_target_hs, set())

        configured_support = _support_rates(
            mapping=mapping,
            observed_source_codes=observed_source_codes,
            observed_target_codes=observed_target_codes,
        )

        reversed_mapping = mapping.rename(
            columns={
                "source_cmd": "target_cmd",
                "target_cmd": "source_cmd",
            }
        ).copy()

        reversed_support = _support_rates(
            mapping=reversed_mapping,
            observed_source_codes=observed_source_codes,
            observed_target_codes=observed_target_codes,
        )

        print()
        print("Configured direction support:")
        print(
            f"  {configured_source_hs} ({HS_TO_OFFICIAL.get(configured_source_hs)}) "
            f"→ {configured_target_hs} ({HS_TO_OFFICIAL.get(configured_target_hs)})"
        )
        print(f"  source_code_hit_rate: {configured_support['source_code_hit_rate']:.3f}")
        print(f"  target_code_hit_rate: {configured_support['target_code_hit_rate']:.3f}")
        print(f"  mean_hit_rate:        {configured_support['mean_hit_rate']:.3f}")

        print()
        print("Reversed interpretation support:")
        print(f"  source_code_hit_rate: {reversed_support['source_code_hit_rate']:.3f}")
        print(f"  target_code_hit_rate: {reversed_support['target_code_hit_rate']:.3f}")
        print(f"  mean_hit_rate:        {reversed_support['mean_hit_rate']:.3f}")

        if configured_support["mean_hit_rate"] >= reversed_support["mean_hit_rate"]:
            conclusion = "configured_direction_supported"
        else:
            conclusion = "reversed_direction_has_higher_support"

        print()
        print("Conclusion:", conclusion)

        audit_rows.append(
            {
                "mapping_key": mapping_key,
                "filename": filename,
                "configured_source_hs": configured_source_hs,
                "configured_target_hs": configured_target_hs,
                "configured_mean_hit_rate": configured_support["mean_hit_rate"],
                "reversed_mean_hit_rate": reversed_support["mean_hit_rate"],
                "conclusion": conclusion,
            }
        )

    audit = pd.DataFrame(audit_rows)

    print()
    print("=" * 80)
    print("Overall direction audit")
    print("=" * 80)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()