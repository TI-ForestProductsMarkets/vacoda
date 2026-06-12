"""
Build a standardized HS concordance table from Harvard/Growth Lab static
consolidated Comtrade concordances.

This creates an unoptimized, static concordance table with temporary equal
weights. It is useful for coverage diagnostics, but it is not the final
Harvard optimized conversion-weight table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.pipeline import PipelineConfig


HS_YEAR_TO_H = {
    "1992": "H0",
    "1996": "H1",
    "2002": "H2",
    "2007": "H3",
    "2012": "H4",
    "2017": "H5",
    "2022": "H6",
}


def parse_adjustment(adjustment: str) -> tuple[str, str]:
    """
    Parse strings like '1996 to 1992'.

    Returns
    -------
    tuple[str, str]
        after_year, before_year
    """

    parts = str(adjustment).strip().split(" to ")

    if len(parts) != 2:
        raise ValueError(f"Cannot parse adjustment value: {adjustment!r}")

    after_year, before_year = parts

    if after_year not in HS_YEAR_TO_H:
        raise ValueError(f"Unknown HS year in adjustment: {after_year!r}")

    if before_year not in HS_YEAR_TO_H:
        raise ValueError(f"Unknown HS year in adjustment: {before_year!r}")

    return after_year, before_year


def build_static_adjacent_concordance(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert Harvard static concordance format to vacoda standard format.

    Input columns:
        code.after
        code.before
        Relationship
        adjustment

    Output columns:
        source_hs
        source_cmd
        target_hs
        target_cmd
        relationship
        adjustment
        weight
    """

    required = {"code.after", "code.before", "Relationship", "adjustment"}
    missing = required - set(raw.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = raw.copy()

    parsed = df["adjustment"].apply(parse_adjustment)

    df["after_year"] = parsed.apply(lambda x: x[0])
    df["before_year"] = parsed.apply(lambda x: x[1])

    df["source_hs"] = df["after_year"].map(HS_YEAR_TO_H)
    df["target_hs"] = df["before_year"].map(HS_YEAR_TO_H)

    df["source_cmd"] = (
        df["code.after"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["target_cmd"] = (
        df["code.before"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["relationship"] = df["Relationship"].astype("string").str.strip()
    df["adjustment"] = df["adjustment"].astype("string").str.strip()

    df = df[
        [
            "source_hs",
            "source_cmd",
            "target_hs",
            "target_cmd",
            "relationship",
            "adjustment",
        ]
    ].drop_duplicates()

    # Temporary equal weights within each source_hs/source_cmd/target_hs group.
    # This is NOT Harvard optimized weighting. It is a static baseline.
    df["n_targets"] = (
        df.groupby(["source_hs", "source_cmd", "target_hs"])["target_cmd"]
        .transform("nunique")
    )

    df["weight"] = 1.0 / df["n_targets"]

    return df[
        [
            "source_hs",
            "source_cmd",
            "target_hs",
            "target_cmd",
            "weight",
            "relationship",
            "adjustment",
        ]
    ]


def main() -> None:
    pipeline_cfg = PipelineConfig.from_yaml()
    cfg = HsSyncConfig.from_yaml(
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    src = cfg.reference_path("comtrade.harvard_static_concordance_source")
    out = cfg.reference_path("comtrade.static_adjacent_concordance_output")

    raw = pd.read_csv(src)

    concordance = build_static_adjacent_concordance(raw)

    out.parent.mkdir(parents=True, exist_ok=True)
    concordance.to_parquet(out, index=False)

    print(f"Saved: {out}")
    print(f"Shape: {concordance.shape}")
    print(concordance.head())


if __name__ == "__main__":
    main()