from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from vacoda.config.base import (
    get_config_path,
    get_step_config_path,
    load_yaml_config,
    require_dict,
)
from vacoda.config.cif_fob import CifFobConfig
from vacoda.config.pipeline import PipelineEngineConfig, PipelineScopeConfig


FEATURES = [
    "log_lagged_bilateral_trade_value",
    "log_lagged_pair_product_trade_value",
    "lagged_pair_product_median_factor",
    "lagged_pair_median_factor",
    "years_since_latest_pair_product_factor",
    "years_since_latest_pair_factor",
    "lagged_pair_product_factor_observations",
    "lagged_pair_factor_observations",
]

KEYS = ["year", "exporter_iso", "importer_iso", "target_cmd"]


def load_runtime_config() -> CifFobConfig:
    raw = load_yaml_config(get_config_path("pipeline"))
    scope = PipelineScopeConfig.from_dict(require_dict(raw.get("scope"), key="scope"))
    engine = PipelineEngineConfig.from_dict(require_dict(raw.get("engine"), key="engine"))
    return CifFobConfig.from_yaml(
        get_step_config_path("cif_fob"),
        scope=scope,
        engine=engine,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Targeted CIF/FOB feature-distribution audit for one validation year."
    )
    parser.add_argument("--estimated-rates", type=Path, default=None)
    parser.add_argument("--training-features", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=2022)
    parser.add_argument("--tail-threshold", type=float, default=1.25)
    return parser.parse_args()


def qid(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [str(path)],
    ).fetchall()
    return {str(row[0]) for row in rows}


def load_data(
    *,
    estimated_rates_path: Path,
    training_features_path: Path,
    threads: int,
) -> tuple[pd.DataFrame, list[str]]:
    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(threads)}")
        er_cols = parquet_columns(con, estimated_rates_path)
        tf_cols = parquet_columns(con, training_features_path)

        holdout_cols = sorted(
            c for c in er_cols if c.endswith("_holdout_predicted_factor")
        )

        er_select = sorted(
            set(KEYS)
            | {
                "observed_cif_fob_factor",
                "clean_estimation_eligible",
                "estimation_evidence_value",
            }
            | set(holdout_cols)
        )
        tf_select = sorted(set(KEYS) | (set(FEATURES) & tf_cols))

        er = con.execute(
            f"SELECT {', '.join(qid(c) for c in er_select)} FROM read_parquet(?)",
            [str(estimated_rates_path)],
        ).df()
        tf = con.execute(
            f"SELECT {', '.join(qid(c) for c in tf_select)} FROM read_parquet(?)",
            [str(training_features_path)],
        ).df()
    finally:
        con.close()

    dup = tf.duplicated(KEYS, keep=False)
    if dup.any():
        raise RuntimeError(
            f"Training features contain {int(dup.sum()):,} duplicate canonical rows."
        )

    frame = er.merge(tf, on=KEYS, how="left", validate="one_to_one")
    frame = frame.loc[
        frame["clean_estimation_eligible"].fillna(False).astype(bool)
    ].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["observed_cif_fob_factor"] = pd.to_numeric(
        frame["observed_cif_fob_factor"], errors="coerce"
    )
    return frame, holdout_cols


def safe_quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def year_feature_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, part in frame.groupby("year", sort=True):
        for feature in FEATURES:
            if feature not in part.columns:
                continue
            values = pd.to_numeric(part[feature], errors="coerce")
            rows.append(
                {
                    "year": int(year),
                    "feature": feature,
                    "n_rows": len(part),
                    "n_available": int(values.notna().sum()),
                    "missing_share": float(values.isna().mean()),
                    "p05": safe_quantile(values, 0.05),
                    "p25": safe_quantile(values, 0.25),
                    "median": safe_quantile(values, 0.50),
                    "p75": safe_quantile(values, 0.75),
                    "p95": safe_quantile(values, 0.95),
                }
            )
    return pd.DataFrame(rows)


def feature_shift_summary(frame: pd.DataFrame, target_year: int) -> pd.DataFrame:
    target = frame.loc[frame["year"].eq(target_year)]
    reference = frame.loc[~frame["year"].eq(target_year)]
    rows: list[dict[str, Any]] = []

    for feature in FEATURES:
        if feature not in frame.columns:
            continue
        target_raw = pd.to_numeric(target[feature], errors="coerce")
        reference_raw = pd.to_numeric(reference[feature], errors="coerce")
        target_values = target_raw.dropna()
        reference_values = reference_raw.dropna()

        if target_values.empty or reference_values.empty:
            ks_stat = np.nan
            ks_pvalue = np.nan
        else:
            ks = ks_2samp(target_values, reference_values, alternative="two-sided")
            ks_stat = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)

        target_median = float(target_values.median()) if not target_values.empty else np.nan
        reference_median = (
            float(reference_values.median()) if not reference_values.empty else np.nan
        )

        rows.append(
            {
                "feature": feature,
                "target_year": target_year,
                "target_n_available": len(target_values),
                "reference_n_available": len(reference_values),
                "target_missing_share": float(target_raw.isna().mean()),
                "reference_missing_share": float(reference_raw.isna().mean()),
                "target_median": target_median,
                "reference_median": reference_median,
                "median_difference": target_median - reference_median,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "ks_statistic", ascending=False, na_position="last"
    )


def factor_by_missingness(
    frame: pd.DataFrame,
    tail_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = frame.copy()
    frame["is_upper_tail"] = frame["observed_cif_fob_factor"].ge(tail_threshold)

    for year, part in frame.groupby("year", sort=True):
        for feature in FEATURES:
            if feature not in part.columns:
                continue
            missing = pd.to_numeric(part[feature], errors="coerce").isna()
            for label, mask in {"available": ~missing, "missing": missing}.items():
                group = part.loc[mask]
                if group.empty:
                    continue
                rows.append(
                    {
                        "year": int(year),
                        "feature": feature,
                        "availability": label,
                        "n_rows": len(group),
                        "tail_share": float(group["is_upper_tail"].mean()),
                        "median_factor": float(
                            group["observed_cif_fob_factor"].median()
                        ),
                        "p95_factor": float(
                            group["observed_cif_fob_factor"].quantile(0.95)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def model_name(column: str) -> str:
    suffix = "_holdout_predicted_factor"
    return column[: -len(suffix)] if column.endswith(suffix) else column


def build_target_residuals(
    *,
    frame: pd.DataFrame,
    holdout_columns: list[str],
    target_year: int,
    tail_threshold: float,
) -> pd.DataFrame:
    target = frame.loc[frame["year"].eq(target_year)].copy()
    target["is_upper_tail"] = target["observed_cif_fob_factor"].ge(tail_threshold)
    observed = pd.to_numeric(target["observed_cif_fob_factor"], errors="coerce")
    output: list[pd.DataFrame] = []

    for column in holdout_columns:
        predicted = pd.to_numeric(target[column], errors="coerce")
        valid = observed.gt(0) & predicted.gt(0)
        if not valid.any():
            continue

        part = target.loc[valid].copy()
        part["model_name"] = model_name(column)
        part["predicted_factor"] = predicted.loc[valid].to_numpy(dtype=float)
        part["log_residual"] = np.log(
            part["observed_cif_fob_factor"].to_numpy(dtype=float)
        ) - np.log(part["predicted_factor"].to_numpy(dtype=float))
        part["absolute_percentage_error"] = (
            np.abs(part["predicted_factor"] - part["observed_cif_fob_factor"])
            / part["observed_cif_fob_factor"].clip(lower=1e-12)
        )
        output.append(part)

    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def quantile_band(values: pd.Series, q: int = 5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series("missing", index=values.index, dtype="string")
    available = numeric.notna()
    if available.sum() < q:
        return result
    try:
        result.loc[available] = pd.qcut(
            numeric.loc[available], q=q, duplicates="drop"
        ).astype("string")
    except ValueError:
        result.loc[available] = "available"
    return result


def residuals_by_feature_band(residuals: pd.DataFrame) -> pd.DataFrame:
    if residuals.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for model, model_frame in residuals.groupby("model_name", sort=True):
        for feature in FEATURES:
            if feature not in model_frame.columns:
                continue
            working = model_frame.copy()
            working["feature_band"] = quantile_band(working[feature])
            for band, group in working.groupby("feature_band", dropna=False, sort=True):
                rows.append(
                    {
                        "model_name": model,
                        "feature": feature,
                        "feature_band": str(band),
                        "n_rows": len(group),
                        "tail_share": float(group["is_upper_tail"].mean()),
                        "median_log_residual": float(group["log_residual"].median()),
                        "mean_log_residual": float(group["log_residual"].mean()),
                        "median_ape": float(
                            group["absolute_percentage_error"].median()
                        ),
                        "p90_ape": float(
                            group["absolute_percentage_error"].quantile(0.90)
                        ),
                        "underprediction_share": float(
                            group["log_residual"].gt(0).mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def model_residual_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    if residuals.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    tagged = residuals.assign(
        subset=np.where(residuals["is_upper_tail"], "upper_tail", "non_tail")
    )
    for (model, subset), group in tagged.groupby(["model_name", "subset"]):
        rows.append(
            {
                "model_name": model,
                "subset": subset,
                "n_rows": len(group),
                "median_ape": float(group["absolute_percentage_error"].median()),
                "p90_ape": float(group["absolute_percentage_error"].quantile(0.90)),
                "p95_ape": float(group["absolute_percentage_error"].quantile(0.95)),
                "median_log_residual": float(group["log_residual"].median()),
                "underprediction_share": float(group["log_residual"].gt(0).mean()),
                "observed_p95": float(group["observed_cif_fob_factor"].quantile(0.95)),
                "predicted_p95": float(group["predicted_factor"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config()

    estimated_rates_path = args.estimated_rates or cfg.product_path("estimated_rates")
    training_features_path = args.training_features or cfg.product_path(
        "training_features"
    )
    output_dir = args.output_dir or (
        cfg.output_dir() / f"targeted_{args.target_year}_audit"
    )

    frame, holdout_columns = load_data(
        estimated_rates_path=estimated_rates_path,
        training_features_path=training_features_path,
        threads=cfg.engine.threads,
    )

    distribution = year_feature_distribution(frame)
    shift = feature_shift_summary(frame, args.target_year)
    missingness = factor_by_missingness(frame, args.tail_threshold)
    residuals = build_target_residuals(
        frame=frame,
        holdout_columns=holdout_columns,
        target_year=args.target_year,
        tail_threshold=args.tail_threshold,
    )
    band_summary = residuals_by_feature_band(residuals)
    residual_summary = model_residual_summary(residuals)

    output_dir.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(output_dir / "year_feature_distribution.csv", index=False)
    shift.to_csv(
        output_dir / f"{args.target_year}_feature_shift_summary.csv", index=False
    )
    missingness.to_csv(output_dir / "factor_by_feature_missingness.csv", index=False)
    residual_summary.to_csv(
        output_dir / f"{args.target_year}_model_residual_summary.csv", index=False
    )
    band_summary.to_csv(
        output_dir / f"{args.target_year}_residuals_by_feature_band.csv", index=False
    )
    if not residuals.empty:
        residuals.to_parquet(
            output_dir / f"{args.target_year}_holdout_residuals.parquet",
            index=False,
        )

    print("\n" + "=" * 96)
    print(f"TARGETED {args.target_year} FEATURE-DISTRIBUTION AUDIT")
    print("=" * 96)
    print("\nLargest feature shifts:")
    print(
        shift[
            [
                "feature",
                "target_missing_share",
                "reference_missing_share",
                "target_median",
                "reference_median",
                "ks_statistic",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    if not residual_summary.empty:
        print("\nTarget-year residual summary:")
        print(residual_summary.to_string(index=False))
    print(f"\nArtifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
