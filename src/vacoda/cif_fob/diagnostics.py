"""
Diagnostics for primary model temporal holdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from vacoda.config.cif_fob import CifFobConfig


def _read_primary_run_name(
    *,
    estimated_rates_path: Path,
) -> str:
    con = duckdb.connect()

    try:
        row = con.execute(
            """
            SELECT DISTINCT primary_estimation_run
            FROM read_parquet(?)
            WHERE primary_estimation_run IS NOT NULL
            """,
            [str(estimated_rates_path)],
        ).fetchall()
    finally:
        con.close()

    values = [
        str(item[0]).strip()
        for item in row
        if item[0] is not None
    ]

    if len(values) != 1:
        raise RuntimeError(
            "Estimated-rates artifact must contain exactly one "
            f"primary estimation run; found {values!r}."
        )

    return values[0]

def build_primary_holdout_diagnostics(
    *,
    estimated_rates_path: Path,
    primary_run_name: str,
) -> pd.DataFrame:
    """
    Build row-level diagnostics for the primary model's temporal holdout.

    This is the proper input table for heterogeneity plots.
    """
    estimated_rates_path = Path(estimated_rates_path)

    predicted_col = "holdout_predicted_factor"
    predicted_log_col = "holdout_predicted_log_factor"

    q = f"""
    SELECT
        year,
        exporter_iso,
        importer_iso,
        target_cmd,
        hs4_product_group,
        factor_source,

        observed_cif_fob_factor,
        LOG(observed_cif_fob_factor)
            AS log_observed_cif_fob_factor,

        {predicted_col}
            AS predicted_cif_fob_factor,

        {predicted_log_col}
            AS predicted_log_cif_fob_factor,

        ABS(
            {predicted_col}
            - observed_cif_fob_factor
        )
        / NULLIF(
            observed_cif_fob_factor,
            0
        ) AS ape,

        ABS(
            {predicted_log_col}
            - LOG(observed_cif_fob_factor)
        ) AS absolute_log_error,

        estimation_evidence_value,
        import_direct_covered_primary_value,
        export_value_fob,
        import_value_cif,

        log_distance,
        EXP(log_distance) AS distance_km,

        contiguity,
        exporter_landlocked,
        importer_landlocked,

        pair_uses_weight_proxy,
        pair_uses_estimated_weight,
        import_weight_estimated,

        lagged_pair_factor_found,
        lagged_pair_factor_observations,
        years_since_latest_pair_factor,

        gatt_pair,
        wto_pair,
        eu_pair,
        fta_wto,

        primary_estimation_run

    FROM read_parquet('{estimated_rates_path}')
    WHERE is_temporal_holdout
      AND clean_estimation_eligible
      AND factor_source = 'direct'
      AND observed_cif_fob_factor IS NOT NULL
      AND observed_cif_fob_factor > 0
      AND {predicted_col} IS NOT NULL
      AND {predicted_log_col} IS NOT NULL
    """

    diagnostics = duckdb.query(q).df()

    if diagnostics.empty:
        raise RuntimeError(
            "Primary holdout diagnostics are empty. "
            "Check is_temporal_holdout, clean_estimation_eligible, "
            "factor_source, and primary model prediction columns."
        )

    diagnostics = add_diagnostic_bands(diagnostics)

    diagnostics["model_error_direction"] = np.where(
        diagnostics["predicted_cif_fob_factor"]
        > diagnostics["observed_cif_fob_factor"],
        "over_prediction",
        "under_prediction",
    )

    diagnostics["primary_run_name"] = primary_run_name

    return diagnostics


def add_diagnostic_bands(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Add reusable diagnostic bands for heterogeneity plots."""
    data = diagnostics.copy()

    data["observed_factor_band"] = pd.cut(
        data["observed_cif_fob_factor"],
        bins=[
            1.0,
            1.05,
            1.10,
            1.20,
            1.50,
            2.00,
            3.00,
        ],
        labels=[
            "1.00–1.05",
            "1.05–1.10",
            "1.10–1.20",
            "1.20–1.50",
            "1.50–2.00",
            "2.00–3.00",
        ],
        include_lowest=True,
    )

    data["distance_band"] = pd.cut(
        data["distance_km"],
        bins=[
            0,
            1_000,
            3_000,
            7_000,
            12_000,
            np.inf,
        ],
        labels=[
            "<1k km",
            "1k–3k km",
            "3k–7k km",
            "7k–12k km",
            ">12k km",
        ],
        include_lowest=True,
    )

    data["value_band"] = pd.cut(
        data["estimation_evidence_value"],
        bins=[
            0,
            10_000,
            100_000,
            1_000_000,
            10_000_000,
            np.inf,
        ],
        labels=[
            "<10k",
            "10k–100k",
            "100k–1M",
            "1M–10M",
            ">10M",
        ],
        include_lowest=True,
    )

    data["lagged_pair_history_band"] = np.where(
        data["lagged_pair_factor_found"].fillna(False),
        "Lagged pair history available",
        "No lagged pair history",
    )

    data["weight_quality_band"] = np.select(
        [
            data["pair_uses_weight_proxy"].fillna(False),
            data["pair_uses_estimated_weight"].fillna(False),
        ],
        [
            "Uses proxy weight",
            "Uses estimated weight",
        ],
        default="Reported/direct weight",
    )

    data["policy_integration_band"] = np.select(
        [
            data["eu_pair"].fillna(0).astype(int).eq(1),
            data["fta_wto"].fillna(0).astype(int).eq(1),
            data["wto_pair"].fillna(0).astype(int).eq(1),
            data["gatt_pair"].fillna(0).astype(int).eq(1),
        ],
        [
            "EU pair",
            "FTA pair",
            "WTO pair",
            "GATT pair",
        ],
        default="No listed integration",
    )

    return data


def write_primary_holdout_diagnostics(
    *,
    cfg: CifFobConfig,
    estimated_rates_path: Path,
) -> dict[str, Any]:
    """
    Build and write primary-model holdout diagnostics.
    """
    primary_run_name = _read_primary_run_name(
        estimated_rates_path=estimated_rates_path,
    )

    diagnostics = build_primary_holdout_diagnostics(
        estimated_rates_path=estimated_rates_path,
        primary_run_name=primary_run_name,
    )

    output_path = (
        estimated_rates_path.parent
        / cfg.product_path("diagnostics").name
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics.to_parquet(
        output_path,
        index=False,
    )

    return {
        "status": "completed",
        "diagnostics_path": output_path,
        "output_path": output_path,
        "primary_run_name": primary_run_name,
        "n_rows": len(diagnostics),
        "year_min": (
            int(diagnostics["year"].min())
            if "year" in diagnostics.columns
            and not diagnostics.empty
            else None
        ),
        "year_max": (
            int(diagnostics["year"].max())
            if "year" in diagnostics.columns
            and not diagnostics.empty
            else None
        ),
        "median_ape": (
            float(diagnostics["ape"].median())
            if "ape" in diagnostics.columns
            and not diagnostics.empty
            else None
        ),
        "mean_ape": (
            float(diagnostics["ape"].mean())
            if "ape" in diagnostics.columns
            and not diagnostics.empty
            else None
        ),
        "p90_ape": (
            float(diagnostics["ape"].quantile(0.90))
            if "ape" in diagnostics.columns
            and not diagnostics.empty
            else None
        ),
    }