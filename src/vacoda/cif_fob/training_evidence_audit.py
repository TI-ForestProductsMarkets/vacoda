"""
vacoda.cif_fob.training_evidence_audit

Pre-model evidence representativeness audit for the CIF/FOB stage.

This module evaluates whether the rows used for model development are
representative of the broader population that can receive CIF/FOB
predictions.

The central comparison is:

    prediction population
        rows with complete predictors and eligible for prediction

    selected direct-factor evidence
        rows with selected direct CIF/FOB observed factors

    clean direct training sample
        selected direct-factor rows that pass all estimation filters

The resulting audit table is intended to identify pre-model evidence
imbalance by geography, product group, value band, distance band, and year.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

def write_training_evidence_audit(
    *,
    input_path: Path,
    output_path: Path,
    threads: int = 16,
) -> dict[str, Any]:
    """
    Build and write the CIF/FOB training-evidence audit.

    Parameters
    ----------
    input_path:
        Path to the CIF/FOB training-features Parquet artifact.

    output_path:
        Path where the audit CSV should be written.

    threads:
        DuckDB thread count.

    Returns
    -------
    dict[str, Any]
        Standard workflow-style result dictionary.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    audit = build_training_evidence_audit(
        input_path=input_path,
        threads=threads,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        output_path,
        index=False,
    )

    return {
        "status": "completed",
        "training_evidence_audit_path": output_path,
        "output_path": output_path,
        "n_rows": len(audit),
        "n_dimensions": (
            int(audit["dimension"].nunique())
            if "dimension" in audit.columns
            and not audit.empty
            else 0
        ),
        "minimum_clean_representation_ratio": (
            float(audit["clean_representation_ratio"].min())
            if "clean_representation_ratio" in audit.columns
            and not audit.empty
            else None
        ),
        "maximum_clean_representation_ratio": (
            float(audit["clean_representation_ratio"].max())
            if "clean_representation_ratio" in audit.columns
            and not audit.empty
            else None
        ),
    }


def build_training_evidence_audit(
    *,
    input_path: Path,
    threads: int = 16,
) -> pd.DataFrame:
    """
    Build the CIF/FOB training-evidence representativeness audit.

    The audit compares three populations:

    1. prediction population
       Rows with complete predictors and therefore eligible to receive
       model predictions.

    2. selected direct-factor evidence
       Rows with selected direct CIF/FOB factor evidence.

    3. clean direct training sample
       Rows that pass the final clean-estimation filters and are direct
       observations.

    Returns one long table with one row per dimension × group.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training-features artifact not found: {input_path}"
        )

    base = _load_base_audit_frame(
        input_path=input_path,
        threads=threads,
    )

    if base.empty:
        raise RuntimeError(
            "Training-evidence audit input is empty after loading "
            f"{input_path}."
        )

    base = _add_audit_bands(base)

    dimensions = [
        "year",
        "exporter_continent",
        "importer_continent",
        "hs4_product_group",
        "distance_band",
        "value_band",
        "observed_factor_band",
        "lagged_pair_history_band",
        "weight_quality_band",
        "policy_integration_band",
    ]

    available_dimensions = [
        dimension
        for dimension in dimensions
        if dimension in base.columns
    ]

    audit_parts = [
        _summarize_dimension(
            base,
            dimension=dimension,
        )
        for dimension in available_dimensions
    ]

    audit = pd.concat(
        audit_parts,
        ignore_index=True,
    )

    audit = _add_representation_flags(audit)

    audit = audit.sort_values(
        [
            "dimension",
            "clean_representation_ratio",
            "prediction_rows",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    return audit


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------


def _load_base_audit_frame(
    *,
    input_path: Path,
    threads: int,
) -> pd.DataFrame:
    """
    Load the columns needed for training-evidence representativeness.

    This intentionally works from training_features, not estimated_rates,
    because the audit is a pre-estimation diagnostic.
    """
    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        query = """
        SELECT
            year,
            exporter_iso,
            importer_iso,
            target_cmd,
            hs4_product_group,
            product_group,
            target_cmd_text,

            exporter_continent,
            importer_continent,

            observed_cif_fob_factor,
            factor_source,

            model_eligible,
            initial_predictors_complete,
            direct_eligible,
            training_eligible,

            export_value_fob,
            import_value_cif,
            import_direct_covered_primary_value,
            route_trade_value_usd,

            distance_model_km,
            log_distance,

            pair_uses_weight_proxy,
            pair_uses_estimated_weight,
            export_uses_weight_proxy,
            import_uses_weight_proxy,
            export_uses_estimated_weight,
            import_uses_estimated_weight,

            lagged_pair_factor_found,
            lagged_pair_factor_observations,
            lagged_pair_factor_years,
            years_since_latest_pair_factor,

            gatt_pair,
            wto_pair,
            eu_pair,
            fta_wto,

            predictor_completeness_reason,
            model_eligibility_reason

        FROM read_parquet(?)
        """

        data = con.execute(
            query,
            [str(input_path)],
        ).df()

    finally:
        con.close()

    # These boolean definitions are the core audit populations.
    data["prediction_population"] = (
        data["initial_predictors_complete"]
        .fillna(False)
        .astype(bool)
    )

    data["selected_direct_evidence"] = (
        data["model_eligible"]
        .fillna(False)
        .astype(bool)
        & data["factor_source"]
        .astype("string")
        .str.lower()
        .eq("direct")
    )

    # This mirrors the clean direct population as closely as possible from
    # training_features. If estimate_rates later applies additional filters,
    # this remains the pre-estimation view.
    data["clean_direct_training_sample"] = (
        data["selected_direct_evidence"]
        & data["training_eligible"]
        .fillna(False)
        .astype(bool)
        & data["initial_predictors_complete"]
        .fillna(False)
        .astype(bool)
    )

    return data


# ----------------------------------------------------------------------
# Band construction
# ----------------------------------------------------------------------


def _add_audit_bands(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add reusable ordered bands for representativeness auditing."""
    data = data.copy()

    data["year"] = data["year"].astype("Int64").astype("string")

    data["exporter_continent"] = (
        data["exporter_continent"]
        .astype("string")
        .fillna("Unknown")
    )

    data["importer_continent"] = (
        data["importer_continent"]
        .astype("string")
        .fillna("Unknown")
    )

    data["hs4_product_group"] = (
        data["hs4_product_group"]
        .astype("string")
        .fillna("Unknown")
    )

    evidence_value = _resolve_evidence_value(data)

    data["audit_evidence_value_usd"] = evidence_value

    data["value_band"] = pd.cut(
        evidence_value,
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

    data["value_band"] = (
        data["value_band"]
        .astype("string")
        .fillna("Missing value")
    )

    distance_km = _resolve_distance_km(data)

    data["audit_distance_km"] = distance_km

    data["distance_band"] = pd.cut(
        distance_km,
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

    data["distance_band"] = (
        data["distance_band"]
        .astype("string")
        .fillna("Missing distance")
    )

    data["observed_factor_band"] = pd.cut(
        pd.to_numeric(
            data["observed_cif_fob_factor"],
            errors="coerce",
        ),
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

    data["observed_factor_band"] = (
        data["observed_factor_band"]
        .astype("string")
        .fillna("No observed factor")
    )

    data["lagged_pair_history_band"] = np.where(
        data["lagged_pair_factor_found"]
        .fillna(False)
        .astype(bool),
        "Lagged pair history available",
        "No lagged pair history",
    )

    data["weight_quality_band"] = np.select(
        [
            data["pair_uses_weight_proxy"]
            .fillna(False)
            .astype(bool),
            data["pair_uses_estimated_weight"]
            .fillna(False)
            .astype(bool),
        ],
        [
            "Uses proxy weight",
            "Uses estimated weight",
        ],
        default="Reported/direct weight",
    )

    data["policy_integration_band"] = np.select(
        [
            _as_int_flag(data["eu_pair"]).eq(1),
            _as_int_flag(data["fta_wto"]).eq(1),
            _as_int_flag(data["wto_pair"]).eq(1),
            _as_int_flag(data["gatt_pair"]).eq(1),
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


def _resolve_evidence_value(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Resolve the value variable used for evidence-band auditing.

    Preference order:
    1. route_trade_value_usd
    2. import_direct_covered_primary_value
    3. max(export_value_fob, import_value_cif)
    """
    candidates = []

    for column in [
        "route_trade_value_usd",
        "import_direct_covered_primary_value",
        "export_value_fob",
        "import_value_cif",
    ]:
        if column in data.columns:
            candidates.append(
                pd.to_numeric(
                    data[column],
                    errors="coerce",
                )
            )

    if not candidates:
        return pd.Series(
            np.nan,
            index=data.index,
            dtype="float64",
        )

    stacked = pd.concat(
        candidates,
        axis=1,
    )

    return stacked.max(
        axis=1,
        skipna=True,
    )


def _resolve_distance_km(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Resolve route distance in kilometres.

    Preference order:
    1. distance_model_km
    2. exp(log_distance)
    """
    if "distance_model_km" in data.columns:
        distance = pd.to_numeric(
            data["distance_model_km"],
            errors="coerce",
        )

        if distance.notna().any():
            return distance

    if "log_distance" in data.columns:
        log_distance = pd.to_numeric(
            data["log_distance"],
            errors="coerce",
        )
        return np.exp(log_distance)

    return pd.Series(
        np.nan,
        index=data.index,
        dtype="float64",
    )


def _as_int_flag(
    values: pd.Series,
) -> pd.Series:
    """Safely convert a nullable flag column to integer-like values."""
    return (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )


# ----------------------------------------------------------------------
# Dimension summaries
# ----------------------------------------------------------------------


def _summarize_dimension(
    data: pd.DataFrame,
    *,
    dimension: str,
) -> pd.DataFrame:
    """
    Summarize prediction and training evidence shares for one dimension.
    """
    work = data[
        [
            dimension,
            "prediction_population",
            "selected_direct_evidence",
            "clean_direct_training_sample",
        ]
    ].copy()

    work[dimension] = (
        work[dimension]
        .astype("string")
        .fillna("Unknown")
    )

    grouped = (
        work
        .groupby(
            dimension,
            dropna=False,
            observed=True,
        )
        .agg(
            prediction_rows=(
                "prediction_population",
                "sum",
            ),
            direct_evidence_rows=(
                "selected_direct_evidence",
                "sum",
            ),
            clean_training_rows=(
                "clean_direct_training_sample",
                "sum",
            ),
            total_rows=(
                "prediction_population",
                "size",
            ),
        )
        .reset_index()
        .rename(
            columns={
                dimension: "group",
            }
        )
    )

    totals = {
        "prediction_rows": float(
            grouped["prediction_rows"].sum()
        ),
        "direct_evidence_rows": float(
            grouped["direct_evidence_rows"].sum()
        ),
        "clean_training_rows": float(
            grouped["clean_training_rows"].sum()
        ),
        "total_rows": float(
            grouped["total_rows"].sum()
        ),
    }

    grouped["dimension"] = dimension

    grouped["prediction_share"] = _safe_share(
        grouped["prediction_rows"],
        totals["prediction_rows"],
    )

    grouped["direct_evidence_share"] = _safe_share(
        grouped["direct_evidence_rows"],
        totals["direct_evidence_rows"],
    )

    grouped["clean_training_share"] = _safe_share(
        grouped["clean_training_rows"],
        totals["clean_training_rows"],
    )

    grouped["row_share"] = _safe_share(
        grouped["total_rows"],
        totals["total_rows"],
    )

    grouped["direct_representation_ratio"] = _safe_ratio(
        grouped["direct_evidence_share"],
        grouped["prediction_share"],
    )

    grouped["clean_representation_ratio"] = _safe_ratio(
        grouped["clean_training_share"],
        grouped["prediction_share"],
    )

    grouped["direct_coverage_within_group"] = _safe_ratio(
        grouped["direct_evidence_rows"],
        grouped["prediction_rows"],
    )

    grouped["clean_training_coverage_within_group"] = _safe_ratio(
        grouped["clean_training_rows"],
        grouped["prediction_rows"],
    )

    columns = [
        "dimension",
        "group",
        "total_rows",
        "prediction_rows",
        "direct_evidence_rows",
        "clean_training_rows",
        "row_share",
        "prediction_share",
        "direct_evidence_share",
        "clean_training_share",
        "direct_representation_ratio",
        "clean_representation_ratio",
        "direct_coverage_within_group",
        "clean_training_coverage_within_group",
    ]

    return grouped[columns]


def _safe_share(
    numerator: pd.Series,
    denominator: float,
) -> pd.Series:
    """Return numerator / denominator, guarding against zero totals."""
    if denominator <= 0:
        return pd.Series(
            0.0,
            index=numerator.index,
        )

    return numerator.astype(float) / denominator


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Return numerator / denominator with stable zero handling."""
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)

    ratio = numerator / denominator.replace(
        {
            0.0: np.nan,
        }
    )

    return ratio.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


def _add_representation_flags(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Add coarse interpretive labels for representation ratios."""
    audit = audit.copy()

    audit["clean_representation_flag"] = np.select(
        [
            audit["clean_representation_ratio"].isna(),
            audit["clean_representation_ratio"].lt(0.5),
            audit["clean_representation_ratio"].between(
                0.5,
                0.8,
                inclusive="left",
            ),
            audit["clean_representation_ratio"].between(
                0.8,
                1.25,
                inclusive="both",
            ),
            audit["clean_representation_ratio"].gt(1.25),
        ],
        [
            "No prediction share",
            "Strongly underrepresented",
            "Moderately underrepresented",
            "Approximately represented",
            "Overrepresented",
        ],
        default="Unclassified",
    )

    audit["direct_representation_flag"] = np.select(
        [
            audit["direct_representation_ratio"].isna(),
            audit["direct_representation_ratio"].lt(0.5),
            audit["direct_representation_ratio"].between(
                0.5,
                0.8,
                inclusive="left",
            ),
            audit["direct_representation_ratio"].between(
                0.8,
                1.25,
                inclusive="both",
            ),
            audit["direct_representation_ratio"].gt(1.25),
        ],
        [
            "No prediction share",
            "Strongly underrepresented",
            "Moderately underrepresented",
            "Approximately represented",
            "Overrepresented",
        ],
        default="Unclassified",
    )

    return audit