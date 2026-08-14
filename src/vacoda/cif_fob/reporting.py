"""
vacoda.cif_fob.reporting

Write the human-readable CIF/FOB decision report.

The normal CIF/FOB workflow owns this report. The separate upper-tail audit
script remains separate and is not invoked or read here.

The report summarizes decision-relevant evidence from the normal CIF/FOB
workflow, including the applied correction and readiness for trade sync:
- scope and configured references;
- explicit direct/mirror factor selection and training-sample funnel;
- predictor configuration and feature coverage;
- candidate model configuration;
- expanding-window temporal validation;
- upper-tail validation metrics already produced by estimate_rates.py;
- primary-model decision;
- final estimated-rate coverage;
- corrected-trade audit, coverage, value impact, and uncertainty;
- limitations and downstream readiness.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from vacoda.config.cif_fob import CifFobConfig
from vacoda.config.paths import get_figure_dir
from vacoda.visualization.distributions import (
    plot_nebula_boxplot,
    plot_model_metric_ranking,
    plot_observed_vs_predicted_p95,
    plot_grouped_error_bars,
    plot_model_error_by_group,
    plot_cif_fob_corrected_agreement_bands
)
from vacoda.visualization.relationships import (
    plot_central_vs_tail_tradeoff,
    plot_observed_vs_predicted_density,
    plot_observed_vs_predicted_with_tolerance_bands,
    plot_temporal_validation_metric,
    plot_corrected_trade_comparison,
)
from vacoda.visualization.audits import plot_representation_ratio_bars
from vacoda.cif_fob.display import (
    DISTANCE_BAND_ORDER,
    VALUE_BAND_ORDER,
    PRODUCT_MEDIAN_THRESHOLD_BINS,

)
from vacoda.utils.markdown import markdown_image

# ======================================================================
# Formatting helpers
# ======================================================================


def _fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(value):,}"


def _fmt_float(
    value: Any,
    digits: int = 4,
) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_fraction_percent(
    value: Any,
    digits: int = 2,
) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def _fmt_percent_value(
    value: Any,
    digits: int = 2,
) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}%"


def _markdown_table(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    formatters: dict[str, Any] | None = None,
) -> str:
    available = [
        column
        for column in columns
        if column in frame.columns
    ]

    if not available or frame.empty:
        return "_No rows available._"

    table = frame[available].copy()
    formatters = formatters or {}

    headers = [str(column) for column in available]
    rendered_rows: list[list[str]] = []

    for _, row in table.iterrows():
        rendered: list[str] = []

        for column in available:
            value = row[column]
            formatter = formatters.get(column)

            if formatter is not None:
                text = formatter(value)
            elif value is None or pd.isna(value):
                text = ""
            else:
                text = str(value)

            rendered.append(
                text.replace("|", r"\|").replace("\n", " ")
            )

        rendered_rows.append(rendered)

    widths = [
        max(
            len(headers[index]),
            *(
                len(row[index])
                for row in rendered_rows
            ),
        )
        for index in range(len(headers))
    ]

    header = "| " + " | ".join(
        headers[index].ljust(widths[index])
        for index in range(len(headers))
    ) + " |"

    separator = "| " + " | ".join(
        "-" * widths[index]
        for index in range(len(headers))
    ) + " |"

    body = [
        "| " + " | ".join(
            row[index].ljust(widths[index])
            for index in range(len(headers))
        ) + " |"
        for row in rendered_rows
    ]

    return "\n".join([header, separator, *body])


def _metric(
    result: dict[str, Any],
    key: str,
) -> Any:
    return result.get(key)



def _quote_identifier(value: str) -> str:
    """Return a safely quoted DuckDB identifier."""
    return '"' + value.replace('"', '""') + '"'

def _assign_evidence_value_band(
    values: pd.Series,
    *,
    boundaries: tuple[int, ...],
) -> pd.Series:
    """Assign evidence-value bands from the weighting configuration."""

    if len(boundaries) != 4:
        raise ValueError(
            "The current report expects four evidence-value boundaries."
        )

    numeric = pd.to_numeric(values, errors="coerce")

    return pd.cut(
        numeric,
        bins=[-np.inf, *boundaries, np.inf],
        labels=VALUE_BAND_ORDER,
        right=False,
        ordered=True,
    )

def _inspect_value_band_model_comparison(
    *,
    path: Path,
    threads: int,
    holdout_year: int,
    boundaries: tuple[int, ...],
    prediction_columns_by_run: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare all completed estimation runs within evidence-value bands
    for the latest temporal holdout.

    Parameters
    ----------
    prediction_columns_by_run
        Mapping returned by estimate_rates.py. Each key is a concrete
        estimation run and each value contains its prediction-column names.

    Returns
    -------
    summary
        One row per estimation run and evidence-value band.

    long_frame
        Row-level model errors for plotting.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Estimated-rate artifact not found: {path}"
        )

    if not prediction_columns_by_run:
        raise RuntimeError(
            "No estimation-run prediction-column mapping was supplied "
            "for model comparison."
        )

    prediction_columns: dict[str, str] = {}

    for run_name, columns in prediction_columns_by_run.items():
        holdout_column = columns.get(
            "holdout_predicted_factor"
        )

        if holdout_column:
            prediction_columns[str(run_name)] = str(
                holdout_column
            )

    if not prediction_columns:
        raise RuntimeError(
            "The prediction-column mapping contains no "
            "'holdout_predicted_factor' columns."
        )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        available_columns = {
            str(row[0])
            for row in con.execute(
                """
                DESCRIBE
                SELECT *
                FROM read_parquet(?)
                """,
                [str(path)],
            ).fetchall()
        }

        required_base_columns = {
            "year",
            "clean_estimation_eligible",
            "estimation_evidence_value",
            "observed_cif_fob_factor",
        }

        missing_base_columns = sorted(
            required_base_columns - available_columns
        )

        if missing_base_columns:
            raise RuntimeError(
                "Estimated-rate artifact is missing required "
                "model-comparison columns: "
                + ", ".join(missing_base_columns)
            )

        missing_prediction_columns = {
            run_name: column
            for run_name, column in prediction_columns.items()
            if column not in available_columns
        }

        if missing_prediction_columns:
            details = ", ".join(
                f"{run_name}: {column}"
                for run_name, column
                in missing_prediction_columns.items()
            )

            raise RuntimeError(
                "Estimated-rate artifact is missing prediction "
                "columns for completed estimation runs: "
                + details
            )

        prediction_select_sql = ",\n".join(
            (
                f"{_quote_identifier(column)} "
                f"AS {_quote_identifier(run_name)}"
            )
            for run_name, column in prediction_columns.items()
        )

        comparison = con.execute(
            f"""
            SELECT
                year,
                estimation_evidence_value,
                observed_cif_fob_factor,
                {prediction_select_sql}

            FROM read_parquet(?)

            WHERE year = ?
              AND clean_estimation_eligible
              AND observed_cif_fob_factor IS NOT NULL
              AND observed_cif_fob_factor > 0
            """,
            [
                str(path),
                int(holdout_year),
            ],
        ).df()

    finally:
        con.close()

    if comparison.empty:
        raise RuntimeError(
            "No latest-holdout observations are available for "
            "model comparison."
        )

    comparison["value_band"] = (
        _assign_evidence_value_band(
            comparison["estimation_evidence_value"],
            boundaries=boundaries,
        )
    )

    long_frame = comparison.melt(
        id_vars=[
            "year",
            "estimation_evidence_value",
            "observed_cif_fob_factor",
            "value_band",
        ],
        value_vars=list(prediction_columns),
        var_name="model_name",
        value_name="predicted_factor",
    )

    long_frame["observed_cif_fob_factor"] = (
        pd.to_numeric(
            long_frame["observed_cif_fob_factor"],
            errors="coerce",
        )
    )

    long_frame["predicted_factor"] = pd.to_numeric(
        long_frame["predicted_factor"],
        errors="coerce",
    )

    long_frame["ape"] = (
        (
            long_frame["predicted_factor"]
            - long_frame["observed_cif_fob_factor"]
        ).abs()
        / long_frame["observed_cif_fob_factor"]
    )

    long_frame = long_frame.loc[
        long_frame["value_band"].notna()
        & long_frame["observed_cif_fob_factor"].gt(0)
        & np.isfinite(long_frame["predicted_factor"])
        & np.isfinite(long_frame["ape"])
    ].copy()

    if long_frame.empty:
        raise RuntimeError(
            "No finite latest-holdout predictions are available "
            "for model comparison."
        )

    summary = (
        long_frame
        .groupby(
            [
                "model_name",
                "value_band",
            ],
            observed=False,
            sort=False,
        )
        .agg(
            observations=("ape", "size"),
            median_ape=("ape", "median"),
            mean_ape=("ape", "mean"),
            p90_ape=(
                "ape",
                lambda values: values.quantile(0.90),
            ),
        )
        .reset_index()
    )

    return summary, long_frame

def _inspect_population_journey(
    *,
    cfg: CifFobConfig,
) -> dict[str, Any]:
    """
    Inspect the end-to-end source population before factor selection.

    The early stages contain two different kinds of reduction:

    1. filtering, where source rows are excluded by scope or policy;
    2. grain changes, where multiple source records are consolidated into
       one canonical trade group or one bilateral mirror group.

    The report distinguishes these transitions so aggregation is not
    described as lost trade information.
    """
    input_path = cfg.input_path()
    prepared_path = cfg.product_path("prepared_input")
    mirror_path = cfg.product_path("mirror_pairs")

    for label, path in [
        ("HS-synced CIF/FOB input", input_path),
        ("prepared CIF/FOB input", prepared_path),
        ("mirror-pair artifact", mirror_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    columns = cfg.source_data.columns
    flows = cfg.source_data.supported_flows

    year_col = _quote_identifier(columns.year)
    flow_col = _quote_identifier(columns.flow)
    is_reported_col = _quote_identifier(
        columns.is_reported
    )
    is_aggregate_col = _quote_identifier(
        columns.is_aggregate
    )

    min_year = int(cfg.scope.min_year)
    max_year = int(cfg.scope.max_year)

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        source_row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_source_input_rows,

                COUNT(*) FILTER (
                    WHERE {year_col}
                          BETWEEN {min_year}
                              AND {max_year}
                ) AS n_rows_in_year_scope,

                COUNT(*) FILTER (
                    WHERE {year_col}
                          BETWEEN {min_year}
                              AND {max_year}
                      AND {flow_col} IN (?, ?)
                ) AS n_rows_in_supported_flows,

                COUNT(*) FILTER (
                    WHERE {year_col}
                          BETWEEN {min_year}
                              AND {max_year}
                      AND {flow_col} IN (?, ?)
                      AND COALESCE(
                          CAST(
                              {is_reported_col}
                              AS BOOLEAN
                          ),
                          FALSE
                      )
                ) AS n_reported_rows_in_supported_scope,

                COUNT(*) FILTER (
                    WHERE {year_col}
                          BETWEEN {min_year}
                              AND {max_year}
                      AND {flow_col} IN (?, ?)
                      AND COALESCE(
                          CAST(
                              {is_reported_col}
                              AS BOOLEAN
                          ),
                          FALSE
                      )
                      AND NOT COALESCE(
                          CAST(
                              {is_aggregate_col}
                              AS BOOLEAN
                          ),
                          FALSE
                      )
                ) AS n_selected_reported_nonaggregate_rows

            FROM read_parquet(?)
            """,
            [
                flows.import_code,
                flows.export_code,
                flows.import_code,
                flows.export_code,
                flows.import_code,
                flows.export_code,
                str(input_path),
            ],
        ).fetchone()

        prepared_row = con.execute(
            """
            SELECT
                COUNT(*) AS n_canonical_trade_groups,
                COALESCE(
                    SUM(source_row_count),
                    0
                ) AS n_represented_source_rows,
                COUNT(*) FILTER (
                    WHERE source_row_count = 1
                ) AS n_single_component_groups,
                COUNT(*) FILTER (
                    WHERE source_row_count > 1
                ) AS n_multi_component_groups

            FROM read_parquet(?)
            """,
            [str(prepared_path)],
        ).fetchone()

        mirror_row = con.execute(
            """
            SELECT
                COUNT(*) AS n_bilateral_mirror_groups,

                COUNT(*) FILTER (
                    WHERE has_export_report
                      AND has_import_report
                ) AS n_groups_with_both_reports,

                COUNT(*) FILTER (
                    WHERE has_export_report
                      AND NOT has_import_report
                ) AS n_export_only_groups,

                COUNT(*) FILTER (
                    WHERE has_import_report
                      AND NOT has_export_report
                ) AS n_import_only_groups

            FROM read_parquet(?)
            """,
            [str(mirror_path)],
        ).fetchone()

    finally:
        con.close()

    if source_row is None:
        raise RuntimeError(
            "Could not inspect the HS-synced CIF/FOB input."
        )

    if prepared_row is None:
        raise RuntimeError(
            "Could not inspect the prepared CIF/FOB artifact."
        )

    if mirror_row is None:
        raise RuntimeError(
            "Could not inspect the mirror-pair artifact."
        )

    names = [
        "n_source_input_rows",
        "n_rows_in_year_scope",
        "n_rows_in_supported_flows",
        "n_reported_rows_in_supported_scope",
        "n_selected_reported_nonaggregate_rows",
        "n_canonical_trade_groups",
        "n_represented_source_rows",
        "n_single_component_groups",
        "n_multi_component_groups",
        "n_bilateral_mirror_groups",
        "n_groups_with_both_reports",
        "n_export_only_groups",
        "n_import_only_groups",
    ]

    values = [
        *source_row,
        *prepared_row,
        *mirror_row,
    ]

    return dict(zip(names, values, strict=True))


def _build_population_journey_table(
    metrics: dict[str, Any],
) -> pd.DataFrame:
    """
    Build the source-to-bilateral workflow population table.

    Percentage reduction is calculated against the immediately preceding
    stage. For aggregation stages it describes row-count consolidation,
    not loss of represented trade information.
    """
    rows = [
        {
            "Workflow stage": "HS-synced input",
            "Rows/groups": metrics[
                "n_source_input_rows"
            ],
            "Transition type": "Starting population",
            "Interpretation": (
                "All HS-synced source rows entering the CIF/FOB stage."
            ),
        },
        {
            "Workflow stage": "Scope filter",
            "Rows/groups": metrics[
                "n_rows_in_year_scope"
            ],
            "Transition type": "Scope filter",
            "Interpretation": (
                "Rows retained within the configured minimum and "
                "maximum years."
            ),
        },
        {
            "Workflow stage": "Flow filter",
            "Rows/groups": metrics[
                "n_rows_in_supported_flows"
            ],
            "Transition type": "Flow filter",
            "Interpretation": (
                "Only canonical import and export flows are retained."
            ),
        },
        {
            "Workflow stage": "Reporting status filter",
            "Rows/groups": metrics[
                "n_reported_rows_in_supported_scope"
            ],
            "Transition type": "Reporting-status filter",
            "Interpretation": (
                "Only explicitly reported observations are retained."
            ),
        },
        {
            "Workflow stage": "Aggregate row filter",
            "Rows/groups": metrics[
                "n_selected_reported_nonaggregate_rows"
            ],
            "Transition type": "Aggregate-row filter",
            "Interpretation": (
                "Aggregate observations are excluded before canonical "
                "group construction."
            ),
        },
        {
            "Workflow stage": "Aggregation/grain change",
            "Rows/groups": metrics[
                "n_canonical_trade_groups"
            ],
            "Transition type": "Aggregation / grain change",
            "Interpretation": (
                "Source components are consolidated to one "
                "`year–reporter–partner–product–flow` record."
            ),
        },
        {
            "Workflow stage": "Pairing/grain change",
            "Rows/groups": metrics[
                "n_bilateral_mirror_groups"
            ],
            "Transition type": "Pairing / grain change",
            "Interpretation": (
                "Import and export perspectives are aligned to one "
                "`exporter–importer–product–year` group."
            ),
        },
    ]

    table = pd.DataFrame(rows)
    table["Rows/groups"] = pd.to_numeric(
        table["Rows/groups"],
        errors="coerce",
    ).fillna(0).astype(int)

    previous = table["Rows/groups"].shift(1)

    table["Rows/groups reduced"] = (
        previous - table["Rows/groups"]
    )
    table.loc[
        table.index[0],
        "Rows/groups reduced",
    ] = 0

    table["Reduction from previous stage"] = np.where(
        previous.gt(0),
        table["Rows/groups reduced"] / previous,
        0.0,
    )

    return table

# ======================================================================
# Artifact inspection
# ======================================================================

def _load_corrected_trade_factor_distribution_data(
    *,
    path: Path,
    threads: int,
) -> pd.DataFrame:
    """
    Load positive applied CIF/FOB factors from corrected trade
    for distribution diagnostics.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Corrected-trade artifact not found: {path}"
        )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        df = con.execute(
            """
            SELECT
                cif_fob_factor_applied

            FROM read_parquet(?)

            WHERE cif_fob_correction_applied
              AND cif_fob_factor_applied IS NOT NULL
              AND cif_fob_factor_applied > 0
            """,
            [str(path)],
        ).df()

    finally:
        con.close()

    if df.empty:
        raise RuntimeError(
            "No positive applied CIF/FOB factors available "
            "for corrected-trade nebula plot."
        )

    return df

def _inspect_product_median_threshold_sensitivity(
    *,
    path: Path,
    threads: int,
    thresholds: tuple[int, ...] = (5, 10, 20, 30, 50),
) -> pd.DataFrame:
    """
    Show how the product-median fallback hierarchy would change under
    alternative minimum-observation thresholds.

    This does not rebuild the feature artifact. It applies alternative
    thresholds to the candidate medians and observation counts already
    retained in training_features.parquet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training-feature artifact not found: {path}"
        )

    normalized_thresholds = tuple(
        sorted(
            {
                int(threshold)
                for threshold in thresholds
                if int(threshold) > 0
            }
        )
    )

    if not normalized_thresholds:
        raise ValueError(
            "At least one positive threshold must be supplied."
        )

    values_sql = ", ".join(
        f"({threshold})"
        for threshold in normalized_thresholds
    )

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        result = con.execute(
            f"""
            WITH thresholds(minimum_observations) AS (
                VALUES {values_sql}
            ),

            candidates AS (
                SELECT
                    lagged_product_median_unit_value,
                    lagged_product_observations,

                    product_group_year_median_unit_value,
                    product_group_year_observations,

                    global_year_median_unit_value,
                    global_year_observations

                FROM read_parquet(?)
            ),

            resolved AS (
                SELECT
                    t.minimum_observations,

                    CASE
                        WHEN
                            c.lagged_product_observations
                                >= t.minimum_observations
                            AND c.lagged_product_median_unit_value
                                IS NOT NULL
                            THEN 'lagged_product'

                        WHEN
                            c.product_group_year_observations
                                >= t.minimum_observations
                            AND c.product_group_year_median_unit_value
                                IS NOT NULL
                            THEN 'product_group_year'

                        WHEN
                            c.global_year_observations
                                >= t.minimum_observations
                            AND c.global_year_median_unit_value
                                IS NOT NULL
                            THEN 'global_year'

                        ELSE 'missing'
                    END AS source

                FROM candidates AS c
                CROSS JOIN thresholds AS t
            ),

            summarized AS (
                SELECT
                    minimum_observations,
                    source,
                    COUNT(*) AS rows

                FROM resolved

                GROUP BY
                    minimum_observations,
                    source
            ),

            totals AS (
                SELECT
                    minimum_observations,
                    SUM(rows) AS total_rows

                FROM summarized

                GROUP BY minimum_observations
            )

            SELECT
                s.minimum_observations,
                s.source,
                s.rows,
                100.0 * s.rows / t.total_rows AS share_percent

            FROM summarized AS s

            INNER JOIN totals AS t
                USING (minimum_observations)

            ORDER BY
                s.minimum_observations,
                s.source
            """,
            [str(path)],
        ).df()

    finally:
        con.close()

    source_labels = {
        "lagged_product": "Same HS6, earlier years",
        "product_group_year": "Same year, HS4 fallback",
        "global_year": "Same year, global fallback",
        "missing": "No median available",
    }

    result["source_label"] = (
        result["source"]
        .map(source_labels)
        .fillna(result["source"])
    )

    shares = (
        result
        .pivot(
            index="minimum_observations",
            columns="source_label",
            values="share_percent",
        )
        .fillna(0.0)
        .reset_index()
        .rename(
            columns={
                "minimum_observations": "Minimum observations",
                "Same HS6, earlier years": "HS6 share",
                "Same year, HS4 fallback": "HS4 fallback share",
                "Same year, global fallback": "Global fallback share",
                "No median available": "Missing share",
            }
        )
    )

    expected_columns = [
        "Minimum observations",
        "HS6 share",
        "HS4 fallback share",
        "Global fallback share",
        "Missing share",
    ]

    for column in expected_columns:
        if column not in shares.columns:
            shares[column] = 0.0

    return (
        shares[expected_columns]
        .sort_values("Minimum observations")
        .reset_index(drop=True)
    )

def _inspect_training_features(
    *,
    path: Path,
    threads: int,
) -> dict[str, Any]:
    """
    Inspect predictor coverage and the selected observed-factor pool.

    ``model_eligible`` is an internal pipeline flag. In the report it is
    described as "rows with a selected observed-factor candidate", because
    it includes both selected direct factors and selected mirror factors.
    It does not mean that every such row is used to fit the model.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training-feature artifact not found: {path}"
        )

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        row = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,
                COUNT(DISTINCT year) AS n_years,
                MIN(year) AS minimum_year,
                MAX(year) AS maximum_year,

                COUNT(*) FILTER (
                    WHERE model_eligible
                ) AS n_factor_candidates,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND LOWER(CAST(factor_source AS VARCHAR)) = 'direct'
                ) AS n_selected_direct_factors,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND LOWER(CAST(factor_source AS VARCHAR)) = 'mirror'
                ) AS n_selected_mirror_factors,

                COUNT(*) FILTER (
                    WHERE pair_geography_found
                ) AS n_pair_geography_found,

                COUNT(*) FILTER (
                    WHERE exporter_geography_found
                ) AS n_exporter_geography_found,

                COUNT(*) FILTER (
                    WHERE importer_geography_found
                ) AS n_importer_geography_found,

                COUNT(*) FILTER (
                    WHERE oil_price_found
                ) AS n_oil_price_found,

                COUNT(*) FILTER (
                    WHERE product_median_unit_value IS NOT NULL
                ) AS n_product_feature_found,

                COUNT(*) FILTER (
                    WHERE initial_predictors_complete
                ) AS n_initial_predictors_complete,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND initial_predictors_complete
                ) AS n_factor_candidates_predictors_complete,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND NOT initial_predictors_complete
                ) AS n_factor_candidates_predictors_incomplete,

                COUNT(*) FILTER (
                    WHERE lagged_pair_median_factor IS NOT NULL
                ) AS n_lagged_pair_factor_found,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND lagged_pair_median_factor IS NOT NULL
                ) AS n_factor_candidates_lagged_pair_factor_found,

                COUNT(*) FILTER (
                    WHERE lagged_pair_product_median_factor IS NOT NULL
                ) AS n_lagged_pair_product_factor_found,

                COUNT(*) FILTER (
                    WHERE log_lagged_bilateral_trade_value IS NOT NULL
                ) AS n_lagged_bilateral_trade_value_found,

                COUNT(*) FILTER (
                    WHERE log_lagged_pair_product_trade_value IS NOT NULL
                ) AS n_lagged_pair_product_trade_value_found,

                COUNT(*) FILTER (
                    WHERE distance_model_source
                          = 'gravity_2021_distw_harmonic'
                ) AS n_gravity_harmonic_distance,

                COUNT(*) FILTER (
                    WHERE distance_model_source = 'weighted'
                ) AS n_legacy_weighted_distance,

                COUNT(*) FILTER (
                    WHERE distance_model_source = 'capital'
                ) AS n_legacy_capital_fallback

            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()

    finally:
        con.close()

    if row is None:
        raise RuntimeError(
            f"Could not inspect training-feature artifact: {path}"
        )

    names = [
        "n_rows",
        "n_years",
        "minimum_year",
        "maximum_year",
        "n_factor_candidates",
        "n_selected_direct_factors",
        "n_selected_mirror_factors",
        "n_pair_geography_found",
        "n_exporter_geography_found",
        "n_importer_geography_found",
        "n_oil_price_found",
        "n_product_feature_found",
        "n_initial_predictors_complete",
        "n_factor_candidates_predictors_complete",
        "n_factor_candidates_predictors_incomplete",
        "n_lagged_pair_factor_found",
        "n_factor_candidates_lagged_pair_factor_found",
        "n_lagged_pair_product_factor_found",
        "n_lagged_bilateral_trade_value_found",
        "n_lagged_pair_product_trade_value_found",
        "n_gravity_harmonic_distance",
        "n_legacy_weighted_distance",
        "n_legacy_capital_fallback",
    ]

    result = dict(zip(names, row, strict=True))
    n_rows = int(result["n_rows"] or 0)
    n_factor_candidates = int(
        result["n_factor_candidates"] or 0
    )

    def coverage(
        count_name: str,
        denominator: int,
    ) -> float:
        if denominator <= 0:
            return 0.0

        return (
            100.0
            * int(result[count_name] or 0)
            / denominator
        )

    result.update(
        {
            "factor_candidate_share_percent": coverage(
                "n_factor_candidates",
                n_rows,
            ),
            "selected_direct_share_percent": coverage(
                "n_selected_direct_factors",
                n_factor_candidates,
            ),
            "selected_mirror_share_percent": coverage(
                "n_selected_mirror_factors",
                n_factor_candidates,
            ),
            "pair_geography_coverage_percent": coverage(
                "n_pair_geography_found",
                n_rows,
            ),
            "exporter_geography_coverage_percent": coverage(
                "n_exporter_geography_found",
                n_rows,
            ),
            "importer_geography_coverage_percent": coverage(
                "n_importer_geography_found",
                n_rows,
            ),
            "oil_price_coverage_percent": coverage(
                "n_oil_price_found",
                n_rows,
            ),
            "product_feature_coverage_percent": coverage(
                "n_product_feature_found",
                n_rows,
            ),
            "initial_predictor_coverage_percent": coverage(
                "n_initial_predictors_complete",
                n_rows,
            ),
            "factor_candidate_predictor_coverage_percent": coverage(
                "n_factor_candidates_predictors_complete",
                n_factor_candidates,
            ),
            "lagged_pair_factor_coverage_percent": coverage(
                "n_lagged_pair_factor_found",
                n_rows,
            ),
            "factor_candidate_lagged_pair_factor_coverage_percent": coverage(
                "n_factor_candidates_lagged_pair_factor_found",
                n_factor_candidates,
            ),
            "lagged_pair_product_factor_coverage_percent": coverage(
                "n_lagged_pair_product_factor_found",
                n_rows,
            ),
        }
    )

    selected_total = (
        int(result["n_selected_direct_factors"] or 0)
        + int(result["n_selected_mirror_factors"] or 0)
    )

    if selected_total != n_factor_candidates:
        raise RuntimeError(
            "Selected direct and mirror factor counts do not sum to "
            "the factor-candidate count."
        )

    return result

def _inspect_product_median_unit_value_sources(
    *,
    path: Path,
    threads: int,
) -> pd.DataFrame:
    """Summarize product median unit-value fallback-source usage."""

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        sources = con.execute(
            """
            SELECT
                CAST(product_median_unit_value_source AS VARCHAR) AS source,
                COUNT(*) AS rows,
                100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS share_percent

            FROM read_parquet(?)

            WHERE product_median_unit_value_source IS NOT NULL

            GROUP BY 1
            ORDER BY rows DESC
            """,
            [str(path)],
        ).df()

    finally:
        con.close()

    labels = {
        "lagged_product": "Same HS6, earlier years",
        "product_group_year": "Same year, HS4 group",
        "global_year": "Same year, global fallback",
    }

    sources["Source"] = sources["source"].map(labels).fillna(sources["source"])
    sources["Rows"] = sources["rows"]
    sources["Share"] = sources["share_percent"]

    return sources[["Source", "Rows", "Share"]]

def _inspect_direct_predictor_failures(
    *,
    path: Path,
    threads: int,
) -> pd.DataFrame:
    """
    Summarize why selected direct-factor rows fail predictor completeness.

    ``predictor_completeness_reason`` is created by the training-feature
    step as an ordered, mutually exclusive diagnostic. It reports the first
    failed requirement in the feature builder's completeness hierarchy.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training-feature artifact not found: {path}"
        )

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        available_columns = {
            str(row[0])
            for row in con.execute(
                """
                DESCRIBE
                SELECT *
                FROM read_parquet(?)
                """,
                [str(path)],
            ).fetchall()
        }

        if "predictor_completeness_reason" not in available_columns:
            raise RuntimeError(
                "Training-feature artifact does not contain "
                "'predictor_completeness_reason'. Rebuild the training "
                "features with the current feature builder before writing "
                "the decision report."
            )

        failures = con.execute(
            """
            SELECT
                CAST(
                    predictor_completeness_reason AS VARCHAR
                ) AS predictor_failure_reason,
                COUNT(*) AS n_rows

            FROM read_parquet(?)

            WHERE model_eligible
              AND LOWER(CAST(factor_source AS VARCHAR)) = 'direct'
              AND NOT initial_predictors_complete

            GROUP BY 1
            ORDER BY n_rows DESC, predictor_failure_reason
            """,
            [str(path)],
        ).df()

    finally:
        con.close()

    if failures.empty:
        return pd.DataFrame(
            columns=[
                "Predictor requirement not met",
                "Direct rows excluded",
                "Share of excluded direct rows",
            ]
        )

    reason_labels = {
        "missing_pair_geography": "Pair geography unavailable",
        "missing_exporter_geography": (
            "Exporter country geography unavailable"
        ),
        "missing_importer_geography": (
            "Importer country geography unavailable"
        ),
        "missing_oil_price": "Annual oil price unavailable",
        "missing_product_feature": (
            "Product median unit-value feature unavailable"
        ),
        "missing_log_distance": "Log distance unavailable",
        "missing_log_distance_squared": (
            "Squared log-distance term unavailable"
        ),
        "missing_contiguity": "Contiguity indicator unavailable",
        "missing_exporter_landlocked": (
            "Exporter landlocked indicator unavailable"
        ),
        "missing_importer_landlocked": (
            "Importer landlocked indicator unavailable"
        ),
        "missing_oil_distance_interaction": (
            "Oil-price × distance interaction unavailable"
        ),
        "missing_hs4_product_group": (
            "HS4 product-group predictor unavailable"
        ),
        "complete": "Complete",
    }

    failures["n_rows"] = pd.to_numeric(
        failures["n_rows"],
        errors="coerce",
    ).fillna(0).astype(int)

    total_failures = int(failures["n_rows"].sum())

    failures["Predictor requirement not met"] = (
        failures["predictor_failure_reason"]
        .map(reason_labels)
        .fillna(failures["predictor_failure_reason"])
    )
    failures["Direct rows excluded"] = failures["n_rows"]
    failures["Share of excluded direct rows"] = (
        failures["n_rows"] / total_failures
        if total_failures
        else 0.0
    )

    return failures[
        [
            "Predictor requirement not met",
            "Direct rows excluded",
            "Share of excluded direct rows",
        ]
    ]

def _inspect_estimated_rates(
    *,
    path: Path,
    threads: int,) -> dict[str, Any]:
    """
    Inspect the final prediction artifact and every cumulative estimation gate.

    The cumulative counts make the reduction from the selected factor pool
    to the final clean training sample fully explicit.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Estimated-rate artifact not found: {path}"
        )

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        row = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE model_eligible
                ) AS n_factor_candidates,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND LOWER(CAST(factor_source AS VARCHAR)) = 'direct'
                ) AS n_selected_direct_factors,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND LOWER(CAST(factor_source AS VARCHAR)) = 'mirror'
                ) AS n_selected_mirror_factors,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND prediction_eligible
                ) AS n_all_sources_after_predictor_completeness,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND LOWER(CAST(factor_source AS VARCHAR)) = 'direct'
                      AND prediction_eligible
                ) AS n_direct_after_predictor_completeness,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND prediction_eligible
                      AND passes_factor_source_rule
                ) AS n_after_factor_source_rule,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND prediction_eligible
                      AND passes_factor_source_rule
                      AND passes_minimum_value
                ) AS n_after_minimum_value_rule,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND prediction_eligible
                      AND passes_factor_source_rule
                      AND passes_minimum_value
                      AND passes_factor_lower_bound
                ) AS n_after_lower_factor_bound,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND prediction_eligible
                      AND passes_factor_source_rule
                      AND passes_minimum_value
                      AND passes_factor_lower_bound
                      AND passes_factor_upper_bound
                ) AS n_after_upper_factor_bound,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND prediction_eligible
                      AND passes_factor_source_rule
                      AND passes_minimum_value
                      AND passes_factor_lower_bound
                      AND passes_factor_upper_bound
                      AND passes_proxy_rule
                ) AS n_after_proxy_rule,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND prediction_eligible
                      AND passes_factor_source_rule
                      AND passes_minimum_value
                      AND passes_factor_lower_bound
                      AND passes_factor_upper_bound
                      AND passes_proxy_rule
                      AND estimation_source_weight > 0
                      AND observed_cif_fob_factor IS NOT NULL
                ) AS n_after_positive_weight_and_observed_factor,

                COUNT(*) FILTER (
                    WHERE clean_estimation_eligible
                ) AS n_clean_estimation_observations,

                COUNT(*) FILTER (
                    WHERE clean_estimation_eligible
                      AND LOWER(CAST(factor_source AS VARCHAR)) = 'direct'
                ) AS n_clean_direct_observations,

                COUNT(*) FILTER (
                    WHERE clean_estimation_eligible
                      AND LOWER(CAST(factor_source AS VARCHAR)) = 'mirror'
                ) AS n_clean_mirror_observations,

                COUNT(*) FILTER (
                    WHERE prediction_eligible
                ) AS n_prediction_eligible,

                COUNT(*) FILTER (
                    WHERE estimated_cif_fob_factor IS NOT NULL
                ) AS n_with_estimated_factor,

                MEDIAN(estimated_cif_fob_factor)
                    FILTER (
                        WHERE estimated_cif_fob_factor IS NOT NULL
                    ) AS median_estimated_factor,

                QUANTILE_CONT(
                    estimated_cif_fob_factor,
                    0.05
                ) FILTER (
                    WHERE estimated_cif_fob_factor IS NOT NULL
                ) AS estimated_factor_p05,

                QUANTILE_CONT(
                    estimated_cif_fob_factor,
                    0.95
                ) FILTER (
                    WHERE estimated_cif_fob_factor IS NOT NULL
                ) AS estimated_factor_p95,

                MIN(estimated_cif_fob_factor)
                    FILTER (
                        WHERE estimated_cif_fob_factor IS NOT NULL
                    ) AS minimum_estimated_factor,

                MAX(estimated_cif_fob_factor)
                    FILTER (
                        WHERE estimated_cif_fob_factor IS NOT NULL
                    ) AS maximum_estimated_factor,

            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()

        training_factor_direction = con.execute(
            """
            SELECT
                CASE
                    WHEN observed_cif_fob_factor > 1.000001
                        THEN 'CIF above FOB'
                    WHEN observed_cif_fob_factor < 0.999999
                        THEN 'CIF below FOB'
                    ELSE 'Approximately equal'
                END AS "Direction",

                COUNT(*) AS "Rows",

                100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                    AS "Share",

                MEDIAN(observed_cif_fob_factor)
                    AS "Median factor",

                QUANTILE_CONT(
                    observed_cif_fob_factor,
                    0.10
                ) AS "Factor P10",

                QUANTILE_CONT(
                    observed_cif_fob_factor,
                    0.90
                ) AS "Factor P90"

            FROM read_parquet(?)

            WHERE clean_estimation_eligible
              AND LOWER(
                    CAST(factor_source AS VARCHAR)
                  ) = 'direct'

            GROUP BY "Direction"

            ORDER BY
                CASE "Direction"
                    WHEN 'CIF above FOB' THEN 1
                    WHEN 'Approximately equal' THEN 2
                    ELSE 3
                END
            """,
            [str(path)],
        ).df()

    finally:
        con.close()

    if row is None:
        raise RuntimeError(
            f"Could not inspect estimated-rate artifact: {path}"
        )

    names = [
        "n_rows",
        "n_factor_candidates",
        "n_selected_direct_factors",
        "n_selected_mirror_factors",
        "n_all_sources_after_predictor_completeness",
        "n_direct_after_predictor_completeness",
        "n_after_factor_source_rule",
        "n_after_minimum_value_rule",
        "n_after_lower_factor_bound",
        "n_after_upper_factor_bound",
        "n_after_proxy_rule",
        "n_after_positive_weight_and_observed_factor",
        "n_clean_estimation_observations",
        "n_clean_direct_observations",
        "n_clean_mirror_observations",
        "n_prediction_eligible",
        "n_with_estimated_factor",
        "median_estimated_factor",
        "estimated_factor_p05",
        "estimated_factor_p95",
        "minimum_estimated_factor",
        "maximum_estimated_factor",
    ]

    result = dict(zip(names, row, strict=True))
    n_rows = int(result["n_rows"] or 0)
    n_factor_candidates = int(
        result["n_factor_candidates"] or 0
    )

    if (
        int(result["n_direct_after_predictor_completeness"] or 0)
        != int(result["n_after_factor_source_rule"] or 0)
    ):
        raise RuntimeError(
            "Direct predictor-complete count does not match the "
            "configured factor-source-rule count. Check whether "
            "training_factor_sources is still configured as direct-only."
        )

    result.update(
        {
            "factor_candidate_share_percent": (
                100.0 * n_factor_candidates / n_rows
                if n_rows
                else 0.0
            ),
            "clean_estimation_share_of_candidates_percent": (
                100.0
                * int(
                    result["n_clean_estimation_observations"] or 0
                )
                / n_factor_candidates
                if n_factor_candidates
                else 0.0
            ),
            "estimated_factor_coverage_percent": (
                100.0
                * int(result["n_with_estimated_factor"] or 0)
                / n_rows
                if n_rows
                else 0.0
            ),
        }
    )

    result["training_factor_direction"] = (
            training_factor_direction
        )

    return result

def _inspect_correction_application_reasons(
    *,
    corrected_trade_path: Path,
    training_features_path: Path,
    threads: int,
) -> dict[str, pd.DataFrame]:
    """
    Explain which bilateral rows received CIF/FOB correction and why
    correction was unavailable for the remaining rows.
    """

    for label, path in [
        ("Corrected-trade artifact", corrected_trade_path),
        ("Training-feature artifact", training_features_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        application_row = con.execute(
            """
            SELECT
                COUNT(*) AS total_rows,

                COUNT(*) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS corrected_rows,

                COUNT(*) FILTER (
                    WHERE cif_fob_factor_applied_source = 'model_estimated'
                ) AS model_corrected_rows,

                COUNT(*) FILTER (
                    WHERE cif_fob_correction_applied
                    AND cif_fob_factor_applied_source <> 'model_estimated'
                ) AS non_model_corrected_rows,

                COUNT(*) FILTER (
                    WHERE NOT cif_fob_correction_applied
                    AND import_value_cif IS NULL
                ) AS no_importer_cif_rows,

                COUNT(*) FILTER (
                    WHERE NOT cif_fob_correction_applied
                    AND import_value_cif IS NOT NULL
                ) AS no_model_factor_rows

            FROM read_parquet(?)
            """,
            [str(corrected_trade_path)],
        ).fetchone()

        predictor_row = con.execute(
            """
            SELECT
                COUNT(*) AS total_rows,

                COUNT(*) FILTER (
                    WHERE pair_geography_found
                ) AS rows_with_pair_geography,

                COUNT(*) FILTER (
                    WHERE NOT pair_geography_found
                ) AS rows_without_pair_geography,

                COUNT(*) FILTER (
                    WHERE initial_predictors_complete
                ) AS rows_with_complete_initial_predictors,

                COUNT(*) FILTER (
                    WHERE NOT pair_geography_found
                      AND import_value_cif IS NOT NULL
                ) AS importer_rows_without_pair_geography

            FROM read_parquet(?)
            """,
            [str(training_features_path)],
        ).fetchone()

    finally:
        con.close()

    if application_row is None or predictor_row is None:
        raise RuntimeError(
            "Could not inspect CIF/FOB correction application reasons."
        )

    (
        total_rows,
        corrected_rows,
        model_corrected_rows,
        non_model_corrected_rows,
        no_importer_cif_rows,
        no_model_factor_rows,
    ) = application_row

    (
        predictor_total_rows,
        rows_with_pair_geography,
        rows_without_pair_geography,
        rows_with_complete_initial_predictors,
        importer_rows_without_pair_geography,
    ) = predictor_row


    if int(non_model_corrected_rows or 0) != 0:
        raise RuntimeError(
            "Corrected CIF/FOB trade contains non-model production "
            f"corrections: {int(non_model_corrected_rows):,} rows."
        )
    if int(total_rows) != int(predictor_total_rows):
        raise RuntimeError(
            "Corrected-trade and training-feature artifacts have "
            "different bilateral row counts."
        )
    
    application_outcomes = pd.DataFrame(
        [
            {
                "Application outcome": (
                    "Corrected using model-estimated factor"
                ),
                "Rows": model_corrected_rows,
                "Share": 100.0 * model_corrected_rows / total_rows,
                "Explanation": (
                    "Prediction-eligible importer rows receive the "
                    "primary model's estimated CIF/FOB factor. "
                    "Observed direct factors are retained as model evidence "
                    "and are not directly applied in production."
                ),
            },
            {
                "Application outcome": (
                    "Uncorrected rows: no importer CIF value"
                ),
                "Rows": no_importer_cif_rows,
                "Share": 100.0 * no_importer_cif_rows / total_rows,
                "Explanation": (
                    "Export-only bilateral group. There is no importer CIF "
                    "value to convert; the exporter value is already FOB."
                ),
            },
            {
                "Application outcome": (
                    "Uncorrected rows: no model-estimated factor"
                ),
                "Rows": no_model_factor_rows,
                "Share": 100.0 * no_model_factor_rows / total_rows,
                "Explanation": (
                    "An importer CIF value exists, but the row is not "
                    "prediction-eligible and therefore no model-estimated "
                    "CIF/FOB factor is available."
                ),
            },
        ]
    )

    rows_without_geography_no_import = (
        int(rows_without_pair_geography)
        - int(importer_rows_without_pair_geography)
    )

    predictor_diagnostic = pd.DataFrame(
        [
            {
                "Prediction coverage path": "Total bilateral rows",
                "Rows": total_rows,
                "Share": 100.0,
                "Interpretation": (
                    "Full exporter–importer–product–year population."
                ),
            },
            {
                "Prediction coverage path": (
                    "├─ Pair geography available"
                ),
                "Rows": rows_with_pair_geography,
                "Share": (
                    100.0 * rows_with_pair_geography / total_rows
                ),
                "Interpretation": (
                    "Model predictors are complete."
                ),
            },
            {
                "Prediction coverage path": (
                    "└─ Pair geography unavailable"
                ),
                "Rows": rows_without_pair_geography,
                "Share": (
                    100.0 * rows_without_pair_geography / total_rows
                ),
                "Interpretation": (
                    "No model prediction can be generated."
                ),
            },
            {
                "Prediction coverage path": (
                    "       ├─ Importer CIF exists"
                ),
                "Rows": importer_rows_without_pair_geography,
                "Share": (
                    100.0
                    * importer_rows_without_pair_geography
                    / total_rows
                ),
                "Interpretation": (
                    "Required model predictors are incomplete, so no model-estimated "
                    "CIF/FOB factor can be generated. Importer-containing rows in this "
                    "group therefore remain uncorrected."
                ),
            },
            {
                "Prediction coverage path": (
                    "       └─ No importer CIF exists"
                ),
                "Rows": rows_without_geography_no_import,
                "Share": (
                    100.0
                    * rows_without_geography_no_import
                    / total_rows
                ),
                "Interpretation": (
                    "No correction was needed."
                ),
            },
        ]
    )

    return {
        "application_outcomes": application_outcomes,
        "predictor_diagnostic": predictor_diagnostic,
    }

def _inspect_corrected_trade(
    *,
    path: Path,
    threads: int,
    value_band_boundaries: tuple[int, ...],
) -> dict[str, Any]:
    """
    Inspect the applied CIF/FOB correction for reporting.

    This function produces descriptive summaries only. Validation of the
    corrected-trade artifact remains the responsibility of apply_weights.py.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Corrected-trade artifact not found: {path}"
        )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        available_columns = {
            str(row[0])
            for row in con.execute(
                """
                DESCRIBE
                SELECT *
                FROM read_parquet(?)
                """,
                [str(path)],
            ).fetchall()
        }

        required_columns = {
            "year",
            "estimation_evidence_value",
            "cif_fob_correction_applied",
            "cif_fob_factor_applied",
            "import_value_cif",
            "import_value_fob_corrected",
            "cif_fob_uncertainty_calibration_rows",
            "cif_fob_factor_pi80_log_error_quantile",
            "cif_fob_factor_pi80_relative_width",
            "import_value_fob_corrected_pi80_lower",
            "import_value_fob_corrected_pi80_upper",
            "cif_fob_factor_pi90_log_error_quantile",
            "cif_fob_factor_pi90_relative_width",
            "import_value_fob_corrected_pi90_lower",
            "import_value_fob_corrected_pi90_upper",
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                "Corrected-trade artifact is missing reporting columns: "
                + ", ".join(missing_columns)
            )

        overall_row = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS n_corrected_rows,

                COUNT(*) FILTER (
                    WHERE NOT cif_fob_correction_applied
                ) AS n_uncorrected_rows,

                SUM(import_value_cif) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS original_import_cif_value,

                SUM(import_value_fob_corrected) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS corrected_import_fob_value,

                MEDIAN(cif_fob_factor_applied) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS median_factor_applied,

                QUANTILE_CONT(
                    cif_fob_factor_applied,
                    0.05
                ) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS factor_p05,

                QUANTILE_CONT(
                    cif_fob_factor_applied,
                    0.95
                ) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS factor_p95,

                MEDIAN(
                    1.0 - 1.0 / cif_fob_factor_applied
                ) FILTER (
                    WHERE cif_fob_correction_applied
                      AND cif_fob_factor_applied > 0
                ) AS median_correction_fraction

            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()

        by_year = con.execute(
            """
            SELECT
                year AS "Year",

                COUNT(*) AS "Rows",

                COUNT(*) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS "Corrected rows",

                100.0
                * COUNT(*) FILTER (
                    WHERE cif_fob_correction_applied
                )
                / NULLIF(COUNT(*), 0)
                    AS "Correction coverage",

                SUM(import_value_cif) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS "Original CIF value",

                SUM(import_value_fob_corrected) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS "Corrected FOB value",

                100.0 * (
                    1.0
                    - SUM(import_value_fob_corrected) FILTER (
                        WHERE cif_fob_correction_applied
                    )
                    / NULLIF(
                        SUM(import_value_cif) FILTER (
                            WHERE cif_fob_correction_applied
                        ),
                        0
                    )
                ) AS "Value reduction",

                MEDIAN(cif_fob_factor_applied) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS "Median factor",

                MEDIAN(
                    1.0 - 1.0 / cif_fob_factor_applied
                ) FILTER (
                    WHERE cif_fob_correction_applied
                      AND cif_fob_factor_applied > 0
                ) AS "Median correction"

            FROM read_parquet(?)

            GROUP BY year
            ORDER BY year
            """,
            [str(path)],
        ).df()

        value_band_data = con.execute(
            """
            SELECT
                estimation_evidence_value,
                cif_fob_correction_applied,
                import_value_cif,
                import_value_fob_corrected,
                cif_fob_factor_applied

            FROM read_parquet(?)
            """,
            [str(path)],
        ).df()

        uncertainty_rows: list[dict[str, Any]] = []

        for suffix, level in [
            ("80", 0.80),
            ("90", 0.90),
        ]:
            uncertainty_row = con.execute(
                f"""
                SELECT
                    MAX(
                        cif_fob_uncertainty_calibration_rows
                    ) AS calibration_rows,

                    MAX(
                        cif_fob_factor_pi{suffix}_log_error_quantile
                    ) AS log_error_quantile,

                    MEDIAN(
                        cif_fob_factor_pi{suffix}_relative_width
                    ) FILTER (
                        WHERE cif_fob_correction_applied
                    ) AS median_factor_relative_width,

                    MEDIAN(
                        (
                            import_value_fob_corrected_pi{suffix}_upper
                            - import_value_fob_corrected_pi{suffix}_lower
                        )
                        / NULLIF(
                            import_value_fob_corrected,
                            0
                        )
                    ) FILTER (
                        WHERE cif_fob_correction_applied
                          AND import_value_fob_corrected > 0
                    ) AS median_corrected_value_relative_width

                FROM read_parquet(?)
                """,
                [str(path)],
            ).fetchone()

            if uncertainty_row is None:
                raise RuntimeError(
                    "Could not inspect corrected-trade uncertainty "
                    f"summary for the {suffix}% interval."
                )

            uncertainty_rows.append(
                {
                    "Prediction interval": f"{level:.0%}",
                    "Calibration rows": uncertainty_row[0],
                    "Log-error quantile": uncertainty_row[1],
                    "Median factor interval width": uncertainty_row[2],
                    "Median corrected-value interval width": (
                        uncertainty_row[3]
                    ),
                }
            )

    finally:
        con.close()

    if overall_row is None:
        raise RuntimeError(
            f"Could not inspect corrected-trade artifact: {path}"
        )

    (
        n_rows,
        n_corrected_rows,
        n_uncorrected_rows,
        original_import_cif_value,
        corrected_import_fob_value,
        median_factor_applied,
        factor_p05,
        factor_p95,
        median_correction_fraction,
    ) = overall_row

    n_rows = int(n_rows or 0)
    n_corrected_rows = int(n_corrected_rows or 0)
    n_uncorrected_rows = int(n_uncorrected_rows or 0)

    correction_coverage_percent = (
        100.0 * n_corrected_rows / n_rows
        if n_rows > 0
        else 0.0
    )

    total_cif_fob_reduction = None
    total_cif_fob_reduction_percent = None

    if (
        original_import_cif_value is not None
        and corrected_import_fob_value is not None
    ):
        total_cif_fob_reduction = (
            original_import_cif_value
            - corrected_import_fob_value
        )

        if original_import_cif_value > 0:
            total_cif_fob_reduction_percent = (
                100.0
                * total_cif_fob_reduction
                / original_import_cif_value
            )

    value_band_data["Evidence-value band"] = (
        _assign_evidence_value_band(
            value_band_data["estimation_evidence_value"],
            boundaries=value_band_boundaries,
        )
    )

    value_band_data["cif_fob_correction_applied"] = (
        value_band_data["cif_fob_correction_applied"]
        .fillna(False)
        .astype(bool)
    )

    value_band_data["import_value_cif"] = pd.to_numeric(
        value_band_data["import_value_cif"],
        errors="coerce",
    )

    value_band_data["import_value_fob_corrected"] = pd.to_numeric(
        value_band_data["import_value_fob_corrected"],
        errors="coerce",
    )

    value_band_data["cif_fob_factor_applied"] = pd.to_numeric(
        value_band_data["cif_fob_factor_applied"],
        errors="coerce",
    )

    value_band_rows: list[dict[str, Any]] = []

    for value_band in VALUE_BAND_ORDER:
        group = value_band_data.loc[
            value_band_data["Evidence-value band"] == value_band
        ]

        corrected = group.loc[
            group["cif_fob_correction_applied"]
        ]

        group_rows = int(len(group))
        corrected_rows = int(len(corrected))

        original_value = corrected[
            "import_value_cif"
        ].sum(min_count=1)

        corrected_value = corrected[
            "import_value_fob_corrected"
        ].sum(min_count=1)

        value_reduction_percent = None

        if (
            pd.notna(original_value)
            and original_value > 0
            and pd.notna(corrected_value)
        ):
            value_reduction_percent = (
                100.0
                * (original_value - corrected_value)
                / original_value
            )

        valid_factors = corrected[
            "cif_fob_factor_applied"
        ].loc[
            corrected["cif_fob_factor_applied"].gt(0)
        ]

        median_factor = (
            valid_factors.median()
            if not valid_factors.empty
            else None
        )

        median_correction_fraction_band = (
            (
                1.0 - 1.0 / valid_factors
            ).median()
            if not valid_factors.empty
            else None
        )

        value_band_rows.append(
            {
                "Evidence-value band": value_band,
                "Rows": group_rows,
                "Corrected rows": corrected_rows,
                "Correction coverage": (
                    100.0 * corrected_rows / group_rows
                    if group_rows > 0
                    else 0.0
                ),
                "Original CIF value": original_value,
                "Corrected FOB value": corrected_value,
                "Value reduction": value_reduction_percent,
                "Median factor": median_factor,
                "Median correction": (
                    median_correction_fraction_band
                ),
            }
        )

    by_value_band = pd.DataFrame(
        value_band_rows
    )

    uncertainty = pd.DataFrame(
        uncertainty_rows
    )

    overall = {
        "n_rows": n_rows,
        "n_corrected_rows": n_corrected_rows,
        "n_uncorrected_rows": n_uncorrected_rows,
        "correction_coverage_percent": (
            correction_coverage_percent
        ),
        "original_import_cif_value": (
            original_import_cif_value
        ),
        "corrected_import_fob_value": (
            corrected_import_fob_value
        ),
        "total_cif_fob_reduction": (
            total_cif_fob_reduction
        ),
        "total_cif_fob_reduction_percent": (
            total_cif_fob_reduction_percent
        ),
        "median_factor_applied": median_factor_applied,
        "factor_p05": factor_p05,
        "factor_p95": factor_p95,
        "median_correction_fraction": (
            median_correction_fraction
        ),
    }

    return {
        "overall": overall,
        "by_year": by_year,
        "by_value_band": by_value_band,
        "uncertainty": uncertainty,
    }



def _inspect_trade_sync_readiness(
    *,
    path: Path,
    threads: int,
) -> dict[str, Any]:
    """
    Inspect the corrected CIF/FOB artifact as the input to Trade Sync.

    A usable report is positive, finite, and non-missing. The audit keeps
    structural presence separate from usable-value availability and measures
    the residual exporter/importer disagreement before and after CIF/FOB
    correction.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Corrected-trade artifact not found: {path}"
        )

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        available_columns = {
            str(row[0])
            for row in con.execute(
                """
                DESCRIBE
                SELECT *
                FROM read_parquet(?)
                """,
                [str(path)],
            ).fetchall()
        }

        required_columns = {
            "year",
            "export_value_fob",
            "import_value_cif",
            "import_value_fob_corrected",
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                "Corrected-trade artifact is missing Trade Sync "
                "readiness columns: "
                + ", ".join(missing_columns)
            )

        parquet_path_sql = str(path).replace("'", "''")

        # DuckDB does not allow prepared parameters inside CREATE VIEW.
        # Embed the safely escaped path literal for this DDL statement.
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW trade_sync_input AS
            SELECT
                *,
                export_value_fob IS NOT NULL
                    AND isfinite(export_value_fob)
                    AND export_value_fob > 0
                    AS has_export_fob,

                import_value_cif IS NOT NULL
                    AND isfinite(import_value_cif)
                    AND import_value_cif > 0
                    AS has_import_cif,

                import_value_fob_corrected IS NOT NULL
                    AND isfinite(import_value_fob_corrected)
                    AND import_value_fob_corrected > 0
                    AS has_import_fob_corrected

            FROM read_parquet('{parquet_path_sql}')
            """
        )

        availability = con.execute(
            """
            SELECT
                CASE
                    WHEN has_export_fob
                         AND has_import_fob_corrected
                        THEN 'Both exporter and corrected importer'
                    WHEN has_export_fob
                         AND NOT has_import_fob_corrected
                        THEN 'Exporter only'
                    WHEN NOT has_export_fob
                         AND has_import_fob_corrected
                        THEN 'Corrected importer only'
                    ELSE 'Neither usable'
                END AS "Trade Sync status",

                COUNT(*) AS "Rows",

                100.0 * COUNT(*)
                    / SUM(COUNT(*)) OVER ()
                    AS "Share"

            FROM trade_sync_input

            GROUP BY 1

            ORDER BY
                CASE "Trade Sync status"
                    WHEN 'Both exporter and corrected importer' THEN 1
                    WHEN 'Corrected importer only' THEN 2
                    WHEN 'Exporter only' THEN 3
                    ELSE 4
                END
            """
        ).df()

        
        neither_usable_breakdown = con.execute(
                """
                WITH classified AS (
                    SELECT
                        CASE
                            WHEN export_value_fob IS NULL
                                THEN 'Exporter FOB missing'
                            WHEN NOT isfinite(export_value_fob)
                                THEN 'Exporter FOB non-finite'
                            WHEN export_value_fob <= 0
                                THEN 'Exporter FOB zero or negative'
                            ELSE 'Exporter FOB usable'
                        END AS exporter_reason,

                        CASE
                            WHEN import_value_cif IS NULL
                                THEN 'Importer CIF missing'
                            WHEN NOT isfinite(import_value_cif)
                                THEN 'Importer CIF non-finite'
                            WHEN import_value_cif <= 0
                                THEN 'Importer CIF zero or negative'
                            WHEN estimated_cif_fob_factor IS NULL
                                THEN 'No estimated CIF/FOB factor'
                            WHEN NOT isfinite(estimated_cif_fob_factor)
                                THEN 'Estimated factor non-finite'
                            WHEN estimated_cif_fob_factor <= 0
                                THEN 'Estimated factor zero or negative'
                            WHEN import_value_fob_corrected IS NULL
                                THEN 'Corrected importer FOB missing'
                            WHEN NOT isfinite(import_value_fob_corrected)
                                THEN 'Corrected importer FOB non-finite'
                            WHEN import_value_fob_corrected <= 0
                                THEN 'Corrected importer FOB zero or negative'
                            ELSE 'Corrected importer FOB usable'
                        END AS importer_reason

                    FROM trade_sync_input
                )

                SELECT
                    exporter_reason || ' + ' || importer_reason
                        AS "Reason combination",

                    COUNT(*) AS "Rows",

                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS "Share"

                FROM classified

                WHERE exporter_reason <> 'Exporter FOB usable'
                AND importer_reason <> 'Corrected importer FOB usable'

                GROUP BY
                    exporter_reason,
                    importer_reason

                ORDER BY "Rows" DESC
                """
            ).df()


        coverage_row = con.execute(
            """
            SELECT
                COUNT(*) AS total_rows,

                COUNT(*) FILTER (
                    WHERE has_export_fob
                ) AS rows_with_export_fob,

                COUNT(*) FILTER (
                    WHERE has_import_cif
                ) AS rows_with_import_cif,

                COUNT(*) FILTER (
                    WHERE has_import_fob_corrected
                ) AS rows_with_corrected_import_fob,

                COUNT(*) FILTER (
                    WHERE has_export_fob
                      AND has_import_cif
                ) AS original_usable_mirror_pairs,

                COUNT(*) FILTER (
                    WHERE has_export_fob
                      AND has_import_fob_corrected
                ) AS usable_trade_sync_pairs,

                COUNT(*) FILTER (
                    WHERE has_import_cif
                      AND NOT has_import_fob_corrected
                ) AS imports_present_not_corrected

            FROM trade_sync_input
            """
        ).fetchone()


        direction_before_after = con.execute(
            """
            WITH paired AS (
                SELECT
                    import_value_cif,
                    import_value_fob_corrected,
                    export_value_fob

                FROM trade_sync_input

                WHERE has_export_fob
                AND has_import_cif
                AND has_import_fob_corrected
            ),

            long_form AS (
                SELECT
                    'Before correction' AS stage,

                    CASE
                        WHEN import_value_cif
                            > export_value_fob * 1.000001
                            THEN 'Importer value above exporter FOB'

                        WHEN import_value_cif
                            < export_value_fob * 0.999999
                            THEN 'Importer value below exporter FOB'

                        ELSE 'Approximately equal'
                    END AS direction

                FROM paired

                UNION ALL

                SELECT
                    'After correction' AS stage,

                    CASE
                        WHEN import_value_fob_corrected
                            > export_value_fob * 1.000001
                            THEN 'Importer value above exporter FOB'

                        WHEN import_value_fob_corrected
                            < export_value_fob * 0.999999
                            THEN 'Importer value below exporter FOB'

                        ELSE 'Approximately equal'
                    END AS direction

                FROM paired
            )

            SELECT
                stage AS "Stage",
                direction AS "Direction",
                COUNT(*) AS "Rows",

                100.0 * COUNT(*)
                    / SUM(COUNT(*)) OVER (
                        PARTITION BY stage
                    ) AS "Share"

            FROM long_form

            GROUP BY
                stage,
                direction

            ORDER BY
                CASE stage
                    WHEN 'Before correction' THEN 1
                    ELSE 2
                END,

                CASE direction
                    WHEN 'Importer value above exporter FOB' THEN 1
                    WHEN 'Approximately equal' THEN 2
                    ELSE 3
                END
            """
        ).df()

        agreement_row = con.execute(
            """
            WITH paired AS (
                SELECT
                    ABS(
                        LN(import_value_cif / export_value_fob)
                    ) AS absolute_log_gap_before,

                    ABS(
                        LN(
                            import_value_fob_corrected
                            / export_value_fob
                        )
                    ) AS absolute_log_gap_after,

                    GREATEST(
                        import_value_cif,
                        export_value_fob
                    ) / LEAST(
                        import_value_cif,
                        export_value_fob
                    ) AS factor_gap_before,

                    GREATEST(
                        import_value_fob_corrected,
                        export_value_fob
                    ) / LEAST(
                        import_value_fob_corrected,
                        export_value_fob
                    ) AS factor_gap_after

                FROM trade_sync_input

                WHERE has_export_fob
                  AND has_import_cif
                  AND has_import_fob_corrected
            )

            SELECT
                COUNT(*) AS paired_rows,

                MEDIAN(factor_gap_before)
                    AS median_factor_gap_before,

                MEDIAN(factor_gap_after)
                    AS median_factor_gap_after,

                QUANTILE_CONT(factor_gap_before, 0.90)
                    AS p90_factor_gap_before,

                QUANTILE_CONT(factor_gap_after, 0.90)
                    AS p90_factor_gap_after,

                MEDIAN(absolute_log_gap_before)
                    AS median_absolute_log_gap_before,

                MEDIAN(absolute_log_gap_after)
                    AS median_absolute_log_gap_after,

                COUNT(*) FILTER (
                    WHERE absolute_log_gap_after
                        < absolute_log_gap_before
                ) AS rows_improved,

                COUNT(*) FILTER (
                    WHERE absolute_log_gap_after
                        = absolute_log_gap_before
                ) AS rows_unchanged,

                COUNT(*) FILTER (
                    WHERE absolute_log_gap_after
                        > absolute_log_gap_before
                ) AS rows_worsened

            FROM paired
            """
        ).fetchone()

        agreement_bands = con.execute(
            """
            WITH paired AS (
                SELECT
                    GREATEST(
                        import_value_fob_corrected,
                        export_value_fob
                    ) / LEAST(
                        import_value_fob_corrected,
                        export_value_fob
                    ) AS factor_gap

                FROM trade_sync_input

                WHERE has_export_fob
                  AND has_import_fob_corrected
            ),

            banded AS (
                SELECT
                    CASE
                        WHEN factor_gap <= 1.05 THEN 'Within 5%'
                        WHEN factor_gap <= 1.10 THEN '5–10%'
                        WHEN factor_gap <= 1.25 THEN '10–25%'
                        WHEN factor_gap <= 1.50 THEN '25–50%'
                        WHEN factor_gap <= 2.00 THEN '50–100%'
                        ELSE 'More than factor 2'
                    END AS agreement_band

                FROM paired
            )

            SELECT
                agreement_band AS "Agreement band",
                COUNT(*) AS "Rows",
                100.0 * COUNT(*)
                    / SUM(COUNT(*)) OVER ()
                    AS "Share"

            FROM banded

            GROUP BY agreement_band

            ORDER BY
                CASE agreement_band
                    WHEN 'Within 5%' THEN 1
                    WHEN '5–10%' THEN 2
                    WHEN '10–25%' THEN 3
                    WHEN '25–50%' THEN 4
                    WHEN '50–100%' THEN 5
                    ELSE 6
                END
            """
        ).df()

        by_year = con.execute(
            """
            SELECT
                year AS "Year",
                COUNT(*) AS "Rows",

                COUNT(*) FILTER (
                    WHERE has_export_fob
                      AND has_import_fob_corrected
                ) AS "Both reports",

                COUNT(*) FILTER (
                    WHERE has_export_fob
                      AND NOT has_import_fob_corrected
                ) AS "Exporter only",

                COUNT(*) FILTER (
                    WHERE NOT has_export_fob
                      AND has_import_fob_corrected
                ) AS "Corrected importer only",

                COUNT(*) FILTER (
                    WHERE NOT has_export_fob
                      AND NOT has_import_fob_corrected
                ) AS "Neither usable",

                100.0 * COUNT(*) FILTER (
                    WHERE has_export_fob
                      AND has_import_fob_corrected
                ) / NULLIF(COUNT(*), 0)
                    AS "Both-reports share"

            FROM trade_sync_input

            GROUP BY year
            ORDER BY year
            """
        ).df()

    finally:
        con.close()

    if coverage_row is None or agreement_row is None:
        raise RuntimeError(
            "Could not inspect Trade Sync readiness."
        )

    coverage_names = [
        "total_rows",
        "rows_with_export_fob",
        "rows_with_import_cif",
        "rows_with_corrected_import_fob",
        "original_usable_mirror_pairs",
        "usable_trade_sync_pairs",
        "imports_present_not_corrected",
    ]
    coverage = dict(
        zip(coverage_names, coverage_row, strict=True)
    )

    agreement_names = [
        "paired_rows",
        "median_factor_gap_before",
        "median_factor_gap_after",
        "p90_factor_gap_before",
        "p90_factor_gap_after",
        "median_absolute_log_gap_before",
        "median_absolute_log_gap_after",
        "rows_improved",
        "rows_unchanged",
        "rows_worsened",
    ]
    agreement = dict(
        zip(agreement_names, agreement_row, strict=True)
    )

    paired_rows = int(agreement["paired_rows"] or 0)
    agreement["percent_improved"] = (
        100.0
        * int(agreement["rows_improved"] or 0)
        / paired_rows
        if paired_rows
        else 0.0
    )
    agreement["percent_unchanged"] = (
        100.0
        * int(agreement["rows_unchanged"] or 0)
        / paired_rows
        if paired_rows
        else 0.0
    )
    agreement["percent_worsened"] = (
        100.0
        * int(agreement["rows_worsened"] or 0)
        / paired_rows
        if paired_rows
        else 0.0
    )

    agreement_summary = pd.DataFrame(
        [
            {
                "Metric": "Paired rows evaluated",
                "Before correction": agreement["paired_rows"],
                "After correction": agreement["paired_rows"],
            },
            {
                "Metric": "Median exporter/importer factor gap",
                "Before correction": (
                    agreement["median_factor_gap_before"]
                ),
                "After correction": (
                    agreement["median_factor_gap_after"]
                ),
            },
            {
                "Metric": "P90 exporter/importer factor gap",
                "Before correction": (
                    agreement["p90_factor_gap_before"]
                ),
                "After correction": (
                    agreement["p90_factor_gap_after"]
                ),
            },
            {
                "Metric": "Median absolute log gap",
                "Before correction": (
                    agreement["median_absolute_log_gap_before"]
                ),
                "After correction": (
                    agreement["median_absolute_log_gap_after"]
                ),
            },
        ]
    )

    change_summary = pd.DataFrame(
        [
            {
                "Outcome": "Improved",
                "Rows": agreement["rows_improved"],
                "Share": agreement["percent_improved"],
            },
            {
                "Outcome": "Unchanged",
                "Rows": agreement["rows_unchanged"],
                "Share": agreement["percent_unchanged"],
            },
            {
                "Outcome": "Worsened",
                "Rows": agreement["rows_worsened"],
                "Share": agreement["percent_worsened"],
            },
        ]
    )

    return {
        "availability": availability,
        "coverage": coverage,
        "agreement": agreement,
        "agreement_summary": agreement_summary,
        "change_summary": change_summary,
        "agreement_bands": agreement_bands,
        "by_year": by_year,
        "neither_usable_breakdown": neither_usable_breakdown,
        "direction_before_after": direction_before_after,
    }

def _prepare_corrected_trade_report(
    *,
    cfg: CifFobConfig,
    corrected_trade_result: dict[str, Any],
    figure_dir: Path,
    logger,
) -> dict[str, Any]:
    """
    Inspect corrected trade, prepare report tables, and write its figures.

    The application audit itself remains owned by apply_weights.py.
    This function only prepares human-readable reporting material.
    """
    corrected_trade_path = corrected_trade_result.get(
        "corrected_trade_path"
    )

    if corrected_trade_path is None:
        raise RuntimeError(
            "Corrected-trade reporting requires "
            "'corrected_trade_path' in corrected_trade_result."
        )

    corrected_trade_path = Path(
        corrected_trade_path
    )

    if not corrected_trade_path.exists():
        raise FileNotFoundError(
            "Corrected-trade artifact not found: "
            f"{corrected_trade_path}"
        )

    inspection = _inspect_corrected_trade(
        path=corrected_trade_path,
        threads=cfg.engine.threads,
        value_band_boundaries=(
            cfg.method.estimation
            .representation_weighting
            .value_band
            .boundaries_usd
        ),
    )

    application_reason_inspection = (
        _inspect_correction_application_reasons(
            corrected_trade_path=corrected_trade_path,
            training_features_path=cfg.product_path(
                "training_features"
            ),
            threads=cfg.engine.threads,
        )
    )

    overall = inspection["overall"]
    by_year = inspection["by_year"]
    by_value_band = inspection["by_value_band"]
    uncertainty = inspection["uncertainty"]

    trade_sync_readiness = _inspect_trade_sync_readiness(
        path=corrected_trade_path,
        threads=cfg.engine.threads,
    )

    audit_table = pd.DataFrame(
        [
            {
                "Audit check": "Application audit passed",
                "Result": (
                    "Yes"
                    if corrected_trade_result.get(
                        "audit_passed",
                        False,
                    )
                    else "No"
                ),
            },
            {
                "Audit check": "Ready for trade sync",
                "Result": (
                    "Yes"
                    if corrected_trade_result.get(
                        "ready_for_trade_sync",
                        False,
                    )
                    else "No"
                ),
            },
            {
                "Audit check": "Duplicate bilateral keys",
                "Result": _fmt_int(
                    corrected_trade_result.get(
                        "n_duplicate_keys"
                    )
                ),
            },
            {
                "Audit check": "Eligible rows not corrected",
                "Result": _fmt_int(
                    corrected_trade_result.get(
                        "n_eligible_not_corrected"
                    )
                ),
            },
            {
                "Audit check": "Ineligible rows corrected",
                "Result": _fmt_int(
                    corrected_trade_result.get(
                        "n_ineligible_corrected"
                    )
                ),
            },
            {
                "Audit check": "Invalid corrected rows",
                "Result": _fmt_int(
                    corrected_trade_result.get(
                        "n_invalid_corrected_rows"
                    )
                ),
            },
            {
                "Audit check": "Formula failures",
                "Result": _fmt_int(
                    corrected_trade_result.get(
                        "n_formula_failures"
                    )
                ),
            },
            {
                "Audit check": "Interval failures",
                "Result": _fmt_int(
                    corrected_trade_result.get(
                        "n_interval_failures"
                    )
                ),
            },
            
        ]
    )

    summary_table = pd.DataFrame(
        [
            {
                "Metric": "Output rows",
                "Value": _fmt_int(
                    overall["n_rows"]
                ),
            },
            {
                "Metric": "Corrected rows",
                "Value": _fmt_int(
                    overall["n_corrected_rows"]
                ),
            },
            {
                "Metric": "Uncorrected rows",
                "Value": _fmt_int(
                    overall["n_uncorrected_rows"]
                ),
            },
            {
                "Metric": "Correction coverage",
                "Value": _fmt_percent_value(
                    overall[
                        "correction_coverage_percent"
                    ]
                ),
            },
            {
                "Metric": "Original CIF import value",
                "Value": _fmt_float(
                    overall[
                        "original_import_cif_value"
                    ],
                    digits=2,
                ),
            },
            {
                "Metric": "Corrected FOB-equivalent value",
                "Value": _fmt_float(
                    overall[
                        "corrected_import_fob_value"
                    ],
                    digits=2,
                ),
            },
            {
                "Metric": "Estimated value reduction",
                "Value": _fmt_float(
                    overall[
                        "total_cif_fob_reduction"
                    ],
                    digits=2,
                ),
            },
            {
                "Metric": "Estimated value reduction share",
                "Value": _fmt_percent_value(
                    overall[
                        "total_cif_fob_reduction_percent"
                    ]
                ),
            },
            {
                "Metric": "Median applied factor",
                "Value": _fmt_float(
                    overall[
                        "median_factor_applied"
                    ]
                ),
            },
            {
                "Metric": "Applied factor P05",
                "Value": _fmt_float(
                    overall["factor_p05"]
                ),
            },
            {
                "Metric": "Applied factor P95",
                "Value": _fmt_float(
                    overall["factor_p95"]
                ),
            },
            {
                "Metric": "Median row-level correction",
                "Value": _fmt_fraction_percent(
                    overall[
                        "median_correction_fraction"
                    ]
                ),
            },
        ]
    )

    audit_markdown = _markdown_table(
        audit_table,
        columns=[
            "Audit check",
            "Result",
        ],
    )

    summary_markdown = _markdown_table(
        summary_table,
        columns=[
            "Metric",
            "Value",
        ],
    )

    application_outcomes_markdown = _markdown_table(
        application_reason_inspection["application_outcomes"],
        columns=[
            "Application outcome",
            "Rows",
            "Share",
            "Explanation",
        ],
        formatters={
            "Rows": _fmt_int,
            "Share": _fmt_percent_value,
        },
    )

    predictor_diagnostic_markdown = _markdown_table(
        application_reason_inspection["predictor_diagnostic"],
        columns=[
            "Prediction coverage path",
            "Rows",
            "Share",
            "Interpretation",
        ],
        formatters={
            "Rows": _fmt_int,
            "Share": _fmt_percent_value,
        },
    )

    by_year_markdown = _markdown_table(
        by_year,
        columns=[
            "Year",
            "Rows",
            "Corrected rows",
            "Correction coverage",
            "Original CIF value",
            "Corrected FOB value",
            "Value reduction",
            "Median factor",
            "Median correction",
        ],
        formatters={
            "Year": _fmt_int,
            "Rows": _fmt_int,
            "Corrected rows": _fmt_int,
            "Correction coverage": _fmt_percent_value,
            "Original CIF value": (
                lambda value: _fmt_float(
                    value,
                    digits=2,
                )
            ),
            "Corrected FOB value": (
                lambda value: _fmt_float(
                    value,
                    digits=2,
                )
            ),
            "Value reduction": _fmt_percent_value,
            "Median factor": _fmt_float,
            "Median correction": (
                _fmt_fraction_percent
            ),
        },
    )

    by_value_band_markdown = _markdown_table(
        by_value_band,
        columns=[
            "Evidence-value band",
            "Rows",
            "Corrected rows",
            "Correction coverage",
            "Original CIF value",
            "Corrected FOB value",
            "Value reduction",
            "Median factor",
            "Median correction",
        ],
        formatters={
            "Rows": _fmt_int,
            "Corrected rows": _fmt_int,
            "Correction coverage": _fmt_percent_value,
            "Original CIF value": (
                lambda value: _fmt_float(
                    value,
                    digits=2,
                )
            ),
            "Corrected FOB value": (
                lambda value: _fmt_float(
                    value,
                    digits=2,
                )
            ),
            "Value reduction": _fmt_percent_value,
            "Median factor": _fmt_float,
            "Median correction": (
                _fmt_fraction_percent
            ),
        },
    )

    uncertainty_markdown = _markdown_table(
        uncertainty,
        columns=[
            "Prediction interval",
            "Calibration rows",
            "Log-error quantile",
            "Median factor interval width",
            "Median corrected-value interval width",
        ],
        formatters={
            "Calibration rows": _fmt_int,
            "Log-error quantile": (
                lambda value: _fmt_float(
                    value,
                    digits=6,
                )
            ),
            "Median factor interval width": (
                _fmt_fraction_percent
            ),
            "Median corrected-value interval width": (
                _fmt_fraction_percent
            ),
        },
    )

    trade_sync_availability_markdown = _markdown_table(
        trade_sync_readiness["availability"],
        columns=[
            "Trade Sync status",
            "Rows",
            "Share",
        ],
        formatters={
            "Rows": _fmt_int,
            "Share": _fmt_percent_value,
        },
    )

    coverage = trade_sync_readiness["coverage"]
    trade_sync_coverage_table = pd.DataFrame(
        [
            {
                "Metric": "Total bilateral rows",
                "Rows": coverage["total_rows"],
            },
            {
                "Metric": "Rows with usable exporter FOB value",
                "Rows": coverage["rows_with_export_fob"],
            },
            {
                "Metric": "Rows with usable importer CIF value",
                "Rows": coverage["rows_with_import_cif"],
            },
            {
                "Metric": "Rows with usable corrected importer FOB value",
                "Rows": coverage[
                    "rows_with_corrected_import_fob"
                ],
            },
            {
                "Metric": "Original usable mirror pairs",
                "Rows": coverage[
                    "original_usable_mirror_pairs"
                ],
            },
            {
                "Metric": "Usable Trade Sync pairs",
                "Rows": coverage[
                    "usable_trade_sync_pairs"
                ],
            },
            {
                "Metric": "Importer reports present but not corrected",
                "Rows": coverage[
                    "imports_present_not_corrected"
                ],
            },
        ]
    )

    trade_sync_coverage_markdown = _markdown_table(
        trade_sync_coverage_table,
        columns=[
            "Metric",
            "Rows",
        ],
        formatters={
            "Rows": _fmt_int,
        },
    )

    neither_usable_breakdown_markdown = _markdown_table(
        trade_sync_readiness["neither_usable_breakdown"],
        columns=[
            "Reason combination",
            "Rows",
            "Share",
        ],
        formatters={
            "Rows": _fmt_int,
            "Share": _fmt_percent_value,
        },
    )
    

    direction_before_after_markdown = _markdown_table(
        trade_sync_readiness["direction_before_after"],
        columns=[
            "Stage",
            "Direction",
            "Rows",
            "Share",
        ],
        formatters={
            "Rows": _fmt_int,
            "Share": _fmt_percent_value,
        },
    )

    trade_sync_agreement_markdown = _markdown_table(
        trade_sync_readiness["agreement_summary"],
        columns=[
            "Metric",
            "Before correction",
            "After correction",
        ],
        formatters={
            "Before correction": (
                lambda value: (
                    _fmt_int(value)
                    if float(value).is_integer()
                    else _fmt_float(value, digits=6)
                )
            ),
            "After correction": (
                lambda value: (
                    _fmt_int(value)
                    if float(value).is_integer()
                    else _fmt_float(value, digits=6)
                )
            ),
        },
    )

    trade_sync_change_markdown = _markdown_table(
        trade_sync_readiness["change_summary"],
        columns=[
            "Outcome",
            "Rows",
            "Share",
        ],
        formatters={
            "Rows": _fmt_int,
            "Share": _fmt_percent_value,
        },
    )

    trade_sync_bands_markdown = _markdown_table(
        trade_sync_readiness["agreement_bands"],
        columns=[
            "Agreement band",
            "Rows",
            "Share",
        ],
        formatters={
            "Rows": _fmt_int,
            "Share": _fmt_percent_value,
        },
    )

    trade_sync_by_year_markdown = _markdown_table(
        trade_sync_readiness["by_year"],
        columns=[
            "Year",
            "Rows",
            "Both reports",
            "Exporter only",
            "Corrected importer only",
            "Neither usable",
            "Both-reports share",
        ],
        formatters={
            "Year": _fmt_int,
            "Rows": _fmt_int,
            "Both reports": _fmt_int,
            "Exporter only": _fmt_int,
            "Corrected importer only": _fmt_int,
            "Neither usable": _fmt_int,
            "Both-reports share": _fmt_percent_value,
        },
    )

    figure_paths = _write_corrected_trade_figures(

            cfg=cfg,
            corrected_trade_path=corrected_trade_path,
            corrected_trade_by_year=by_year,
            corrected_trade_by_value_band=by_value_band,
            corrected_trade_agreement_bands=trade_sync_readiness["agreement_bands"],
            figure_dir=figure_dir,
            logger=logger,
        )

    return {
        "path": corrected_trade_path,
        "inspection": inspection,
        "audit_markdown": audit_markdown,
        "summary_markdown": summary_markdown,
        "by_year_markdown": by_year_markdown,
        "by_value_band_markdown": (
            by_value_band_markdown
        ),
        "uncertainty_markdown": (
            uncertainty_markdown
        ),
        "trade_sync_readiness": trade_sync_readiness,
        "trade_sync_availability_markdown": (
            trade_sync_availability_markdown
        ),
        "trade_sync_coverage_markdown": (
            trade_sync_coverage_markdown
        ),
        "trade_sync_agreement_markdown": (
            trade_sync_agreement_markdown
        ),
        "trade_sync_change_markdown": (
            trade_sync_change_markdown
        ),
        "trade_sync_bands_markdown": (
            trade_sync_bands_markdown
        ),
        "trade_sync_by_year_markdown": (
            trade_sync_by_year_markdown
        ),
        "figure_paths": figure_paths,

        "application_outcomes_markdown": (
            application_outcomes_markdown
        ),
        "predictor_diagnostic_markdown": (
            predictor_diagnostic_markdown
        ),
        "trade_sync_total_rows": (
            trade_sync_readiness["coverage"]["total_rows"]
        ),
        "neither_usable_breakdown_markdown": (
            neither_usable_breakdown_markdown
        ),
        "direction_before_after_markdown": (
            direction_before_after_markdown
        ),
    }

def _load_validation_artifacts(
    *,
    summary_path: Path,
    folds_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not summary_path.exists():
        raise FileNotFoundError(
            "Temporal-validation summary not found: "
            f"{summary_path}"
        )

    if not folds_path.exists():
        raise FileNotFoundError(
            "Temporal-validation folds not found: "
            f"{folds_path}"
        )

    summary = pd.read_csv(summary_path)
    folds = pd.read_csv(folds_path)

    if summary.empty:
        raise RuntimeError(
            "Temporal-validation summary contains no rows."
        )

    if folds.empty:
        raise RuntimeError(
            "Temporal-validation folds contain no rows."
        )

    return summary, folds

def _save_cif_fob_figure(
    *,
    fig,
    root_dir: Path,
    category: str,
    target_hs: str,
    stem: str,
    figure_paths: dict[str, Path],
    key: str,
    logger=None,
    dpi: int = 180,
) -> Path:
    """Save a CIF/FOB figure in a category subdirectory and close it."""
    category_dir = root_dir / category
    category_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_prefix = target_hs.lower()
    path = category_dir / f"{target_prefix}_{stem}.png"

    try:
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)

    figure_paths[key] = path

    # if logger is not None:
    #     logger.info(
    #         "PLOT WRITTEN — %s: %s",
    #         key,
    #         path,
    #     )

    return path

def _write_primary_observed_predicted_figures(
    *,
    cfg: CifFobConfig,
    diagnostics_result: dict[str, Any],
    figure_dir: Path,
    logger,
) -> dict[str, Path]:
    """
    Write observed-versus-predicted figures from the primary holdout.

    These are standalone model-comparison artifacts and are not embedded
    individually in the decision report.
    """
    diagnostics_path = diagnostics_result.get(
        "diagnostics_path"
    )

    if diagnostics_path is None:
        logger.warning(
            "Diagnostics result does not contain diagnostics_path."
        )
        return {}

    diagnostics_path = Path(
        diagnostics_path
    )

    if not diagnostics_path.exists():
        logger.warning(
            "Diagnostics artifact not found: %s",
            diagnostics_path,
        )
        return {}

    diagnostics = pd.read_parquet(
        diagnostics_path
    )

    if diagnostics.empty:
        logger.warning(
            "Diagnostics artifact is empty: %s",
            diagnostics_path,
        )
        return {}

    required_columns = {
        "observed_cif_fob_factor",
        "predicted_cif_fob_factor",
    }

    missing_columns = sorted(
        required_columns - set(diagnostics.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Diagnostics artifact is missing required "
            "observed/predicted columns: "
            + ", ".join(missing_columns)
        )

    figure_paths: dict[str, Path] = {}

    # Observed-versus-predicted density
    fig, _ = plot_observed_vs_predicted_density(
        diagnostics,
        observed_column="observed_cif_fob_factor",
        predicted_column="predicted_cif_fob_factor",
        title=(
            "Primary model: observed vs predicted "
            "CIF/FOB factor density"
        ),
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_observed_vs_predicted_density",
        figure_paths=figure_paths,
        key="observed_vs_predicted_density",
        logger=logger,
    )

    # Observed-versus-predicted tolerance bands
    fig, _ = plot_observed_vs_predicted_with_tolerance_bands(
        diagnostics,
        observed_column="observed_cif_fob_factor",
        predicted_column="predicted_cif_fob_factor",
        tolerance_bands=(0.02, 0.05, 0.10, 0.20),
        lower_quantile=0.001,
        upper_quantile=0.995,
        display_lower=1.0,
        display_upper=1.95,
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_observed_vs_predicted_tolerance_bands",
        figure_paths=figure_paths,
        key="observed_vs_predicted_tolerance_bands",
        logger=logger,
    )

    logger.info(
        "Primary observed-versus-predicted figures written: %s",
        len(figure_paths),
    )

    return figure_paths

def _write_primary_diagnostics_figures(
    *,
    cfg: CifFobConfig,
    diagnostics_result: dict[str, Any],
    figure_dir: Path,
    logger,
) -> dict[str, Path]:
    """
    Write detailed primary-model holdout diagnostics.

    These figures are saved as standalone run artifacts. They are not
    embedded individually in the decision report.
    """
    diagnostics_path = diagnostics_result.get(
        "diagnostics_path"
    )

    if diagnostics_path is None:
        logger.warning(
            "Diagnostics result does not contain diagnostics_path."
        )
        return {}

    diagnostics_path = Path(
        diagnostics_path
    )

    if not diagnostics_path.exists():
        logger.warning(
            "Diagnostics artifact not found: %s",
            diagnostics_path,
        )
        return {}

    diagnostics = pd.read_parquet(
        diagnostics_path
    )

    if diagnostics.empty:
        logger.warning(
            "Diagnostics artifact is empty: %s",
            diagnostics_path,
        )
        return {}

    figure_paths: dict[str, Path] = {}

    # Error by observed factor band
    fig, _, _ = plot_grouped_error_bars(
        diagnostics,
        group_column="observed_factor_band",
        error_column="ape",
        min_observations=100,
        top_n=None,
        title=(
            "Primary-model error by observed "
            "CIF/FOB factor band"
        ),
        x_label="Median APE (%)",
        y_label="Observed factor band",
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="holdout_diagnostics",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_error_by_observed_factor_band",
        figure_paths=figure_paths,
        key="error_by_observed_factor_band",
        logger=logger,
    )

    # Error by evidence-value band
    fig, _, _ = plot_grouped_error_bars(
        diagnostics,
        group_column="value_band",
        error_column="ape",
        min_observations=100,
        top_n=None,
        title="Primary-model error by evidence-value band",
        x_label="Median APE (%)",
        y_label="Evidence-value band",
        group_order=VALUE_BAND_ORDER,
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="holdout_diagnostics",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_error_by_value_band",
        figure_paths=figure_paths,
        key="error_by_value_band",
        logger=logger,
    )

    # Error by route-distance band
    fig, _, _ = plot_grouped_error_bars(
        diagnostics,
        group_column="distance_band",
        error_column="ape",
        min_observations=100,
        top_n=None,
        title="Primary-model error by route-distance band",
        x_label="Median APE (%)",
        y_label="Distance band",
        group_order=DISTANCE_BAND_ORDER,
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="holdout_diagnostics",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_error_by_distance_band",
        figure_paths=figure_paths,
        key="error_by_distance_band",
        logger=logger,
    )

    # Error by HS4 product group
    fig, _, _ = plot_grouped_error_bars(
        diagnostics,
        group_column="hs4_product_group",
        error_column="ape",
        min_observations=300,
        top_n=20,
        title="Primary-model error by HS4 product group",
        x_label="Median APE (%)",
        y_label="HS4 product group",
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="holdout_diagnostics",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_error_by_hs4_product_group",
        figure_paths=figure_paths,
        key="error_by_hs4_product_group",
        logger=logger,
    )

    logger.info(
        "Primary-model holdout diagnostic figures written: %s",
        len(figure_paths),
    )

    return figure_paths

def _write_corrected_trade_figures(
    *,
    cfg: CifFobConfig,
    corrected_trade_path: Path,
    corrected_trade_by_year: pd.DataFrame,
    corrected_trade_by_value_band: pd.DataFrame,
    corrected_trade_agreement_bands: pd.DataFrame | None,
    figure_dir: Path,
    logger,
) -> dict[str, Path]:
    """
    Write figures describing the applied CIF/FOB correction.
    """
    figure_paths: dict[str, Path] = {}

    # -------------------------------------------------
    # 1. By year: coverage vs value reduction
    # -------------------------------------------------
    if not corrected_trade_by_year.empty:
        by_year = corrected_trade_by_year.copy()

        by_year["Year"] = pd.to_numeric(
            by_year["Year"],
            errors="coerce",
        )

        by_year = by_year.loc[
            by_year["Year"].notna()
        ].sort_values("Year")

        by_year["Year"] = by_year["Year"].astype(int).astype(str)

        fig, _ = plot_corrected_trade_comparison(
            frame=by_year,
            category_column="Year",
            coverage_column="Correction coverage",
            reduction_column="Value reduction",
            title=(
                "Applied CIF/FOB correction by year: "
                "coverage versus value reduction"
            ),
            x_label="Year",
            rotate_xticks=0,
        )

        _save_cif_fob_figure(
            fig=fig,
            root_dir=figure_dir,
            category="corrected_trade",
            target_hs=cfg.scope.target_hs,
            stem="corrected_trade_by_year_comparison",
            figure_paths=figure_paths,
            key="corrected_trade_by_year_comparison",
            logger=logger,
        )
    else:
        logger.warning(
            "No annual corrected-trade summary is available for plotting."
        )

    # -------------------------------------------------
    # 2. By evidence-value band: coverage vs reduction
    # -------------------------------------------------
    if not corrected_trade_by_value_band.empty:
        by_value_band = corrected_trade_by_value_band.copy()

        if "Evidence-value band" in by_value_band.columns:
            by_value_band["Evidence-value band"] = pd.Categorical(
                by_value_band["Evidence-value band"],
                categories=VALUE_BAND_ORDER,
                ordered=True,
            )
            by_value_band = by_value_band.sort_values(
                "Evidence-value band"
            )
            by_value_band["Evidence-value band"] = (
                by_value_band["Evidence-value band"].astype(str)
            )

        fig, _ = plot_corrected_trade_comparison(
            frame=by_value_band,
            category_column="Evidence-value band",
            coverage_column="Correction coverage",
            reduction_column="Value reduction",
            title=(
                "Applied CIF/FOB correction by evidence-value band: "
                "coverage versus value reduction"
            ),
            x_label="Evidence-value band",
            rotate_xticks=0,
        )

        _save_cif_fob_figure(
            fig=fig,
            root_dir=figure_dir,
            category="corrected_trade",
            target_hs=cfg.scope.target_hs,
            stem="corrected_trade_by_value_band_comparison",
            figure_paths=figure_paths,
            key="corrected_trade_by_value_band_comparison",
            logger=logger,
        )
    else:
        logger.warning(
            "No value-band corrected-trade summary is available for plotting."
        )


    # -------------------------------------------------
    # 3. Trade-sync agreement bands after CIF/FOB correction
    # -------------------------------------------------
    if (
        corrected_trade_agreement_bands is not None
        and not corrected_trade_agreement_bands.empty
    ):
        agreement_band_plot_frame = (
            corrected_trade_agreement_bands.rename(
                columns={
                    "Agreement band": "agreement_band",
                    "Rows": "rows",
                    "Share": "percent",
                }
            )
        )

        fig, _ = plot_cif_fob_corrected_agreement_bands(
            agreement_band_plot_frame,
            x_label="Value-disagreement band",
            y_label="Residual bilateral value disagreement after CIF/FOB correction",
            title=(
                "Residual exporter/importer disagreement after "
                "CIF/FOB correction"
            ),
        )

        _save_cif_fob_figure(
            fig=fig,
            root_dir=figure_dir,
            category="corrected_trade",
            target_hs=cfg.scope.target_hs,
            stem="trade_sync_agreement_bands",
            figure_paths=figure_paths,
            key="trade_sync_agreement_bands",
            logger=logger,
        )
    else:
        logger.warning(
            "No trade-sync agreement-band summary is available for plotting."
        )


    # -------------------------------------------------
    # 3. Nebula plot of applied CIF/FOB factors
    # -------------------------------------------------
    try:
        corrected_factor_distribution_df = (
            _load_corrected_trade_factor_distribution_data(
                path=corrected_trade_path,
                threads=cfg.engine.threads,
            )
        )

        fig, _, _ = plot_nebula_boxplot(
            corrected_factor_distribution_df,
            factor_col="cif_fob_factor_applied",
            upper_display=float(
                cfg.method.estimation.maximum_factor_rule.fixed_value
            ),
            lower_display=0.0,
            x_label= "Applied CIF/FOB factor",
            title=(
                "Distribution of applied CIF/FOB factors "
                "in corrected trade"
            ),
        )

        _save_cif_fob_figure(
            fig=fig,
            root_dir=figure_dir,
            category="corrected_trade",
            target_hs=cfg.scope.target_hs,
            stem="corrected_trade_factor_distribution",
            figure_paths=figure_paths,
            key="corrected_trade_factor_distribution",
            logger=logger,
        )

    except Exception as exc:
        logger.warning(
            "Could not write corrected-trade factor-distribution figure: %s",
            exc,
        )

    return figure_paths


def _write_training_evidence_audit_figures(
    *,
    cfg: CifFobConfig,
    training_evidence_audit_result: dict[str, Any],
    figure_dir: Path,
    logger,
) -> dict[str, Path]:
    """Write training-evidence representativeness audit figures."""

    audit_path = training_evidence_audit_result.get("training_evidence_audit_path")
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(audit_path, Path):
        audit_path = Path(str(audit_path))

    if not audit_path.exists():
        logger.warning(
            "Training-evidence audit artifact not found: %s",
            audit_path,
        )
        return {}

    audit = pd.read_csv(audit_path)

    if audit.empty:
        logger.warning(
            "Training-evidence audit artifact is empty: %s",
            audit_path,
        )
        return {}

    target_prefix = cfg.scope.target_hs.lower()

    figure_paths: dict[str, Path] = {}

    # Exporter continent representation
    fig, _, _ = plot_representation_ratio_bars(
        audit,
        dimension="exporter_continent",
        ratio_column="clean_representation_ratio",
        min_prediction_rows=100,
        top_n=None,
        title="Training-evidence representation by exporter continent",
        x_label="Clean-training representation ratio",
        y_label="Exporter continent",
    )

    path = (
        figure_dir
        / f"{target_prefix}_cif_fob_training_representation_exporter_continent.png"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths["training_representation_exporter_continent"] = path

    # Importer continent representation
    fig, _, _ = plot_representation_ratio_bars(
        audit,
        dimension="importer_continent",
        ratio_column="clean_representation_ratio",
        min_prediction_rows=100,
        top_n=None,
        title="Training-evidence representation by importer continent",
        x_label="Clean-training representation ratio",
        y_label="Importer continent",
    )

    path = (
        figure_dir
        / f"{target_prefix}_cif_fob_training_representation_importer_continent.png"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths["training_representation_importer_continent"] = path

    # Value-band representation
    fig, _, _ = plot_representation_ratio_bars(
        audit,
        dimension="value_band",
        ratio_column="clean_representation_ratio",
        min_prediction_rows=100,
        top_n=None,
        title="Training-evidence representation by evidence-value band",
        x_label="Clean-training representation ratio",
        y_label="Evidence-value band",
        group_order=VALUE_BAND_ORDER,
    )

    path = (
        figure_dir
        / f"{target_prefix}_cif_fob_training_representation_value_band.png"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths["training_representation_value_band"] = path

    # Distance-band representation
    fig, _, _ = plot_representation_ratio_bars(
        audit,
        dimension="distance_band",
        ratio_column="clean_representation_ratio",
        min_prediction_rows=100,
        top_n=None,
        title="Training-evidence representation by route-distance band",
        x_label="Clean-training representation ratio",
        y_label="Distance band",
        group_order=DISTANCE_BAND_ORDER
    )
    path = (
        figure_dir
        / f"{target_prefix}_cif_fob_training_representation_distance_band.png"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths["training_representation_distance_band"] = path

    # HS4 product group: show most underrepresented groups.
    fig, _, _ = plot_representation_ratio_bars(
        audit,
        dimension="hs4_product_group",
        ratio_column="clean_representation_ratio",
        min_prediction_rows=300,
        top_n=20,
        title="Most underrepresented HS4 product groups in training evidence",
        x_label="Clean-training representation ratio",
        y_label="HS4 product group",
        sort_by_ratio=True,
    )

    path = (
        figure_dir
        / f"{target_prefix}_cif_fob_training_representation_hs4_product_group.png"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths["training_representation_hs4_product_group"] = path

    logger.info(
        "Training-evidence audit figures written: %s",
        len(figure_paths),
    )

    return figure_paths


def _write_value_band_model_comparison_figures(
    *,
    cfg: CifFobConfig,
    estimated_rates_path: Path,
    folds: pd.DataFrame,
    prediction_columns_by_run: dict[str, dict[str, str]],
    figure_dir: Path,
    logger,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Path],
]:
    """
    Compare the configured primary run with its weighted counterpart.

    If no weighted counterpart was completed, return empty comparison
    material and write no weighting-comparison figures.
    """
    latest_holdout_year = int(
        pd.to_numeric(
            folds["validation_year"],
            errors="coerce",
        ).max()
    )

    (
        all_summary,
        all_long,
    ) = _inspect_value_band_model_comparison(
        path=estimated_rates_path,
        threads=cfg.engine.threads,
        holdout_year=latest_holdout_year,
        boundaries=(
            cfg.method.estimation
            .representation_weighting
            .value_band
            .boundaries_usd
        ),
        prediction_columns_by_run=prediction_columns_by_run,
    )

    figure_paths: dict[str, Path] = {}

    available_runs = {
        str(value)
        for value in all_summary["model_name"].dropna().unique()
    }

    primary_algorithm = str(
        cfg.method.estimation.models.primary
    ).strip().lower()

    primary_predictor_set = str(
        getattr(
            cfg.method.estimation,
            "primary_predictor_set",
            "",
        )
        or ""
    ).strip()

    expected_primary_run = (
        f"{primary_algorithm}__{primary_predictor_set}"
        if primary_predictor_set
        else primary_algorithm
    )

    if expected_primary_run in available_runs:
        primary_run = expected_primary_run
    else:
        primary_candidates = sorted(
            run_name
            for run_name in available_runs
            if run_name.split("__", 1)[0].lower()
            == primary_algorithm
            and (
                not primary_predictor_set
                or (
                    "__" in run_name
                    and run_name.split("__", 1)[1]
                    == primary_predictor_set
                )
            )
        )

        if len(primary_candidates) != 1:
            logger.warning(
                "Skipping value-band weighting comparison: "
                "could not uniquely resolve primary run. "
                "Expected=%s, candidates=%s",
                expected_primary_run,
                primary_candidates,
            )

            return (
                pd.DataFrame(),
                pd.DataFrame(),
                figure_paths,
            )

        primary_run = primary_candidates[0]

    primary_run_parts = primary_run.split("__", 1)
    primary_run_predictor_set = (
        primary_run_parts[1]
        if len(primary_run_parts) == 2
        else ""
    )

    weighted_candidates = sorted(
        run_name
        for run_name in available_runs
        if run_name != primary_run
        and "weighted" in run_name.split("__", 1)[0].lower()
        and (
            not primary_run_predictor_set
            or (
                "__" in run_name
                and run_name.split("__", 1)[1]
                == primary_run_predictor_set
            )
        )
    )

    if not weighted_candidates:
        logger.info(
            "Skipping value-band weighting comparison: "
            "no weighted counterpart is available for primary run %s.",
            primary_run,
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            figure_paths,
        )

    if len(weighted_candidates) > 1:
        logger.warning(
            "Skipping value-band weighting comparison: "
            "multiple weighted counterparts were found for %s: %s",
            primary_run,
            weighted_candidates,
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            figure_paths,
        )

    weighted_run = weighted_candidates[0]

    comparison_long = all_long.loc[
        all_long["model_name"].isin(
            [primary_run, weighted_run]
        )
    ].copy()

    primary_summary = (
        all_summary.loc[
            all_summary["model_name"].eq(primary_run),
            [
                "value_band",
                "observations",
                "median_ape",
                "p90_ape",
            ],
        ]
        .rename(
            columns={
                "observations": "Observations",
                "median_ape": "Primary median APE",
                "p90_ape": "Primary P90 APE",
            }
        )
    )

    weighted_summary = (
        all_summary.loc[
            all_summary["model_name"].eq(weighted_run),
            [
                "value_band",
                "observations",
                "median_ape",
                "p90_ape",
            ],
        ]
        .rename(
            columns={
                "observations": "Weighted observations",
                "median_ape": "Weighted median APE",
                "p90_ape": "Weighted P90 APE",
            }
        )
    )

    comparison_table = primary_summary.merge(
        weighted_summary,
        on="value_band",
        how="inner",
        validate="one_to_one",
    )

    if not (
        comparison_table["Observations"]
        == comparison_table["Weighted observations"]
    ).all():
        raise RuntimeError(
            "Primary and weighted models were not evaluated on "
            "the same observations within every evidence-value band."
        )

    comparison_table["Change in median APE"] = (
        comparison_table["Weighted median APE"]
        - comparison_table["Primary median APE"]
    )

    comparison_table["Change in P90 APE"] = (
        comparison_table["Weighted P90 APE"]
        - comparison_table["Primary P90 APE"]
    )

    comparison_table["Evidence-value band"] = pd.Categorical(
        comparison_table["value_band"],
        categories=VALUE_BAND_ORDER,
        ordered=True,
    )

    comparison_table = (
        comparison_table[
            [
                "Evidence-value band",
                "Observations",
                "Primary median APE",
                "Weighted median APE",
                "Change in median APE",
                "Primary P90 APE",
                "Weighted P90 APE",
                "Change in P90 APE",
            ]
        ]
        .sort_values("Evidence-value band")
        .reset_index(drop=True)
    )

    fig, _, _ = plot_model_error_by_group(
        comparison_long,
        group_column="value_band",
        model_column="model_name",
        error_column="ape",
        aggregation="median",
        group_order=VALUE_BAND_ORDER,
        title="Primary versus weighted model by evidence-value band",
        x_label="Evidence-value band",
        y_label="Median APE (%)",
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="primary_weighted_by_value_band_median_ape",
        figure_paths=figure_paths,
        key="primary_weighted_by_value_band_median_ape",
        logger=logger,
    )

    fig, _, _ = plot_model_error_by_group(
        comparison_long,
        group_column="value_band",
        model_column="model_name",
        error_column="ape",
        aggregation="p90",
        group_order=VALUE_BAND_ORDER,
        title="Primary versus weighted model by evidence-value band",
        x_label="Evidence-value band",
        y_label="90th percentile APE (%)",
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="primary_weighted_by_value_band_p90_ape",
        figure_paths=figure_paths,
        key="primary_weighted_by_value_band_p90_ape",
        logger=logger,
    )

    return (
        comparison_table,
        comparison_long,
        figure_paths,
    )


def _write_value_band_model_comparison_figures_deprecated(
    *,
    cfg: CifFobConfig,
    estimated_rates_path: Path,
    folds: pd.DataFrame,
    prediction_columns_by_run: dict[str, dict[str, str]],
    figure_dir: Path,
    logger,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Path],
]:
    """
    Compare all completed estimation runs by evidence-value band
    and write median-APE and P90-APE figures.
    """
    latest_holdout_year = int(
        pd.to_numeric(
            folds["validation_year"],
            errors="coerce",
        ).max()
    )

    (
        comparison_summary,
        comparison_long,
    ) = _inspect_value_band_model_comparison(
        path=estimated_rates_path,
        threads=cfg.engine.threads,
        holdout_year=latest_holdout_year,
        boundaries=(
            cfg.method.estimation
            .representation_weighting
            .value_band
            .boundaries_usd
        ),
        prediction_columns_by_run=prediction_columns_by_run,
    )

    figure_paths: dict[str, Path] = {}

    fig, _, _ = plot_model_error_by_group(
        comparison_long,
        group_column="value_band",
        model_column="model_name",
        error_column="ape",
        aggregation="median",
        group_order=VALUE_BAND_ORDER,
        title="Model comparison by evidence-value band",
        x_label="Evidence-value band",
        y_label="Median APE (%)",
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="model_comparison_by_value_band_median_ape",
        figure_paths=figure_paths,
        key="model_comparison_by_value_band_median_ape",
        logger=logger,
    )

    fig, _, _ = plot_model_error_by_group(
        comparison_long,
        group_column="value_band",
        model_column="model_name",
        error_column="ape",
        aggregation="p90",
        group_order=VALUE_BAND_ORDER,
        title="Model comparison by evidence-value band",
        x_label="Evidence-value band",
        y_label="90th percentile APE (%)",
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="model_comparison_by_value_band_p90_ape",
        figure_paths=figure_paths,
        key="model_comparison_by_value_band_p90_ape",
        logger=logger,
    )

    return (
        comparison_summary,
        comparison_long,
        figure_paths,
    )


def _load_direct_factor_distribution_data(
    *,
    path: Path,
    threads: int,
) -> pd.DataFrame:
    """
    Load positive directly observed CIF/FOB factors for distribution
    diagnostics.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training-feature artifact not found: {path}"
        )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        df = con.execute(
            """
            SELECT
                observed_cif_fob_factor

            FROM read_parquet(?)

            WHERE model_eligible
              AND LOWER(CAST(factor_source AS VARCHAR)) = 'direct'
              AND observed_cif_fob_factor IS NOT NULL
              AND observed_cif_fob_factor > 0
            """,
            [str(path)],
        ).df()

    finally:
        con.close()

    if df.empty:
        raise RuntimeError(
            "No positive direct CIF/FOB factors available for plotting."
        )

    return df


def _summarize_direct_factor_distribution(
    *,
    df: pd.DataFrame,
    factor_col: str = "observed_cif_fob_factor",
) -> dict[str, Any]:
    """Summarize the full positive direct-factor distribution."""
    factor = pd.to_numeric(
        df[factor_col],
        errors="coerce",
    ).dropna()

    factor = factor[factor > 0]

    if factor.empty:
        raise RuntimeError(
            "No positive direct CIF/FOB factors available for summary."
        )

    n = len(factor)
    q1 = float(factor.quantile(0.25))
    median = float(factor.quantile(0.50))
    q3 = float(factor.quantile(0.75))
    iqr = q3 - q1

    def n_above(cutoff: float) -> int:
        return int((factor > cutoff).sum())

    return {
        "n_positive": n,
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "q3_plus_1_5_iqr": q3 + 1.5 * iqr,
        "q3_plus_2_5_iqr": q3 + 2.5 * iqr,
        "p95": float(factor.quantile(0.95)),
        "p99": float(factor.quantile(0.99)),
        "maximum": float(factor.max()),
        "n_above_3": n_above(3.0),
        "share_above_3_percent": 100.0 * n_above(3.0) / n,
        "n_above_10": n_above(10.0),
        "share_above_10_percent": 100.0 * n_above(10.0) / n,
    }

# ======================================================================
# Decision helpers
# ======================================================================


def _rank_models(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    ranked = summary.copy()

    metadata_columns = {
        "model_name",
        "model_algorithm",
        "predictor_set_name",
    }

    numeric_columns = [
        column
        for column in ranked.columns
        if column not in metadata_columns
    ]

    for column in numeric_columns:
        ranked[column] = pd.to_numeric(
            ranked[column],
            errors="coerce",
        )

    ranked = ranked.sort_values(
        [
            "mean_mae_log_factor",
            "mean_median_ape",
            "mean_p90_ape",
        ],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    ranked.insert(
        0,
        "overall_rank",
        np.arange(1, len(ranked) + 1),
    )

    return ranked


def _enabled_predictor_set_names(
    cfg: CifFobConfig,
) -> set[str]:
    """Return configured predictor-set names flagged as enabled."""
    predictor_sets_cfg = getattr(
        cfg.method.estimation,
        "predictor_sets",
        {},
    )

    if not predictor_sets_cfg:
        return set()

    return {
        name
        for name, predictor_set in predictor_sets_cfg.items()
        if bool(getattr(predictor_set, "enabled", False))
    }


def _filter_to_enabled_predictor_sets(
    frame: pd.DataFrame,
    *,
    enabled_predictor_sets: set[str],
) -> pd.DataFrame:
    """Keep only rows whose predictor set is enabled in config."""
    if frame.empty or not enabled_predictor_sets:
        return frame

    filtered = frame.copy()

    if "predictor_set_name" not in filtered.columns:
        if "model_name" not in filtered.columns:
            return filtered

        split_parts = (
            filtered["model_name"]
            .astype(str)
            .str.split("__", n=1, expand=True)
        )
        filtered["predictor_set_name"] = (
            split_parts[1]
            if split_parts.shape[1] > 1
            else ""
        )

    mask = filtered["predictor_set_name"].astype(str).isin(
        enabled_predictor_sets
    )

    return filtered.loc[mask].copy()


def _best_model(
    summary: pd.DataFrame,
    *,
    metric: str,
    higher_is_better: bool = False,
) -> str:
    if metric not in summary.columns:
        return "n/a"

    values = pd.to_numeric(
        summary[metric],
        errors="coerce",
    )

    valid = values.notna()

    if not valid.any():
        return "n/a"

    index = (
        values[valid].idxmax()
        if higher_is_better
        else values[valid].idxmin()
    )

    return str(summary.loc[index, "model_name"])


def _resolve_cif_fob_figure_dir(
    *,
    cfg: CifFobConfig,
    run_data_dir: Path,
) -> Path:
    """Resolve the figure directory matching one CIF/FOB run."""

    run_name = run_data_dir.name

    figure_dir = (
        get_figure_dir()
        / "cif_fob"
        / f"target_{cfg.scope.target_hs.lower()}"
        / "cif_fob_runs"
        / run_name
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return figure_dir

def _primary_decision_text(
    *,
    primary_model: str,
    ranked: pd.DataFrame,
) -> str:
    primary_rows = ranked.loc[
        ranked["model_name"].eq(primary_model)
    ]

    if primary_rows.empty:
        return (
            f"The configured primary model `{primary_model}` is absent "
            "from the validation summary. Review the configuration and "
            "validation artifacts before applying the correction."
        )

    primary_rank = int(
        primary_rows.iloc[0]["overall_rank"]
    )

    best_model = str(
        ranked.iloc[0]["model_name"]
    )

    if primary_rank == 1:
        return (
            f"The configured primary model `{primary_model}` ranks first "
            "under the report's overall ordering: mean log-factor MAE, "
            "followed by median APE and P90 APE. It remains the preferred "
            "global estimator for downstream CIF/FOB correction."
        )

    return (
        f"The configured primary model `{primary_model}` ranks "
        f"{primary_rank} of {len(ranked)}. The best-ranked candidate is "
        f"`{best_model}`. The primary-model configuration should be "
        "reviewed before treating the current choice as final."
    )


def _tail_tradeoff_text(
    *,
    primary_model: str,
    ranked: pd.DataFrame,
) -> str:
    required = {
        "model_name",
        "mean_upper_tail_median_ape",
        "mean_upper_tail_p90_ape",
        "mean_median_ape",
    }

    if not required.issubset(ranked.columns):
        return (
            "Upper-tail metrics were not fully available in the "
            "validation summary."
        )

    tail_best = _best_model(
        ranked,
        metric="mean_upper_tail_median_ape",
    )

    primary_row = ranked.loc[
        ranked["model_name"].eq(primary_model)
    ]
    tail_row = ranked.loc[
        ranked["model_name"].eq(tail_best)
    ]

    if primary_row.empty or tail_row.empty:
        return "Upper-tail model comparison could not be resolved."

    primary_tail = float(
        primary_row.iloc[0]["mean_upper_tail_median_ape"]
    )
    best_tail = float(
        tail_row.iloc[0]["mean_upper_tail_median_ape"]
    )

    if tail_best == primary_model:
        return (
            f"The primary model `{primary_model}` also has the lowest "
            "mean upper-tail median APE among enabled candidates."
        )

    primary_central = float(
        primary_row.iloc[0]["mean_median_ape"]
    )
    tail_central = float(
        tail_row.iloc[0]["mean_median_ape"]
    )

    return (
        f"`{tail_best}` has the best mean upper-tail median APE "
        f"({_fmt_fraction_percent(best_tail)} versus "
        f"{_fmt_fraction_percent(primary_tail)} for `{primary_model}`), "
        "but its overall median APE is "
        f"{_fmt_fraction_percent(tail_central)} versus "
        f"{_fmt_fraction_percent(primary_central)} for the primary model. "
        "This is evidence of a central-versus-tail trade-off rather than "
        "a universal improvement."
    )




def _build_primary_diagnostics_section(
    *,
    diagnostics_result: dict[str, Any] | None,
) -> list[str]:
    """Build a compact diagnostics summary without embedding detailed plots."""
    status = (
        diagnostics_result.get("status")
        if diagnostics_result is not None
        else None
    )

    if status not in {"completed", "reused"}:
        return [
            "## 9. Primary-model diagnostics",
            "",
            "_Primary-model diagnostics were not available._",
            "",
        ]

    return [
        "## 9. Primary-model diagnostics",
        "",
        (
            "The diagnostics summarize performance on the latest temporal "
            "holdout. Detailed heterogeneity artifacts are retained separately "
            "and are not reproduced in this decision report."
        ),
        "",
        (
            f"- **Diagnostics rows:** "
            f"{_fmt_int(diagnostics_result.get('n_rows'))}"
        ),
        (
            f"- **Diagnostics years:** "
            f"{diagnostics_result.get('year_min')}-"
            f"{diagnostics_result.get('year_max')}"
        ),
        (
            f"- **Median APE:** "
            f"{_fmt_fraction_percent(diagnostics_result.get('median_ape'))}"
        ),
        (
            f"- **Mean APE:** "
            f"{_fmt_fraction_percent(diagnostics_result.get('mean_ape'))}"
        ),
        (
            f"- **P90 APE:** "
            f"{_fmt_fraction_percent(diagnostics_result.get('p90_ape'))}"
        ),
        "",
    ]


def _build_corrected_trade_section(
    *,
    corrected_trade_report: dict[str, Any],
    year_figure: Path | None,
    value_band_figure: Path | None,
    report_path: Path,
) -> list[str]:
    """Build the applied-correction section from prepared reporting material."""
    
    from textwrap import dedent
    equation = dedent(
        r"""
        $$
        V^{\mathrm{import,FOB\text{-}corrected}}_{ijpt}
        =
        \frac{
            V^{\mathrm{import,CIF}}_{ijpt}
        }{
            F^{\mathrm{CIF/FOB}}_{ijpt}
        }
        $$
        """
    ).strip()
    
    lines = [
        "## 10. Applied CIF/FOB correction",
        "",
        (
            "Observed direct CIF/FOB factors are used as empirical evidence "
            "for model training and validation. They are not directly applied "
            "to production rows; FOB-equivalent importer values are generated "
            "only from the selected model's estimated CIF/FOB factor."
        ),
        "",
        "",
        equation,
        "",
        "### 10.1 Application audit",
        "",
        corrected_trade_report["audit_markdown"],
        "",
        "### 10.2 Overall application summary",
        "",
        corrected_trade_report["summary_markdown"],
        "",
        (
            f"The uncorrected population contains "
            f"{_fmt_int(corrected_trade_report['inspection']['overall']['n_uncorrected_rows'])} "
            "rows and consists of two fundamentally different cases. "
            "Export-only groups have no importer CIF value and therefore require "
            "no CIF-to-FOB conversion. The remaining importer-containing rows "
            "could not be corrected because no estimated factor was available."
        ),
        "",
        corrected_trade_report[
            "application_outcomes_markdown"
        ],
        "",
        "#### Prediction coverage behind the correction",
        "",
        (
            "The following diagnostic explains why an estimated factor was "
            "unavailable for some importer observations. Pair geography and "
            "complete initial-predictor coverage are identical, showing that "
            "missing bilateral geography is the binding limitation."
        ),
        "",
        corrected_trade_report[
            "predictor_diagnostic_markdown"
        ],
        "",
        "### 10.3 Application by year",
        "",
        corrected_trade_report["by_year_markdown"],
        "",
    ]

    if year_figure is not None:
        lines.extend(
            [
                markdown_image(
                    "Applied CIF/FOB correction by year",
                    year_figure,
                    document_path=report_path,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "### 10.4 Application by evidence-value band",
            "",
            (
                "This breakdown includes only rows with a non-missing "
                "`estimation_evidence_value`. It is therefore not a partition "
                "of the full corrected-trade population. Correction coverage "
                "and value reduction measure different quantities: the former "
                "is a row share, while the latter is a value-weighted change."
            ),
            "",
            corrected_trade_report["by_value_band_markdown"],
            "",
        ]
    )

    if value_band_figure is not None:
        lines.extend(
            [
                markdown_image(
                    "Applied CIF/FOB correction by evidence-value band",
                    value_band_figure,
                    document_path=report_path,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "### 10.5 Prediction-interval calibration",
            "",
            corrected_trade_report["uncertainty_markdown"],
            "",
            (
                "The 80% and 90% intervals are calibrated from absolute "
                "log-factor errors in the latest temporal holdout. They "
                "represent marginal empirical coverage rather than a "
                "row-specific probability that an estimate is correct."
            ),
            "",
            "### 10.6 Trade Sync readiness and residual mirror disagreement",
            "",
            (
                "For downstream Trade Sync, the full population of "
                f"{_fmt_int(corrected_trade_report['trade_sync_total_rows'])} "
                "rows is considered. A usable report is defined as a "
                "positive, finite value. Structural report presence is therefore "
                "kept separate from usable-value availability."
            ),
            "",
            "#### Post-correction bilateral availability",
            "",
            (
                "The following four mutually exclusive categories partition the full "
                "population of 5,487,981 bilateral "
                "`exporter–importer–product–year` rows. A usable value is defined as "
                "positive, finite, and non-missing."
            ),
            "",
            corrected_trade_report[
                "trade_sync_availability_markdown"
            ],
            "",
            "##### Breakdown of rows with neither usable value",
            "",
            (
                "The percentages in the following table are calculated relative to "
                "the 90,726 rows classified as neither usable, not relative to the "
                "full bilateral population."
            ),
            "",
            corrected_trade_report[
                "neither_usable_breakdown_markdown"
            ],
            "",
            (
                "Almost all neither-usable rows have no exporter FOB value and no "
                "estimated CIF/FOB factor for the importer side. The remaining seven "
                "rows are edge cases involving missing, zero, or negative values."
            ),
            "",
            "#### Structural and usable-value coverage",
            "",
            corrected_trade_report[
                "trade_sync_coverage_markdown"
            ],
            "",
            "#### Importer/exporter value direction before and after correction",
            "",
            (
                "This table compares the importer-reported value with the "
                "exporter-reported FOB value for the same paired rows before and "
                "after CIF/FOB correction. Before correction, the importer value is "
                "the reported CIF value; after correction, it is the estimated "
                "FOB-equivalent value."
            ),
            "",
            corrected_trade_report[
                "direction_before_after_markdown"
            ],
            "",
            (
                "Rows where the importer value was already below the exporter FOB "
                "value cannot be reconciled through a standard CIF-to-FOB adjustment, "
                "because applying a factor greater than or equal to one lowers the "
                "importer value further."
            ),
            "",
            "#### Exporter/importer agreement before and after correction",
            "",
            corrected_trade_report[
                "trade_sync_agreement_markdown"
            ],
            "",
            corrected_trade_report[
                "trade_sync_change_markdown"
            ],
            "",
            (
                "CIF/FOB correction places importer-reported values on an "
                "FOB-equivalent valuation basis. The before/after "
                "comparison therefore measures how much bilateral disagreement "
                "remains."
            ),
            "",
            "The bilateral value-disagreement factor is defined as:",
            "",
            "$$",
            r"D_{ijpt}",
            r"=",
            r"\frac{",
            r"\max\left(",
            r"V^{\mathrm{export,FOB}}_{ijpt},",
            r"V^{\mathrm{import,FOB\text{-}corrected}}_{ijpt}",
            r"\right)",
            r"}{",
            r"\min\left(",
            r"V^{\mathrm{export,FOB}}_{ijpt},",
            r"V^{\mathrm{import,FOB\text{-}corrected}}_{ijpt}",
            r"\right)",
            r"}",
            "$$",
            "",
            (
                "where $D_{ijpt} \\geq 1$. A value of $D_{ijpt}=1$ indicates "
                "perfect agreement, while $D_{ijpt}=2$ means that one reported "
                "bilateral trade value is twice the other. The maximum is "
                "always in the numerator and the minimum is always in the "
                "denominator, so the disagreement factor cannot fall below 1."
            ),
            "",
            "#### Residual agreement bands after correction",
            "",
            corrected_trade_report[
                "trade_sync_bands_markdown"
            ],
            "",
            "#### Trade Sync availability by year",
            "",
            corrected_trade_report[
                "trade_sync_by_year_markdown"
            ],
            "",
            (
                "A low both-report share in the latest year may reflect "
                "incomplete reporting rather than failure of the CIF/FOB "
                "correction. Downstream Trade Sync should preserve the year "
                "dimension when estimating reporter reliability."
            ),
            "",
        ]
    )

    return lines


# ======================================================================
# Report writer
# ======================================================================


def write_cif_fob_decision_report(
    *,
    cfg: CifFobConfig,
    run_id: str,
    training_features_result: dict[str, Any],
    estimated_rates_result: dict[str, Any],
    diagnostics_result: dict[str, Any] | None = None,
    training_evidence_audit_result: dict[str, Any] | None = None,
    corrected_trade_result: dict[str, Any],
    logger,
) -> dict[str, Any]:
    """
    Write the timestamped and latest CIF/FOB Markdown decision report.
    """
    if not cfg.artifacts.enable:
        return {
            "status": "disabled",
            "report_path": None,
            "latest_report_path": None,
        }

    training_features_path = training_features_result.get("training_features_path")
    estimated_rates_path = Path(estimated_rates_result["estimated_rates_path"])

    run_data_dir = estimated_rates_path.parent

    figure_dir = _resolve_cif_fob_figure_dir(
        cfg=cfg,
        run_data_dir=run_data_dir,
    )
    if not isinstance(training_features_path, Path):
        training_features_path = cfg.product_path(
            "training_features"
        )

    if not isinstance(estimated_rates_path, Path):
        estimated_rates_path = cfg.product_path(
            "estimated_rates"
        )

    estimated_rates_path = Path(
        estimated_rates_result["estimated_rates_path"]
    )

    run_data_dir = estimated_rates_path.parent

    summary_path = Path(
        estimated_rates_result.get(
            "temporal_validation_summary_path",
            run_data_dir
            / cfg.product_path(
                "temporal_validation_summary"
            ).name,
        )
    )

    folds_path = Path(
        estimated_rates_result.get(
            "temporal_validation_folds_path",
            run_data_dir
            / cfg.product_path(
                "temporal_validation_folds"
            ).name,
        )
    )

    population_metrics = _inspect_population_journey(
        cfg=cfg,
    )
    population_journey_table = _build_population_journey_table(
        population_metrics
    )
    population_journey_table = population_journey_table.rename(
        columns={
            "Rows/groups reduced": "Rows reduction",
            "Reduction from previous stage": "% Reduction",
            "Transition type": "Type",
        }
    )

    ## Get product-median threshold sensitivity metrics for the configured thresholds
    product_median_threshold_sensitivity = _inspect_product_median_threshold_sensitivity(
            path=training_features_path,
            threads=cfg.engine.threads,
            thresholds=PRODUCT_MEDIAN_THRESHOLD_BINS
        )
    
    configured_product_minimum = int(cfg.features.product.median_unit_value.minimum_observations)
    product_median_threshold_sensitivity["Setting"] = product_median_threshold_sensitivity[
                                "Minimum observations"
                            ].eq(configured_product_minimum).map(
                                {
                                    True: "Configured",
                                    False: "",
                                }
                            )



    feature_metrics = _inspect_training_features(
        path=training_features_path,
        threads=cfg.engine.threads,
    )
    product_unit_value_sources = (_inspect_product_median_unit_value_sources(
        path=training_features_path,
        threads=cfg.engine.threads,
    )
    )
    direct_predictor_failures = _inspect_direct_predictor_failures(
        path=training_features_path,
        threads=cfg.engine.threads,
    )
    rate_metrics = _inspect_estimated_rates(
        path=estimated_rates_path,
        threads=cfg.engine.threads,
    )
    summary, folds = _load_validation_artifacts(
        summary_path=summary_path,
        folds_path=folds_path,
    )

    predictor_sets_cfg = getattr(
        cfg.method.estimation,
        "predictor_sets",
        {},
    )
    enabled_predictor_sets = _enabled_predictor_set_names(cfg)

    if predictor_sets_cfg:
        summary = _filter_to_enabled_predictor_sets(
            summary,
            enabled_predictor_sets=enabled_predictor_sets,
        )
        folds = _filter_to_enabled_predictor_sets(
            folds,
            enabled_predictor_sets=enabled_predictor_sets,
        )

    direct_factor_distribution_df = (
        _load_direct_factor_distribution_data(
            path=training_features_path,
            threads=cfg.engine.threads,
        )
    )

    direct_factor_distribution_metrics = (
        _summarize_direct_factor_distribution(
            df=direct_factor_distribution_df,
        )
    )

    figure_dir = _resolve_cif_fob_figure_dir(cfg=cfg,
                    run_data_dir=run_data_dir)
    figure_paths: dict[str, Path] = {}

    # Prepare Hold-out diagnostic material and figures.
    diagnostics_status = (
        diagnostics_result.get("status")
        if diagnostics_result is not None
        else None
    )

    if diagnostics_status in {
        "completed",
        "reused",
    }:
        logger.info(
            "PLOTS — Writing primary-model holdout "
            "diagnostic figures (status=%s)",
            diagnostics_status,
        )

        figure_paths.update(
            _write_primary_diagnostics_figures(
                cfg=cfg,
                diagnostics_result=diagnostics_result,
                figure_dir=figure_dir,
                logger=logger,
            )
        )
        figure_paths.update(
            _write_primary_observed_predicted_figures(
                cfg=cfg,
                diagnostics_result=diagnostics_result,
                figure_dir=figure_dir,
                logger=logger,
            )
        )
    else:
        logger.info(
            "PLOTS — Skipping primary-model holdout "
            "diagnostic figures (status=%s)",
            diagnostics_status or "no result",
        )

    # Prepare corrected-trade report material and figures.
    corrected_trade_report = (
            _prepare_corrected_trade_report(
                    cfg=cfg,
                    corrected_trade_result=corrected_trade_result,
                    figure_dir=figure_dir,
                    logger=logger,
                )
            )

    figure_paths.update(corrected_trade_report["figure_paths"])

    corrected_trade_year_figure = figure_paths.get(
        "corrected_trade_by_year_comparison"
    )

    corrected_trade_value_band_figure = figure_paths.get(
        "corrected_trade_by_value_band_comparison"
    )

    # Write training-evidence audit figures when available.
    training_evidence_status = (
        training_evidence_audit_result.get("status")
        if training_evidence_audit_result is not None
        else None
    )

    if training_evidence_status in {"completed", "reused"}:
        logger.info(
            "PLOTS — Writing training-evidence audit figures "
            "(status=%s)",
            training_evidence_status,
        )

        figure_paths.update(
            _write_training_evidence_audit_figures(
                cfg=cfg,
                training_evidence_audit_result=training_evidence_audit_result,
                figure_dir=figure_dir / "training_evidence_audit",
                logger=logger,
            )
        )
    else:
        logger.info(
            "PLOTS — Skipping training-evidence audit figures "
            "(status=%s)",
            training_evidence_status or "no result",
        )

    
    # Write value-band model-comparison figures.
    latest_holdout_year = int(
    pd.to_numeric(
        folds["validation_year"],
        errors="coerce",
    ).max()
    )

    prediction_columns_by_run = estimated_rates_result.get(
        "prediction_columns_by_run",
        {},
    )

    if not prediction_columns_by_run:
        raise RuntimeError(
            "estimated_rates_result does not contain "
            "'prediction_columns_by_run'."
        )

    (
    value_band_comparison,
    value_band_comparison_long,
    value_band_figure_paths,
    ) = _write_value_band_model_comparison_figures(
        cfg=cfg,
        estimated_rates_path=estimated_rates_path,
        folds=folds,
        prediction_columns_by_run=prediction_columns_by_run,
        figure_dir=figure_dir,
        logger=logger,
    )

    figure_paths.update(
        value_band_figure_paths
    )

    #############################################################################
    # FIGURE PLOTTING
    #############################################################################

    ## 1. Plot the distribution of directly observed CIF/FOB factors
    fig, _, _ = plot_nebula_boxplot(
        direct_factor_distribution_df,
        factor_col="observed_cif_fob_factor",
        upper_display=float(
            cfg.method.estimation.maximum_factor
        ),
        lower_display=0.0,
        title=(
            "Distribution of directly observed CIF/FOB factors"
        ),
    )

    direct_factor_distribution_figure = _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="input_diagnostics",
        target_hs=cfg.scope.target_hs,
        stem="direct_cif_fob_factor_distribution",
        figure_paths=figure_paths,
        key="direct_factor_distribution",
        logger=logger
    )

    ## 2. Plot the model ranking by mean log-factor MAE
    primary_run_name = str(cfg.method.estimation.models.primary).strip().lower() 
    fig, _, _ = plot_model_metric_ranking(
        summary,
        metric="mean_mae_log_factor",
        primary_run_name=None,
        top_n=18,
        higher_is_better=False,
        title="CIF/FOB model ranking by mean log MAE",
        metric_label="Mean log MAE",
    )
    
    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_model_ranking_mean_log_mae",
        figure_paths=figure_paths,
        key="model_ranking_mean_log_mae",
        logger=logger,
    )

    ## 3. Plot the model ranking by median APE
    fig, _, _ = plot_model_metric_ranking(
        summary,
        metric="mean_median_ape",
        primary_run_name=primary_run_name,
        top_n=18,
        higher_is_better=False,
        title="CIF/FOB model ranking by median APE",
        metric_label="Median APE",
    )
    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_model_ranking_median_ape",
        figure_paths=figure_paths,
        key="model_ranking_median_ape",
        logger=logger,
    )


    ## 4. Plot the model ranking by RMSE
    fig, _, _ = plot_model_metric_ranking(
        summary,
        metric="mean_rmse_log_factor",
        primary_run_name=primary_run_name,
        top_n=18,
        higher_is_better=False,
        title="CIF/FOB model ranking by RMSE-Log",
        metric_label="RMSE-Log",
    )
    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_model_ranking_rmse",
        figure_paths=figure_paths,
        key="model_ranking_rmse",
        logger=logger,
    )

    ## 4. Plot the model ranking by R2 
    fig, _, _ = plot_model_metric_ranking(
        summary,
        metric="mean_r2_log_factor",
        primary_run_name=primary_run_name,
        top_n=18,
        higher_is_better=True,
        title="CIF/FOB model ranking by R2-Log",
        metric_label="R2-Log",
    )
    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_model_ranking_r2",
        figure_paths=figure_paths,
        key="model_ranking_r2",
        logger=logger,
    )

    ## 5. Plot the model ranking by upper-tail median APE
    highlight_runs = {
        "hist_gradient_boosting__model_5_full_history": ("Primary: HGB M5"),
        "hist_gradient_boosting_squared__model_5_full_history": ("Tail challenger: HGB² M5"),
        "log_linear_regression__model_5_full_history": ("OLS baseline: M5"),
    }

    fig, _, _ = plot_central_vs_tail_tradeoff(
        summary,
        x_metric="mean_median_ape",
        y_metric="mean_upper_tail_median_ape",
        primary_run_name="hist_gradient_boosting__model_5_full_history",
        highlight_runs=highlight_runs,
        x_label="Median APE",
        y_label="Upper-tail median APE",
        title="CIF/FOB central accuracy vs upper-tail accuracy",
    )
    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_central_vs_tail_tradeoff",
        figure_paths=figure_paths,
        key="central_vs_tail_tradeoff",
        logger=logger,
    )


    ## 6.  Plot the temporal validation stability of the primary model and a few other candidates
    selected_runs = [
        "hist_gradient_boosting__model_5_full_history",
        "hist_gradient_boosting_squared__model_5_full_history",
        "log_linear_regression__model_5_full_history",]

    fig, _, _ = plot_temporal_validation_metric(
        folds,
        metric="median_ape",
        selected_runs=selected_runs,
        primary_run_name=primary_run_name,
        metric_label="Median APE",
        title="CIF/FOB temporal validation stability",
    )

    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="temporal_validation",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_temporal_validation_median_ape",
        figure_paths=figure_paths,
        key="temporal_validation_median_ape",
        logger=logger,
    )

    ## 7. Plot Observed vs Predicted P95 for the primary model and a few other candidates
    fig, _, _ = plot_observed_vs_predicted_p95(
        summary,
        primary_run_name=primary_run_name,
        top_n=12,
        ranking_metric="mean_mae_log_factor",
        observed_column="mean_observed_p95",
        predicted_column="mean_predicted_p95",
        title="CIF/FOB observed vs predicted upper-tail factor",
    )
    _save_cif_fob_figure(
        fig=fig,
        root_dir=figure_dir,
        category="model_comparison",
        target_hs=cfg.scope.target_hs,
        stem="cif_fob_observed_vs_predicted_p95",
        figure_paths=figure_paths,
        key="observed_vs_predicted_p95",
        logger=logger
    )

    #####################################################################################

    ranked = _rank_models(summary)
    model_display_names = {
        "hist_gradient_boosting": "HGB absolute-log",
        "hist_gradient_boosting_squared": "HGB squared-log",
        "hist_gradient_boosting_squared_level": "HGB squared-level",
        "hist_gradient_boosting_squared_regularized": (
            "HGB squared-log regularized"
        ),
        "xgboost_log_squared": "XGBoost squared-log",
        "log_linear_regression": "Log-linear OLS",
        "log_linear_regression_smearing": "Log-linear OLS smearing",
    }

    if "model_algorithm" not in ranked.columns:
        ranked["model_algorithm"] = (
            ranked["model_name"]
            .astype(str)
            .str.split("__", n=1)
            .str[0]
        )

    if "predictor_set_name" not in ranked.columns:
        split_parts = (
            ranked["model_name"]
            .astype(str)
            .str.split("__", n=1, expand=True)
        )
        ranked["predictor_set_name"] = (
            split_parts[1]
            if split_parts.shape[1] > 1
            else ""
        )

    ranked["algorithm_display_name"] = (
        ranked["model_algorithm"]
        .map(model_display_names)
        .fillna(ranked["model_algorithm"])
    )

    ranked["model_display_name"] =  ranked["algorithm_display_name"].astype(str)

    primary_algorithm = str(
        cfg.method.estimation.models.primary
    ).strip().lower()

    primary_predictor_set = str(
        getattr(
            cfg.method.estimation,
            "primary_predictor_set",
            "",
        )
        or ""
    ).strip()

    primary_model = (
        f"{primary_algorithm}__{primary_predictor_set}"
        if primary_predictor_set
        else primary_algorithm
    )

    if primary_model not in set(ranked["model_name"].astype(str)):
        primary_model = primary_algorithm

    enabled_models = dict(
        cfg.method.estimation.models.enabled()
    )

    primary_folds = folds.loc[
        folds["model_name"].astype(str).eq(primary_model)
    ].copy()

    report_path = cfg.artifact_path(run_id)
    latest_path = cfg.artifact_latest_path()

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_rows = pd.DataFrame(
        [
            {
                "Reference role": "Bilateral geography",
                "Path": str(
                    cfg.features.pair_geography.resolve_path()
                ),
            },
            {
                "Reference role": "Country characteristics",
                "Path": str(
                    cfg.features.geography.resolve_path()
                ),
            },
            {
                "Reference role": "Pair geography crosswalk",
                "Path": str(
                    cfg.features.pair_geography_crosswalk.resolve_path()
                ),
            },
            {
                "Reference role": "Country geography crosswalk",
                "Path": str(
                    cfg.features.geography_crosswalk.resolve_path()
                ),
            },
            {
                "Reference role": "Annual oil price",
                "Path": str(
                    cfg.features.oil_price.resolve_path()
                ),
            },
        ]
    )

    model_rows = pd.DataFrame(
        [
            {
                "Model": model_display_names.get(name, name),
                "Config class": type(model_cfg).__name__,
                "Enabled": bool(model_cfg.enabled),
                "Primary": name == primary_model,
                "Target transformation": getattr(
                    model_cfg,
                    "target_transformation",
                    getattr(
                        model_cfg,
                        "target_transformation",
                        "log",
                    ),
                ),
                "Loss/objective": getattr(
                    model_cfg,
                    "loss",
                    getattr(
                        model_cfg,
                        "objective",
                        getattr(
                            model_cfg,
                            "estimator_method",
                            "n/a",
                        ),
                    ),
                ),
            }
            for name, model_cfg in enabled_models.items()
        ]
    )

    if predictor_sets_cfg:
        predictor_set_rows = pd.DataFrame(
            [
                {
                    "Predictor set": name,
                    "Enabled": bool(getattr(predictor_set, "enabled", False)),
                    "Numeric predictors": ", ".join(
                        getattr(predictor_set, "numeric", [])
                    ),
                    "Categorical predictors": ", ".join(
                        getattr(predictor_set, "categorical", [])
                    ),
                    "Predictor count": (
                        len(getattr(predictor_set, "numeric", []))
                        + len(getattr(predictor_set, "categorical", []))
                    ),
                    "Primary": name == primary_predictor_set,
                }
                for name, predictor_set in predictor_sets_cfg.items()
            ]
        )
    else:
        predictor_set_rows = pd.DataFrame(
            [
                {
                    "Predictor set": "global",
                    "Enabled": True,
                    "Numeric predictors": ", ".join(
                        cfg.method.predictors.all()
                    ),
                    "Categorical predictors": ", ".join(
                        sorted(
                            {
                                value
                                for model_cfg in enabled_models.values()
                                for value in getattr(
                                    model_cfg,
                                    "categorical_predictors",
                                    (),
                                )
                            }
                        )
                    ),
                    "Predictor count": len(
                        cfg.method.predictors.all()
                    ),
                    "Primary": True,
                }
            ]
        )

    predictor_set_details_rows = predictor_set_rows.loc[
        predictor_set_rows["Enabled"].astype(bool)
    ].copy()

    coverage_rows = pd.DataFrame(
        [
            {
                "Feature area": "Pair geography",
                "Rows found": feature_metrics[
                    "n_pair_geography_found"
                ],
                "Coverage": feature_metrics[
                    "pair_geography_coverage_percent"
                ],
            },
            {
                "Feature area": "Exporter geography",
                "Rows found": feature_metrics[
                    "n_exporter_geography_found"
                ],
                "Coverage": feature_metrics[
                    "exporter_geography_coverage_percent"
                ],
            },
            {
                "Feature area": "Importer geography",
                "Rows found": feature_metrics[
                    "n_importer_geography_found"
                ],
                "Coverage": feature_metrics[
                    "importer_geography_coverage_percent"
                ],
            },
            {
                "Feature area": "Oil price",
                "Rows found": feature_metrics[
                    "n_oil_price_found"
                ],
                "Coverage": feature_metrics[
                    "oil_price_coverage_percent"
                ],
            },
            {
                "Feature area": "Product feature",
                "Rows found": feature_metrics[
                    "n_product_feature_found"
                ],
                "Coverage": feature_metrics[
                    "product_feature_coverage_percent"
                ],
            },
            {
                "Feature area": "Complete initial predictors",
                "Rows found": feature_metrics[
                    "n_initial_predictors_complete"
                ],
                "Coverage": feature_metrics[
                    "initial_predictor_coverage_percent"
                ],
            },
            {
                "Feature area": (
                    "Complete predictors among selected-factor candidates"
                ),
                "Rows found": feature_metrics[
                    "n_factor_candidates_predictors_complete"
                ],
                "Coverage": feature_metrics[
                    "factor_candidate_predictor_coverage_percent"
                ],
            },
        ]
    )

    training_factor_sources = [
        str(source).strip().lower()
        for source in cfg.method.estimation.training_factor_sources
    ]

    paired_population = int(
        feature_metrics["n_rows"]
    )

    selected_direct = int(
        feature_metrics["n_selected_direct_factors"]
    )

    selected_mirror = int(
        feature_metrics["n_selected_mirror_factors"]
    )

    selected_total = int(
        feature_metrics["n_factor_candidates"]
    )

    no_selected_factor = (
        paired_population
        - selected_total
    )


    source_selection_rows = pd.DataFrame(
        [
            {
                "Evidence population": "Selected direct factors",
                "Rows": selected_direct,
                "Share of paired groups": (
                    selected_direct / paired_population
                    if paired_population
                    else 0.0
                ),
                "Share of selected-factor pool": (
                    selected_direct / selected_total
                    if selected_total
                    else 0.0
                ),
                "Meaning": (
                    "Direct importer CIF/FOB factor selected as the "
                    "observed target."
                ),
            },
            {
                "Evidence population": "Selected mirror factors",
                "Rows": selected_mirror,
                "Share of paired groups": (
                    selected_mirror / paired_population
                    if paired_population
                    else 0.0
                ),
                "Share of selected-factor pool": (
                    selected_mirror / selected_total
                    if selected_total
                    else 0.0
                ),
                "Meaning": (
                    "Mirror factor retained where an eligible direct "
                    "factor was not selected."
                ),
            },
            {
                "Evidence population": (
                    "All rows with a selected observed-factor candidate"
                ),
                "Rows": selected_total,
                "Share of paired groups": (
                    selected_total / paired_population
                    if paired_population
                    else 0.0
                ),
                "Share of selected-factor pool": 1.0,
                "Meaning": (
                    "Internal `model_eligible` pool; not yet the model "
                    "training sample."
                ),
            },
            {
                "Evidence population": "No selected observed factor",
                "Rows": no_selected_factor,
                "Share of paired groups": (
                    no_selected_factor / paired_population
                    if paired_population
                    else 0.0
                ),
                "Share of selected-factor pool": np.nan,
                "Meaning": (
                    "No usable observed CIF/FOB target was selected. "
                    "These rows may still receive model predictions when "
                    "their predictors are complete."
                ),
            },
        ]
    )

    estimation_funnel_rows = pd.DataFrame(
        [
            {
                "Cumulative estimation stage": (
                    "Selected observed-factor candidates"
                ),
                "Rows remaining": rate_metrics[
                    "n_factor_candidates"
                ],
            },
            {
                "Cumulative estimation stage": (
                    "After restricting training to direct factors"
                ),
                "Rows remaining": rate_metrics[
                    "n_selected_direct_factors"
                ],
            },
            {
                "Cumulative estimation stage": (
                    "After requiring all initial predictors"
                ),
                "Rows remaining": rate_metrics[
                    "n_direct_after_predictor_completeness"
                ],
            },
            {
                "Cumulative estimation stage": (
                    "After requiring minimum evidence value"
                ),
                "Rows remaining": rate_metrics[
                    "n_after_minimum_value_rule"
                ],
            },
            {
                "Cumulative estimation stage": (
                    "After CIF/FOB lower factor bound"
                ),
                "Rows remaining": rate_metrics[
                    "n_after_lower_factor_bound"
                ],
            },
            {
                "Cumulative estimation stage": (
                    "After CIF/FOB upper factor bound"
                ),
                "Rows remaining": rate_metrics[
                    "n_after_upper_factor_bound"
                ],
            },
            {
                "Cumulative estimation stage": (
                    "After proxy-weight rule"
                ),
                "Rows remaining": rate_metrics[
                    "n_after_proxy_rule"
                ],
            },
            {
                "Cumulative estimation stage": (
                    "Final clean observations used for fitting/backtesting"
                ),
                "Rows remaining": rate_metrics[
                    "n_clean_estimation_observations"
                ],
            },
        ]
    )

    estimation_funnel_rows["Rows remaining"] = pd.to_numeric(
        estimation_funnel_rows["Rows remaining"],
        errors="coerce",
    ).fillna(0).astype(int)

    previous_rows = estimation_funnel_rows[
        "Rows remaining"
    ].shift(1)

    estimation_funnel_rows["Rows removed at stage"] = (
        previous_rows
        - estimation_funnel_rows["Rows remaining"]
    )

    estimation_funnel_rows.loc[
        estimation_funnel_rows.index[0],
        "Rows removed at stage",
    ] = 0

    estimation_funnel_rows["Data lost at stage"] = np.where(
        previous_rows.gt(0),
        (
            estimation_funnel_rows["Rows removed at stage"]
            / previous_rows
        ),
        0.0,
    )


    direct_candidates = int(
        rate_metrics["n_selected_direct_factors"] or 0
    )
    direct_complete = int(
        rate_metrics["n_direct_after_predictor_completeness"] or 0
    )
    direct_predictor_loss = (
        direct_candidates - direct_complete
    )
    direct_predictor_loss_share = (
        direct_predictor_loss / direct_candidates
        if direct_candidates
        else 0.0
    )

    overall_table = ranked.rename(
        columns={
            "overall_rank": "Rank",
            "model_display_name": "Model",
            "model_algorithm": "Algorithm",
            "predictor_set_name": "Predictor set",
            "mean_mae_log_factor": "Mean log MAE",
            "mean_rmse_log_factor": "RMSE log",
            "mean_r2_log_factor": "R² log",
            "mean_median_ape": "Median APE",
            "mean_mean_ape": "Mean APE",
            "mean_p90_ape": "P90 APE",
            "mean_p95_ape": "P95 APE",
            "mean_spearman_correlation": "Spearman",
        }
    )

    tail_table = ranked.rename(
        columns={
            "overall_rank": "Rank",
            "model_display_name": "Model",
            "predictor_set_name": "Predictor set",
            "mean_observed_p95": "Observed P95",
            "mean_predicted_p95": "Predicted P95",
            "mean_upper_tail_median_ape": "Tail median APE",
            "mean_upper_tail_p90_ape": "Tail P90 APE",
        }
    )

    combined_table = overall_table.merge(
        tail_table[
            [
                "Rank",
                "Model",
                "Predictor set",
                "Observed P95",
                "Predicted P95",
                "Tail median APE",
                "Tail P90 APE",
            ]
        ],
        on=["Rank", "Model","Predictor set"],
        how="left",
    )

    value_band_comparison_table = value_band_comparison.copy()

    primary_folds_table = primary_folds.sort_values(
        "validation_year"
    ).rename(
        columns={
            "validation_year": "Year",
            "n_train": "Train rows",
            "n_holdout": "Holdout rows",
            "mae_log_factor": "Mean log MAE",
            "rmse_log_factor": "RMSE log",
            "r2_log_factor": "R² log",
            "mean_rmse_log_factor": "RMSE log",
            "mean_r2_log_factor": "R² log",
            "median_ape": "Median APE",
            "mean_ape": "Mean APE",
            "p90_ape": "P90 APE",
            "p95_ape": "P95 APE",
            "spearman_correlation": "Spearman",
            "observed_p95": "Observed P95",
            "predicted_p95": "Predicted P95",
            "upper_tail_median_ape": "Tail median APE",
            "upper_tail_p90_ape": "Tail P90 APE",
            "n_upper_tail": "Tail rows",
        }
    )

    fold_columns = [
        "Year",
        "Train rows",
        "Holdout rows",
        "Mean log MAE",
        "RMSE log",
        "R² log",
        "Median APE",
        "Mean APE",
        "P90 APE",
        "P95 APE",
        "Spearman",
        "Observed P95",
        "Predicted P95",
        "Tail median APE",
        "Tail P90 APE",
        "Tail rows",
    ]

    fold_formatters = {
        "Year": lambda value: ("n/a" if value is None or pd.isna(value) else str(int(value))),
        "Train rows": _fmt_int,
        "Holdout rows": _fmt_int,
        "Mean log MAE": _fmt_float,
        "RMSE log": _fmt_float,
        "R² log": _fmt_float,
        "Median APE": _fmt_fraction_percent,
        "Mean APE": _fmt_fraction_percent,
        "P90 APE": _fmt_fraction_percent,
        "P95 APE": _fmt_fraction_percent,
        "Spearman": _fmt_float,
        "Observed P95": _fmt_float,
        "Predicted P95": _fmt_float,
        "Tail median APE": _fmt_fraction_percent,
        "Tail P90 APE": _fmt_fraction_percent,
        "Tail rows": _fmt_int,
    }

    best_mae = _best_model(
        ranked,
        metric="mean_mae_log_factor",
    )
    best_rmse = _best_model(
        ranked,
        metric="mean_rmse_log_factor",
    )

    best_r2 = _best_model(
        ranked,
        metric="mean_r2_log_factor",
        higher_is_better=True,
    )
    best_median_ape = _best_model(
        ranked,
        metric="mean_median_ape",
    )
    best_p90 = _best_model(
        ranked,
        metric="mean_p90_ape",
    )
    best_p95 = _best_model(
        ranked,
        metric="mean_p95_ape",
    )
    best_spearman = _best_model(
        ranked,
        metric="mean_spearman_correlation",
        higher_is_better=True,
    )
    best_tail = _best_model(
        ranked,
        metric="mean_upper_tail_median_ape",
    )

    predictor_set_detail_lines: list[str] = []

    for _, row in predictor_set_details_rows.iterrows():
        predictor_set_detail_lines.extend(
            [
                f"#### `{row['Predictor set']}`",
                "",
                f"- **Primary:** `{row['Primary']}`",
                f"- **Predictor count:** {_fmt_int(row['Predictor count'])}",
                f"- **Categorical predictors:** "
                f"{row['Categorical predictors'] or 'none'}",
                "- **Numeric predictors:**",
                *[
                    f"  - `{predictor.strip()}`"
                    for predictor in str(
                        row["Numeric predictors"]
                    ).split(",")
                    if predictor.strip()
                ],
                "",
            ]
        )

    if not predictor_set_detail_lines:
        predictor_set_detail_lines = [
            "_No enabled predictor sets configured._",
            "",
        ]

    paired_groups = int(
        population_metrics["n_bilateral_mirror_groups"]
    )

    bilateral_reporting_rows = pd.DataFrame(
        [
            {
                "Reporting case": "Both importer and exporter reports",
                "Groups": population_metrics[
                    "n_groups_with_both_reports"
                ],
                "Share of paired groups": (
                    population_metrics["n_groups_with_both_reports"]
                    / paired_groups
                ),
                "Potential observed-factor evidence": (
                    "Direct or mirror"
                ),
            },
            {
                "Reporting case": "Import-only",
                "Groups": population_metrics[
                    "n_import_only_groups"
                ],
                "Share of paired groups": (
                    population_metrics["n_import_only_groups"]
                    / paired_groups
                ),
                "Potential observed-factor evidence": (
                    "Direct only"
                ),
            },
            {
                "Reporting case": "Export-only",
                "Groups": population_metrics[
                    "n_export_only_groups"
                ],
                "Share of paired groups": (
                    population_metrics["n_export_only_groups"]
                    / paired_groups
                ),
                "Potential observed-factor evidence": (
                    "Neither direct nor mirror"
                ),
            },
        ]
    )

    direct_structural_share = (
        population_metrics["n_groups_with_both_reports"]
        + population_metrics["n_import_only_groups"]
    ) / paired_groups

    mirror_structural_share = (
        population_metrics["n_groups_with_both_reports"]
        / paired_groups
    )

    maximum_factor_rule = cfg.method.estimation.maximum_factor_rule

    if maximum_factor_rule.method == "iqr_cap":
        resolved_maximum_factor = (
            float(direct_factor_distribution_metrics["q3"])
            + float(maximum_factor_rule.iqr_multiplier)
            * float(direct_factor_distribution_metrics["iqr"])
        )
    else:
        resolved_maximum_factor = None

    upper_bound_removed = (
        int(rate_metrics["n_after_lower_factor_bound"])
        - int(rate_metrics["n_after_upper_factor_bound"])
    )

    upper_bound_population = int(
        rate_metrics["n_after_lower_factor_bound"]
    )

    primary_diagnostics_section = _build_primary_diagnostics_section(
        diagnostics_result=diagnostics_result,
    )

    corrected_trade_section = _build_corrected_trade_section(
        corrected_trade_report=corrected_trade_report,
        year_figure=corrected_trade_year_figure,
        value_band_figure=corrected_trade_value_band_figure,
        report_path=report_path,
    )

    lines = [
        f"# VACODA CIF/FOB Decision Report — {cfg.scope.target_hs}",
        "",
        "## 1. Run overview",
        "",
        f"- **Run ID:** `{run_id}`",
        f"- **Target HS:** `{cfg.scope.target_hs}`",
        (
            f"- **Configured year range:** "
            f"{cfg.scope.min_year}–{cfg.scope.max_year}"
        ),
        f"- **Primary model run:** `{primary_model}`",
        f"- **Primary algorithm:** `{primary_algorithm}`",
        (
            f"- **Primary predictor set:** `{primary_predictor_set}`"
            if primary_predictor_set
            else "- **Primary predictor set:** `global`"
        ),
        "- **Enabled candidate models:**",
        *[
            f"  - `{name}`"
            for name in enabled_models
        ],
        f"- **Training-feature artifact:** `{training_features_path}`",
        f"- **Estimated-rate artifact:** `{estimated_rates_path}`",
        f"- **Temporal-validation summary:** `{summary_path}`",
        f"- **Temporal-validation folds:** `{folds_path}`",
        f"- **Figure directory:** `{figure_dir}`",
        "",
        "## 2. Method position within VACODA",
        "",
        (
            "The CIF/FOB stage estimates a conversion factor from "
            "observed direct and mirror evidence, enriches that evidence "
            "with bilateral geography, country characteristics, product "
            "information, oil prices, and lagged route history, and then "
            "uses expanding-window temporal validation to compare candidate "
            "estimators. The selected primary model provides the factor "
            "used by the downstream CIF/FOB application step."
        ),
        "",
        (
            "CEPII Gravity is used here as a **bilateral predictor source**, "
            "not as a standalone structural gravity model."
        ),
        "",
        "## 3. Configured reference sources",
        "",
        _markdown_table(
            reference_rows,
            columns=["Reference role", "Path"],
        ),
        "",
        "## 4. Configured predictor sets",
        "",
        (
            "The report distinguishes estimator algorithms from predictor "
            "sets. All configured predictor sets are listed below with "
            "their enable status; detailed breakdowns in Section 4.1 are "
            "shown only for enabled predictor sets."
        ),
        "",
        _markdown_table(
            predictor_set_rows,
            columns=[
                "Predictor set",
                "Enabled",
                "Predictor count",
                "Primary",
                "Categorical predictors",
            ],
            formatters={
                "Predictor count": _fmt_int,
            },
        ),
        "",
        "### 4.1 Predictor-set details",
        "",
        *predictor_set_detail_lines,
        "",
        "## 5. CIF/FOB data-population journey",
        "",
        (
            "The CIF/FOB workflow changes data grain at several stages. "
            "A reduction caused by filtering represents excluded source "
            "rows. A reduction caused by aggregation or bilateral pairing "
            "represents consolidation to a new analytical grain and should "
            "not be interpreted as lost trade information."
        ),
        "",
        "### 5.1 Source input to bilateral mirror groups",
        "",
        _markdown_table(
            population_journey_table,
            columns=[
                "Workflow stage",
                "Rows/groups",
                "Rows reduction",
                "% Reduction",
                "Interpretation",
            ],
            formatters={
                "Rows/groups": _fmt_int,
                "Rows reduction": _fmt_int,
                "% Reduction": (
                    _fmt_fraction_percent
                ),
            },
        ),
        "",
        (
            f"The canonical prepared artifact represents "
            f"{_fmt_int(population_metrics['n_represented_source_rows'])} "
            f"selected source rows in "
            f"{_fmt_int(population_metrics['n_canonical_trade_groups'])} "
            f"canonical groups. It contains "
            f"{_fmt_int(population_metrics['n_single_component_groups'])} "
            f"single-component groups and "
            f"{_fmt_int(population_metrics['n_multi_component_groups'])} "
            f"multi-component groups."
        ),
        "",
        _markdown_table(
            bilateral_reporting_rows,
            columns=[
                "Reporting case",
                "Groups",
                "Share of paired groups",
                "Potential observed-factor evidence",
            ],
            formatters={
                "Groups": _fmt_int,
                "Share of paired groups": _fmt_fraction_percent,
            },
        ),
        "",
        (
            "Structurally, both-report and import-only groups can "
            "potentially provide direct-factor evidence, representing "
            f"{_fmt_fraction_percent(direct_structural_share)} "
            "of paired groups. Mirror-factor evidence is possible "
            "only for both-report groups, representing "
            f"{_fmt_fraction_percent(mirror_structural_share)} "
            "of paired groups. These are maximum structural candidate "
            "populations before value, weight, concordance, and factor-quality "
            "requirements are applied."
        ),
        "",
        (
            f"The bilateral mirror artifact contains "
            f"{_fmt_int(population_metrics['n_bilateral_mirror_groups'])} "
            f"groups: "
            f"{_fmt_int(population_metrics['n_groups_with_both_reports'])} "
            f"with both exporter and importer reports, "
            f"{_fmt_int(population_metrics['n_export_only_groups'])} "
            f"export-only groups, and "
            f"{_fmt_int(population_metrics['n_import_only_groups'])} "
            f"import-only groups."
        ),
        "",
        (
            f"- **Observed years in the bilateral feature artifact:** "
            f"{int(feature_metrics['minimum_year'])}–"
            f"{int(feature_metrics['maximum_year'])} "
            f"({_fmt_int(feature_metrics['n_years'])} years)"
        ),
        (
            "- **Configured training factor sources:** "
            + ", ".join(
                f"`{source}`"
                for source in training_factor_sources
            )
        ),
        (
            f"- **Minimum evidence value:** "
            f"{_fmt_float(cfg.method.estimation.minimum_evidence_value_usd, 2)} USD"
        ),
        (
            f"- **Minimum permitted CIF/FOB factor:** "
            f"{_fmt_float(cfg.method.estimation.minimum_factor, 3)}"
        ),
        (
            f"- **Upper-factor rule:** "
            f"`{cfg.method.estimation.maximum_factor_rule.method}`"
        ),
        (
            f"- **Upper-factor IQR multiplier:** "
            f"{_fmt_float(
                cfg.method.estimation.maximum_factor_rule.iqr_multiplier,
                2,
            )}"
        ),
        (
            f"- **Exclude proxy weights:** "
            f"{cfg.method.estimation.exclude_proxy_weights}"
        ),
        "",
        "### 5.2 Selected observed-factor candidates",
        "",
        (
            "The internal `model_eligible` flag means that a row has a "
            "selected observed-factor candidate. It does not mean that "
            "the row is used to train the model. For observed-target "
            "construction, direct evidence has priority; a mirror factor is "
            "selected only where an eligible direct factor was not selected. "
            "This priority applies to model-evidence construction only and "
            "does not determine which factor is applied in downstream "
            "production."
        ),
        "",
        _markdown_table(
            source_selection_rows,
            columns=[
                "Evidence population",
                "Rows",
                "Share of paired groups",
                "Share of selected-factor pool",
                "Meaning",
            ],
            formatters={
                "Rows": _fmt_int,
                "Share of paired groups": _fmt_fraction_percent,
                "Share of selected-factor pool": _fmt_fraction_percent,
                },
        ),
        "",
        (
            f"Of the {_fmt_int(paired_population)} paired bilateral groups, "
            f"{_fmt_int(selected_total)} "
            f"({_fmt_fraction_percent(selected_total / paired_population)}) "
            "contain a selected observed CIF/FOB factor. "
            f"Direct factors account for {_fmt_int(selected_direct)} groups "
            f"({_fmt_fraction_percent(selected_direct / paired_population)} "
            "of all paired groups), while mirror factors account for "
            f"{_fmt_int(selected_mirror)} groups "
            f"({_fmt_fraction_percent(selected_mirror / paired_population)}). "
            f"The remaining {_fmt_int(no_selected_factor)} groups "
            f"({_fmt_fraction_percent(no_selected_factor / paired_population)}) "
            "do not contain a selected observed factor."
        ),
        "",
        (
            "The current estimator is configured to train on "
            + ", ".join(
                f"`{source}`"
                for source in training_factor_sources
            )
            + " evidence only. The report therefore applies the source "
            "restriction before evaluating predictor completeness. This "
            "makes the loss of direct-factor observations explicit. Mirror "
            "candidates remain in the shared artifact for diagnostics and "
            "possible alternative experiments, but they are not part of "
            "the current training population."
        ),
        "",
        "### 5.3 Cumulative reduction to the clean training sample",
        "",
        _markdown_table(
            estimation_funnel_rows,
            columns=[
                "Cumulative estimation stage",
                "Rows remaining",
                "Rows removed at stage",
                "Data lost at stage",
            ],
            formatters={
                "Rows remaining": _fmt_int,
                "Rows removed at stage": _fmt_int,
                "Data lost at stage": _fmt_fraction_percent,
            },
        ),
        "",
        (
            f"After restricting the candidate pool to direct factors, "
            f"{_fmt_int(direct_candidates)} direct-factor candidates "
            f"remain. Predictor completeness is then evaluated only within "
            f"those direct rows. {_fmt_int(direct_predictor_loss)} "
            f"({_fmt_fraction_percent(direct_predictor_loss_share)}) "
            f"do not satisfy the "
            f"initial predictor-completeness requirement, leaving "
            f"{_fmt_int(direct_complete)} direct rows before the "
            f"remaining evidence and factor-range filters."
        ),
        "",
        "#### Direct-factor rows excluded by predictor completeness",
        "",
        (
            "The table below reports the first failed requirement in the "
            "ordered `predictor_completeness_reason` diagnostic. The "
            "categories are mutually exclusive and therefore sum to the "
            "number of excluded direct-factor rows."
        ),
        "",
        _markdown_table(
            direct_predictor_failures,
            columns=[
                "Predictor requirement not met",
                "Direct rows excluded",
                "Share of excluded direct rows",
            ],
            formatters={
                "Direct rows excluded": _fmt_int,
                "Share of excluded direct rows": (
                    _fmt_fraction_percent
                ),
            },
        ),
        "",
        (
            f"The final fitting and temporal-validation sample contains "
            f"{_fmt_int(rate_metrics['n_clean_estimation_observations'])} "
            f"observations, equal to "
            f"{_fmt_percent_value(rate_metrics['clean_estimation_share_of_candidates_percent'])} "
            f"of the selected-factor candidate pool."
        ),
        "",
        "#### Direction of observed factors in the final clean training sample",
        "",
        (
            "The table below describes the observed direct CIF/FOB factors in "
            "the final sample used for model fitting and temporal validation. "
            "Factors above one indicate that the observed CIF value exceeds the "
            "corresponding FOB value; factors approximately equal to one indicate "
            "no measurable CIF uplift."
        ),
        "",
        _markdown_table(
            rate_metrics["training_factor_direction"],
            columns=[
                "Direction",
                "Rows",
                "Share",
                "Median factor",
                "Factor P10",
                "Factor P90",
            ],
            formatters={
                "Rows": _fmt_int,
                "Share": _fmt_percent_value,
                "Median factor": lambda value: _fmt_float(value, 6),
                "Factor P10": lambda value: _fmt_float(value, 6),
                "Factor P90": lambda value: _fmt_float(value, 6),
            },
        ),
        "",
        "### 5.4 Direct CIF/FOB factor distribution and upper-bound evidence",
        "",
        (
            "The figure below shows the distribution of directly observed "
            "CIF/FOB factors before the final upper-bound training filter "
            "is applied. The upper panel shows the point-cloud distribution, "
            "while the lower panel shows the corresponding boxplot. The "
            "shaded region marks the interquartile range, containing the "
            "middle 50% of displayed direct observations."
        ),
        "",
        markdown_image(
            "Direct CIF/FOB factor distribution",
            direct_factor_distribution_figure,
            document_path=report_path,
        ),
        "",
        (
            f"Across the full positive direct-factor pool "
            f"({_fmt_int(direct_factor_distribution_metrics['n_positive'])} "
            f"rows), the median observed CIF/FOB factor is "
            f"{_fmt_float(direct_factor_distribution_metrics['median'], 3)}, "
            f"with Q1 "
            f"{_fmt_float(direct_factor_distribution_metrics['q1'], 3)} "
            f"and Q3 "
            f"{_fmt_float(direct_factor_distribution_metrics['q3'], 3)}."
        ),
        (
            f"The full-distribution IQR is "
            f"{_fmt_float(direct_factor_distribution_metrics['iqr'], 3)}. "
            f"A strict boxplot-style upper threshold Q3 + 1.5×IQR is "
            f"{_fmt_float(
                direct_factor_distribution_metrics['q3_plus_1_5_iqr'],
                3,
            )}, while the configured `iqr_cap` rule uses "
            f"Q3 + "
            f"{_fmt_float(maximum_factor_rule.iqr_multiplier, 2)}×IQR. "
            f"For this run, that rule resolves to an upper training bound "
            f"of {_fmt_float(resolved_maximum_factor, 3)}."
        ),
        (
            f"Applying this resolved upper bound removes "
            f"{_fmt_int(upper_bound_removed)} observations "
            f"({_fmt_fraction_percent(
                upper_bound_removed / upper_bound_population
                if upper_bound_population
                else 0.0
            )}) "
            f"after the lower-bound filter."
        ),

        (
            f"Clean observations by source: "
            f"{_fmt_int(rate_metrics['n_clean_direct_observations'])} direct, "
            f"{_fmt_int(rate_metrics['n_clean_mirror_observations'])} mirror."
        ),
        "",
        "### 5.5 Predictor coverage for downstream prediction",
        "",
        _markdown_table(
            coverage_rows,
            columns=["Feature area", "Rows found", "Coverage"],
            formatters={
                "Rows found": _fmt_int,
                "Coverage": _fmt_percent_value,
            },
        ),
        "",
        "#### Product median unit-value source",
        "",
        (
            f"The lagged-product median is used only when at least "
            f"{cfg.features.product.median_unit_value.minimum_observations} "
            f"historical observations are available; otherwise the fallback "
            f"hierarchy is applied."
        ),
        "",
        _markdown_table(
            product_unit_value_sources,
            columns=["Source", "Rows", "Share"],
            formatters={
                "Rows": _fmt_int,
                "Share": lambda value: _fmt_percent_value(value, digits=5),
            },
        ),
        "",
        (
            f"Predictor-complete rows can receive model predictions even "
            f"when they do not contain an observed training factor. This "
            f"is why the prediction population "
            f"({_fmt_int(rate_metrics['n_prediction_eligible'])}) is much "
            f"larger than the clean training population "
            f"({_fmt_int(rate_metrics['n_clean_estimation_observations'])})."
        ),
        "",
        "#### Product-median minimum-observation sensitivity",
        "",
        (
            "The minimum-observation threshold controls the trade-off between "
            "retaining HS6-specific historical information and falling back to "
            "broader HS4 or global medians."
        ),
        "",
        _markdown_table(
            product_median_threshold_sensitivity,
            columns=[
                "Minimum observations",
                "Setting",
                "HS6 share",
                "HS4 fallback share",
                "Global fallback share",
                "Missing share",
            ],
            formatters={
                "Minimum observations": _fmt_int,
                "HS6 share": lambda value: _fmt_percent_value(
                    value,
                    digits=2,
                ),
                "HS4 fallback share": lambda value: _fmt_percent_value(
                    value,
                    digits=2,
                ),
                "Global fallback share": lambda value: _fmt_percent_value(
                    value,
                    digits=4,
                ),
                "Missing share": lambda value: _fmt_percent_value(
                    value,
                    digits=4,
                ),
            },
        ),
        "",
        (
            f"The configured threshold is "
            f"**{configured_product_minimum} observations**. "
            "A lower threshold preserves more HS6-specific medians but accepts "
            "estimates based on fewer observations; a higher threshold increases "
            "fallback to broader product groups."
        ),

        "",
        (
            f"The downstream prediction population is a separate branch "
            f"from the training funnel: "
            f"{_fmt_int(rate_metrics['n_prediction_eligible'])} bilateral "
            f"groups have complete predictors and can receive estimated "
            f"CIF/FOB factors, including groups without an observed "
            f"training factor."
        ),
        "",
        "### 5.6 Bilateral-distance source usage",
        "",
        (
            f"- Gravity 2021 harmonic-distance rows: "
            f"{_fmt_int(feature_metrics['n_gravity_harmonic_distance'])}"
        ),
        (
            f"- Legacy weighted-distance rows: "
            f"{_fmt_int(feature_metrics['n_legacy_weighted_distance'])}"
        ),
        (
            f"- Legacy capital-distance fallback rows: "
            f"{_fmt_int(feature_metrics['n_legacy_capital_fallback'])}"
        ),
        "",
        "### 5.7 Historical feature availability",
        "",
        (
            f"- Rows with lagged pair factor: "
            f"{_fmt_int(feature_metrics['n_lagged_pair_factor_found'])} "
            f"({_fmt_percent_value(feature_metrics['lagged_pair_factor_coverage_percent'])} "
            f"of all rows)"
        ),
        (
            f"- Selected-factor candidates with lagged pair factor: "
            f"{_fmt_int(feature_metrics['n_factor_candidates_lagged_pair_factor_found'])} "
            f"({_fmt_percent_value(feature_metrics['factor_candidate_lagged_pair_factor_coverage_percent'])} "
            f"of selected-factor candidates)"
        ),
        (
            f"- Rows with lagged pair-product factor: "
            f"{_fmt_int(feature_metrics['n_lagged_pair_product_factor_found'])} "
            f"({_fmt_percent_value(feature_metrics['lagged_pair_product_factor_coverage_percent'])})"
        ),
        (
            f"- Rows with lagged bilateral trade value: "
            f"{_fmt_int(feature_metrics['n_lagged_bilateral_trade_value_found'])}"
        ),
        (
            f"- Rows with lagged pair-product trade value: "
            f"{_fmt_int(feature_metrics['n_lagged_pair_product_trade_value_found'])}"
        ),
        "",
        "## 6. Candidate model configuration",
        "",
        (
            "Candidate models define estimator algorithms. In the validation "
            "section, each algorithm may appear once per configured "
            "predictor set."
        ),
        "",
        _markdown_table(
            model_rows,
            columns=[
                "Model",
                "Config class",
                "Enabled",
                "Primary",
                "Target transformation",
                "Loss/objective",
            ],
        ),
        "",
        "## 7. Expanding-window temporal validation",
        "",
        "### Overall predictive performance",
        "",
        _markdown_table(
            combined_table,
            columns=[
                "Rank",
                "Model",
                "Algorithm",
                "Predictor set",
                "Mean log MAE",
                "RMSE log",
                "R² log",
                "Median APE",
                "Mean APE",
                "P90 APE",
                "P95 APE",
                "Spearman",
                "Observed P95",
                "Predicted P95",
                "Tail median APE",
                "Tail P90 APE",
            ],
            formatters={
                "Rank": _fmt_int,
                "Mean log MAE": _fmt_float,
                "RMSE log": _fmt_float,
                "R² log": _fmt_float,
                "Median APE": _fmt_fraction_percent,
                "Mean APE": _fmt_fraction_percent,
                "P90 APE": _fmt_fraction_percent,
                "P95 APE": _fmt_fraction_percent,
                "Spearman": _fmt_float,
                "Observed P95": _fmt_float,
                "Predicted P95": _fmt_float,
                "Tail median APE": _fmt_fraction_percent,
                "Tail P90 APE": _fmt_fraction_percent,
            },
        ),

        "",

        f"",
        f"- Lowest mean log-factor MAE: `{best_mae}`",
        f"- Lowest mean log-factor RMSE: `{best_rmse}`",
        f"- Highest mean log-factor R²: `{best_r2}`",
        f"- Lowest mean median APE: `{best_median_ape}`",
        f"- Lowest mean P90 APE: `{best_p90}`",
        f"- Lowest mean P95 APE: `{best_p95}`",
        f"- Highest mean Spearman correlation: `{best_spearman}`",
        f"- Lowest mean upper-tail median APE: `{best_tail}`",
        "",
        "## 8. Primary-model validation by year",
        "",
        _markdown_table(
            primary_folds_table,
            columns=fold_columns,
            formatters=fold_formatters,
        ),
        "",
        *primary_diagnostics_section,
        *corrected_trade_section,
        "## 11. Decision assessment",
        "",
        _primary_decision_text(
            primary_model=primary_model,
            ranked=ranked,
        ),
        "",
        _tail_tradeoff_text(
            primary_model=primary_model,
            ranked=ranked,
        ),
        "",
        (
            "The upper-tail figures in this report come from the normal "
            "temporal-validation metrics produced by `estimate_rates.py`. "
            "The separate, deeper upper-tail audit remains an independent "
            "diagnostic script and is intentionally not part of this "
            "workflow report."
        ),
        "",
        "## 12. Final estimated-rate artifact",
        "",
        (
            f"- **Output rows:** "
            f"{_fmt_int(rate_metrics['n_rows'])}"
        ),
        (
            f"- **Clean observations used for fitting/backtesting:** "
            f"{_fmt_int(rate_metrics['n_clean_estimation_observations'])}"
        ),
        (
            f"- **Clean direct training observations:** "
            f"{_fmt_int(rate_metrics['n_clean_direct_observations'])}"
        ),
        (
            f"- **Clean mirror training observations:** "
            f"{_fmt_int(rate_metrics['n_clean_mirror_observations'])}"
        ),
        (
            f"- **Rows eligible to receive predictions:** "
            f"{_fmt_int(rate_metrics['n_prediction_eligible'])}"
        ),
        (
            f"- **Rows with estimated factors:** "
            f"{_fmt_int(rate_metrics['n_with_estimated_factor'])}"
        ),
        (
            f"- **Estimated-factor coverage:** "
            f"{_fmt_percent_value(rate_metrics['estimated_factor_coverage_percent'])}"
        ),
        (
            f"- **Estimated factor median:** "
            f"{_fmt_float(rate_metrics['median_estimated_factor'])}"
        ),
        (
            f"- **Estimated factor P05/P95:** "
            f"{_fmt_float(rate_metrics['estimated_factor_p05'])} / "
            f"{_fmt_float(rate_metrics['estimated_factor_p95'])}"
        ),
        (
            f"- **Estimated factor minimum/maximum:** "
            f"{_fmt_float(rate_metrics['minimum_estimated_factor'])} / "
            f"{_fmt_float(rate_metrics['maximum_estimated_factor'])}"
        ),
        "",
    ]

    value_band_section: list[str] = []
    if not value_band_comparison_table.empty:
        value_band_section.extend(
        [
            "",
            "### Value-band weighting comparison",
            "",
            (
                "The following comparison evaluates the baseline and "
                "value-band-weighted HGB models on the same latest-year "
                "temporal holdout observations. Negative changes indicate "
                "lower error under value-band weighting."
            ),
            "",
            _markdown_table(
                value_band_comparison_table,
                columns=[
                    "Evidence-value band",
                    "Observations",
                    "Baseline median APE",
                    "Weighted median APE",
                    "Change in median APE",
                    "Baseline P90 APE",
                    "Weighted P90 APE",
                    "Change in P90 APE",
                ],
                formatters={
                    "Observations": _fmt_int,
                    "Baseline median APE": _fmt_fraction_percent,
                    "Weighted median APE": _fmt_fraction_percent,
                    "Change in median APE": (
                        lambda value: (
                            f"{100.0 * float(value):+.2f} pp"
                            if not pd.isna(value)
                            else "n/a"
                        )
                    ),
                    "Baseline P90 APE": _fmt_fraction_percent,
                    "Weighted P90 APE": _fmt_fraction_percent,
                    "Change in P90 APE": (
                        lambda value: (
                            f"{100.0 * float(value):+.2f} pp"
                            if not pd.isna(value)
                            else "n/a"
                        )
                    ),
                },
            ),
            "",
        ]
    )

    comparison_figure = figure_paths.get(
        "hgb_weighting_comparison_by_value_band"
    )

    if comparison_figure is not None:
        value_band_section.extend(
            [
                markdown_image(
                    "Baseline versus weighted HGB by evidence-value band",
                    comparison_figure,
                    document_path=report_path,
                ),
                "",
            ]
        )

    primary_validation_header = "## 8. Primary-model validation by year"
    insertion_index = lines.index(primary_validation_header)
    lines[insertion_index:insertion_index] = value_band_section

    report_text = "\n".join(lines)
    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    latest_report_path: Path | None = None

    if cfg.artifacts.write_latest:
        latest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copyfile(
            report_path,
            latest_path,
        )
        latest_report_path = latest_path

    logger.info(
        "CIF/FOB decision report written: %s",
        report_path,
    )
    logger.info(
        "CIF/FOB direct-factor distribution figure written: %s",
        direct_factor_distribution_figure,
    )

    if latest_report_path is not None:
        logger.info(
            "CIF/FOB latest decision report written: %s",
            latest_report_path,
        )

    return {
        "status": "completed",
        "audit_passed": True,
        "report_path": report_path,
        "latest_report_path": latest_report_path,
        "primary_model": primary_model,
        "best_ranked_model": str(
            ranked.iloc[0]["model_name"]
        ),
        "ready_for_trade_correction": (
            int(rate_metrics["n_with_estimated_factor"] or 0) > 0
        ),
    }
