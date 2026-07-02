"""
vacoda.cif_fob.reporting

Write the human-readable CIF/FOB decision report.

The normal CIF/FOB workflow owns this report. The separate upper-tail audit
script remains separate and is not invoked or read here.

The report summarizes all normal workflow evidence available before the
CIF/FOB correction is applied:
- scope and configured references;
- explicit direct/mirror factor selection and training-sample funnel;
- predictor configuration and feature coverage;
- candidate model configuration;
- expanding-window temporal validation;
- upper-tail validation metrics already produced by estimate_rates.py;
- primary-model decision;
- final estimated-rate coverage;
- limitations and downstream readiness.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from vacoda.config.cif_fob import CifFobConfig


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
    threads: int,
) -> dict[str, Any]:
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
                    ) AS maximum_estimated_factor

            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()

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

    return result


def _load_validation_artifacts(
    *,
    summary_path: Path,
    folds_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


# ======================================================================
# Decision helpers
# ======================================================================


def _rank_models(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    ranked = summary.copy()

    numeric_columns = [
        column
        for column in ranked.columns
        if column != "model_name"
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


# ======================================================================
# Report writer
# ======================================================================


def write_cif_fob_decision_report(
    *,
    cfg: CifFobConfig,
    run_id: str,
    training_features_result: dict[str, Any],
    estimated_rates_result: dict[str, Any],
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

    training_features_path = training_features_result.get(
        "training_features_path"
    )
    estimated_rates_path = estimated_rates_result.get(
        "estimated_rates_path"
    )

    if not isinstance(training_features_path, Path):
        training_features_path = cfg.product_path(
            "training_features"
        )

    if not isinstance(estimated_rates_path, Path):
        estimated_rates_path = cfg.product_path(
            "estimated_rates"
        )

    summary_path = cfg.product_path(
        "temporal_validation_summary"
    )
    folds_path = cfg.product_path(
        "temporal_validation_folds"
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

    feature_metrics = _inspect_training_features(
        path=training_features_path,
        threads=cfg.engine.threads,
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

    ranked["model_display_name"] = (
        ranked["model_name"]
        .map(model_display_names)
        .fillna(ranked["model_name"])
    )
    primary_model = str(
        cfg.method.estimation.models.primary
    ).strip().lower()
    enabled_models = dict(
        cfg.method.estimation.models.enabled()
    )

    primary_folds = folds.loc[
        folds["model_name"].eq(primary_model)
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

    source_selection_rows = pd.DataFrame(
        [
            {
                "Evidence population": "Selected direct factors",
                "Rows": feature_metrics["n_selected_direct_factors"],
                "Meaning": (
                    "Direct importer CIF/FOB factor selected as the "
                    "observed target."
                ),
            },
            {
                "Evidence population": "Selected mirror factors",
                "Rows": feature_metrics["n_selected_mirror_factors"],
                "Meaning": (
                    "Mirror factor retained where an eligible direct "
                    "factor was not selected."
                ),
            },
            {
                "Evidence population": (
                    "All rows with a selected observed-factor candidate"
                ),
                "Rows": feature_metrics["n_factor_candidates"],
                "Meaning": (
                    "Internal `model_eligible` pool; not yet the model "
                    "training sample."
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
            "mean_mae_log_factor": "Mean log MAE",
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
                "Observed P95",
                "Predicted P95",
                "Tail median APE",
                "Tail P90 APE",
            ]
        ],
        on=["Rank", "Model"],
        how="left",
    )

    primary_folds_table = primary_folds.sort_values(
        "validation_year"
    ).rename(
        columns={
            "validation_year": "Year",
            "n_train": "Train rows",
            "n_holdout": "Holdout rows",
            "mae_log_factor": "Mean log MAE",
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

    predictors = cfg.method.predictors.all()

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
        f"- **Primary model:** `{primary_model}`",
        "- **Enabled candidate models:**",
        *[
            f"  - `{name}`"
            for name in enabled_models
        ],
        f"- **Training-feature artifact:** `{training_features_path}`",
        f"- **Estimated-rate artifact:** `{estimated_rates_path}`",
        f"- **Temporal-validation summary:** `{summary_path}`",
        f"- **Temporal-validation folds:** `{folds_path}`",
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
        "## 4. Configured predictors",
        "",
        "\n".join(
            f"- `{predictor}`"
            for predictor in predictors
        ),
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
            f"- **Configured training factor sources:** "
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
            f"- **Permitted observed CIF/FOB factor range:** "
            f"{_fmt_float(cfg.method.estimation.minimum_factor, 3)}–"
            f"{_fmt_float(cfg.method.estimation.maximum_factor, 3)}"
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
            "the row is used to train the model. Direct evidence has "
            "priority; a mirror factor is selected only where an eligible "
            "direct factor was not selected."
        ),
        "",
        _markdown_table(
            source_selection_rows,
            columns=[
                "Evidence population",
                "Rows",
                "Meaning",
            ],
            formatters={
                "Rows": _fmt_int,
            },
        ),
        "",
        (
            f"The selected-factor pool therefore contains "
            f"{_fmt_int(feature_metrics['n_selected_direct_factors'])} "
            f"direct rows plus "
            f"{_fmt_int(feature_metrics['n_selected_mirror_factors'])} "
            f"mirror rows, giving "
            f"{_fmt_int(feature_metrics['n_factor_candidates'])} "
            f"candidate rows in total."
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
        (
            f"Clean observations by source: "
            f"{_fmt_int(rate_metrics['n_clean_direct_observations'])} direct, "
            f"{_fmt_int(rate_metrics['n_clean_mirror_observations'])} mirror."
        ),
        "",
        "### 5.4 Predictor coverage for downstream prediction",
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
        (
            f"Predictor-complete rows can receive model predictions even "
            f"when they do not contain an observed training factor. This "
            f"is why the prediction population "
            f"({_fmt_int(rate_metrics['n_prediction_eligible'])}) is much "
            f"larger than the clean training population "
            f"({_fmt_int(rate_metrics['n_clean_estimation_observations'])})."
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
        "### 5.5 Bilateral-distance source usage",
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
        "### 5.6 Historical feature availability",
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
                "Mean log MAE",
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
        f"",
        f"- Lowest mean log-factor MAE: `{best_mae}`",
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
        "## 9. Decision assessment",
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
        "## 10. Final estimated-rate artifact",
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
        "## 11. Known limitations",
        "",
        (
            "- All tested global estimators should be interpreted as "
            "conditional point estimators. High observed CIF/FOB factors "
            "remain harder to reproduce than the central majority."
        ),
        (
            "- Better upper-tail performance in a candidate model does not "
            "automatically justify global adoption when central and overall "
            "validation performance deteriorate."
        ),
        (
            "- The current predictors do not directly observe shipment "
            "mode, shipment size, port routing, temporary freight disruption, "
            "insurance conditions, or all reporting-quality mechanisms."
        ),
        (
            "- A separate upper-tail audit may be consulted for deeper "
            "credibility and persistence analysis, but it is not required "
            "for this workflow report."
        ),
        "",
        "## 12. Downstream readiness",
        "",
        (
            "The estimated-rate artifact is ready for the CIF/FOB "
            "application step."
            if int(rate_metrics["n_with_estimated_factor"] or 0) > 0
            else (
                "The estimated-rate artifact contains no usable estimated "
                "factors and is not ready for downstream application."
            )
        ),
        "",
    ]

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
