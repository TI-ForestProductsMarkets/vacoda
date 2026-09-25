"""
vacoda.steps.validation.quantity_audit
---------------------------------------
Audit the quantity-selection product produced by Validation.

This step is intentionally independent from the selector. It verifies that
quantity selection preserved the upstream population, that the recorded
candidate decision can be reproduced from the selector's own row-level
selection evidence, and that the selected quantity was applied correctly to
every bilateral/unilateral availability case.
"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import pandas as pd

from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal


# ======================================================================
# Helpers
# ======================================================================


def _safe_identifier(name: str) -> str:
    """Return a safely quoted SQL identifier."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
        raise ValueError(
            f"Unsafe Validation quantity-audit column identifier: {name!r}"
        )

    return f'"{name}"'


def _table_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> list[str]:
    """Return parquet column names in file order."""

    path_sql = sql_literal(path)

    return [
        str(row[0])
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet({path_sql})
            """
        ).fetchall()
    ]


def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    path: Path,
    required_columns: set[str],
    label: str,
) -> None:
    """Require columns in one audit input."""

    available = set(
        _table_columns(
            con=con,
            path=path,
        )
    )

    missing = sorted(required_columns - available)

    if missing:
        raise RuntimeError(
            f"{label} is missing required column(s): "
            + ", ".join(missing)
        )


def _nearly_equal_sql(
    left: str,
    right: str,
    *,
    relative_tolerance: float = 1e-9,
) -> str:
    """Return SQL for strict numeric agreement with small floating tolerance."""

    tolerance = float(relative_tolerance)

    return f"""
    (
        {left} IS NOT NULL
        AND {right} IS NOT NULL
        AND isfinite(CAST({left} AS DOUBLE))
        AND isfinite(CAST({right} AS DOUBLE))
        AND ABS(CAST({left} AS DOUBLE) - CAST({right} AS DOUBLE))
            <= {tolerance:.17g}
               * GREATEST(
                    ABS(CAST({right} AS DOUBLE)),
                    1.0
               )
    )
    """


def _metric_close(
    left: Any,
    right: Any,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Compare aggregate decision metrics."""

    if left is None or right is None:
        return left is None and right is None

    if pd.isna(left) or pd.isna(right):
        return pd.isna(left) and pd.isna(right)

    return abs(float(left) - float(right)) <= tolerance


# ======================================================================
# Public step
# ======================================================================


def audit_quantity_selection(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    structural_audit_path: Path,
    quantity_selection_path: Path,
    selection_evidence_path: Path,
    run_id: str,
    output_path: Path | None = None,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Audit the Validation quantity-selection product and fail on violations."""

    start = perf_counter()

    input_path = Path(input_path)
    structural_audit_path = Path(structural_audit_path)
    quantity_selection_path = Path(quantity_selection_path)
    selection_evidence_path = Path(selection_evidence_path)

    output_path = (
        Path(output_path)
        if output_path is not None
        else cfg.product_path("quantity_selection_audit")
    )

    required_files = {
        "Validation input": input_path,
        "Validation structural audit": structural_audit_path,
        "quantity-selection output": quantity_selection_path,
        "quantity-selection evidence": selection_evidence_path,
    }

    for label, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        not force
        and output_path.is_file()
        and output_path.stat().st_size > 0
    ):
        logger.info(
            "%s REUSE — Validation quantity-selection audit: %s",
            get_symbols("reuse"),
            output_path,
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "output_path": output_path,
            "quantity_selection_audit_path": output_path,
            "reused": True,
        }

    logger.info(
        "%s REBUILD — Validation quantity-selection audit",
        get_symbols("rebuilt"),
    )
    logger.info("  Input: %s", input_path)
    logger.info("  Structural audit: %s", structural_audit_path)
    logger.info("  Quantity selection: %s", quantity_selection_path)
    logger.info("  Selection evidence: %s", selection_evidence_path)
    logger.info("  Output: %s", output_path)

    columns_cfg = cfg.diagnostics.columns

    year_col = str(columns_cfg.year)
    product_col = str(columns_cfg.product)
    export_quantity_col = str(columns_cfg.export_quantity)
    import_quantity_col = str(columns_cfg.import_quantity)

    key_columns = (
        year_col,
        "exporter_iso",
        "importer_iso",
        product_col,
    )

    simple_mean_col = "quantity_simple_mean_kg"
    value_weighted_col = "quantity_value_weighted_kg"
    reliability_weighted_col = "quantity_reliability_weighted_kg"

    selected_candidate_col = "quantity_selected_candidate"
    selected_quantity_col = "quantity_selected_kg"
    selected_source_col = "quantity_selected_source"

    conditioned_columns = {
        "simple_mean": (
            "candidate_simple_mean_conditioned_uv_abs_residual"
        ),
        "value_weighted": (
            "candidate_value_weighted_conditioned_uv_abs_residual"
        ),
        "quantity_reliability_weighted": (
            "candidate_quantity_reliability_weighted_"
            "conditioned_uv_abs_residual"
        ),
    }

    uv_columns = {
        "simple_mean": "candidate_simple_mean_uv_log_distance",
        "value_weighted": "candidate_value_weighted_uv_log_distance",
        "quantity_reliability_weighted": (
            "candidate_quantity_reliability_weighted_uv_log_distance"
        ),
    }

    candidate_quantity_columns = {
        "simple_mean": simple_mean_col,
        "value_weighted": value_weighted_col,
        "quantity_reliability_weighted": reliability_weighted_col,
    }

    expected_candidates = (
        "simple_mean",
        "value_weighted",
        "quantity_reliability_weighted",
    )

    expected_selection_rule = (
        "minimum median quantity-conditioned UV absolute residual; "
        "tie-break P95 then P90"
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"PRAGMA threads={int(cfg.engine.threads)}"
        )

        input_sql = sql_literal(input_path)
        structural_audit_sql = sql_literal(structural_audit_path)
        selection_sql = sql_literal(quantity_selection_path)
        evidence_sql = sql_literal(selection_evidence_path)

        # --------------------------------------------------------------
        # Upstream structural gate
        # --------------------------------------------------------------

        structural_row = con.execute(
            f"""
            SELECT audit_passed
            FROM read_parquet({structural_audit_sql})
            LIMIT 1
            """
        ).fetchone()

        structural_audit_passed = bool(
            structural_row is not None
            and structural_row[0]
        )

        if not structural_audit_passed:
            raise RuntimeError(
                "Quantity-selection audit requires a passing structural audit."
            )

        # --------------------------------------------------------------
        # Contracts
        # --------------------------------------------------------------

        input_required = {
            *key_columns,
            export_quantity_col,
            import_quantity_col,
            simple_mean_col,
            value_weighted_col,
            reliability_weighted_col,
        }

        output_required = {
            *input_required,
            selected_candidate_col,
            selected_quantity_col,
            selected_source_col,
            "has_export_quantity_selection",
            "has_import_quantity_selection",
            "both_quantities_observed_selection",
            *conditioned_columns.values(),
            *uv_columns.values(),
        }

        evidence_required = {
            "candidate",
            "n_conditioned_rows",
            "median_conditioned_uv_abs_residual",
            "p90_conditioned_uv_abs_residual",
            "p95_conditioned_uv_abs_residual",
            "n_uv_rows",
            "median_uv_log_distance",
            "p95_uv_log_distance",
            "run_id",
            "selected",
            "selection_rule",
            "decision_rank",
        }

        _require_columns(
            con=con,
            path=input_path,
            required_columns=input_required,
            label="Validation input",
        )
        _require_columns(
            con=con,
            path=quantity_selection_path,
            required_columns=output_required,
            label="Quantity-selection output",
        )
        _require_columns(
            con=con,
            path=selection_evidence_path,
            required_columns=evidence_required,
            label="Quantity-selection evidence",
        )

        input_columns = _table_columns(
            con=con,
            path=input_path,
        )
        output_columns = set(
            _table_columns(
                con=con,
                path=quantity_selection_path,
            )
        )

        missing_input_columns = sorted(
            set(input_columns) - output_columns
        )
        input_columns_preserved = not missing_input_columns

        # --------------------------------------------------------------
        # Population and key preservation
        # --------------------------------------------------------------

        n_input_rows = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet({input_sql})"
            ).fetchone()[0]
        )
        n_output_rows = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet({selection_sql})"
            ).fetchone()[0]
        )

        row_count_preserved = n_input_rows == n_output_rows

        key_sql = [
            _safe_identifier(column)
            for column in key_columns
        ]
        key_list_sql = ", ".join(key_sql)

        n_null_output_keys = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({selection_sql})
                WHERE {' OR '.join(f'{column} IS NULL' for column in key_sql)}
                """
            ).fetchone()[0]
        )

        duplicate_output_rows = int(
            con.execute(
                f"""
                SELECT COALESCE(SUM(n_rows - 1), 0)
                FROM (
                    SELECT
                        {key_list_sql},
                        COUNT(*) AS n_rows
                    FROM read_parquet({selection_sql})
                    GROUP BY {key_list_sql}
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )

        canonical_key_unique = (
            n_null_output_keys == 0
            and duplicate_output_rows == 0
        )

        unmatched_input_keys = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT {key_list_sql}
                    FROM read_parquet({input_sql})
                    EXCEPT
                    SELECT {key_list_sql}
                    FROM read_parquet({selection_sql})
                )
                """
            ).fetchone()[0]
        )

        unmatched_output_keys = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT {key_list_sql}
                    FROM read_parquet({selection_sql})
                    EXCEPT
                    SELECT {key_list_sql}
                    FROM read_parquet({input_sql})
                )
                """
            ).fetchone()[0]
        )

        bilateral_population_preserved = (
            unmatched_input_keys == 0
            and unmatched_output_keys == 0
        )

        # --------------------------------------------------------------
        # Upstream columns must remain unchanged
        # --------------------------------------------------------------

        comparable_input_columns = [
            column
            for column in input_columns
            if column in output_columns
            and column not in key_columns
        ]

        join_condition = " AND ".join(
            f"i.{column} = o.{column}"
            for column in key_sql
        )

        if comparable_input_columns:
            mismatch_expression = " OR ".join(
                (
                    f"i.{_safe_identifier(column)} "
                    f"IS DISTINCT FROM "
                    f"o.{_safe_identifier(column)}"
                )
                for column in comparable_input_columns
            )

            upstream_value_mismatch_rows = int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql}) AS i
                    JOIN read_parquet({selection_sql}) AS o
                      ON {join_condition}
                    WHERE {mismatch_expression}
                    """
                ).fetchone()[0]
            )
        else:
            upstream_value_mismatch_rows = 0

        upstream_values_preserved = (
            input_columns_preserved
            and upstream_value_mismatch_rows == 0
        )

        # --------------------------------------------------------------
        # Evidence-table integrity
        # --------------------------------------------------------------

        evidence = con.execute(
            f"""
            SELECT *
            FROM read_parquet({evidence_sql})
            ORDER BY decision_rank
            """
        ).fetchdf()

        n_evidence_rows = len(evidence)
        candidate_set_valid = (
            n_evidence_rows == 3
            and set(evidence["candidate"].astype(str))
                == set(expected_candidates)
        )

        evidence_run_ids = {
            str(value)
            for value in evidence["run_id"].dropna().tolist()
        }
        evidence_run_id_consistent = (
            n_evidence_rows == 3
            and len(evidence_run_ids) == 1
        )

        selection_rule_valid = bool(
            n_evidence_rows == 3
            and evidence["selection_rule"]
                .astype(str)
                .eq(expected_selection_rule)
                .all()
        )

        selected_count = int(
            evidence["selected"].fillna(False).astype(bool).sum()
        )

        selected_flag_valid = selected_count == 1

        ranks = {
            int(value)
            for value in evidence["decision_rank"].dropna().tolist()
        }
        decision_ranks_valid = ranks == {1, 2, 3}

        rank_one = evidence[
            evidence["decision_rank"].eq(1)
        ]
        selected_rows = evidence[
            evidence["selected"].fillna(False).astype(bool)
        ]

        selected_matches_rank_one = bool(
            len(rank_one) == 1
            and len(selected_rows) == 1
            and str(rank_one.iloc[0]["candidate"])
                == str(selected_rows.iloc[0]["candidate"])
        )

        selected_candidate = (
            str(selected_rows.iloc[0]["candidate"])
            if len(selected_rows) == 1
            else None
        )

        selected_candidate_valid = (
            selected_candidate in expected_candidates
        )

        evidence_metrics_complete = bool(
            candidate_set_valid
            and evidence[
                [
                    "n_conditioned_rows",
                    "median_conditioned_uv_abs_residual",
                    "p90_conditioned_uv_abs_residual",
                    "p95_conditioned_uv_abs_residual",
                    "n_uv_rows",
                    "median_uv_log_distance",
                    "p95_uv_log_distance",
                ]
            ].notna().all().all()
            and (evidence["n_conditioned_rows"] > 0).all()
            and (evidence["n_uv_rows"] > 0).all()
        )

        # --------------------------------------------------------------
        # Reproduce selector decision metrics from row-level evidence
        # --------------------------------------------------------------

        recomputed_rows: list[dict[str, Any]] = []

        for candidate in expected_candidates:
            conditioned_col = _safe_identifier(
                conditioned_columns[candidate]
            )
            uv_col = _safe_identifier(
                uv_columns[candidate]
            )

            row = con.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (
                        WHERE both_quantities_observed_selection
                          AND {conditioned_col} IS NOT NULL
                    ) AS n_conditioned_rows,

                    MEDIAN({conditioned_col}) FILTER (
                        WHERE both_quantities_observed_selection
                          AND {conditioned_col} IS NOT NULL
                    ) AS median_conditioned_uv_abs_residual,

                    QUANTILE_CONT({conditioned_col}, 0.90) FILTER (
                        WHERE both_quantities_observed_selection
                          AND {conditioned_col} IS NOT NULL
                    ) AS p90_conditioned_uv_abs_residual,

                    QUANTILE_CONT({conditioned_col}, 0.95) FILTER (
                        WHERE both_quantities_observed_selection
                          AND {conditioned_col} IS NOT NULL
                    ) AS p95_conditioned_uv_abs_residual,

                    COUNT(*) FILTER (
                        WHERE both_quantities_observed_selection
                          AND {uv_col} IS NOT NULL
                    ) AS n_uv_rows,

                    MEDIAN({uv_col}) FILTER (
                        WHERE both_quantities_observed_selection
                          AND {uv_col} IS NOT NULL
                    ) AS median_uv_log_distance,

                    QUANTILE_CONT({uv_col}, 0.95) FILTER (
                        WHERE both_quantities_observed_selection
                          AND {uv_col} IS NOT NULL
                    ) AS p95_uv_log_distance

                FROM read_parquet({selection_sql})
                """
            ).fetchone()

            recomputed_rows.append(
                {
                    "candidate": candidate,
                    "n_conditioned_rows": int(row[0]),
                    "median_conditioned_uv_abs_residual": row[1],
                    "p90_conditioned_uv_abs_residual": row[2],
                    "p95_conditioned_uv_abs_residual": row[3],
                    "n_uv_rows": int(row[4]),
                    "median_uv_log_distance": row[5],
                    "p95_uv_log_distance": row[6],
                }
            )

        recomputed = pd.DataFrame(recomputed_rows)

        evidence_by_candidate = evidence.set_index("candidate")

        evidence_metric_mismatches = 0

        metric_columns = (
            "median_conditioned_uv_abs_residual",
            "p90_conditioned_uv_abs_residual",
            "p95_conditioned_uv_abs_residual",
            "median_uv_log_distance",
            "p95_uv_log_distance",
        )

        count_columns = (
            "n_conditioned_rows",
            "n_uv_rows",
        )

        for row in recomputed.itertuples(index=False):
            stored = evidence_by_candidate.loc[row.candidate]

            for column in count_columns:
                if int(getattr(row, column)) != int(stored[column]):
                    evidence_metric_mismatches += 1

            for column in metric_columns:
                if not _metric_close(
                    getattr(row, column),
                    stored[column],
                ):
                    evidence_metric_mismatches += 1

        selection_evidence_reproduced = (
            evidence_metric_mismatches == 0
        )

        # Independent ranking from recomputed metrics.
        recomputed_ranked = recomputed.sort_values(
            [
                "median_conditioned_uv_abs_residual",
                "p95_conditioned_uv_abs_residual",
                "p90_conditioned_uv_abs_residual",
            ],
            ascending=True,
            kind="mergesort",
        ).reset_index(drop=True)

        decision_tied = False

        if len(recomputed_ranked) > 1:
            first = recomputed_ranked.iloc[0]
            second = recomputed_ranked.iloc[1]
            decision_tied = all(
                abs(float(first[column]) - float(second[column])) <= 1e-12
                for column in (
                    "median_conditioned_uv_abs_residual",
                    "p95_conditioned_uv_abs_residual",
                    "p90_conditioned_uv_abs_residual",
                )
            )

        recomputed_selected_candidate = (
            None
            if decision_tied
            else str(recomputed_ranked.iloc[0]["candidate"])
        )

        selection_decision_reproduced = bool(
            not decision_tied
            and selected_candidate is not None
            and recomputed_selected_candidate == selected_candidate
        )

        # --------------------------------------------------------------
        # Row-level selected-candidate contract
        # --------------------------------------------------------------

        selected_candidate_values = con.execute(
            f"""
            SELECT DISTINCT {selected_candidate_col}
            FROM read_parquet({selection_sql})
            ORDER BY {selected_candidate_col}
            """
        ).fetchall()

        row_level_candidate_constant = bool(
            selected_candidate is not None
            and len(selected_candidate_values) == 1
            and str(selected_candidate_values[0][0]) == selected_candidate
        )

        selected_quantity_sql = (
            _safe_identifier(candidate_quantity_columns[selected_candidate])
            if selected_candidate_valid
            else None
        )

        export_sql = _safe_identifier(export_quantity_col)
        import_sql = _safe_identifier(import_quantity_col)
        selected_quantity_identifier = _safe_identifier(selected_quantity_col)
        selected_source_identifier = _safe_identifier(selected_source_col)

        has_export_sql = f"""
        (
            {export_sql} IS NOT NULL
            AND {export_sql} > 0
            AND isfinite(CAST({export_sql} AS DOUBLE))
        )
        """

        has_import_sql = f"""
        (
            {import_sql} IS NOT NULL
            AND {import_sql} > 0
            AND isfinite(CAST({import_sql} AS DOUBLE))
        )
        """

        if selected_quantity_sql is not None:
            bilateral_match_sql = _nearly_equal_sql(
                selected_quantity_identifier,
                selected_quantity_sql,
            )
        else:
            bilateral_match_sql = "FALSE"

        export_match_sql = _nearly_equal_sql(
            selected_quantity_identifier,
            export_sql,
        )
        import_match_sql = _nearly_equal_sql(
            selected_quantity_identifier,
            import_sql,
        )

        contract = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE {has_export_sql}
                      AND {has_import_sql}
                ) AS n_bilateral_rows,

                COUNT(*) FILTER (
                    WHERE {has_export_sql}
                      AND {has_import_sql}
                      AND NOT ({bilateral_match_sql})
                ) AS n_invalid_bilateral_assignments,

                COUNT(*) FILTER (
                    WHERE {has_export_sql}
                      AND {has_import_sql}
                      AND {selected_source_identifier}
                            IS DISTINCT FROM '{selected_candidate or ''}'
                ) AS n_invalid_bilateral_sources,

                COUNT(*) FILTER (
                    WHERE {has_export_sql}
                      AND NOT {has_import_sql}
                ) AS n_export_only_rows,

                COUNT(*) FILTER (
                    WHERE {has_export_sql}
                      AND NOT {has_import_sql}
                      AND NOT ({export_match_sql})
                ) AS n_invalid_export_only_assignments,

                COUNT(*) FILTER (
                    WHERE {has_export_sql}
                      AND NOT {has_import_sql}
                      AND {selected_source_identifier}
                            IS DISTINCT FROM 'export_only'
                ) AS n_invalid_export_only_sources,

                COUNT(*) FILTER (
                    WHERE NOT {has_export_sql}
                      AND {has_import_sql}
                ) AS n_import_only_rows,

                COUNT(*) FILTER (
                    WHERE NOT {has_export_sql}
                      AND {has_import_sql}
                      AND NOT ({import_match_sql})
                ) AS n_invalid_import_only_assignments,

                COUNT(*) FILTER (
                    WHERE NOT {has_export_sql}
                      AND {has_import_sql}
                      AND {selected_source_identifier}
                            IS DISTINCT FROM 'import_only'
                ) AS n_invalid_import_only_sources,

                COUNT(*) FILTER (
                    WHERE NOT {has_export_sql}
                      AND NOT {has_import_sql}
                ) AS n_missing_rows,

                COUNT(*) FILTER (
                    WHERE NOT {has_export_sql}
                      AND NOT {has_import_sql}
                      AND {selected_quantity_identifier} IS NOT NULL
                ) AS n_invalid_missing_assignments,

                COUNT(*) FILTER (
                    WHERE NOT {has_export_sql}
                      AND NOT {has_import_sql}
                      AND {selected_source_identifier}
                            IS DISTINCT FROM 'missing'
                ) AS n_invalid_missing_sources,

                COUNT(*) FILTER (
                    WHERE (
                        {has_export_sql}
                        OR {has_import_sql}
                    )
                    AND (
                        {selected_quantity_identifier} IS NULL
                        OR {selected_quantity_identifier} <= 0
                        OR NOT isfinite(
                            CAST({selected_quantity_identifier} AS DOUBLE)
                        )
                    )
                ) AS n_invalid_selected_quantities

            FROM read_parquet({selection_sql})
            """
        ).fetchone()

        (
            n_bilateral_rows,
            n_invalid_bilateral_assignments,
            n_invalid_bilateral_sources,
            n_export_only_rows,
            n_invalid_export_only_assignments,
            n_invalid_export_only_sources,
            n_import_only_rows,
            n_invalid_import_only_assignments,
            n_invalid_import_only_sources,
            n_missing_rows,
            n_invalid_missing_assignments,
            n_invalid_missing_sources,
            n_invalid_selected_quantities,
        ) = map(int, contract)

        availability_partition_valid = (
            n_bilateral_rows
            + n_export_only_rows
            + n_import_only_rows
            + n_missing_rows
            == n_output_rows
        )

        assignment_contract_valid = all(
            value == 0
            for value in (
                n_invalid_bilateral_assignments,
                n_invalid_bilateral_sources,
                n_invalid_export_only_assignments,
                n_invalid_export_only_sources,
                n_invalid_import_only_assignments,
                n_invalid_import_only_sources,
                n_invalid_missing_assignments,
                n_invalid_missing_sources,
                n_invalid_selected_quantities,
            )
        )

        # --------------------------------------------------------------
        # Final decision
        # --------------------------------------------------------------

        audit_passed = all(
            (
                structural_audit_passed,
                row_count_preserved,
                canonical_key_unique,
                bilateral_population_preserved,
                input_columns_preserved,
                upstream_values_preserved,
                candidate_set_valid,
                evidence_run_id_consistent,
                selection_rule_valid,
                selected_flag_valid,
                decision_ranks_valid,
                selected_matches_rank_one,
                selected_candidate_valid,
                evidence_metrics_complete,
                selection_evidence_reproduced,
                selection_decision_reproduced,
                row_level_candidate_constant,
                availability_partition_valid,
                assignment_contract_valid,
            )
        )

        elapsed = perf_counter() - start

        audit_row = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "audit_passed": audit_passed,
                    "structural_audit_passed": structural_audit_passed,
                    "n_input_rows": n_input_rows,
                    "n_output_rows": n_output_rows,
                    "row_count_preserved": row_count_preserved,
                    "n_null_output_keys": n_null_output_keys,
                    "duplicate_output_rows": duplicate_output_rows,
                    "canonical_key_unique": canonical_key_unique,
                    "unmatched_input_keys": unmatched_input_keys,
                    "unmatched_output_keys": unmatched_output_keys,
                    "bilateral_population_preserved": (
                        bilateral_population_preserved
                    ),
                    "input_columns_preserved": input_columns_preserved,
                    "n_missing_input_columns": len(missing_input_columns),
                    "upstream_value_mismatch_rows": (
                        upstream_value_mismatch_rows
                    ),
                    "upstream_values_preserved": upstream_values_preserved,
                    "n_evidence_rows": n_evidence_rows,
                    "candidate_set_valid": candidate_set_valid,
                    "evidence_run_id_consistent": (
                        evidence_run_id_consistent
                    ),
                    "selection_evidence_run_id": (
                        next(iter(evidence_run_ids))
                        if evidence_run_ids
                        else None
                    ),
                    "selection_rule_valid": selection_rule_valid,
                    "selected_flag_valid": selected_flag_valid,
                    "decision_ranks_valid": decision_ranks_valid,
                    "selected_matches_rank_one": selected_matches_rank_one,
                    "selected_candidate": selected_candidate,
                    "selected_candidate_valid": selected_candidate_valid,
                    "evidence_metrics_complete": evidence_metrics_complete,
                    "evidence_metric_mismatches": evidence_metric_mismatches,
                    "selection_evidence_reproduced": (
                        selection_evidence_reproduced
                    ),
                    "decision_tied": decision_tied,
                    "recomputed_selected_candidate": (
                        recomputed_selected_candidate
                    ),
                    "selection_decision_reproduced": (
                        selection_decision_reproduced
                    ),
                    "row_level_candidate_constant": (
                        row_level_candidate_constant
                    ),
                    "n_bilateral_rows": n_bilateral_rows,
                    "n_export_only_rows": n_export_only_rows,
                    "n_import_only_rows": n_import_only_rows,
                    "n_missing_rows": n_missing_rows,
                    "availability_partition_valid": (
                        availability_partition_valid
                    ),
                    "n_invalid_bilateral_assignments": (
                        n_invalid_bilateral_assignments
                    ),
                    "n_invalid_bilateral_sources": (
                        n_invalid_bilateral_sources
                    ),
                    "n_invalid_export_only_assignments": (
                        n_invalid_export_only_assignments
                    ),
                    "n_invalid_export_only_sources": (
                        n_invalid_export_only_sources
                    ),
                    "n_invalid_import_only_assignments": (
                        n_invalid_import_only_assignments
                    ),
                    "n_invalid_import_only_sources": (
                        n_invalid_import_only_sources
                    ),
                    "n_invalid_missing_assignments": (
                        n_invalid_missing_assignments
                    ),
                    "n_invalid_missing_sources": (
                        n_invalid_missing_sources
                    ),
                    "n_invalid_selected_quantities": (
                        n_invalid_selected_quantities
                    ),
                    "assignment_contract_valid": assignment_contract_valid,
                    "duration_seconds": elapsed,
                }
            ]
        )

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        if temporary_path.exists():
            temporary_path.unlink()

        temporary_sql = sql_literal(temporary_path)

        con.register(
            "_quantity_selection_audit",
            audit_row,
        )

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM _quantity_selection_audit
            )
            TO {temporary_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        temporary_path.replace(output_path)

    finally:
        con.close()

    logger.info("  Input rows: %s", f"{n_input_rows:,}")
    logger.info("  Output rows: %s", f"{n_output_rows:,}")
    logger.info("  Selected candidate: %s", selected_candidate)
    logger.info(
        "  Evidence metric mismatches: %s",
        f"{evidence_metric_mismatches:,}",
    )
    logger.info(
        "  Invalid bilateral assignments: %s",
        f"{n_invalid_bilateral_assignments:,}",
    )
    logger.info(
        "  Invalid unilateral assignments: %s",
        f"{n_invalid_export_only_assignments + n_invalid_import_only_assignments:,}",
    )
    logger.info(
        "  Invalid missing assignments: %s",
        f"{n_invalid_missing_assignments:,}",
    )
    logger.info("  Output: %s", output_path)

    if not audit_passed:
        logger.error(
            "%s FAIL — Quantity-selection audit failed.",
            get_symbols("cross"),
        )
        raise RuntimeError(
            "Validation quantity-selection audit failed. "
            f"See audit artifact: {output_path}"
        )

    logger.info(
        "%s CHECK — Quantity-selection audit passed.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "quantity_selection_audit_path": output_path,
        "audit_passed": True,
        "selected_candidate": selected_candidate,
        "n_rows": n_output_rows,
        "duration_seconds": elapsed,
        "reused": False,
    }
