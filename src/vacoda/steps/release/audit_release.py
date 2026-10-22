"""VACODA Release candidate audit; publication is a separate workflow step.

Checks the two staged datasets against Validation Prepared and Inference.
An audit row is written even if hard checks fail. No candidate is modified.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb

from vacoda.steps.release.release_data import PUBLIC_COLUMN_NAMES
from vacoda.config.steps.release import ReleaseConfig
from vacoda.utils.sql import sql_literal

KEY = ("year", "exporter_iso", "importer_iso", "target_cmd")


def _schema(con: duckdb.DuckDBPyConnection, path: Path) -> tuple[str, ...]:
    return tuple(str(row[0]) for row in con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
    ).fetchall())


def audit_release_candidates(
    *,
    cfg: ReleaseConfig,
    prepared_input_path: Path,
    inference_input_path: Path,
    candidate_standard_path: Path,
    candidate_enriched_path: Path,
    run_id: str,
    release_id: str,
    logger,
) -> dict[str, Any]:
    """Audit staged candidates and return audit_passed only on success.

    Both output schemas are checked in exact YAML column order. Source and
    candidate populations must have identical, complete, unique keys.
    Standard suppresses manual-review predictions; enriched retains them.
    """
    start = perf_counter()
    paths = {
        "prepared": Path(prepared_input_path),
        "inference": Path(inference_input_path),
        "standard": Path(candidate_standard_path),
        "enriched": Path(candidate_enriched_path),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Release audit {name} not found: {path}")

    audit_path = cfg.product_path("release_audit", release_id=release_id)
    if audit_path.suffix.lower() != ".parquet":
        raise ValueError("Release audit product must be Parquet.")
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")
        actual = {name: _schema(con, path) for name, path in paths.items()}
        expected_standard = tuple(cfg.schema.standard)
        expected_enriched = tuple(cfg.schema.enriched)

        def public_schema(columns):
            return tuple(PUBLIC_COLUMN_NAMES.get(c, c) for c in columns)

        schema_ok = {
            "standard_schema_exact": actual["standard"] == public_schema(expected_standard),
            "enriched_schema_exact": actual["enriched"] == public_schema(expected_enriched),
        }
        needed = {
            "prepared": set(KEY) | {"trade_value_reconciled", "trade_synced_quantity_kg"},
            "inference": set(KEY) | {"model_prediction_quantity_kg", "model_prediction_available",
                                     "manual_review_required", "hs6_seen", "hs4_seen"},
            "standard": set(KEY) | {"trade_value_reconciled", "trade_synced_quantity_kg",
                                    "model_prediction_quantity_kg", "model_prediction_available",
                                    "manual_review_required"},
            "enriched": set(KEY) | {"trade_value_reconciled", "trade_synced_quantity_kg",
                                    "model_prediction_quantity_kg", "model_prediction_available",
                                    "manual_review_required", "hs6_seen", "hs4_seen"},
        }
        missing = {
            name: sorted(
                PUBLIC_COLUMN_NAMES.get(field, field)
                for field in fields
                if (
                    PUBLIC_COLUMN_NAMES.get(field, field)
                    if name in ("standard", "enriched")
                    else field
                ) not in actual[name]
            )
            for name, fields in needed.items()
        }
        if any(missing.values()):
            raise ValueError(f"Release audit missing essential fields: {missing}")

        for name, path in paths.items():
            if name in ("standard", "enriched"):
                aliases = ", ".join(
                    f'"{public}" AS "{internal}"'
                    for internal, public in PUBLIC_COLUMN_NAMES.items()
                    if internal in needed[name] or internal in expected_enriched
                    if public in actual[name]
                )

                select = f"*, {aliases}" if aliases else "*"

            else:
                select = "*"

            con.execute(
                f"CREATE TEMP VIEW {name} AS "
                f"SELECT {select} FROM read_parquet({sql_literal(path)})"
            )

        key_expr = ", ".join(KEY)
        keys_missing = " OR ".join(f"{key} IS NULL" for key in KEY)
        metrics: dict[str, int] = {}
        for name in paths:
            n, missing_count, distinct_count = con.execute(f"""
                SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE {keys_missing}),
                       COUNT(DISTINCT ({key_expr}))
                FROM {name}
            """).fetchone()
            metrics[f"{name}_rows"] = int(n)
            metrics[f"{name}_missing_key_rows"] = int(missing_count)
            metrics[f"{name}_duplicate_key_rows"] = int(n - distinct_count)

        join = " AND ".join(f"s.{k} = e.{k}" for k in KEY)
        metrics["standard_enriched_key_mismatches"] = int(con.execute(f"""
            SELECT COUNT(*) FROM standard s FULL OUTER JOIN enriched e
            ON {join} WHERE s.year IS NULL OR e.year IS NULL
        """).fetchone()[0])
        for left, right in (("prepared", "standard"), ("inference", "standard")):
            clause = " AND ".join(f"a.{k} = b.{k}" for k in KEY)
            metrics[f"{left}_standard_key_mismatches"] = int(con.execute(f"""
                SELECT COUNT(*) FROM {left} a FULL OUTER JOIN {right} b
                ON {clause} WHERE a.year IS NULL OR b.year IS NULL
            """).fetchone()[0])

        metrics["standard_enriched_shared_mismatches"] = int(con.execute(f"""
            SELECT COUNT(*) FROM standard s JOIN enriched e ON {join}
            WHERE {" OR ".join(f's."{c}" IS DISTINCT FROM e."{c}"' for c in expected_standard if c != 'model_prediction_quantity_kg')}
        """).fetchone()[0])
        # Compare publication policy against original inference, not merely
        # against the already-generated enriched candidate.
        join_si = " AND ".join(f"s.{k} = i.{k}" for k in KEY)
        join_ei = " AND ".join(f"e.{k} = i.{k}" for k in KEY)
        metrics["standard_prediction_mismatches"] = int(con.execute(f"""
            SELECT COUNT(*) FROM standard s JOIN inference i ON {join_si}
            WHERE s.model_prediction_quantity_kg IS DISTINCT FROM
                CASE WHEN i.manual_review_required IS TRUE THEN NULL
                     ELSE i.model_prediction_quantity_kg END
        """).fetchone()[0])
        metrics["enriched_prediction_mismatches"] = int(con.execute(f"""
            SELECT COUNT(*) FROM enriched e JOIN inference i ON {join_ei}
            WHERE e.model_prediction_quantity_kg
                  IS DISTINCT FROM i.model_prediction_quantity_kg
        """).fetchone()[0])
        metrics["manual_review_rule_mismatches"] = int(con.execute("""
            SELECT COUNT(*) FROM inference
            WHERE manual_review_required IS DISTINCT FROM
                  (hs6_seen IS FALSE AND hs4_seen IS FALSE)
        """).fetchone()[0])
        metrics["manual_review_rows"] = int(con.execute("""
            SELECT COUNT(*) FROM inference WHERE manual_review_required IS TRUE
        """).fetchone()[0])
        metrics["manual_review_hs6_codes"] = int(con.execute("""
            SELECT COUNT(DISTINCT target_cmd) FROM inference
            WHERE manual_review_required IS TRUE
        """).fetchone()[0])

        # Independent comparison of canonical trade values and quantities.
        join_ps = " AND ".join(f"p.{k} = s.{k}" for k in KEY)
        metrics["prepared_standard_trade_mismatches"] = int(con.execute(f"""
            SELECT COUNT(*) FROM prepared p JOIN standard s ON {join_ps}
            WHERE p.trade_value_reconciled IS DISTINCT FROM s.trade_value_reconciled
               OR p.trade_synced_quantity_kg IS DISTINCT FROM s.trade_synced_quantity_kg
        """).fetchone()[0])
        metrics["inference_standard_flag_mismatches"] = int(con.execute(f"""
            SELECT COUNT(*) FROM inference i JOIN standard s ON {join_si}
            WHERE i.model_prediction_available IS DISTINCT FROM s.model_prediction_available
               OR i.manual_review_required IS DISTINCT FROM s.manual_review_required
        """).fetchone()[0])

        # All non-derived enriched evidence must match its original source.
        from vacoda.steps.release.release_data import PREPARED_FIELDS, INFERENCE_FIELDS, DERIVED
        prepared_extra = sorted((set(expected_enriched) & PREPARED_FIELDS)
                                - {"trade_value_reconciled", "trade_synced_quantity_kg"})
        inference_extra = sorted((set(expected_enriched) & INFERENCE_FIELDS)
                                 - {"model_prediction_quantity_kg", "model_prediction_available",
                                    "manual_review_required"})
        metrics["prepared_enriched_evidence_mismatches"] = int(con.execute(f"""
            SELECT COUNT(*) FROM prepared p JOIN enriched e
            ON {" AND ".join(f"p.{k} = e.{k}" for k in KEY)}
            WHERE {" OR ".join(f'p."{c}" IS DISTINCT FROM e."{c}"' for c in prepared_extra)}
        """).fetchone()[0]) if prepared_extra else 0
        metrics["inference_enriched_evidence_mismatches"] = int(con.execute(f"""
            SELECT COUNT(*) FROM inference i JOIN enriched e
            ON {" AND ".join(f"i.{k} = e.{k}" for k in KEY)}
            WHERE {" OR ".join(f'i."{c}" IS DISTINCT FROM e."{c}"' for c in inference_extra)}
        """).fetchone()[0]) if inference_extra else 0

        n = metrics["prepared_rows"]
        checks = {
            **schema_ok,
            "population_nonempty": n > 0,
            "population_equal": all(metrics[f"{name}_rows"] == n for name in paths),
            "keys_complete_unique": all(
                metrics[f"{name}_missing_key_rows"] == 0
                and metrics[f"{name}_duplicate_key_rows"] == 0
                for name in paths
            ),
            "canonical_key_sets_match": all(
                metrics[key] == 0 for key in (
                    "standard_enriched_key_mismatches",
                    "prepared_standard_key_mismatches",
                    "inference_standard_key_mismatches",
                )
            ),
            "standard_enriched_shared_fields_consistent":
                metrics["standard_enriched_shared_mismatches"] == 0,
            "standard_prediction_policy_consistent":
                metrics["standard_prediction_mismatches"] == 0,
            "enriched_prediction_preserved":
                metrics["enriched_prediction_mismatches"] == 0,
            "manual_review_rule_consistent": metrics["manual_review_rule_mismatches"] == 0,
            "prepared_trade_fields_preserved":
                metrics["prepared_standard_trade_mismatches"] == 0,
            "inference_flags_preserved":
                metrics["inference_standard_flag_mismatches"] == 0,
            "enriched_prepared_evidence_preserved":
                metrics["prepared_enriched_evidence_mismatches"] == 0,
            "enriched_inference_evidence_preserved":
                metrics["inference_enriched_evidence_mismatches"] == 0,
        }
        audit_passed = all(checks.values())
        audit_values: dict[str, Any] = {
            "run_id": run_id,
            "release_id": release_id,
            **metrics,
            **checks,
            "audit_passed": audit_passed,
        }
        select = []
        for name, value in audit_values.items():
            if isinstance(value, bool):
                sql = "TRUE" if value else "FALSE"
            elif isinstance(value, int):
                sql = str(value)
            else:
                sql = sql_literal(str(value))
            select.append(f"{sql} AS \"{name}\"")
        tmp = audit_path.with_name(audit_path.name + ".tmp")
        try:
            tmp.unlink(missing_ok=True)
            con.execute(f"COPY (SELECT {', '.join(select)}) TO {sql_literal(tmp)} "
                        "(FORMAT PARQUET, COMPRESSION ZSTD)")
            tmp.replace(audit_path)
        finally:
            tmp.unlink(missing_ok=True)

        failed_checks = [name for name, ok in checks.items() if not ok]
        logger.info("Release candidate audit: rows=%s, manual-review=%s (%s HS6 codes)",
                    f"{n:,}", f"{metrics['manual_review_rows']:,}",
                    f"{metrics['manual_review_hs6_codes']:,}")
        logger.info("Release audit artifact: %s", audit_path)
        if failed_checks:
            raise RuntimeError("Release candidate audit failed: " + ", ".join(failed_checks))
        logger.info("Release candidate audit PASSED; candidates may be published.")
        return {
            "status": "completed",
            "run_id": run_id,
            "release_id": release_id,
            "audit_path": audit_path,
            "audit_passed": True,
            "n_rows": n,
            "n_manual_review_rows": metrics["manual_review_rows"],
            "n_manual_review_hs6_codes": metrics["manual_review_hs6_codes"],
            "duration_seconds": perf_counter() - start,
        }
    finally:
        con.close()
