"""VACODA Release: document audited and published datasets.

Reporting is descriptive. The Release audit and publisher, not this module,
authorize publication.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
import shutil

import duckdb

from vacoda.config.steps.release import ReleaseConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.markdown import insert_markdown_toc
from vacoda.utils.sql import sql_literal


def _require_nonempty(label: str, path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value).replace("|", r"\|").replace("\n", " ")


def _markdown_table(headers: list[str], rows: list[tuple[Any, ...]]) -> list[str]:
    if not rows:
        return ["_No rows available._"]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *[
            "| " + " | ".join(_cell(value) for value in row) + " |"
            for row in rows
        ],
    ]


def build_release_report(
    *,
    cfg: ReleaseConfig,
    standard_path: Path,
    enriched_path: Path,
    audit_path: Path,
    diagnostics_path: Path,
    run_id: str,
    release_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Write the descriptive Markdown Release decision/reporting artifact.

    Both datasets have already passed the release audit and have been published.
    This function does not alter either dataset or the publication decision.
    """
    start = perf_counter()

    standard_path = Path(standard_path)
    enriched_path = Path(enriched_path)
    audit_path = Path(audit_path)
    diagnostics_path = Path(diagnostics_path)

    expected = {
        "standard": cfg.product_path("release", release_id=release_id),
        "enriched": cfg.enriched_path(release_id=release_id),
        "audit": cfg.product_path("release_audit", release_id=release_id),
        "diagnostics": cfg.product_path("diagnostics", release_id=release_id),
    }
    actual = {
        "standard": standard_path,
        "enriched": enriched_path,
        "audit": audit_path,
        "diagnostics": diagnostics_path,
    }
    for name, path in actual.items():
        if path != expected[name]:
            raise ValueError(
                f"Release {name} path does not match release identity: {path}"
            )
        _require_nonempty(name, path)

    standard_metadata_path = cfg.metadata_path(release_id=release_id)
    enriched_metadata_path = cfg.metadata_path(
        enriched=True, release_id=release_id
    )
    _require_nonempty("standard metadata", standard_metadata_path)
    _require_nonempty("enriched metadata", enriched_metadata_path)

    if not cfg.artifacts.enable:
        raise RuntimeError("Release reporting is required by the workflow.")

    report_path = cfg.artifacts_path(run_id)
    latest_path = cfg.artifacts_latest_path()

    # Run IDs are unique to an execution, but release identity is also included
    # to prevent a report for one published release from overwriting another.
    report_path = report_path.with_name(
        f"{report_path.stem}_{release_id}{report_path.suffix}"
    )

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")
        audit_sql = sql_literal(audit_path)
        diagnostics_sql = sql_literal(diagnostics_path)

        audit_rows = con.execute(
            f"SELECT * FROM read_parquet({audit_sql})"
        ).fetchall()
        audit_columns = [
            description[0] for description in con.description
        ]
        if len(audit_rows) != 1:
            raise RuntimeError("Release audit must contain exactly one row.")
        audit = dict(zip(audit_columns, audit_rows[0]))
        if (
            audit.get("run_id") != run_id
            or audit.get("release_id") != release_id
            or audit.get("audit_passed") is not True
        ):
            raise RuntimeError(
                "Release reporting requires a successful audit for this run and release."
            )

        diagnostics_rows = con.execute(f"""
            SELECT *
            FROM read_parquet({diagnostics_sql})
            ORDER BY
                CASE WHEN period_type = 'all_years' THEN 1 ELSE 0 END,
                year
        """).fetchall()
        diagnostics_columns = [
            description[0] for description in con.description
        ]
        if not diagnostics_rows:
            raise RuntimeError("Release diagnostics are empty.")

        summary = [
            dict(zip(diagnostics_columns, row))
            for row in diagnostics_rows
        ]
        overall_matches = [
            row for row in summary if row["period_type"] == "all_years"
        ]
        if len(overall_matches) != 1:
            raise RuntimeError("Expected one all-years Release summary.")
        overall = overall_matches[0]
        annual = [row for row in summary if row["period_type"] == "year"]

        # Audit flags are recorded, not reinterpreted as new approval rules.
        checks = [
            (name, bool(value))
            for name, value in audit.items()
            if name not in {"audit_passed", "run_id", "release_id"}
            and isinstance(value, bool)
        ]
        n_failed = sum(not passed for _, passed in checks)
        if n_failed:
            raise RuntimeError(
                "Audit artifact unexpectedly contains failed checks: "
                + ", ".join(name for name, ok in checks if not ok)
            )

        decision = "PUBLISHED — AUDIT PASSED"
        lines: list[str] = [
            "# VACODA Release Report",
            "",
            f"**Decision:** {decision}  ",
            f"**Run ID:** `{run_id}`  ",
            f"**Release identity:** `{release_id}`  ",
            f"**Target HS:** {cfg.scope.target_hs}  ",
            f"**Years:** {cfg.scope.min_year}–{cfg.scope.max_year}  ",
            "**Generated (UTC):** "
            + datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "",
            "__TOC__",
            "",
            "## 1. Purpose and publication policy",
            "",
            "Release publishes two complementary datasets from Validation "
            "Prepared and Inference. The standard product exposes the "
            "essential trade and model fields. The enriched product "
            "adds bilateral provenance, reconciliation and model evidence.",
            "",
            "The Release audit was performed on both candidate datasets "
            "**before publication**. This report records that result and "
            "describes the published artifacts; it does not make a new "
            "scientific or publication approval decision.",
            "",
            "When `manual_review_required` is TRUE, the **standard** "
            "`model_prediction_quantity_kg` is NULL; the **enriched** "
            "dataset preserves the raw model prediction. "
            "`model_prediction_available` records whether a raw prediction "
            "was generated, regardless of suppression in the standard product.",
            "",
            "Manual review is required only when both model support flags "
            "are explicitly FALSE (`hs6_seen IS FALSE AND hs4_seen IS FALSE`). "
            "Unevaluated NULL support flags do not independently trigger review.",
            "",
            "## 2. Published files and upstream provenance",
            "",
            *[
                f"- **{label}:** `{path}`"
                for label, path in (
                    ("Standard dataset", standard_path),
                    ("Enriched dataset", enriched_path),
                    ("Standard metadata", standard_metadata_path),
                    ("Enriched metadata", enriched_metadata_path),
                    ("Release audit", audit_path),
                    ("Release diagnostics", diagnostics_path),
                )
            ],
            "",
            "Both Release datasets are constructed from Validation Prepared "
            "and Validation Inference and joined on "
            "`(year, exporter_iso, importer_iso, target_cmd)`. "
            "The metadata sidecars contain additional provenance.",
            "",
            "## 3. Publication summary",
            "",
            *(_markdown_table(
                ["Measure", "Count"],
                [
                    ("Standard observations", overall["standard_rows"]),
                    ("Enriched observations", overall["enriched_rows"]),
                    ("HS6 codes", overall["hs6_codes"]),
                    ("Trade-sync quantity available", overall["synced_quantity_rows"]),
                    ("Raw model prediction available", overall["raw_model_prediction_rows"]),
                    ("Standard prediction retained", overall["standard_prediction_rows"]),
                    ("Manual-review observations", overall["manual_review_rows"]),
                    ("Manual-review HS6 codes", overall["manual_review_hs6_codes"]),
                    ("Model-input valid rows", overall["model_input_valid_rows"]),
                    ("Support not evaluated", overall["support_not_evaluated_rows"]),
                ],
            )),
            "",
            "## 4. Year-by-year overview",
            "",
            *(_markdown_table(
                ["Year", "Records", "Sync quantities", "Raw predictions",
                 "Standard predictions", "Manual review"],
                [
                    (
                        row["year"],
                        row["standard_rows"],
                        row["synced_quantity_rows"],
                        row["raw_model_prediction_rows"],
                        row["standard_prediction_rows"],
                        row["manual_review_rows"],
                    )
                    for row in annual
                ],
            )),
            "",
            "## 5. Audit results",
            "",
            f"**Audit passed:** TRUE. **Checks recorded:** {len(checks)}. "
            f"**Failed checks:** {n_failed}.",
            "",
            *(_markdown_table(
                ["Invariant", "Passed"],
                [(name.replace("_", " "), "Yes" if passed else "No")
                 for name, passed in checks],
            )),
            "",
            "## 6. Released schemas",
            "",
            f"**Standard:** {len(cfg.schema.standard)} columns.  ",
            f"**Enriched:** {len(cfg.schema.enriched)} columns "
            "(standard + additional).",
            "",
            "### Standard columns",
            "",
            *[f"- `{column}`" for column in cfg.schema.standard],
            "",
            "### Additional enriched columns",
            "",
            *[f"- `{column}`" for column in cfg.schema.enriched_additional],
            "",
            "## 7. Interpretation and limitations",
            "",
            "- Model predictions are independent model evidence, not a "
            "replacement for `trade_synced_quantity_kg`.",
            "- Manual-review flags identify explicitly unseen HS6 and HS4 "
            "categories; they do not, by themselves, invalidate the dataset.",
            "- Residual statistics compare the observed synchronized "
            "quantity with the model on rows where comparison is possible.",
            "- Unit-value diagnostics should be interpreted using the "
            "value basis and reference definitions recorded in the "
            "release metadata.",
            "- Aggregate statistics and report tables are informational; "
            "they do not override the technical Release audit.",
            "",
            "## 8. Final status",
            "",
            "**Published — technical Release audit passed.**",
            "",
            "Both datasets and metadata sidecars were present at report "
            "generation time. Publication was authorized by the Release "
            "audit in the previous workflow step.",
            "",
        ]
        insert_markdown_toc(lines, min_level=2, max_level=3)

        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists() and not force:
            # Identical run/release combination should not silently overwrite
            # a previously generated report.
            raise FileExistsError(
                f"Release report already exists: {report_path}"
            )

        tmp = report_path.with_name(report_path.name + ".tmp")
        try:
            tmp.write_text("\n".join(lines), encoding="utf-8")
            if tmp.stat().st_size == 0:
                raise RuntimeError("Release report is empty.")
            tmp.replace(report_path)
        finally:
            tmp.unlink(missing_ok=True)

        latest_report_path: Path | None = None
        if cfg.artifacts.write_latest:
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            latest_tmp = latest_path.with_name(latest_path.name + ".tmp")
            try:
                shutil.copyfile(report_path, latest_tmp)
                latest_tmp.replace(latest_path)
            finally:
                latest_tmp.unlink(missing_ok=True)
            latest_report_path = latest_path

        logger.info("%s Release report created.", get_symbols("check"))
        logger.info("  Report: %s", report_path)
        logger.info("  Audit passed: %s", audit["audit_passed"])
        logger.info("  Decision: %s", decision)
        if latest_report_path is not None:
            logger.info("  Latest report: %s", latest_report_path)

        return {
            "status": "completed",
            "run_id": run_id,
            "release_id": release_id,
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": latest_report_path,
            "decision": decision,
            "audit_passed": True,
            "n_rows": int(overall["standard_rows"]),
            "n_manual_review_rows": int(overall["manual_review_rows"]),
            "n_failed_audit_checks": n_failed,
            "duration_seconds": perf_counter() - start,
        }
    finally:
        con.close()
