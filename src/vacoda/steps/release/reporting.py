"""Compact, evidence-led reporting for published VACODA Release datasets.

Reporting is descriptive; publication authority belongs to the Release audit.
Figures follow the shared VACODA figure-directory convention.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
import shutil

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vacoda.config.paths import get_figure_dir
from vacoda.config.steps.release import ReleaseConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.markdown import insert_markdown_toc, markdown_image
from vacoda.utils.sql import sql_literal


SIDECARS = ("journey", "availability", "review", "missing_reasons", "distributions")
STAGE_LABELS = {
    "hs_sync": "HS Sync", "quantity": "Quantity", "cif_fob": "CIF/FOB",
    "trade_sync": "Trade Sync", "value": "Value", "validation_prepared": "Validation prepared",
    "validation_inference": "Validation inference", "release_enriched": "Release",
}
FIELD_LABELS = {
    "trade_value_reconciled_constant_usd": "Trade value (constant USD)",
    "trade_synced_quantity_kg": "Synchronized quantity (kg)",
    "model_prediction_quantity_kg": "Model prediction (kg)",
    "trade_sync_uv_usd_per_kg": "Observed unit value (USD/kg)",
    "model_uv_usd_per_kg": "Model unit value (USD/kg)",
    "quantity_abs_log_residual": "Absolute log quantity residual",
}
REASON_LABELS = {
    "export_missing_and_import_corrected_missing": "Export missing; corrected import missing",
    "export_nonpositive": "Export value nonpositive",
    "export_missing_import_corrected_present": "Export missing; corrected import present (not necessarily usable)",
    "other_missing_reconciled_value": "Other missing-value cases",
    "source_not_provided": "CIF/FOB evidence not provided",
}


def _require(label: str, path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} missing or empty: {path}")


def _fetch(con: duckdb.DuckDBPyConnection, path: Path) -> list[dict[str, Any]]:
    _require(path.name, path)
    result = con.execute(f"SELECT * FROM read_parquet({sql_literal(path)})")
    columns = [item[0] for item in result.description]
    return [dict(zip(columns, values)) for values in result.fetchall()]


def _fmt(value: Any, *, year: bool = False, decimals: int = 2) -> str:
    if value is None:
        return "—"
    if year:
        return str(int(value))
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{decimals}f}"
    return str(value).replace("|", r"\|").replace("\n", " ")


def _table(headers: list[str], rows: list[tuple[Any, ...]]) -> list[str]:
    if not rows:
        return ["_No observations._"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return lines


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_journey(journey: list[dict[str, Any]], path: Path) -> None:
    available = [r for r in sorted(journey, key=lambda x: x["stage_order"])
                 if r["stage_status"] == "read" and r["rows"] is not None]
    if not available:
        raise RuntimeError("No stages available for journey figure")
    names = [STAGE_LABELS.get(r["stage"], r["stage"]) for r in available]
    values = [int(r["rows"]) for r in available]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.46 * len(available) + 1)))
    bars = ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("Rows at each stage (different grains; not a loss rate)")
    ax.ticklabel_format(axis="x", style="plain")
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values[::-1]):
        ax.annotate(f"{value:,}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.28)
    _save(fig, path)


def _plot_annual_coverage(annual: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(annual, key=lambda r: r["year"])
    years = [int(r["year"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for key, label in (
        ("synced_quantity_rows", "Synchronized quantity"),
        ("nominal_value_rows", "Reconciled trade value"),
        ("raw_model_prediction_rows", "Raw model prediction"),
    ):
        ax.plot(years, [100 * r[key] / r["standard_rows"] for r in rows], marker="o", label=label)
    ax.set_ylabel("Coverage (% of annual Release rows)")
    ax.set_xlabel("Year")
    ax.set_xticks(years)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left")
    _save(fig, path)


def _plot_residual_profile(distributions: list[dict[str, Any]], path: Path) -> bool:
    match = next((r for r in distributions if r["period_type"] == "all_years"
                  and r["field"] == "quantity_abs_log_residual"), None)
    if not match or not match["usable_rows"]:
        return False
    vals = [float(match[key]) for key in ("p05", "p25", "median", "p75", "p95")]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.hlines(0, vals[0], vals[-1], linewidth=2)
    ax.hlines(0, vals[1], vals[3], linewidth=9)
    ax.plot(vals[2], 0, "o", markersize=9, label="Median")
    ax.plot([vals[0], vals[-1]], [0, 0], "|", markersize=17)
    ax.set_yticks([])
    ax.set_xlabel("Absolute log residual (paired observed vs model quantity)")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper right")
    _save(fig, path)
    return True


def build_release_report(
    *,
    cfg: ReleaseConfig,
    standard_path: Path,
    enriched_path: Path,
    audit_path: Path,
    diagnostics_path: Path,
    run_id: str,
    release_id: str,
    logger,
    force: bool = False,
) -> dict[str, Any]:
    """Write a concise Markdown Release report and diagnostic figures."""
    start = perf_counter()
    standard_path, enriched_path = Path(standard_path), Path(enriched_path)
    audit_path, diagnostics_path = Path(audit_path), Path(diagnostics_path)
    expected = {
        "standard": cfg.product_path("release", release_id=release_id),
        "enriched": cfg.enriched_path(release_id=release_id),
        "audit": cfg.product_path("release_audit", release_id=release_id),
        "diagnostics": cfg.product_path("diagnostics", release_id=release_id),
    }
    for name, path in {"standard": standard_path, "enriched": enriched_path,
                       "audit": audit_path, "diagnostics": diagnostics_path}.items():
        if path != expected[name]:
            raise ValueError(f"Release {name} path does not match release identity: {path}")
        _require(name, path)
    standard_metadata = cfg.metadata_path(release_id=release_id)
    enriched_metadata = cfg.metadata_path(enriched=True, release_id=release_id)
    _require("standard metadata", standard_metadata)
    _require("enriched metadata", enriched_metadata)
    if not cfg.artifacts.enable:
        raise RuntimeError("Release reporting is required by the workflow")
    report_path = cfg.artifacts_path(run_id)
    report_path = report_path.with_name(f"{report_path.stem}_{release_id}{report_path.suffix}")
    latest_path = cfg.artifacts_latest_path()
    if report_path.exists() and not force:
        raise FileExistsError(f"Release report already exists: {report_path}")

    # Match Value reporting: central figure root / stage / target revision.
    figure_dir = get_figure_dir() / "release" / f"target_{cfg.scope.target_hs.lower()}"
    figure_paths = {
        "journey": figure_dir / f"{release_id}_processing_journey.png",
        "annual_coverage": figure_dir / f"{release_id}_annual_coverage.png",
        "residual_profile": figure_dir / f"{release_id}_residual_profile.png",
    }
    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")
        audit_records = _fetch(con, audit_path)
        if len(audit_records) != 1:
            raise RuntimeError("Release audit must contain one row")
        audit = audit_records[0]
        if (audit.get("run_id") != run_id or audit.get("release_id") != release_id
                or audit.get("audit_passed") is not True):
            raise RuntimeError("Release reporting requires the successful matching audit")
        checks = [(k, v) for k, v in audit.items()
                  if k not in {"run_id", "release_id", "audit_passed"} and isinstance(v, bool)]
        if any(not value for _, value in checks):
            raise RuntimeError("Release audit contains failed invariant checks")

        summary = _fetch(con, diagnostics_path)
        overall_rows = [r for r in summary if r["period_type"] == "all_years"]
        if len(overall_rows) != 1:
            raise RuntimeError("Expected exactly one all-years diagnostics row")
        overall = overall_rows[0]
        annual = sorted((r for r in summary if r["period_type"] == "year"),
                        key=lambda r: r["year"])
        sidecars = {
            name: _fetch(con, diagnostics_path.with_name(f"{diagnostics_path.stem}_{name}.parquet"))
            for name in SIDECARS
        }
        journey = sorted(sidecars["journey"], key=lambda r: r["stage_order"])
        availability = sidecars["availability"]
        review = sorted(sidecars["review"], key=lambda r: (-r["review_rows"], r["target_cmd"]))
        reasons = sorted(sidecars["missing_reasons"], key=lambda r: (-r["rows"], r["reason"]))
        dist = [r for r in sidecars["distributions"] if r["period_type"] == "all_years"]
        lookup = {(r["stage"], r["field"]): r for r in availability}
        final_specs = (
            ("Reconciled value", "trade_value_reconciled"),
            ("Synchronized quantity", "trade_synced_quantity_kg"),
            ("Raw model prediction", "model_prediction_quantity_kg"),
        )
        final_coverage = []
        for label, field in final_specs:
            item = lookup.get(("release_enriched", field))
            if item is None:
                raise RuntimeError(f"Release availability missing: {field}")
            final_coverage.append((label, item))
        missing_value_rows = int(overall["enriched_rows"]) - int(overall["nominal_value_rows"])
        if sum(int(r["rows"]) for r in reasons) != missing_value_rows:
            raise RuntimeError("Missing-reason totals differ from missing monetary values")
        if sum(int(r["review_rows"]) for r in review) != overall["manual_review_rows"]:
            raise RuntimeError("Review totals differ from overall diagnostics")

        _plot_journey(journey, figure_paths["journey"])
        _plot_annual_coverage(annual, figure_paths["annual_coverage"])
        residual_figure = _plot_residual_profile(sidecars["distributions"], figure_paths["residual_profile"])
        n_checks = len(checks)
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
            "## Data Journey",
            "",
            f"**Published · Audit passed** | `{release_id}` | {cfg.scope.target_hs} | "
            f"{cfg.scope.min_year}–{cfg.scope.max_year}",
            f"Run `{run_id}` · Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC",
            "", "__TOC__", "",
            "## 1. Data Journey", "",
            "### 1.1 Processing populations", "",
            *_table(["Stage", "Rows / groups", "Read status"], [
                (STAGE_LABELS.get(r["stage"], r["stage"]),
                 _fmt(r["rows"]), r["stage_status"]) for r in journey
            ]), "",
            markdown_image("Processing populations", figure_paths["journey"], document_path=report_path), "",
            "_HS Sync and Quantity are source-row populations; CIF/FOB and subsequent stages are bilateral observations. "
            "Filtering, aggregation and mirror pairing must not be interpreted as missing values._", "",
            "### 1.2 Annual population", "",
            *_table(["Year", "Bilateral rows", "Quantity usable", "Value usable", "Raw predictions"], [
                (_fmt(r["year"], year=True), _fmt(r["standard_rows"]),
                 _fmt(r["synced_quantity_rows"]), _fmt(r["nominal_value_rows"]),
                 _fmt(r["raw_model_prediction_rows"])) for r in annual
            ]), "",
            "## 2. Availability and Missingness", "",
            "### 2.1 Final dataset coverage", "",
            *_table(["Measure", "Usable", "Missing", "Invalid", "Coverage"], [
                (label, _fmt(r["usable"]), _fmt(r["missing"]),
                 _fmt(r["nonpositive_or_invalid"]), f'{r["coverage_pct"]:.2f}%')
                for label, r in final_coverage
            ]), "",
            markdown_image("Annual data availability", figure_paths["annual_coverage"], document_path=report_path), "",
            "### 2.2 Missing reconciled values", "",
            *_table(["Observed CIF/FOB evidence", "Rows"], [
                (REASON_LABELS.get(r["reason"], r["reason"].replace("_", " ")),
                 _fmt(r["rows"])) for r in reasons
            ]), "",
            "_Evidence categories describe observed values, not established causes of missing factors._", "",
            "## 3. Model Predictions and Review", "",
            "### 3.1 Training, inference and publication", "",
            *_table(["Population", "Observations"], [
                ("Training selected", _fmt(overall["training_selected_rows"])),
                ("Raw model predictions", _fmt(overall["raw_model_prediction_rows"])),
                ("Standard predictions retained", _fmt(overall["standard_prediction_rows"])),
                ("Manual review", _fmt(overall["manual_review_rows"])),
                ("Support not evaluated", _fmt(overall["support_not_evaluated_rows"])),
            ]), "",
            "### 3.2 HS6 requiring review", "",
            *_table(["HS6", "Rows", "First year", "Last year", "HS6 unseen", "HS4 unseen"], [
                (r["target_cmd"], _fmt(r["review_rows"]),
                 _fmt(r["first_year"], year=True), _fmt(r["last_year"], year=True),
                 _fmt(r["hs6_unseen_rows"]), _fmt(r["hs4_unseen_rows"])) for r in review
            ]), "",
            "_Review requires both model-support flags explicitly FALSE. Raw predictions remain in enriched; "
            "standard predictions are withheld. HS-mapping provenance is not available in these diagnostics._", "",
            "### 3.3 Prediction discrepancy", "",
        ]
        if residual_figure:
            lines.extend([markdown_image("Absolute log residual profile (P05–P95)",
                                         figure_paths["residual_profile"], document_path=report_path), ""])
        lines.extend([
            "## 4. Statistical Profile", "",
            *_table(["Measure", "Usable", "P05", "Median", "P95"], [
                (FIELD_LABELS.get(r["field"], r["field"].replace("_", " ")),
                 _fmt(r["usable_rows"]), _fmt(r["p05"]), _fmt(r["median"]), _fmt(r["p95"]))
                for r in dist
            ]), "",
            "_Quantiles for different measures need not refer to the same observations._", "",
            "## 5. Publication and Integrity", "",
            "### 5.1 Published products", "",
            *_table(["Product", "Rows", "Columns"], [
                ("Standard", _fmt(overall["standard_rows"]), _fmt(len(cfg.schema.standard))),
                ("Enriched", _fmt(overall["enriched_rows"]), _fmt(len(cfg.schema.enriched))),
            ]), "",
            f"**Audit:** passed · {n_checks} checks · 0 failed. "
            f"**Review predictions withheld from standard:** {_fmt(overall['manual_review_rows'])}.", "",
            "### 5.2 Published artifacts", "",
            *_table(["Artifact", "Path"], [
                ("Standard", f"`{standard_path}`"),
                ("Enriched", f"`{enriched_path}`"),
                ("Standard metadata", f"`{standard_metadata}`"),
                ("Enriched metadata", f"`{enriched_metadata}`"),
                ("Audit", f"`{audit_path}`"),
                ("Diagnostics", f"`{diagnostics_path}`"),
                ("Figures", f"`{figure_dir}`"),
            ]), "",
            "## Appendix — Technical Reference", "",
            "### A. Audit checks", "",
            *_table(["Check", "Passed"], [
                (name.replace("_", " "), _fmt(value)) for name, value in checks
            ]), "",
            "### B. Published schemas", "",
            "**Standard:** " + ", ".join(f"`{name}`" for name in cfg.schema.standard), "",
            "**Enriched additions:** " + ", ".join(f"`{name}`" for name in cfg.schema.enriched_additional), "",
            "_Release diagnostics describe the published data; the upstream Release audit authorized publication._", "",
        ])
        missing_stages = [r["stage"] for r in journey if r["stage_status"] != "read"]
        if missing_stages:
            lines.extend(["**Upstream stages not provided:** " + ", ".join(missing_stages), ""])
        insert_markdown_toc(lines, min_level=2, max_level=3)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_path.with_name(report_path.name + ".tmp")
        try:
            tmp.write_text("\n".join(lines), encoding="utf-8")
            _require("temporary Release report", tmp)
            tmp.replace(report_path)
        finally:
            tmp.unlink(missing_ok=True)
        latest_report_path = None
        if cfg.artifacts.write_latest:
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            latest_tmp = latest_path.with_name(latest_path.name + ".tmp")
            try:
                shutil.copyfile(report_path, latest_tmp)
                latest_tmp.replace(latest_path)
            finally:
                latest_tmp.unlink(missing_ok=True)
            latest_report_path = latest_path
        logger.info("%s Release report created: %s", get_symbols("check"), report_path)
        logger.info("  Figures: %s", figure_dir)
        return {
            "status": "completed", "run_id": run_id, "release_id": release_id,
            "output_path": report_path, "report_path": report_path,
            "latest_report_path": latest_report_path,
            "figure_paths": [path for key, path in figure_paths.items()
                             if key != "residual_profile" or residual_figure],
            "figures": {key: path for key, path in figure_paths.items()
                        if key != "residual_profile" or residual_figure},
            "decision": "PUBLISHED — AUDIT PASSED", "audit_passed": True,
            "n_rows": int(overall["standard_rows"]),
            "n_manual_review_rows": int(overall["manual_review_rows"]),
            "n_failed_audit_checks": 0,
            "duration_seconds": perf_counter() - start,
        }
    finally:
        con.close()
