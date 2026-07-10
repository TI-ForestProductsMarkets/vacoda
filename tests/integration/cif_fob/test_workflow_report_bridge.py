from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from vacoda.cif_fob import workflow


class _Flag:
    def __init__(self, enable: bool):
        self.enable = enable


class _Cfg:
    def __init__(self, root: Path):
        self._root = root

        self.scope = SimpleNamespace(
            target_hs="H5",
            min_year=2012,
            max_year=2025,
        )

        self.engine = SimpleNamespace(threads=1)

        self.products = SimpleNamespace(
            prepared_input=_Flag(True),
            mirror_pairs=_Flag(True),
            training_sample=_Flag(True),
            training_features=_Flag(True),
            training_evidence_audit=_Flag(True),
            estimated_rates=_Flag(True),
            diagnostics=_Flag(False),
            corrected_trade=_Flag(False),
        )

        self.artifacts = SimpleNamespace(
            enable=True,
            kind="reports",
            subdir="cif_fob",
            filename="report.md",
            timestamp=False,
            write_latest=False,
        )

    def input_path(self) -> Path:
        path = self._root / "input.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
        return path

    def output_dir(self) -> Path:
        return self._root

    def product_path(self, key: str) -> Path:
        suffix = ".csv" if key == "training_evidence_audit" else ".parquet"
        return self._root / f"{key}{suffix}"

    def artifact_path(self, run_id: str) -> Path:
        return self._root / f"report_{run_id}.md"

    def artifact_latest_path(self) -> Path:
        return self._root / "report_latest.md"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def test_workflow_passes_training_evidence_audit_to_report_writer(
    tmp_path,
    monkeypatch,
):
    cfg = _Cfg(tmp_path)

    monkeypatch.setattr(workflow, "timed_workflow", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(workflow, "timed_step", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(workflow, "get_symbols", lambda _: "*")

    monkeypatch.setattr(
        workflow,
        "prepare_cif_fob_input",
        lambda **kwargs: {
            "status": "completed",
            "audit_passed": True,
            "prepared_input_path": _touch(cfg.product_path("prepared_input")),
        },
    )

    monkeypatch.setattr(
        workflow,
        "build_mirror_pairs",
        lambda **kwargs: {
            "status": "completed",
            "mirror_pairs_path": _touch(cfg.product_path("mirror_pairs")),
        },
    )

    monkeypatch.setattr(
        workflow,
        "build_training_sample",
        lambda **kwargs: {
            "status": "completed",
            "training_sample_path": _touch(cfg.product_path("training_sample")),
        },
    )

    monkeypatch.setattr(
        workflow,
        "build_training_features",
        lambda **kwargs: {
            "status": "completed",
            "training_features_path": _touch(cfg.product_path("training_features")),
        },
    )

    monkeypatch.setattr(
        workflow,
        "write_training_evidence_audit",
        lambda **kwargs: {
            "status": "completed",
            "training_evidence_audit_path": _touch(cfg.product_path("training_evidence_audit")),
            "n_rows": 1,
        },
    )

    monkeypatch.setattr(
        workflow,
        "estimate_cif_fob_rates",
        lambda **kwargs: {
            "status": "completed",
            "estimated_rates_path": _touch(cfg.product_path("estimated_rates")),
            "primary_estimation_run": "hist_gradient_boosting__model_5_full_history",
        },
    )

    captured = {}

    def fake_report_writer(**kwargs):
        captured.update(kwargs)
        return {
            "status": "completed",
            "report_path": _touch(cfg.artifact_path("RUN_1")),
            "latest_report_path": _touch(cfg.artifact_latest_path()),
        }

    monkeypatch.setattr(
        workflow,
        "write_cif_fob_decision_report",
        fake_report_writer,
    )

    result = workflow.run_cif_fob_workflow(
        cfg=cfg,
        run_id="RUN_1",
    )

    assert "training_evidence_audit_result" in captured
    assert captured["training_evidence_audit_result"]["status"] == "completed"
    assert captured["training_evidence_audit_result"]["training_evidence_audit_path"].exists()

    assert result["status"] == "completed"
    assert result["decision_report_path"].exists()
