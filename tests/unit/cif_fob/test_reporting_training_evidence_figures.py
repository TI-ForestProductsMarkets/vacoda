from pathlib import Path
from types import SimpleNamespace
from vacoda.utils.logger import get_logger

import pandas as pd

from vacoda.steps.cif_fob import reporting


def _fake_cfg(target_hs: str = "H5"):
    return SimpleNamespace(
        scope=SimpleNamespace(target_hs=target_hs),
    )


def test_write_training_evidence_audit_figures_writes_all_figures(
    tmp_path,
    monkeypatch,
):
    audit_path = tmp_path / "training_evidence_audit.csv"
    pd.DataFrame({"dummy": [1]}).to_csv(audit_path, index=False)

    def fake_plot(*args, **kwargs):
        fig, ax = reporting.plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig, ax, None

    monkeypatch.setattr(
        reporting,
        "plot_representation_ratio_bars",
        fake_plot,
    )

    figure_dir = tmp_path / "figures" / "training_evidence_audit"
    result = reporting._write_training_evidence_audit_figures(
        cfg=_fake_cfg("H5"),
        training_evidence_audit_result={
            "training_evidence_audit_path": audit_path,
            "status": "completed",
        },
        figure_dir=figure_dir,
        logger=get_logger("TEST"),
    )

    expected_keys = {
        "training_representation_exporter_continent",
        "training_representation_importer_continent",
        "training_representation_value_band",
        "training_representation_distance_band",
        "training_representation_hs4_product_group",
    }

    assert set(result) == expected_keys

    for path in result.values():
        assert isinstance(path, Path)
        assert path.exists()
        assert path.parent == figure_dir


def test_write_training_evidence_audit_figures_returns_empty_when_missing(
    tmp_path,
):
    result = reporting._write_training_evidence_audit_figures(
        cfg=_fake_cfg("H5"),
        training_evidence_audit_result={
            "training_evidence_audit_path": tmp_path / "missing.csv",
            "status": "completed",
        },
        figure_dir=tmp_path / "figures",
        logger=get_logger("TEST"),
    )

    assert result == {}
