from pathlib import Path

from vacoda.steps.validation.reporting import _build_validation_figure_paths


def test_validation_figure_paths_include_model_performance_by_year():
    figure_paths = _build_validation_figure_paths(
        figure_dir=Path("/tmp/figures"),
        target_hs="4401",
    )

    assert "model_performance_by_year" in figure_paths
    assert (
        figure_paths["model_performance_by_year"].name
        == "target_4401_validation_model_performance_by_year.png"
    )
