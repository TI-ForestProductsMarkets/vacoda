from vacoda.steps.cif_fob.workflow import _resolve_primary_estimation_name


def test_resolve_primary_estimation_name_returns_full_run_name():
    result = {
        "primary_estimation_run": "hist_gradient_boosting__model_5_full_history",
    }

    assert (
        _resolve_primary_estimation_name(
            estimated_rates_result=result,
        )
        == "hist_gradient_boosting__model_5_full_history"
    )


def test_resolve_primary_estimation_name_returns_none_when_missing():
    assert (
        _resolve_primary_estimation_name(
            estimated_rates_result={},
        )
        is None
    )


def test_resolve_primary_estimation_name_rejects_non_run_style_name():
    result = {
        "primary_estimation_run": "hist_gradient_boosting",
    }

    assert (
        _resolve_primary_estimation_name(
            estimated_rates_result=result,
        )
        is None
    )
