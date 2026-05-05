from vacoda.comtrade.config import JobConfig
from vacoda.comtrade.pipeline import _expand_jobs
from vacoda.comtrade.pipeline import _format_output_file


def test_expand_jobs_expands_cmd_code_list_and_year_range() -> None:
    job = JobConfig(
        name="wood_panel",
        params={"cmdCode": ["441012", "441011"]},
        output_file="comtrade_{cmdCode}_{period}.csv",
        all_years_start=2024,
        all_years_end=2025,
    )

    jobs = _expand_jobs([job])

    assert len(jobs) == 4

    observed = {(job.params["cmdCode"], job.params["period"]) for job in jobs}

    assert observed == {
        ("441012", "2024"),
        ("441012", "2025"),
        ("441011", "2024"),
        ("441011", "2025"),
    }


def test_expand_jobs_converts_single_item_cmd_code_list_to_scalar() -> None:
    job = JobConfig(
        name="wood_panel",
        params={"cmdCode": ["441012"]},
        output_file="comtrade_{cmdCode}_{period}.csv",
        all_years_start=2024,
        all_years_end=2024,
    )

    jobs = _expand_jobs([job])

    assert len(jobs) == 1
    assert jobs[0].params["cmdCode"] == "441012"
    assert jobs[0].params["period"] == "2024"



def test_format_output_file_uses_cmd_code_and_period() -> None:
    output = _format_output_file(
        "comtrade_{cmdCode}_{period}.csv",
        {"cmdCode": "441012", "period": "2025"},
    )

    assert output == "comtrade_441012_2025.csv"