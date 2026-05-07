import pytest

from vacoda.comtrade.config import JobConfig
from vacoda.comtrade.pipeline import _expand_jobs
from vacoda.comtrade.pipeline import _format_output_file
from vacoda.comtrade.manifest import quarantine_invalid_outputs
from vacoda.comtrade.pipeline import _handle_existing_output
from types import SimpleNamespace

from types import SimpleNamespace

from vacoda.comtrade.config import JobConfig
from vacoda.comtrade.pipeline import _download_single_job


def test_download_single_job_empty_payload_returns_no_data(tmp_path, monkeypatch):
    def fake_build_query_params(default_params, job_params, auth):
        return {"format": "csv", **job_params}

    def fake_build_auth_headers(auth):
        return {}

    def fake_is_csv_format(params):
        return True

    def fake_download_paginated_csv(
        endpoint,
        params,
        retry,
        runtime,
        pagination,
        headers,
    ):
        return b"", ["https://example.test/redacted"], 0

    monkeypatch.setattr(
        "vacoda.comtrade.pipeline.build_query_params",
        fake_build_query_params,
    )
    monkeypatch.setattr(
        "vacoda.comtrade.pipeline.build_auth_headers",
        fake_build_auth_headers,
    )
    monkeypatch.setattr(
        "vacoda.comtrade.pipeline.is_csv_format",
        fake_is_csv_format,
    )
    monkeypatch.setattr(
        "vacoda.comtrade.pipeline.download_paginated_csv",
        fake_download_paginated_csv,
    )

    cfg = SimpleNamespace(
        default_params={"format": "csv"},
        auth=SimpleNamespace(),
        retry=SimpleNamespace(),
        runtime=SimpleNamespace(),
        pagination=SimpleNamespace(enabled=True),
    )

    job = JobConfig(
        name="wood_panel_direct_440112_1989",
        params={"cmdCode": "440112", "period": "1989"},
        output_file="comtrade_{cmdCode}_{period}.csv",
    )

    output_path = tmp_path / "comtrade_440112_1989.csv"

    row = _download_single_job(
        cfg=cfg,
        endpoint="https://example.test",
        job=job,
        output_path=output_path,
    )

    assert row["status"] == "no_data"
    assert row["rows_downloaded"] == 0
    assert row["validation_reason"] == "api_returned_no_rows"
    assert row["output_path"] is None
    assert not output_path.exists()

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




def test_quarantine_invalid_outputs_moves_zero_byte_file(tmp_path):
    output_path = tmp_path / "bad.csv"
    output_path.write_bytes(b"")

    quarantine_dir = tmp_path / "_quarantine"

    invalid_rows = [
        {
            "job": "bad_job",
            "status": "skipped_existing",
            "output_path": str(output_path),
        }
    ]

    moved = quarantine_invalid_outputs(invalid_rows, quarantine_dir)

    assert moved == 1
    assert not output_path.exists()
    assert (quarantine_dir / "bad.csv").exists()


def test_handle_existing_output_valid_file_skips(tmp_path):
    output_path = tmp_path / "valid.csv"
    output_path.write_text(
        "a,b,c\n"
        "1,2,3\n"
        "4,5,6\n"
        "7,8,9\n",
        encoding="utf-8",
    )

    cfg = SimpleNamespace(
        runtime=SimpleNamespace(
            overwrite_existing=False,
            existing_file_policy="redownload",
        )
    )

    job = JobConfig(
        name="test_job",
        params={"cmdCode": "441012", "period": "2023"},
        output_file="comtrade_{cmdCode}_{period}.csv",
    )

    row = _handle_existing_output(
        output_path=output_path,
        cfg=cfg,
        mode="single",
        job=job,
    )

    assert row is not None
    assert row["status"] == "skipped_existing"
    assert row["validation_reason"] == "valid"
    assert row["file_size_bytes"] > 20


def test_handle_existing_output_zero_byte_redownload_returns_none(tmp_path):
    output_path = tmp_path / "bad.csv"
    output_path.write_bytes(b"")

    cfg = SimpleNamespace(
        runtime=SimpleNamespace(
            overwrite_existing=False,
            existing_file_policy="redownload",
        )
    )

    job = JobConfig(
        name="test_job",
        params={"cmdCode": "441012", "period": "2023"},
        output_file="comtrade_{cmdCode}_{period}.csv",
    )

    row = _handle_existing_output(
        output_path=output_path,
        cfg=cfg,
        mode="single",
        job=job,
    )

    assert row is None


def test_handle_existing_output_zero_byte_fail_raises(tmp_path):
    output_path = tmp_path / "bad.csv"
    output_path.write_bytes(b"")

    cfg = SimpleNamespace(
        runtime=SimpleNamespace(
            overwrite_existing=False,
            existing_file_policy="fail",
        )
    )

    job = JobConfig(
        name="test_job",
        params={"cmdCode": "441012", "period": "2023"},
        output_file="comtrade_{cmdCode}_{period}.csv",
    )

    with pytest.raises(RuntimeError):
        _handle_existing_output(
            output_path=output_path,
            cfg=cfg,
            mode="single",
            job=job,
        )

def test_download_single_job_empty_payload_even_without_row_count_returns_no_data(
    tmp_path,
    monkeypatch,
):
    def fake_build_query_params(default_params, job_params, auth):
        return {"format": "csv", **job_params}

    def fake_build_auth_headers(auth):
        return {}

    def fake_is_csv_format(params):
        return True

    def fake_download_paginated_csv(
        endpoint,
        params,
        retry,
        runtime,
        pagination,
        headers,
    ):
        return b"", ["https://example.test/redacted"], None

    monkeypatch.setattr("vacoda.comtrade.pipeline.build_query_params", fake_build_query_params)
    monkeypatch.setattr("vacoda.comtrade.pipeline.build_auth_headers", fake_build_auth_headers)
    monkeypatch.setattr("vacoda.comtrade.pipeline.is_csv_format", fake_is_csv_format)
    monkeypatch.setattr("vacoda.comtrade.pipeline.download_paginated_csv", fake_download_paginated_csv)

    cfg = SimpleNamespace(
        default_params={"format": "csv"},
        auth=SimpleNamespace(),
        retry=SimpleNamespace(),
        runtime=SimpleNamespace(),
        pagination=SimpleNamespace(enabled=True),
    )

    job = JobConfig(
        name="empty_job",
        params={"cmdCode": "440112", "period": "1989"},
        output_file="comtrade_{cmdCode}_{period}.csv",
    )

    output_path = tmp_path / "empty.csv"

    row = _download_single_job(
        cfg=cfg,
        endpoint="https://example.test",
        job=job,
        output_path=output_path,
    )

    assert row["status"] == "no_data"
    assert not output_path.exists()


def test_download_single_job_non_empty_payload_writes_file(tmp_path, monkeypatch):
    def fake_build_query_params(default_params, job_params, auth):
        return {"format": "csv", **job_params}

    def fake_build_auth_headers(auth):
        return {}

    def fake_is_csv_format(params):
        return True

    def fake_download_paginated_csv(
        endpoint,
        params,
        retry,
        runtime,
        pagination,
        headers,
    ):
        payload = (
            b"period,reporterCode,partnerCode,cmdCode,flowCode,primaryValue\n"
            b"2023,276,840,441012,M,1000\n"
        )
        return payload, ["https://example.test/redacted"], 1

    monkeypatch.setattr("vacoda.comtrade.pipeline.build_query_params", fake_build_query_params)
    monkeypatch.setattr("vacoda.comtrade.pipeline.build_auth_headers", fake_build_auth_headers)
    monkeypatch.setattr("vacoda.comtrade.pipeline.is_csv_format", fake_is_csv_format)
    monkeypatch.setattr("vacoda.comtrade.pipeline.download_paginated_csv", fake_download_paginated_csv)

    cfg = SimpleNamespace(
        default_params={"format": "csv"},
        auth=SimpleNamespace(),
        retry=SimpleNamespace(),
        runtime=SimpleNamespace(),
        pagination=SimpleNamespace(enabled=True),
    )

    job = JobConfig(
        name="non_empty_job",
        params={"cmdCode": "441012", "period": "2023"},
        output_file="comtrade_{cmdCode}_{period}.csv",
    )

    output_path = tmp_path / "comtrade_441012_2023.csv"

    row = _download_single_job(
        cfg=cfg,
        endpoint="https://example.test",
        job=job,
        output_path=output_path,
    )

    assert row["status"] == "downloaded"
    assert row["rows_downloaded"] == 1
    assert output_path.exists()
    assert output_path.stat().st_size > 20