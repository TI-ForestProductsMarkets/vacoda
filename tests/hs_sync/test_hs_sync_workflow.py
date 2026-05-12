# tests/hs_sync/test_hs_sync_workflow.py

from pathlib import Path

import pandas as pd

from vacoda.hs_sync import workflow


def write_input_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "ReporterCode": [276, 276, 250, 250],
            "ReporterISO": ["DEU", "DEU", "FRA", "FRA"],
            "ReporterDesc": ["Germany", "Germany", "France", "France"],
            "RefYear": [2020, 2020, 2021, 2021],
            "ClassificationCode": ["H5", "H5", "H6", "H6"],
            "ClassificationSearchCode": ["HS", "HS", "HS", "HS"],
            "IsOriginalClassification": [True, True, True, True],
            "CmdCode": ["441012", "441011", "441012", "441011"],
            "FlowCode": ["M", "X", "M", "X"],
            "PrimaryValue": [1000.0, 2000.0, 1500.0, 2500.0],
            "NetWgt": [100.0, 200.0, 150.0, 250.0],
        }
    )
    df.to_csv(path, index=False)


def make_test_config(tmp_path: Path) -> dict:
    return {
        "coverage": {
            "input": {
                "stage": "raw",
                "subdir": "api/single",
                "pattern": "*.csv",
            },
            "output": {
                "stage": "interim",
                "subdir": "hs_sync",
            },
            "engine": {
                "name": "pandas",
                "threads": 1,
            },
            "progress": {
                "enabled": False,
            },
            "write": {
                "csv": True,
                "parquet": True,
                "overwrite": True,
                "timestamp": False,
                "allow_empty": True,
            },
            "products": {
                "year_hs_counts": {
                    "enabled": True,
                    "filename": "year_hs_counts_long",
                    "columns": {
                        "year": "RefYear",
                        "hs": "ClassificationCode",
                    },
                },
                "hs_system_overall_summary": {
                    "enabled": True,
                    "depends_on": "year_hs_counts",
                    "filename": "hs_system_overall_summary",
                },
                "year_hs_count_pivot": {
                    "enabled": True,
                    "depends_on": "year_hs_counts",
                    "filename": "year_hs_counts_pivot",
                },
                "year_hs_percentage_pivot": {
                    "enabled": True,
                    "depends_on": "year_hs_count_pivot",
                    "filename": "year_hs_percentage_pivot",
                },
                "year_hs_mix_summary": {
                    "enabled": True,
                    "depends_on": "year_hs_count_pivot",
                    "filename": "year_hs_mix_summary",
                    "thresholds": {
                        "clean": 98.0,
                        "mostly_clean": 95.0,
                    },
                },
                "reporter_year_hs_coverage": {
                    "enabled": True,
                    "filename": "reporter_year_hs_coverage",
                    "columns": {
                        "reporter_code": "ReporterCode",
                        "reporter_iso": "ReporterISO",
                        "reporter_desc": "ReporterDesc",
                        "year": "RefYear",
                        "hs": "ClassificationCode",
                        "hs_search": "ClassificationSearchCode",
                        "original": "IsOriginalClassification",
                        "cmd": "CmdCode",
                        "flow": "FlowCode",
                        "value": "PrimaryValue",
                        "netweight": "NetWgt",
                    },
                },
                "reporter_hs_timeline": {
                    "enabled": True,
                    "depends_on": "reporter_year_hs_coverage",
                    "filename": "reporter_hs_timeline",
                },
                "reporter_year_dominant_hs": {
                    "enabled": True,
                    "depends_on": "reporter_year_hs_coverage",
                    "filename": "reporter_year_dominant_hs",
                },
                "hs_transition_events": {
                    "enabled": True,
                    "depends_on": "reporter_year_hs_coverage",
                    "filename": "hs_transition_events",
                },
                "mixed_classification_reporter_years": {
                    "enabled": True,
                    "depends_on": "reporter_year_hs_coverage",
                    "filename": "mixed_classification_reporter_years",
                },
            },
            "report": {
                "enabled": True,
                "output_kind": "reports",
                "filename": "audit_hs_system_coverage_report.txt",
                "timestamp": False,
                "write_latest": False,
            },
        }
    }


def test_run_hs_system_coverage_audit_writes_outputs(
    tmp_path,
    monkeypatch,
):
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"

    input_dir = data_root / "tmp"
    input_dir.mkdir(parents=True)

    write_input_csv(input_dir / "sample.csv")

    cfg = make_test_config(tmp_path)

    def fake_get_data_dir(stage: str) -> Path:
        return data_root / stage

    def fake_get_output_dir(kind: str) -> Path:
        return output_root / kind

    monkeypatch.setattr(workflow, "get_data_dir", fake_get_data_dir)
    monkeypatch.setattr(workflow, "get_output_dir", fake_get_output_dir)
    monkeypatch.setattr(workflow, "load_hs_sync_config", lambda _: cfg)

    outputs = workflow.run_hs_system_coverage_audit(
        config_path=tmp_path / "dummy.yaml",
        run_id="TEST_RUN",
    )

    out_dir = data_root / "interim" / "hs_sync"
    report_path = output_root / "reports" / "audit_hs_system_coverage_report.txt"

    assert outputs["output_dir"] == out_dir
    assert report_path.exists()

    expected_csvs = [
        "year_hs_counts_long.csv",
        "hs_system_overall_summary.csv",
        "year_hs_counts_pivot.csv",
        "year_hs_percentage_pivot.csv",
        "year_hs_mix_summary.csv",
        "reporter_year_hs_coverage.csv",
        "reporter_hs_timeline.csv",
        "reporter_year_dominant_hs.csv",
        "hs_transition_events.csv",
        "mixed_classification_reporter_years.csv",
    ]

    for filename in expected_csvs:
        assert (out_dir / filename).exists()


def test_workflow_skips_products_when_dependency_missing(
    tmp_path,
    monkeypatch,
):
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"

    input_dir = data_root / "tmp"
    input_dir.mkdir(parents=True)

    write_input_csv(input_dir / "sample.csv")

    cfg = make_test_config(tmp_path)

    cfg["coverage"]["products"]["year_hs_count_pivot"]["enabled"] = False

    def fake_get_data_dir(stage: str) -> Path:
        return data_root / stage

    def fake_get_output_dir(kind: str) -> Path:
        return output_root / kind

    monkeypatch.setattr(workflow, "get_data_dir", fake_get_data_dir)
    monkeypatch.setattr(workflow, "get_output_dir", fake_get_output_dir)
    monkeypatch.setattr(workflow, "load_hs_sync_config", lambda _: cfg)

    workflow.run_hs_system_coverage_audit(
        config_path=tmp_path / "dummy.yaml",
        run_id="TEST_RUN",
    )

    out_dir = data_root / "interim" / "hs_sync"

    assert not (out_dir / "year_hs_counts_pivot.csv").exists()
    assert not (out_dir / "year_hs_percentage_pivot.csv").exists()
    assert not (out_dir / "year_hs_mix_summary.csv").exists()