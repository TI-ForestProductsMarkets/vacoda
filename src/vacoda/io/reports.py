# src/vacoda/io/reports.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _with_timestamp(path: Path, timestamp: str | None = None) -> Path:
    ts = timestamp or _timestamp()
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def write_text_report(
    content: str,
    *,
    report_dir: Path,
    filename: str,
    timestamp: bool = True,
    write_latest: bool = True,
    run_id: str | None = None,
) -> dict[str, Path]:
    """
    Write a text/markdown report with optional timestamp and latest copy.

    Example:
        write_text_report(
            markdown,
            report_dir=Path("/storage/vacoda/output/reports"),
            filename="inspect_hscodes_requiring_weights.md",
        )
    """

    report_dir.mkdir(parents=True, exist_ok=True)

    base_path = report_dir / filename

    if timestamp:
        report_path = _with_timestamp(base_path, timestamp=run_id)
    else:
        report_path = base_path

    report_path.write_text(content, encoding="utf-8")

    outputs = {"report": report_path}

    if write_latest and timestamp:
        latest_path = base_path.with_name(
            f"{base_path.stem}_latest{base_path.suffix}"
        )
        latest_path.write_text(content, encoding="utf-8")
        outputs["latest"] = latest_path

    return outputs


def write_dataframe_report(
    df: pd.DataFrame,
    *,
    report_dir: Path,
    filename: str,
    timestamp: bool = True,
    write_latest: bool = True,
    run_id: str | None = None,
    index: bool = False,
) -> dict[str, Path]:
    """
    Write a DataFrame report as CSV with optional timestamp and latest copy.
    """

    report_dir.mkdir(parents=True, exist_ok=True)

    base_path = report_dir / filename

    if timestamp:
        report_path = _with_timestamp(base_path, timestamp=run_id)
    else:
        report_path = base_path

    df.to_csv(report_path, index=index)

    outputs = {"report": report_path}

    if write_latest and timestamp:
        latest_path = base_path.with_name(
            f"{base_path.stem}_latest{base_path.suffix}"
        )
        df.to_csv(latest_path, index=index)
        outputs["latest"] = latest_path

    return outputs


def write_markdown_and_csv_report(
    *,
    markdown: str,
    df: pd.DataFrame | None,
    report_dir: Path,
    stem: str,
    timestamp: bool = True,
    write_latest: bool = True,
    run_id: str | None = None,
) -> dict[str, Path]:
    """
    Convenience writer for reports that have both a markdown version and
    an optional CSV table.

    Example outputs:
        audit_name_20260601_120000.md
        audit_name_latest.md
        audit_name_20260601_120000.csv
        audit_name_latest.csv
    """

    outputs: dict[str, Path] = {}

    outputs.update(
        {
            f"markdown_{key}": value
            for key, value in write_text_report(
                markdown,
                report_dir=report_dir,
                filename=f"{stem}.md",
                timestamp=timestamp,
                write_latest=write_latest,
                run_id=run_id,
            ).items()
        }
    )

    if df is not None:
        outputs.update(
            {
                f"csv_{key}": value
                for key, value in write_dataframe_report(
                    df,
                    report_dir=report_dir,
                    filename=f"{stem}.csv",
                    timestamp=timestamp,
                    write_latest=write_latest,
                    run_id=run_id,
                ).items()
            }
        )

    return outputs