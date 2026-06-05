# src/vacoda/io/reports.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Serialize a DataFrame into a GitHub-flavored markdown table."""

    def _sanitize(value: object) -> str:
        text = str(value)
        return text.replace("\n", " ").replace("|", r"\|")

    headers = [str(col) for col in df.columns]
    if not headers:
        return ""

    rows_data = [
        [_sanitize(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]

    col_widths = [len(header) for header in headers]
    for row in rows_data:
        for idx, cell in enumerate(row):
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(cell))

    header_row = "| " + " | ".join(
        header.ljust(col_widths[idx]) for idx, header in enumerate(headers)
    ) + " |"
    divider_row = "| " + " | ".join(
        "-" * col_widths[idx] for idx in range(len(headers))
    ) + " |"

    if not rows_data:
        return "\n".join([header_row, divider_row])

    rows: list[str] = []
    for row in rows_data:
        rows.append(
            "| " + " | ".join(
                cell.ljust(col_widths[idx]) for idx, cell in enumerate(row)
            ) + " |"
        )

    return "\n".join([header_row, divider_row, *rows])

class TextReport:
    """Small helper for building plain-text or markdown-style reports."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str = "") -> None:
        self.lines.append(text)

    def section(self, title: str, *, char: str = "=") -> None:
        bar = char * 80
        self.line()
        self.line(bar)
        self.line(title)
        self.line(bar)

    def subsection(self, title: str, *, char: str = "-") -> None:
        bar = char * 80
        self.line()
        self.line(title)
        self.line(bar)

    def dataframe(self, df: pd.DataFrame, *, index: bool = False) -> None:
        self.line(df.to_string(index=index))

    def markdown_dataframe(self, df: pd.DataFrame, *, index: bool = False) -> None:
        if index:
            df = df.reset_index()
        self.line(_dataframe_to_markdown_table(df))

    def render(self) -> str:
        return "\n".join(self.lines)
        
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