from __future__ import annotations

from pathlib import Path


def sql_literal(value: str | Path) -> str:
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def duckdb_scan_relation(
    *,
    path: str | Path,
    file_format: str,
    alias: str | None = None,
    all_varchar: bool = True,
    union_by_name: bool = True,
) -> str:
    path_sql = sql_literal(path)
    fmt = file_format.strip().lower()

    if fmt == "parquet":
        relation = f"read_parquet({path_sql})"

    elif fmt == "csv":
        all_varchar_sql = "TRUE" if all_varchar else "FALSE"
        union_by_name_sql = "TRUE" if union_by_name else "FALSE"

        relation = f"""
        read_csv_auto(
            {path_sql},
            HEADER = TRUE,
            ALL_VARCHAR = {all_varchar_sql},
            union_by_name = {union_by_name_sql}
        )
        """

    else:
        raise ValueError(f"Unsupported DuckDB input format: {file_format!r}")

    if alias:
        return f"{relation} AS {alias}"

    return relation