from __future__ import annotations

from pathlib import Path


def sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"