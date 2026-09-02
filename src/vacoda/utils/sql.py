"""
vacoda.utils.sql
SQL utility functions.

"""
from __future__ import annotations

from pathlib import Path


def sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"

def sql_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + value.replace('"', '""') + '"'