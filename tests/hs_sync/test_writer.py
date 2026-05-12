# tests/hs_sync/test_writer.py

import pandas as pd
import pytest

from vacoda.io.writer import write_csv, write_parquet


def test_write_csv_rejects_empty_dataframe_by_default(tmp_path):
    df = pd.DataFrame(columns=["a", "b"])
    path = tmp_path / "empty.csv"

    with pytest.raises(ValueError, match="DataFrame is empty"):
        write_csv(df, path)


def test_write_csv_allows_empty_dataframe_with_schema(tmp_path):
    df = pd.DataFrame(columns=["a", "b"])
    path = tmp_path / "empty.csv"

    result = write_csv(df, path, allow_empty=True)

    assert result == path
    assert path.exists()

    written = pd.read_csv(path)
    assert list(written.columns) == ["a", "b"]
    assert written.empty


def test_write_csv_rejects_schema_less_dataframe(tmp_path):
    df = pd.DataFrame()
    path = tmp_path / "schema_less.csv"

    with pytest.raises(ValueError, match="no columns"):
        write_csv(df, path, allow_empty=True)


def test_write_parquet_allows_empty_dataframe_with_schema(tmp_path):
    df = pd.DataFrame(columns=["a", "b"])
    path = tmp_path / "empty.parquet"

    result = write_parquet(df, path, allow_empty=True)

    assert result == path
    assert path.exists()

    written = pd.read_parquet(path)
    assert list(written.columns) == ["a", "b"]
    assert written.empty


def test_write_parquet_rejects_schema_less_dataframe(tmp_path):
    df = pd.DataFrame()
    path = tmp_path / "schema_less.parquet"

    with pytest.raises(ValueError, match="no columns"):
        write_parquet(df, path, allow_empty=True)