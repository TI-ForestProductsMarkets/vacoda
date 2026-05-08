"""
Tests for the COMTRADE API client.
Note: These tests are marked as "live_api" and will only run if the COMTRADE_API_KEY environment variable is set. 
They are designed to test the actual API calls and may take longer to run than unit tests.

Run using: pytest -m live_api
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

pytestmark = pytest.mark.live_api


ENV_FILE = Path(".env")

# Load .env before pytest evaluates skipif
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


@pytest.mark.skipif(
    not os.getenv("COMTRADE_API_KEY"),
    reason="COMTRADE_API_KEY not set",
)
def test_live_small_comtrade_query():
    from vacoda.comtrade.api import ComtradeAPIClient

    client = ComtradeAPIClient()

    result = client.query(
        type_code="C",
        freq_code="A",
        cl_code="HS",
        period="2020",        # annual data should usually be year, not 202001
        reporter_code="840",  # USA
        partner_code="156",   # China
        cmd_code="441012",    # Wood panels
    )

    assert "dataset" in result
    assert len(result["dataset"]) > 0