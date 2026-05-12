# tests/hs_sync/test_hs_sync_config.py

import pytest

from vacoda.config.hs_sync import (
    get_coverage_product_columns,
    get_coverage_product_dependency,
    get_coverage_product_filename,
    is_coverage_product_enabled,
)


def make_cfg():
    return {
        "coverage": {
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
            }
        }
    }


def test_get_coverage_product_filename():
    cfg = make_cfg()

    filename = get_coverage_product_filename("year_hs_counts", cfg)

    assert filename == "year_hs_counts_long"


def test_get_coverage_product_filename_raises_if_missing():
    cfg = {
        "coverage": {
            "products": {
                "broken_product": {
                    "enabled": True,
                }
            }
        }
    }

    with pytest.raises(KeyError, match="Missing required 'filename'"):
        get_coverage_product_filename("broken_product", cfg)


def test_get_coverage_product_columns_required_keys():
    cfg = make_cfg()

    columns = get_coverage_product_columns(
        "year_hs_counts",
        cfg,
        required=("year", "hs"),
    )

    assert columns["year"] == "RefYear"
    assert columns["hs"] == "ClassificationCode"


def test_get_coverage_product_columns_raises_for_missing_required_key():
    cfg = {
        "coverage": {
            "products": {
                "year_hs_counts": {
                    "enabled": True,
                    "filename": "year_hs_counts_long",
                    "columns": {
                        "year": "RefYear",
                    },
                }
            }
        }
    }

    with pytest.raises(KeyError, match="Missing required column"):
        get_coverage_product_columns(
            "year_hs_counts",
            cfg,
            required=("year", "hs"),
        )


def test_get_coverage_product_dependency():
    cfg = make_cfg()

    dependency = get_coverage_product_dependency(
        "hs_system_overall_summary",
        cfg,
    )

    assert dependency == "year_hs_counts"


def test_is_coverage_product_enabled():
    cfg = make_cfg()

    assert is_coverage_product_enabled("year_hs_counts", cfg) is True
    assert is_coverage_product_enabled("unknown_product", cfg) is False
    assert is_coverage_product_enabled(
        "unknown_product",
        cfg,
        default=True,
    ) is True