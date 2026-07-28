"""
vacoda.utils.artifacts
----------------------
Shared helpers for representing and validating workflow artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping


def artifact_result(
    *,
    path_key: str,
    path: str | Path,
) -> dict[str, Any]:
    """
    Return a standard result for an artifact-producing step that did not run.

    If the expected artifact already exists, its status is ``provided`` and
    downstream steps may reuse it. Otherwise, its status is ``skipped``.
    """
    artifact_path = Path(path)
    exists = artifact_path.is_file()

    return {
        "status": "provided" if exists else "skipped",
        path_key: artifact_path,
        "output_path": artifact_path if exists else None,
    }


def require_artifact(
    *,
    result: dict[str, Any],
    path_key: str,
    consumer_step: str,
    producer_product: str,
) -> Path:
    """Return an existing upstream artifact or raise a clear error."""
    value = result.get(path_key)

    if value in (None, ""):
        raise RuntimeError(
            f"{consumer_step} requires {producer_product}, but the "
            f"upstream result does not contain {path_key!r}."
        )

    path = Path(value)

    if not path.is_file():
        raise FileNotFoundError(
            f"{consumer_step} requires {producer_product}, but it was "
            f"not found at {path}. Enable the corresponding product or "
            "provide the existing artifact."
        )

    return path


def require_enabled_products(
    *,
    workflow_name: str,
    enabled_products: Mapping[str, bool],
    required_products: Iterable[str],
) -> None:
    """Raise when required workflow products are disabled."""
    disabled = [
        name
        for name in required_products
        if not enabled_products.get(name, False)
    ]

    if disabled:
        raise RuntimeError(
            f"{workflow_name} requires these products to be enabled: "
            + ", ".join(disabled)
        )


def require_product_dependency(
    *,
    workflow_name: str,
    enabled_products: Mapping[str, bool],
    product: str,
    required_product: str,
) -> None:
    """Require one product when another product is enabled."""
    if (
        enabled_products.get(product, False)
        and not enabled_products.get(required_product, False)
    ):
        raise RuntimeError(
            f"{workflow_name} product {product!r} requires "
            f"{required_product!r} to be enabled."
        )

def enabled_coverage_products(self) -> list[str]:
    return [
        product_name
        for product_name, product in self.coverage.products.items()
        if product.enabled
    ]
