from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress_bar(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "Progress",
    unit: str = "it",
    enabled: bool = True,
    leave: bool = True,
) -> Iterator[T]:
    if not enabled:
        yield from iterable
        return

    yield from tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
    )