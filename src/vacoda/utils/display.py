"""
vacoda.utils.display
-------------------
Utility functions for displaying pandas Series and various symbols.

"""
from __future__ import annotations

import pandas as pd
from collections.abc import Iterable, Iterator
from typing import TypeVar, Iterable, Iterator

from tqdm import tqdm

T = TypeVar("T")

_SYMBOLS = {
    "warning": "\u26A0",              # ⚠
    "check": "\u2705",                # ✅
    "cross": "\u274C",                # ❌
    "info": "\u2139",                 # ℹ
    "star": "\u2B50",                 # ⭐
    "fire": "\U0001F525",             # 🔥
    "important": "\U0001F4A5",        # 💥
    "right_arrow": "\u27A1",          # ➡
    "right_finger": "\U0001F449",     # 👉
    "danger": "\U0001F6AB",           # 🚫
    "success": "\U0001F44D",          # 👍
    "failure": "\U0001F44E",          # 👎
    "time": "\U0001F551",             # 🕑
    "duckdb": "\U0001F986",           # 🦆
    "pandas": "\U0001F43C",           # 🐼
    "python": "\U0001F40D",           # 🐍
    "warning_light": "\U0001F6A8",    # 🚨
    "construction": "\U0001F6A7",     # 🚧
    "recycle": "\U0001F504",          # 🔄
    "workflow": "\U0001F3C1",         # 🏁
    "check_mark": "\u2714",           # ✔
    "pipeline_step": "\u25C6",        # ◆
    "play": "\u25B6",                 # ▶
    "directory": "\U0001F4C2",        # 📂
    "checkpoint": "\U0001F4BE",       # 💾
    "output": "\U0001F4E4",           # 📤
    "audit": "\U0001F4DD",            # 📝
    "log": "\U0001F4D1",              # 📑
    "manifest": "\U0001F4C3",         # 📃
    "skip": "\u2298",                 # ⏭
    "flag": "\U0001F6A9",             # 🚩
    "reuse": "\u267B\uFE0F",          # ♻️
    "rebuild": "\U0001F528",          # 🔨
    "build": "\U0001F6E0\uFE0F",    # 🛠️
    "rebuilt": "\U0001F6E0\uFE0F",    # 🛠️
    "search": "\U0001F50E",           # 🔎
}

_ANSI_COLORS = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
}


def color_text(text: str, color: str) -> str:
    try:
        prefix = _ANSI_COLORS[color]
    except KeyError as exc:
        raise ValueError(f"Unknown color: {color}") from exc

    return f"{prefix}{text}{_ANSI_COLORS['reset']}"

def get_symbols(name: str) -> str:
    try:
        return _SYMBOLS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown symbol name: {name}"
        ) from exc


def pretty_print_series(series: pd.Series, float_fmt: str = ",.2f", int_fmt: str = ","):
    """
    Nicely print a pandas Series with thousands separators and aligned output.
    Floats are formatted to 2 decimal places, ints with commas.
    If the key contains 'percentage', append a % sign.
    """
    for k, v in series.items():
        if isinstance(v, float):
            if 'percentage' in k.lower():
                print(f"{k:45} {format(v, float_fmt)}%")
            else:
                print(f"{k:45} {format(v, float_fmt)}")
        elif isinstance(v, int):
            print(f"{k:45} {format(v, int_fmt)}")
        else:
            print(f"{k:45} {v}")


def get_recovery_symbol(pct, threshold):
    if pct >= threshold:
        return "\u2705"   # ✅ strong recovery
    elif pct > 0:
        return "\u26A0"   # ⚠ partial recovery
    else:
        return "\u274C"   # ❌ no recovery

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

def get_table_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> str:
    return "\n".join([
        '<table style="width:100%; border-collapse:collapse;">',
        "<tr>",
        '<td style="width:50%; vertical-align:top; padding-right:16px;">',
        left.to_html(index=False, border=0),
        "</td>",
        '<td style="width:50%; vertical-align:top; padding-left:16px;">',
        right.to_html(index=False, border=0),
        "</td>",
        "</tr>",
        "</table>",
    ])

def get_html_table(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "<em>No rows available.</em>"

    return dataframe.to_html(
        index=False,
        border=0,
        classes="dataframe",
    )


def get_table_figure_layout(
    dataframe: pd.DataFrame,
    image_markdown: str,
    *,
    table_width: int = 42,
) -> str:
    figure_width = 100 - table_width

    return "\n".join([
        '<table style="width:100%; border-collapse:collapse;">',
        "<tr>",
        (
            f'<td style="width:{table_width}%; vertical-align:top; '
            'padding-right:16px;">'
        ),
        get_html_table(dataframe),
        "</td>",
        (
            f'<td style="width:{figure_width}%; vertical-align:top; '
            'padding-left:16px;">'
        ),
        image_markdown,
        "</td>",
        "</tr>",
        "</table>",
    ])