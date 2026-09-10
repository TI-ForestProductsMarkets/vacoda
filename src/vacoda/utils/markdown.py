"""
vacoda.utils.markdown

Utilities for working with Markdown content.
"""

from __future__ import annotations

from pathlib import Path
import re
from vacoda.config.paths import get_markdown_relative_path

DEFAULT_MARKDOWN_IMAGE_WIDTH = 700
DEFAULT_MARKDOWN_IMAGE_HEIGHT = 440

def markdown_image(
    alt_text: str,
    image_path: Path | str,
    *,
    document_path: Path | str,
    width: int | None = DEFAULT_MARKDOWN_IMAGE_WIDTH,
    height: int | None = DEFAULT_MARKDOWN_IMAGE_HEIGHT,
) -> str:
    """Return a local image reference for a Markdown report."""
    image = Path(image_path)

    relative_path = get_markdown_relative_path(
        image,
        document_path=document_path,
    )

    if width is None and height is None:
        return f"![{alt_text}]({relative_path})"

    attributes = [
        f'src="{relative_path}"',
        f'alt="{alt_text}"',
    ]

    if width is not None:
        attributes.append(f'width="{int(width)}"')

    if height is not None:
        attributes.append(f'height="{int(height)}"')

    return f"<img {' '.join(attributes)} />"


def markdown_anchor(text: str) -> str:
    """Return a Markdown-compatible anchor for a heading."""
    text = text.strip().lower()

    # Remove Markdown formatting characters and punctuation.
    text = re.sub(r"[^\w\s-]", "", text)

    # Collapse whitespace into hyphens.
    text = re.sub(r"\s+", "-", text)

    return text

def build_markdown_toc(
    lines: list[str],
    *,
    min_level: int = 2,
    max_level: int = 4,
    title: str = "Table of contents",
    flat_sublevels: bool = False,
) -> list[str]:
    """Build a Markdown table of contents from report heading lines."""

    toc = [
        f"{'#' * min_level} {title}",
        "",
    ]

    heading_pattern = re.compile(
        r"^(#{1,6})\s+(.+)$"
    )

    for line in lines:
        match = heading_pattern.match(
            line
        )

        if match is None:
            continue

        hashes, heading = match.groups()
        level = len(hashes)

        if not min_level <= level <= max_level:
            continue

        anchor = markdown_anchor(
            heading
        )

        toc_heading = re.sub(
            r"[*_`~]",
            "",
            heading,
        )

        if flat_sublevels:
            indent = (
                "&nbsp;"
                * 4
                * (level - min_level)
            )

            toc.append(
                f"{indent}[{toc_heading}](#{anchor})<br>"
            )

        else:
            indent = (
                "  "
                * (level - min_level)
            )

            toc.append(
                f"{indent}- [{toc_heading}](#{anchor})"
            )

    toc.append("")

    return toc


def insert_markdown_toc(
    lines: list[str],
    *,
    placeholder: str = "__TOC__",
    min_level: int = 2,
    max_level: int = 4,
    title: str = "Table of contents",
    flat_sublevels: bool = False,
) -> None:
    """Replace a TOC placeholder with a generated Markdown TOC."""
    try:
        toc_index = lines.index(placeholder)
    except ValueError:
        raise ValueError(
            f"Markdown TOC placeholder not found: {placeholder!r}"
        ) from None

    toc = build_markdown_toc(
        lines,
        min_level=min_level,
        max_level=max_level,
        title=title,
        flat_sublevels=flat_sublevels,
    )

    lines[toc_index:toc_index + 1] = toc