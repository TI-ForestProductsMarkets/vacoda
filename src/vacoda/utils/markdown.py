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

    markdown_heading_pattern = re.compile(
        r"^(#{1,6})\s+(.+)$"
    )

    html_heading_pattern = re.compile(
        r'^<h([1-6])(?:\s+[^>]*)?>(.*?)</h\1>$',
        re.IGNORECASE,
    )

    html_id_pattern = re.compile(
        r'\bid=["\']([^"\']+)["\']',
        re.IGNORECASE,
    )

    for line in lines:

        markdown_match = markdown_heading_pattern.match(
            line
        )

        html_match = html_heading_pattern.match(
            line
        )

        if markdown_match is not None:

            hashes, heading = markdown_match.groups()

            level = len(
                hashes
            )

            anchor = markdown_anchor(
                heading
            )

        elif html_match is not None:

            level_text, heading = html_match.groups()

            level = int(
                level_text
            )

            heading = re.sub(
                r"<[^>]+>",
                "",
                heading,
            )

            id_match = html_id_pattern.search(
                line
            )

            if id_match is not None:
                anchor = id_match.group(
                    1
                )
            else:
                anchor = markdown_anchor(
                    heading
                )

        else:
            continue

        if not min_level <= level <= max_level:
            continue

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

def markdown_styled_heading(
    text: str,
    *,
    level: int = 4,
    font_size: str = "1.20em",
    margin_top: str = "1.2em",
    color: str | None = None,
) -> str:
    """Return a styled HTML heading for Markdown reports."""

    if not 1 <= level <= 6:
        raise ValueError(
            f"Markdown heading level must be between 1 and 6: {level}"
        )

    anchor = markdown_anchor(text)

    style_parts = [
        f"font-size:{font_size}",
        f"margin-top:{margin_top}",
    ]

    if color is not None:
        style_parts.append(
            f"color:{color}"
        )

    style = "; ".join(style_parts)

    return (
        f'<h{level} '
        f'id="{anchor}" '
        f'style="{style};">'
        f'{text}'
        f'</h{level}>'
    )

def markdown_styled_heading_deprecated(
    text: str,
    *,
    level: int = 4,
    font_size: str = "1.20em",
    margin_top: str = "1.2em",
) -> str:
    """Return a styled HTML heading for Markdown reports."""

    if not 1 <= level <= 6:
        raise ValueError(
            f"Markdown heading level must be between 1 and 6: {level}"
        )

    anchor = markdown_anchor(
        text
    )

    return (
        f'<h{level} '
        f'id="{anchor}" '
        f'style="font-size:{font_size}; '
        f'margin-top:{margin_top};">'
        f'{text}'
        f'</h{level}>'
    )


def insert_markdown_toc(
    lines: list[str],
    *,
    placeholder: str = "__TOC__",
    min_level: int = 2,
    max_level: int = 4,
    title: str = "Table of contents",
    flat_sublevels: bool = False,
    heading_styles: dict[int, dict[str, str]] | None = None,
) -> None:
    """Replace a TOC placeholder and optionally style Markdown headings."""

    if heading_styles:
        markdown_heading_pattern = re.compile(
            r"^(#{1,6})\s+(.+)$"
        )

        for index, line in enumerate(lines):
            match = markdown_heading_pattern.match(line)

            if match is None:
                continue

            hashes, heading = match.groups()
            level = len(hashes)

            style = heading_styles.get(level)

            if style is None:
                continue

            lines[index] = markdown_styled_heading(
                heading,
                level=level,
                font_size=style.get("font_size", "1.20em"),
                color=style.get("color"),
            )

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

def insert_markdown_toc_deprecated(
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