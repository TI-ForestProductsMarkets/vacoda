"""
vacoda.utils.markdown

Utilities for working with Markdown content.
"""

from __future__ import annotations

from pathlib import Path

from vacoda.config.paths import get_markdown_relative_path

DEFAULT_MARKDOWN_IMAGE_WIDTH = 900
DEFAULT_MARKDOWN_IMAGE_HEIGHT = 540

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