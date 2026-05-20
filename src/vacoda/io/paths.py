from __future__ import annotations

from pathlib import Path


def resolve_files(
    base_dir: Path,
    patterns: str | list[str],
    *,
    existing_only: bool = False,
) -> list[Path]:
    """Resolve glob patterns or explicit filenames relative to a base directory."""
    if isinstance(patterns, str):
        patterns = [patterns]

    files: list[Path] = []

    for pattern in patterns:
        matches = sorted(base_dir.glob(pattern))

        if matches:
            files.extend(matches)
        else:
            files.append(base_dir / pattern)

    unique = list(dict.fromkeys(files))

    if existing_only:
        missing = [path for path in unique if not path.exists()]
        if missing:
            msg = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"Configured file(s) not found:\n{msg}")

    return unique