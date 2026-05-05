"""
Audit a Comtrade download manifest.

Example:
    python scripts/audit_comtrade_manifest.py

or:
    python scripts/audit_comtrade_manifest.py data/raw/api/single/download_manifest_single.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from vacoda.comtrade.manifest import print_manifest_summary


def main() -> None:
    if len(sys.argv) > 1:
        manifest_path = Path(sys.argv[1])
    else:
        manifest_path = Path("data/raw/api/single/download_manifest_single.json")

    print_manifest_summary(manifest_path)


if __name__ == "__main__":
    main()