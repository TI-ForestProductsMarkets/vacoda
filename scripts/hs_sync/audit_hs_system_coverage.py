"""
scripts/hs_sync/audit_hs_system_coverage.py

Run HS classification year coverage audit.
"""

from vacoda.hs_sync.workflow import run_year_hs_coverage_audit
import multiprocessing


def main() -> None:

    outputs = run_year_hs_coverage_audit(
        input_subdir="api/single",
        output_subdir="hs_sync",
        pattern="*.csv",
        overwrite=True,
    )

    print("\nSaved outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()