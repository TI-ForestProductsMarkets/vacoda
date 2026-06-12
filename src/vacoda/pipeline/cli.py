#!/usr/bin/env python3
"""
Run the VACODA master pipeline.
"""

from __future__ import annotations

import argparse

from vacoda.pipeline.pipeline_workflow import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VACODA master pipeline."
    )

    parser.add_argument(
        "--force-run-all",
        action="store_true",
        help="Run all enabled steps even if valid checkpoints exist.",
    )

    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Run without interactive confirmation.",
    )

    parser.add_argument(
        "--target-hs",
        type=str,
        default=None,
        help="Override pipeline scope.target_hs, e.g. H5.",
    )

    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Override pipeline scope.min_year.",
    )

    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Override pipeline scope.max_year.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override pipeline engine.threads.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        force_run_all=args.force_run_all if args.force_run_all else None,
        require_confirmation=False if args.no_confirm else None,
        target_hs=args.target_hs,
        min_year=args.min_year,
        max_year=args.max_year,
        threads=args.threads,
    )

if __name__ == "__main__":
    main()