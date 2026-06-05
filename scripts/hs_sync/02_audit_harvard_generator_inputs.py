"""
scripts/hs_sync/02_audit_harvard_generator_inputs.py

Audit staged Harvard-generator inputs and generated matrices.

This is a validation/audit step, not a data-producing pipeline step.
"""

from __future__ import annotations

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.hs_sync.harvard_audit import run_harvard_generator_audit_from_config


def main() -> None:
    cfg = HsSyncConfig.from_yaml()
    outputs = run_harvard_generator_audit_from_config(cfg)

    if outputs.get("status") == "disabled":
        print("harvard_generator_audit is disabled in config; skipping.")
        return

    print("Harvard generator audit complete.")
    print(f"Artifact directory: {outputs.get('artifact_dir')}")
    print(f"Artifact: {outputs.get('artifact')}")
    if outputs.get("artifact_latest") is not None:
        print(f"Artifact latest: {outputs.get('artifact_latest')}")

if __name__ == "__main__":
    main()