from pathlib import Path
from vacoda.comtrade.config import load_comtrade_config


def test_load_config_explicit_years(tmp_path):
    config_path = tmp_path / "comtrade.yaml"
    config_path.write_text(
        """
bulk:
  enabled: true
  jobs:
    - name: "hs_all_reporters"
      params:
        typeCode: "C"
        freqCode: "A"
        clCode: "HS"
      years:
        enabled: true
        start: 2024
        end: 2025
      output_file: "Comtrade_bulk_HS_{period}.zip"
  output:
    directory: "data/raw/api/bulk"
    manifest_file: "download_manifest_bulk.json"

single:
  enabled: false
  jobs: []
  output:
    directory: "data/raw/api/single"
    manifest_file: "download_manifest_single.json"

endpoint:
  single: "https://example.com/single"
  single_fallbacks: []
  bulk: "https://example.com/bulk"

default_params:
  format: "csv"

retry:
  max_attempts: 3
  backoff_seconds: 2.0

runtime:
  timeout_seconds: 300
  overwrite_existing: false
  existing_file_policy: "redownload"
  max_workers: 8

auth:
  enabled: false

pagination:
  enabled: true
""",
        encoding="utf-8",
    )

    cfg = load_comtrade_config(config_path)

    job = cfg.bulk.jobs[0]
    assert job.all_years_start == 2024
    assert job.all_years_end == 2025
    assert job.use_reference_years is False

def test_load_config_reference_years(tmp_path):
    config_path = tmp_path / "comtrade.yaml"
    config_path.write_text(
        """
bulk:
  enabled: false
  jobs: []
  output:
    directory: "data/raw/api/bulk"
    manifest_file: "download_manifest_bulk.json"

single:
  enabled: true
  jobs:
    - name: "wood_panel_direct"
      params:
        cmdCode: "441012"
      output_file: "comtrade_{cmdCode}_{period}.csv"
      use_reference_years: true
      fallback_years_start: 1989
      fallback_years_end: 2025
  output:
    directory: "data/raw/api/single"
    manifest_file: "download_manifest_single.json"

endpoint:
  single: "https://example.com/single"
  single_fallbacks: []
  bulk: "https://example.com/bulk"

default_params:
  format: "csv"

retry:
  max_attempts: 3
  backoff_seconds: 2.0

runtime:
  timeout_seconds: 300
  overwrite_existing: false
  existing_file_policy: "redownload"
  max_workers: 8

auth:
  enabled: false

pagination:
  enabled: true
""",
        encoding="utf-8",
    )

    cfg = load_comtrade_config(config_path)

    job = cfg.single.jobs[0]
    assert job.all_years_start is None
    assert job.all_years_end is None
    assert job.use_reference_years is True
    assert job.fallback_years_start == 1989
    assert job.fallback_years_end == 2025