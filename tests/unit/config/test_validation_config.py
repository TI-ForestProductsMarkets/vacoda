from vacoda.config.steps.validation import ValidationStructuralConfig


def test_validation_structural_config_allows_omitted_optional_entries() -> None:
    cfg = ValidationStructuralConfig.from_dict(
        {
            "required_fields": [
                "year",
                "exporter_iso",
                "importer_iso",
                "target_cmd",
            ],
            "canonical_key": [
                "year",
                "exporter_iso",
                "importer_iso",
                "target_cmd",
            ],
            "require_positive_value": True,
            "require_positive_quantity": True,
        }
    )

    assert cfg.required_fields == (
        "year",
        "exporter_iso",
        "importer_iso",
        "target_cmd",
    )
    assert cfg.canonical_key == (
        "year",
        "exporter_iso",
        "importer_iso",
        "target_cmd",
    )
    assert cfg.valid_flows == ("M", "X")
    assert cfg.duplicate_keys.fail_on_duplicates is True
    assert cfg.require_positive_value is True
    assert cfg.require_positive_quantity is True
