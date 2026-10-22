from pathlib import Path

# Internal name: (unit, definition)
VARIABLES = {
    "year": ("Year", "Calendar year of the bilateral trade observation."),
    "exporter_iso": ("Code", "ISO code of the exporting country."),
    "importer_iso": ("Code", "ISO code of the importing country."),
    "target_cmd": ("Code", "Six-digit Harmonized System commodity code."),
    "trade_value_reconciled": ("Nominal USD", "Bilateral trade value reconciled from exporter and importer reports."),
    "trade_synced_quantity_kg": ("kg", "Bilateral trade quantity obtained through VACODA's quantity synchronization method."),
    "model_prediction_quantity_kg": ("kg", "Quantity estimated by the selected machine-learning model."),
    "model_prediction_available": ("Boolean", "Whether a model prediction was generated."),
    "manual_review_required": ("Boolean", "Whether both HS6 and HS4 categories were absent from model training."),

    "export_value_fob": ("Nominal USD", "Exporter-reported trade value on a free-on-board basis."),
    "import_value_cif": ("Nominal USD", "Importer-reported trade value including cost, insurance, and freight."),
    "import_value_fob_corrected": ("Nominal USD", "Importer trade value converted to an estimated FOB-equivalent value."),
    "export_synchronized_quantity_kg": ("kg", "Synchronized quantity from the export reporting side."),
    "import_synchronized_quantity_kg": ("kg", "Synchronized quantity from the import reporting side."),
    "export_synchronized_quantity_source": ("Category", "Source or method used to obtain the export-side quantity."),
    "import_synchronized_quantity_source": ("Category", "Source or method used to obtain the import-side quantity."),
    "pair_uses_estimated_weight": ("Boolean", "Whether either reporting side uses an estimated weight."),
    "pair_uses_weight_proxy": ("Boolean", "Whether either reporting side uses a weight proxy."),
    "trade_value_reconciled_constant_usd": ("Constant USD", "Inflation-adjusted reconciled trade value; reference year must be documented."),
    "cif_fob_factor_applied": ("Ratio", "Factor applied for CIF-to-FOB value adjustment."),
    "cif_fob_factor_applied_source": ("Category", "Source of the applied CIF-to-FOB adjustment factor."),
    "reconciliation_status": ("Category", "Status of the bilateral trade-value reconciliation."),
    "reconciliation_method_applied": ("Category", "Method applied to reconcile exporter and importer trade values."),
    "trade_synced_quantity_method": ("Category", "Method used to obtain the final synchronized bilateral quantity."),
    "trade_synced_quantity_status": ("Category", "Status of bilateral quantity synchronization."),
    "quantity_availability_case": ("Category", "Availability of usable export-side and import-side quantity evidence."),

    "product_year_median_unit_value": ("USD/kg", "Median product unit-value reference calculated from the configured reference window."),
    "product_year_log_unit_value_mad": ("Log-scale", "Median absolute deviation of log unit values around the product reference."),
    "unit_value_reference_observations": ("Count", "Number of observations contributing to the unit-value reference."),
    "trade_sync_uv_usd_per_kg": ("Nominal USD/kg", "Reconciled nominal trade value divided by synchronized trade quantity."),
    "model_uv_usd_per_kg": ("Nominal USD/kg", "Reconciled nominal trade value divided by model-predicted quantity."),
    "trade_sync_uv_mad_distance": ("MAD units", "Absolute log distance between trade unit value and its reference, divided by reference log-UV MAD."),
    "model_uv_mad_distance": ("MAD units", "Absolute log distance between model unit value and its reference, divided by reference log-UV MAD."),

    "training_selected": ("Boolean", "Whether the observation was selected for model training."),
    "model_input_valid": ("Boolean", "Whether the required model inputs are usable."),
    "quantity_log_residual": ("Log-scale", "Signed log difference between synchronized and model-predicted quantities; sign convention requires verification."),
    "quantity_abs_log_residual": ("Log-scale", "Absolute log difference between synchronized and model-predicted quantities."),
    "exporter_seen": ("Boolean", "Whether the exporter was represented during model fitting."),
    "importer_seen": ("Boolean", "Whether the importer was represented during model fitting."),
    "hs6_seen": ("Boolean", "Whether the HS6 category was represented during model fitting."),
    "hs4_seen": ("Boolean", "Whether the HS4 category was represented during model fitting."),
    "validation_model_name": ("Text", "Name of the selected Validation model."),
    "validation_model_type": ("Text", "Algorithm family of the selected Validation model."),
}


def write_data_dictionary(
    *,
    columns: tuple[str, ...],
    public_names: dict[str, str],
    output_path: Path,
    title: str,
) -> Path:
    missing = set(columns) - VARIABLES.keys()

    if missing:
        raise ValueError(f"Missing variable definitions: {sorted(missing)}")

    lines = [
        f"# {title}",
        "",
        "Each row represents bilateral trade between an exporter and importer "
        "for one HS6 commodity and calendar year.",
        "",
        f"**Variables: {len(columns)}**",
        "",
        "| Variable | Unit / Type | Definition |",
        "|---|---|---|",
    ]

    for column in columns:
        name = public_names.get(column, column)
        unit, definition = VARIABLES[column]

        lines.append(
            f"| `{name}` | {unit} | {definition} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return output_path