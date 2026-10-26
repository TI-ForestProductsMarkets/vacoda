from pathlib import Path
from collections import Counter

# Internal name: (unit, definition)
VARIABLES = {
    "year": ("Year", "Calendar year of the bilateral trade observation."),
    "exporter_iso": ("Code", "ISO code of the exporting country."),
    "importer_iso": ("Code", "ISO code of the importing country."),
    "target_cmd": ("Code", "Six-digit Harmonized System commodity code."),
    "trade_value_reconciled": ("Nominal USD", "Bilateral trade value reconciled from exporter and importer reports."),
    "quantity_value_weighted_kg": ("kg", "Bilateral trade quantity obtained through VACODA's quantity synchronization method."),
    "model_prediction_quantity_kg": ("kg", "Quantity estimated by the selected machine-learning model."),
    "model_prediction_available": ("Boolean", "Whether a model prediction was generated."),
    "manual_review_required": ("Boolean", "Whether both HS6 and HS4 categories were absent from model training."),

    "export_value_fob": ("Nominal USD", "Exporter-reported trade value on a free-on-board basis."),
    "import_value_cif": ("Nominal USD", "Importer-reported trade value including cost, insurance, and freight."),
    "import_value_fob_corrected": ("Nominal USD", "Importer trade value converted to an estimated FOB-equivalent value."),
    "export_synchronized_quantity_kg": ("kg", "Quantity reported from the export reporting side."),
    "import_synchronized_quantity_kg": ("kg", "Quantity reported from the import reporting side."),
    "export_synchronized_quantity_source": ("Category", "Source or method used to obtain the export-side quantity."),
    "import_synchronized_quantity_source": ("Category", "Source or method used to obtain the import-side quantity."),
    "pair_uses_estimated_weight": ("Boolean", "Whether either reporting side uses an estimated weight."),
    "trade_value_reconciled_constant_usd": ("Constant USD", "Inflation-adjusted reconciled trade value expressed in {reference_year} USD."),
    "cif_fob_factor_applied": ("Ratio", "Factor applied for CIF-to-FOB value adjustment."),
    "cif_fob_factor_applied_source": (
                                        "Category",
                                        "Evidence used to estimate the CIF-to-FOB factor."
                                        "<br>**Direct-Factor:** Directly reported CIF and FOB values."
                                        "<br>**Mirror-Factor:** Supplementary bilateral observations with agreeing quantities; never used alone."
                                    ),
    "reconciliation_status": ("Category", 
                              "Status of the bilateral trade-value reconciliation."
                               "<br>The categories are:"
                            "<br>**bilateral_reconciled:** Both reporting sides provide usable FOB-basis values; reliability-weighted reconciliation is applied."
                            "<br>**bilateral_reconciled_import_implicit_zero:** Exporter FOB value is usable; importer reports other imports that year but no value for this bilateral product. An implicit zero is used for the importer."
                            "<br>**bilateral_reconciled_export_implicit_zero:** Importer FOB-equivalent value is usable; exporter reports other exports that year but no value for this bilateral product. An implicit zero is used for the exporter."
                            "<br>**export_only:** Only the export-side FOB value is usable and is retained when unmatched flows are permitted."
                            "<br>**import_only:** Only the import-side FOB-equivalent value is usable and is retained when unmatched flows are permitted."
                            "<br>**not_eligible:** No eligible bilateral reconciliation or permitted unilateral retention is available.",
                              ),
    "exporter_adjusted_weight": ("Ratio","Adjusted reliability weight assigned to the exporter FOB value during bilateral reconciliation."),
    "importer_adjusted_weight": ("Ratio","Adjusted reliability weight assigned to the importer FOB-equivalent value, including an implicit zero where applicable."),
    "reconciliation_method_applied": ("Category", 
                                      "Method applied to reconcile exporter and importer trade values."
                                     "<br>**reliability_weighted:** Reports are combined using annual country-direction reliability weights, including the 0.10 importer-weight adjustment."
                                     "<br>**export_report_retained:** The exporter FOB value is retained without bilateral averaging."
                                     "<br>**import_report_retained:** The importer FOB-equivalent value is retained without bilateral averaging."
                                     "<br>**no_usable_fob_basis_value:** No usable FOB-basis reconciliation result is available."
                                      ),
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

VARIABLE_GROUPS = {
    "1. Basic trade information": (
        "Identifies the trade observation and presents the final "
        "reconciled value, synchronized quantity, and model prediction.",
        [
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "trade_value_reconciled",
            "quantity_value_weighted_kg",
            "model_prediction_quantity_kg",
            "model_prediction_available",
            "manual_review_required",
        ],
    ),

    "2. Exporter and importer evidence": (
        "Describes the trade values and quantities reported by the two "
        "countries, including their sources and estimation flags.",
        [
            "export_value_fob",
            "import_value_cif",
            "import_value_fob_corrected",
            "export_synchronized_quantity_kg",
            "import_synchronized_quantity_kg",
            "export_synchronized_quantity_source",
            "import_synchronized_quantity_source",
            "pair_uses_estimated_weight",
        ],
    ),

    "3. Value reconciliation": (
        "Monetary adjustments and bilateral reconciliation.",
        [
            "trade_value_reconciled_constant_usd",
            "cif_fob_factor_applied",
            "cif_fob_factor_applied_source",
            "exporter_adjusted_weight",
            "importer_adjusted_weight",
            "reconciliation_status",
            "reconciliation_method_applied",
        ],
    ),

    "4. Quantity synchronization": (
         "Reporting-side availability and bilateral quantity selection.",
        [
            "trade_synced_quantity_method",
            "trade_synced_quantity_status",
            "quantity_availability_case",
        ],
    ),

    "5. Unit-value references and diagnostics": (
        "Compares trade-derived and model-implied unit values with "
        "the product reference distribution.",
        [
            "product_year_median_unit_value",
            "product_year_log_unit_value_mad",
            "unit_value_reference_observations",
            "trade_sync_uv_usd_per_kg",
            "model_uv_usd_per_kg",
            "trade_sync_uv_mad_distance",
            "model_uv_mad_distance",
        ],
    ),

    "6. Model training and diagnostics": (
        "Describes training eligibility, prediction inputs, "
        "model errors, categorical support, and model identity.",
        [
            "training_selected",
            "model_input_valid",
            "quantity_log_residual",
            "quantity_abs_log_residual",
            "exporter_seen",
            "importer_seen",
            "hs6_seen",
            "hs4_seen",
            "validation_model_name",
            "validation_model_type",
        ],
    ),
}

RECONCILIATION_CATEGORIES = {
    "reconciliation_status": {
        "bilateral_reconciled":
            "Both reporting sides provide usable FOB-basis values; reliability-weighted reconciliation is applied.",
        "bilateral_reconciled_import_implicit_zero":
            "Exporter FOB value is usable; importer reports other imports that year but no value for this bilateral product. An implicit zero is used for the importer.",
        "bilateral_reconciled_export_implicit_zero":
            "Importer FOB-equivalent value is usable; exporter reports other exports that year but no value for this bilateral product. An implicit zero is used for the exporter.",
        "export_only":
            "Only the export-side FOB value is usable and is retained when unmatched flows are permitted.",
        "import_only":
            "Only the import-side FOB-equivalent value is usable and is retained when unmatched flows are permitted.",
        "not_eligible":
            "No eligible bilateral reconciliation or permitted unilateral retention is available.",
    },
    "reconciliation_method_applied": {
        "reliability_weighted":
            "Reports are combined using annual country-direction reliability weights, including the 0.10 importer-weight adjustment.",
        "export_report_retained":
            "The exporter FOB value is retained without bilateral averaging.",
        "import_report_retained":
            "The importer FOB-equivalent value is retained without bilateral averaging.",
        "no_usable_fob_basis_value":
            "No usable FOB-basis reconciliation result is available.",
    },
}

CATEGORY_EXPLANATIONS = {
    "2. Exporter and importer evidence": {
        "title": "Quantity-source categories",
        "applies_to": (
            "export_synchronized_quantity_source",
            "import_synchronized_quantity_source",
        ),
        "values": {
            "quantity_alt_quantity_net_weight":
                "All three reported kilogram candidates agree exactly.",

            "quantity_alt_quantity":
                "Reported primary and alternative kilogram quantities agree exactly.",

            "quantity_net_weight":
                "Reported primary quantity and net weight agree exactly.",

            "alt_quantity_net_weight":
                "Reported alternative quantity and net weight agree exactly.",

            "quantity":
                "Primary quantity selected by reported-source precedence "
                "or estimated fallback.",

            "alt_quantity":
                "Alternative quantity selected by reported-source precedence "
                "or estimated fallback.",

            "net_weight":
                "Net weight selected by reported-source precedence "
                "or estimated fallback.",

            "unavailable":
                "No usable kilogram evidence was available.",
        },
        "note": (
            "VACODA first prioritizes exact agreement among reported kilogram "
            "candidates. If no exact agreement exists, reported values follow "
            "the precedence quantity > alt_quantity > net_weight. Estimated "
            "values are considered only when no reported kilogram candidate "
            "is available, using the same precedence. Gross weight is excluded "
            "from ordinary quantity synchronization."
        ),
    },
}

def write_data_dictionary(
    *,
    columns: tuple[str, ...],
    public_names: dict[str, str],
    output_path: Path,
    title: str,
    reference_year: int,
) -> Path:
    missing = set(columns) - VARIABLES.keys()

    if missing:
        raise ValueError(f"Missing variable definitions: {sorted(missing)}")

    def make_table(fields):
        lines = [
            "| Variable | Unit / Type | Definition |",
            "|---|---|---|",
        ]

        for column in fields:
            name = public_names.get(column, column)
            unit, definition = VARIABLES[column]
            definition = definition.replace(
                "{reference_year}",
                str(reference_year),
            )
            lines.append(
                f"| `{name}` | {unit} | {definition} |"
            )

        return lines

    lines = [
        f"# {title}",
        "",
        "Each row represents bilateral trade between an exporter and importer "
        "for one HS6 commodity and calendar year.",
        "",
        f"**Variables: {len(columns)}**",
        "",
    ]

    standard_fields = set(
        VARIABLE_GROUPS["1. Basic trade information"][1]
    )

    if set(columns) == standard_fields:
        # Standard dataset: one table with all 9 variables
        lines.extend(make_table(columns))

    else:
        # Enriched dataset: all 43 variables in six categories
        grouped_columns = [
            column
            for _, (_, fields) in VARIABLE_GROUPS.items()
            for column in fields
        ]

        missing_from_groups = sorted(set(columns) - set(grouped_columns))
        extra_in_groups = sorted(set(grouped_columns) - set(columns))
        duplicates = sorted(
            name for name, count in Counter(grouped_columns).items()
            if count > 1
        )

        if missing_from_groups or extra_in_groups or duplicates:
            raise ValueError(
                "Enriched dictionary schema mismatch:\n"
                f"  Missing from groups: {missing_from_groups}\n"
                f"  Extra in groups: {extra_in_groups}\n"
                f"  Duplicate group entries: {duplicates}"
            )


        for heading, (description, fields) in VARIABLE_GROUPS.items():
            lines.extend([
                f"## {heading}",
                "",
                description,
                "",
                *make_table(fields),
                "",
            ])

            explanation = CATEGORY_EXPLANATIONS.get(heading)

            if explanation:
                applies_to = ", ".join(
                    f"`{public_names.get(name, name)}`"
                    for name in explanation["applies_to"]
                )

                lines.extend([
                    f"#### {explanation['title']}",
                    "",
                    f"**Applies to:** {applies_to}",
                    "",
                    "| Stored value | Interpretation |",
                    "|---|---|",
                ])

                for value, meaning in explanation["values"].items():
                    lines.append(f"| `{value}` | {meaning} |")

                lines.extend([
                    "",
                    f"*Selection methodology:* {explanation['note']}",
                    "",
                ])

            for field in fields:
                categories = RECONCILIATION_CATEGORIES.get(field)

                if categories is None:
                    continue

                public_name = public_names.get(field, field)

                lines.extend([
                    f"#### `{public_name}`",
                    "",
                    "| Stored value | Interpretation |",
                    "|---|---|",
                ])

                for value, meaning in categories.items():
                    lines.append(f"| `{value}` | {meaning} |")

                lines.append("")

            if heading == "3. Value reconciliation":
                lines.extend([
                    "### 3.2 Calculation",
                    "",
                    "### Reliability-weighted calculation",
                    "",
                    "For eligible bilateral observations:",
                    "",
                    "`trade_value_usd = exporter_trade_weight × reconciliation_export_value "
                    "+ importer_trade_weight × reconciliation_import_value`",
                    "",
                    "The weights depend on annual country-direction "
                    "mean reporting deviations:",
                    "",
                    "`exporter_trade_weight = d_import / (d_export + d_import) - 0.10`",
                    "",
                    "`importer_trade_weight = d_export / (d_export + d_import) + 0.10`",
                    "",
                    "When both deviations are zero, the base weights "
                    "are 0.5 each before adjustment.",
                    "",
                    "For implicit-zero observations, the absent report "
                    "enters reconciliation as zero, while its original "
                    "published evidence field remains missing.",
                    "",
                    "For unilaterally retained observations, "
                    "bilateral weighting is not applied.",
                    "",
                ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return output_path
