"""
vacoda.mappings.trade_map
 This file defines the mapping from original Comtrade column names to the standardized field names used in our processed dataset. 
 The keys are the original column names from the Comtrade API, and the values are the corresponding standardized field names that we use internally in our dataset.
"""


TRADE_MAP_DB = {
    "CD_ID"                    : "record_id",
    "CD_HS"                    : "hs_code_short",
    "CD_Reporter"              : "reporter_code",
    "CD_Partner"               : "partner_code",
    "CD_Commodity"             : "commodity_code",
    "CD_Year"                  : "year",
    "CD_Flow"                  : "flow_code",
    "CD_Unit"                  : "unit_code",
    "CD_Value"                 : "value_usd_raw",
    "CD_ReportedCurrency"      : "reported_currency",
    "CD_ConversionFactor"      : "conversion_factor",
    "CD_Netweight_kg"          : "netweight_kg_raw",
    "CD_Sup_Quantity"          : "sup_quantity",
    "CD_EstimationCode"        : "estimation_code",
    "CD_Valuation"             : "valuation",
    "CD_ReportedClassification": "reported_hs_code_short",
    "CD_PartnerSource"         : "partner_source_raw",
    "CD_Netweight_kg/Value"    : "kg_per_usd",
    "CD_FlowsInDataset"        : "old_flow_code",
}


TRADE_MAP_API = {
    'TypeCode'         : "type_code",
    'FreqCode'         : "freq_code",
    'RefPeriodId'      : "ref_period_id",
    'RefYear'          : "year",
    'RefMonth'         : "month",
    'Period'           : "period",
    'ReporterCode'     : "reporter_code",
    'ReporterISO'      : "reporter_iso",
    'ReporterDesc'     : "reporter_desc",
    'FlowCode'         : "flow_code",
    'FlowDesc'         : "flow_desc",
    'PartnerCode'      : "partner_code",
    'PartnerISO'       : "partner_iso",
    'PartnerDesc'      : "partner_desc",
    'Partner2Code'     : "partner2_code",
    'Partner2ISO'      : "partner2_iso",
    'Partner2Desc'     : "partner2_desc",
    'ClassificationCode': "hs_code_short",
    'ClassificationSearchCode': "hs_code_search",
    'IsOriginalClassification': "is_original_classification",
    'CmdCode'          : "commodity_code",
    'CmdDesc'          : "commodity_desc",
    'aggrLevel'        : "aggregation_level",
    "IsLeaf"           : "is_leaf",
    'CustomsCode'      : "custom_code",
    'CustomsDesc'      : "custom_desc",
    'QtyUnitCode'      : "quantity_unit_code",
    'QtyUnitAbbr'      : "quantity_unit_abbr",
    'Qty'              : "quantity",
    'IsQtyEstimated'   : "estimation_code", #"is_quantity_estimated",
    'AltQtyUnitCode'   : "alt_quantity_unit_code",
    'AltQtyUnitAbbr'   : "alt_quantity_unit_abbr",
    'AltQty'          : "alt_quantity",
    'IsAltQtyEstimated': "is_alt_quantity_estimated",
    'NetWgt'         : "netweight_kg_raw",
    'IsNetWgtEstimated': "is_netweight_estimated",
    'GrossWgt'          : "grossweight_kg",
    'IsGrossWgtEstimated': "is_grossweight_estimated",
    'Cifvalue'         : "cif_value_usd",
    'Fobvalue'         : "fob_value_usd",
    'PrimaryValue'     : "value_usd_raw",
    'LegacyEstimationFlag': "legacy_estimation_flag",
    'IsReported'       : "is_reported",
    'IsAggregate'      : "is_aggregate",
}

   
