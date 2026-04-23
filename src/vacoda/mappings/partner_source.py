"""
Lookup maps for partner_source harmonization.
"""

PARTNER_SOURCE_EXACT_MAP: dict[str, str] = {
    "origin"                         : "Origin",
    "origin/consignment for intra-eu": "Origin/Consignment for Intra-EU",
    "consignment"                    : "Consignment",
    "purchase"                       : "Purchase",
    "sale"                           : "Sale",
    "consumption"                    : "Consumption",
    "last known destination"         : "Last Known Destination",
}

PARTNER_SOURCE_RULE_MAP: dict[str, str] = {
    "last known destinations"         : "Last Known Destination",
    "origins"                         : "Origin",
    "consignment for intra-eu"        : "Consignment for Intra-EU",
}
