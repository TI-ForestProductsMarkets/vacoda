"""
vacoda.mappings
~~~~~~~~~~~~~~~
Central registry of lookup dictionaries and mappings used across the package.

Each submodule is scoped to one domain:
  - partner_source  : partner_source harmonization maps
"""
from vacoda.mappings.partner_source import PARTNER_SOURCE_EXACT_MAP, PARTNER_SOURCE_RULE_MAP

__all__ = [
    "PARTNER_SOURCE_EXACT_MAP",
    "PARTNER_SOURCE_RULE_MAP",
]
