"""
Flow filters for Comtrade v1 get endpoint.
Use these as tokens in configs/comtrade.yaml, e.g. flowCode: "FLOW_ALL".

Flow code hints from API selector:
-1, M, X, RX, RM, MIP, XIP, MOP, XOP, MIF, XIF, DX, FM
"""

# Import/export flow subsets.
FLOW_IMPORTS = ["M"]
FLOW_EXPORTS = ["X"]

# Core merchandise trade directions.
FLOW_BIDIRECTIONAL = FLOW_IMPORTS + FLOW_EXPORTS
FLOW_BASIC  = FLOW_BIDIRECTIONAL
FLOW_BASE = FLOW_BASIC

# Re-export and re-import subset.
FLOW_REEXPORT_REIMPORT = ["RX", "RM"]

# Full merchandise trade including re-export/re-import.
FLOW_INCLUDING_RETRADE = FLOW_BASE + FLOW_REEXPORT_REIMPORT

# Inward/outward processing subset.
FLOW_PROCESSING = ["MIP", "XIP", "MOP", "XOP", "MIF", "XIF"]

# Other adjustment/auxiliary flow subset.
FLOW_OTHER = ["DX", "FM"]

# Full flow list shown in API selector.
FLOW_ALL = ["-1", "M", "X", "RX", "RM", "MIP", "XIP", "MOP", "XOP", "MIF", "XIF", "DX", "FM"]

