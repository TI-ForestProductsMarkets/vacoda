## HS-sync workflow:
```bash
python scripts/hs_sync/run_hs_sync_workflow.py
```
> [!IMPORTANT]
> Download comtrade mapping excel files from `https://unstats.un.org/unsd/classifications/Econ#corresp-hs`
> Copy them under: `vacoda/references/comtrade/hs_correlation_groups/raw`

**This workflow runs:**
1. UNSD/Comtrade HS mapping normalization
2. target-HS coverage diagnostics
3. Comtrade mapping audit
4. equal-split allocation weights
5. complete mapping-weight table
6. HS-sync apply step
7. applied-output integrity audit

The final synced parquet is written to:
    /storage/vacoda/data/interim/hs_sync/target_h5_applied/

Older step-level scripts under dev/, are debugging wrappers only.


## HS System Timeline
HS_HSCode	HS_valid_from	HS_valid_until	        HS_HSCode2
H0	         1/1/1988 0:00	12/31/1995 0:00	        HS1988
H1	         1/1/1996 0:00	12/31/2001 0:00	        HS1996
H2	         1/1/2002 0:00	12/31/2006 0:00	        HS2002
H3	         1/1/2007 0:00	12/31/2011 0:00	        HS2007
H4	         1/1/2012 0:00	12/31/2016 0:00	        HS2012
H5	         1/1/2017 0:00	12/31/2021 0:00	        HS2017
H6	         1/1/2022 0:00	12/31/2026 0:00	        HS2022
