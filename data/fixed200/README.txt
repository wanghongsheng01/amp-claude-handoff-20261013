AMP H5 — FINAL FIXED200 DATA ADMISSION RECEIPT (2026-10-12)
Read APPENDIX_1PAGE.pdf first, then data/SNAPSHOT_DELTAS.csv and DATA_DICTIONARY.csv.
Decision: descriptive-data handoff complete; formal model admission HOLD and scientific NO_GO unchanged. More data alone do not rescue failed EB-context results. Original27 generation origin/selection awaits the author's response; no evidence upgrade.
The fixed positional200 is not a random/representative whole-database sample. There are200 peptide IDs,192 exact-case sequences,141 descriptive80% edit-distance families and68 source-family components; these components are not certified independent biology.
All1788 MIC rows are repeated observations (165 peptide IDs). Ecoli318 rows cover158 IDs. Native unit groups remain separate; NEVER convert mass to molarity, pool units, midpoint ranges, guess SD/SE, or treat MIC rows as independent peptides.
Data inputs preserve case, missing values, raw concentration/reference strings and reported conditions. Two empty references stay unresolved; one is Ecoli. Absence of terminal annotations means unknown, not Free/Amide. No missing-field imputation.
Reproduce tables with Python3 stdlib only, from this directory:
  python3 scripts/clean_data.py --output rebuilt
This recomputes string parsing, label eligibility and original grouping rules; frozen molecular audit status is an explicit upstream input (evidence/FROZEN_CHEMISTRY_AUDIT.jsonl). No molecular audit or model run is concealed. frozen_*.py are unchanged upstream author provenance copies and retain original workspace layout/dependencies; standalone reproduction uses clean_data.py.
The package excludes third-party PDFs, credentials, model weights, train/dev/test sequences, and acquisition executable scripts. Existing200 public database sequences are authorized public data; they are not a new leakage-screened split. test37 was never opened. All earlier failed/negative experiments and acquisition failures remain in their original local files.
See DBAASP_ATTRIBUTION.txt for source policy and citation. Private Library attachment only; no public release. HASHES.sha256 covers packaged files except itself.
