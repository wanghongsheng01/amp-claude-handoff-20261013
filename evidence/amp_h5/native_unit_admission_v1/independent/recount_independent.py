"""Independent cached-label recount; standard library only, no model or split reads."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "dbaasp_bounded_v3" / "prepared"
OUTPUT = Path(__file__).resolve().parent
AA = set("ACDEFGHIKLMNPQRSTVWY")
ATTRIBUTION_PENDING = {16, 36, 1783}  # Literal composition notes; not proven mixtures.


def read(name):
    return [json.loads(line) for line in (INPUT / name).read_text().splitlines()]


peptides = {p["peptide_id"]: p for p in read("peptides.jsonl")}
observations = read("mic_measurements.jsonl")


def count(rows):
    return {
        "rows": len(rows),
        "peptide_ids": len({r["peptide_id"] for r in rows}),
        "exact_case_sequences": len({peptides[r["peptide_id"]]["sequence"] for r in rows}),
        "chemical_annotation_fingerprints": len({r["annotation_fingerprint"] for r in rows}),
        "operational_sequence_families": len({r["sequence_family"] for r in rows}),
        "source_family_union_groups": len({r["source_family_group"] for r in rows}),
    }


def positive_parsed(row):
    values = [row[k] for k in ["lower_raw_unit", "upper_raw_unit"] if row[k] is not None]
    return bool(values) and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values)


def describe(rows):
    out = {"all_units_count_only": count(rows), "by_native_unit": {}}
    for unit in ["µg/ml", "µM"]:
        selected = [r for r in rows if r["unit_raw"] == unit]
        out["by_native_unit"][unit] = {
            "point_and_one_sided_bounds": count(selected),
            "point": count([r for r in selected if r["format"] == "point"]),
            "one_sided_bounds": count([r for r in selected if r["format"] != "point"]),
            "activity_ids": [r["activity_id"] for r in selected],
        }
    return out


core = [r for r in observations
        if r["ecoli_target"]
        and r["raw_activity"].get("activityMeasureValue") == "MIC"
        and r["unit_raw"] in {"µg/ml", "µM"}
        and r["format"] in {"point", "right_bound", "left_bound"}
        and r["reference_index_valid"]
        and peptides[r["peptide_id"]]["sequence"]
        and positive_parsed(r)]
# The supplied monomer cache has no interchain/coordination annotations.
assert all(not p["chemistry"]["interchainBonds"] and not p["chemistry"]["coordinationBonds"]
           for p in peptides.values())
witten = [r for r in core
          if set(peptides[r["peptide_id"]]["sequence"]) <= AA
          and 1 <= peptides[r["peptide_id"]]["length"] < 100]
result = {
    "schema": "AMP_H5_NATIVE_UNITS_INDEPENDENT_RECOUNT_V1",
    "scope": "Existing cached75 only; no conversions, model execution, network or split reads",
    "input_sha256": {name: hashlib.sha256((INPUT / name).read_bytes()).hexdigest()
                     for name in ["peptides.jsonl", "mic_measurements.jsonl"]},
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "attribution_pending_ids": sorted(ATTRIBUTION_PENDING),
    "attribution_basis": "Literal dimer/two-peptide component notes, not proof every assay used a mixture",
    "traceable_parsed_report_core": describe(core),
    "strict_witten_broad_sensitivity": describe(witten),
    "strict_witten_narrow_attribution": describe([r for r in witten if r["peptide_id"] not in ATTRIBUTION_PENDING]),
    "interpretation": "Operational clusters, not proven independent biological samples; units never pooled for modeling",
}
(OUTPUT / "RECOUNT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v["all_units_count_only"] for k, v in result.items()
                  if isinstance(v, dict) and "all_units_count_only" in v}, ensure_ascii=False))
