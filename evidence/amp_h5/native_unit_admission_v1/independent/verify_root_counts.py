"""Verify parent results against independently selected cached observations."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = PACKAGE.parent
independent = json.loads((HERE / "RECOUNT.json").read_text())
parent = json.loads((PACKAGE / "COUNTS.json").read_text())
parent_rows = [json.loads(s) for s in (PACKAGE / "ROW_ADMISSION.jsonl").read_text().splitlines()]
original_rows = [json.loads(s) for s in (ROOT / "dbaasp_bounded_v3/prepared/mic_measurements.jsonl").read_text().splitlines()]
original_rows = [r for r in original_rows if r["ecoli_target"]]
checks = []


def check(name, outcome):
    checks.append({"name": name, "pass": bool(outcome)})


check("all142_Ecoli_rows_retained", len(parent_rows) == len(original_rows) == 142)
old = {r["activity_id"]: r for r in original_rows}
new = {r["activity_id"]: r for r in parent_rows}
check("activity_id_set_unchanged", set(old) == set(new) and len(new) == 142)
check("every_original_field_unchanged", all(all(new[key][field] == value for field, value in row.items())
                                          for key, row in old.items()))
map_fields = {
    "rows": "rows", "peptide_ids": "peptide_IDs", "exact_case_sequences": "exact_sequences",
    "chemical_annotation_fingerprints": "chemical_annotation_fingerprints",
    "operational_sequence_families": "sequence_families", "source_family_union_groups": "source_family_components",
}
for unit, parent_unit in [("µg/ml", "ug_ml_native"), ("µM", "uM_native")]:
    own = independent["strict_witten_narrow_attribution"]["by_native_unit"][unit]
    other = parent["by_native_unit"][parent_unit]
    for own_name, other_name in [
        ("point_and_one_sided_bounds", "plain_Witten_point_plus_bound"),
        ("point", "plain_Witten_primary_points"),
        ("one_sided_bounds", "plain_Witten_censored_reserve"),
    ]:
        check(unit + ":" + own_name + ":all_six_counts_match", all(
            own[own_name][source] == other[other_name][target] for source, target in map_fields.items()))
    selected = {r["activity_id"] for r in parent_rows if r["unit_raw"] == unit
                and (r["plain_Witten_primary_point_eligible"] or r["plain_Witten_censor_eligible"])}
    check(unit + ":exact_eligible_activity_set_matches", selected == set(own["activity_ids"]))
check("all_execution_flags_remain_false", all(not r["execution_admitted"] and not r["training_admitted"] for r in parent_rows))
check("all_retained_censors_remain_nonpoint", all(not r["plain_Witten_primary_point_eligible"]
                                               for r in parent_rows if r["format"] in {"right_bound", "left_bound"}))
check("attribution_pending_literal_ids_preserved", parent["attribution_pending_IDs"] == [16, 36, 1783])
result = {
    "schema": "AMP_H5_NATIVE_UNITS_SCIENTIFIC_INDEPENDENT_SIGNOFF_V1",
    "reviewer": "/root/native_unit_scientific_review; independent of parent rule/count code author",
    "result": "PASS" if all(c["pass"] for c in checks) else "FAIL",
    "checks": checks,
    "scientific_decision": "YES for separately modeled native-unit exploratory reported-endpoint tasks; no conversion prerequisite",
    "execution_decision": "No training this turn; protected H5 isolation and frozen grouped development protocol remain separate prerequisites",
    "recommended_primary": "log10 database-native µg/ml point-report endpoint with strict Witten feature domain and unresolved composition attribution withheld",
    "limitations": [
        "Material mass basis remains unknown; no real parent-molar potency interpretation",
        "Unobserved assay/chemistry variables limit scope; no complete-case prerequisite for a reported-label task",
        "Point selection may be informative; censor bounds retained, not points or independent validation set",
        "Unique IDs and operational families/source groups are not certified biological independence",
        "No test37 access, protected contamination clearance, new collection, natural training or novelty evidence",
    ],
    "reviewed_sha256": {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [PACKAGE / "COUNTING_RULES.json", PACKAGE / "COUNTS.json", PACKAGE / "ROW_ADMISSION.jsonl",
                                  PACKAGE / "ADMISSION_ONE_PAGE.zh.md",
                                  HERE / "RECOUNT.json", HERE / "SCIENCE_REVIEW.zh.md"]},
}
(HERE / "SIGNOFF.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": result["result"], "passed": sum(c["pass"] for c in checks), "total": len(checks)}))
assert result["result"] == "PASS"
