"""Independent scientific input/count/fold audit; no model imports or fitting."""
from pathlib import Path
import collections
import hashlib
import json
import math
import re

HERE = Path(__file__).resolve().parent
R = HERE.parent
H = R.parent


def read_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


source_path = H / "dbaasp_coverage_v5/analysis/NATIVE_ROWS.jsonl"
source = read_jsonl(source_path)
data = read_jsonl(R / "DATA.jsonl")
folds = json.loads((R / "FOLDS.json").read_text())
inventory = json.loads((R / "INVENTORY.json").read_text())
checks = []


def check(name, condition):
    checks.append({"name": name, "pass": bool(condition)})


expected = {r["activity_id"]: r for r in source if
            (r["plain_Witten_primary_point_eligible"] or r["plain_Witten_censor_eligible"])
            and r["activity_id"] != 12605}
actual = {r["activity_id"]: r for r in data}
check("exact_selected_activity_set", set(expected) == set(actual))
check("no_activity_rows_duplicated", len(actual) == len(data))
check("all_cached_row_fields_preserved", all(all(actual[i][k] == v for k, v in row.items())
                                           for i, row in expected.items()))
check("1738_multi_isolate_bound_held_out_of_model", 12605 not in actual)
check("12529_reported_point_retained", 161390 in actual)
for r in data:
    raw = r["concentration_raw"]
    if r["label_kind"] == "point":
        value = float(raw)
    else:
        match = re.fullmatch(r"\s*(>=|>|≤|<=|<|≥)\s*([0-9]+(?:\.[0-9]*)?)\s*", raw)
        check("valid_censor_syntax_" + str(r["activity_id"]), match is not None)
        value = float(match.group(2))
        check("right_direction_" + str(r["activity_id"]), match.group(1) in [">", ">=", "≥"])
    check("positive_unconverted_log10_" + str(r["activity_id"]),
          value > 0 and abs(math.log10(value) - r["target_log10"]) < 1e-12)
    check("original_native_unit_" + str(r["activity_id"]),
          (r["unit_raw"], r["native_unit_group"]) in [("µg/ml", "ug_ml_native"), ("µM", "uM_native")])

counts = {}
for unit in ["ug_ml_native", "uM_native"]:
    rows = [r for r in data if r["native_unit_group"] == unit]
    point = [r for r in rows if r["label_kind"] == "point"]
    right = [r for r in rows if r["label_kind"] == "right"]
    pg = {r["source_family_group"] for r in point}
    cg = {r["source_family_group"] for r in right}
    counts[unit] = {
        "point_rows": len(point), "right_rows": len(right),
        "point_source_groups": len(pg), "censored_source_groups": len(cg),
        "censor_only_source_groups": sorted(cg - pg),
        "minimum_train_censor_groups": min(len(cg - {g}) for g in pg),
        "minimum_train_point_groups": len(pg) - 1,
    }
    check(unit + ":inventory_point", len(point) == inventory[unit]["point"]["rows"])
    check(unit + ":inventory_right", len(right) == inventory[unit]["right"]["rows"])
    check(unit + ":all_and_only_point_groups_are_validation_folds", {f["group"] for f in folds[unit]} == pg)
    check(unit + ":coverage_thresholds", len(cg) >= 3 and len(pg) >= 5 and
          counts[unit]["minimum_train_censor_groups"] >= 2 and len(pg) - 1 >= 3)
    for f in folds[unit]:
        prefix = unit + ":" + f["group"]
        train = [actual[i] for i in f["train"]]
        validation = [actual[i] for i in f["validation"]]
        check(prefix + ":exact_training_set", set(f["train"]) ==
              {r["activity_id"] for r in rows if r["source_family_group"] != f["group"]})
        check(prefix + ":exact_point_validation_set", set(f["validation"]) ==
              {r["activity_id"] for r in point if r["source_family_group"] == f["group"]})
        for key in ["peptide_id", "sequence", "annotation_fingerprint", "sequence_family", "source_family_group"]:
            check(prefix + ":no_overlap:" + key, not ({r[key] for r in train} & {r[key] for r in validation}))
        check(prefix + ":no_known_source_overlap", not
              ({s for r in train for s in r["source_keys"]} & {s for r in validation for s in r["source_keys"]}))

result = {
    "status": "PASS" if all(x["pass"] for x in checks) else "FAIL",
    "checks_passed": sum(x["pass"] for x in checks), "checks_total": len(checks),
    "checks": checks, "independent_counts": counts,
    "no_training_or_model_imports": True,
    "hashes": {str(p.relative_to(H)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [source_path, R / "DATA.jsonl", R / "FOLDS.json", R / "INVENTORY.json", Path(__file__)]},
}
(HERE / "SCIENCE_INPUT_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k:result[k] for k in ["status", "checks_passed", "checks_total", "independent_counts"]}))
assert result["status"] == "PASS"
