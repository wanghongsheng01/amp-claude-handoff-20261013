#!/usr/bin/env python3
"""Pure math and AST checks only; no runner/runtime/model/natural-data import."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import warmup_vae_only as w


def main():
    count = 0

    def check(value, label):
        nonlocal count
        if not value:
            raise AssertionError(label)
        count += 1

    def rejects(fn, label):
        try:
            fn()
        except (ValueError, TypeError):
            check(True, label)
        else:
            check(False, label)

    check("torch" not in sys.modules and "numpy" not in sys.modules, "stdlib import only")
    beta = [w.beta_for_step(i) for i in range(1, 1001)]
    for i, actual in enumerate(beta):
        check(actual == min(1.0, i / 200.0), "exact schedule update " + str(i))
    check((beta[0], beta[199], beta[200], beta[-1]) == (0.0, .995, 1.0, 1.0), "anchors")
    check(sum(x < 1 for x in beta) == 200 and sum(x == 1 for x in beta) == 800, "phase counts")
    check(math.fsum(beta) == 899.5, "schedule integral")
    check(all(a <= b for a, b in zip(beta, beta[1:])), "monotonic")
    for invalid in (0, -1, 1001, 1.0, True, None, "1"):
        rejects(lambda: w.beta_for_step(invalid), "invalid step rejected")
    for r, k in ((3., 7.), (0., -2.), (-4., 0.), (1.5, -.25)):
        check(w.composed_loss(r, k, 1) == r, "beta zero keeps CE")
        check(w.composed_loss(r, k, 201) == r + k, "beta one keeps old loss")
        check(w.composed_loss(r, k, 101) == r + .5 * k, "midpoint objective")
    body = Path(w.__file__).read_text()
    tree = ast.parse(body)
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    check(calls.count("train_vae") == 1, "one inherited train loop call")
    check(not set(calls) & {"train_ddpm", "generate", "posterior_parameters", "reconstruction_export", "init_runtime"},
          "no automatic DDPM/generation/posttraining posterior or broad bootstrap")
    imported = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    imported_names = {alias.name.split('.')[0] for n in imported if isinstance(n, ast.Import) for alias in n.names}
    check(not imported_names & {"torch", "numpy", "rdkit"}, "no scientific import at module top level")
    check("for step in" not in body, "no duplicate training step loop")
    receipt = {"schema": "AMP_H5_WARMUP_ROOT_LAUNCH_V1", "status": "ROOT_AUTHORIZED",
        "candidate_manifest_sha256": "synthetic", "scope": "VAE_ONLY_1000",
        "train_rows":279,"dev_rows":41,"K":8,"dev_M_equal_sequence_accuracy":.7,
        "dev_S_equal_sequence_eight_draw_mean_accuracy":.5}
    flags = ("independent_preaudit_pass", "baseline_full_rms_complete", "root_fresh_process_load_check_pass",
             "single_serial_cpu_launch", "no_other_active_amp_model_job", "baseline_core_predictions_finite")
    for flag in flags:
        receipt[flag] = True
    for key in ("independent_preaudit", "baseline_rms_delivery", "baseline_epsilon_bank_manifest", "root_process_load_check"):
        receipt[key] = {"path": "/tmp/amp_h5/synthetic_metadata.json", "sha256": "synthetic"}
    original_sha = w.sha
    w.sha = lambda path: "synthetic"  # Pure synthetic stub: no file reads, not used in delivered runtime.
    try:
        w.validate_receipt(receipt, "synthetic")
        check(True, "synthetic valid root receipt structure")
        for flag in flags:
            bad = copy.deepcopy(receipt); bad[flag] = False
            rejects(lambda: w.validate_receipt(bad, "synthetic"), "false gate rejects: " + flag)
        for field, value in (("status", "NOT_AUTHORIZED_TEMPLATE_ONLY"), ("K",7), ("train_rows",278),
                             ("dev_rows",40), ("dev_M_equal_sequence_accuracy",None),
                             ("dev_S_equal_sequence_eight_draw_mean_accuracy",float('nan')),
                             ("dev_S_equal_sequence_eight_draw_mean_accuracy",.699)):
            bad = copy.deepcopy(receipt); bad[field] = value
            rejects(lambda: w.validate_receipt(bad, "synthetic"), "invalid admission rejected: " + field)
        rejects(lambda: w.validate_receipt(receipt, "changed"), "manifest drift rejects")
        bad = copy.deepcopy(receipt); bad["baseline_rms_delivery"]["path"] = "/tmp/amp_h5/test37.json"
        rejects(lambda: w.validate_receipt(bad, "synthetic"), "test evidence path rejects")
    finally:
        w.sha = original_sha
    check("torch" not in sys.modules and "numpy" not in sys.modules, "no runtime import after all checks")
    result = {"status": "PASS", "checks": count, "scope": "stdlib pure mathematics, AST and synthetic receipt only",
              "model_imports":0,"model_constructions":0,"model_forwards":0,"training_updates":0,
              "cohort_reads":0,"test_reads":0,"random_draws":0,
              "wrapper_sha256":hashlib.sha256(Path(w.__file__).read_bytes()).hexdigest()}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
