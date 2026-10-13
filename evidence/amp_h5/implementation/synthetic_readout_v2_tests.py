#!/usr/bin/env python3
"""Low-cost synthetic file-contract checks, with no natural AMP input reads."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import build_readout_v2 as v2
from synthetic_readout_tests import fixture, put


def setup(root):
    old = root / "fixture"
    old.mkdir()
    fixture(old)
    paths = {}
    for directory, method in v2.v1.METHODS.items():
        version = "formal_v1" if method == "terminal_LSTM" else "formal_v2"
        dest = root / version / directory
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old / directory), str(dest))
        paths[method] = str(dest)
    return paths, old / "evaluation"


def standard_package(root, source_evaluation, methods, paths):
    root.mkdir()
    generation = json.loads((source_evaluation / "generation_metrics.json").read_text())
    generation["groups"] = [g for g in generation["groups"] if g["method"] in methods]
    rec = json.loads((source_evaluation / "reconstruction_metrics.json").read_text())
    rec["groups"] = [g for g in rec["groups"] if g["method"] in methods]
    rec["groups"] += [{**g, "split": "train", "rows": 279} for g in copy.deepcopy(rec["groups"])]
    put(root, "generation_metrics.json", generation)
    put(root, "reconstruction_metrics.json", rec)
    samples = [v2.v1.record(Path(paths[m]) / "generated.csv") for m in methods]
    reconstruction = [v2.v1.record(Path(paths[m]) / n) for m in methods if m != "terminal_LSTM"
                      for n in ("train_reconstruction.csv", "dev_reconstruction.csv")]
    put(root, "run.json", {"status": "COMPLETE", "sample_inputs": samples, "reconstruction_inputs": reconstruction,
         "train": {"path": "/never/open/train.csv", "sha256": v2.TRAIN_SHA},
         "development": {"path": "/never/open/development.csv", "sha256": v2.DEV_SHA}})
    put(root, "exploratory_gates.json", {"reconstruction_ce_vs_matched_mse": {"status": "EVALUATED", "pass": True},
                                       "generation_ce_vs_terminal_ar": {"status": "EVALUATED", "pass": True}})
    put(root, "MANIFEST.json", {"files": []})


def ar_package(root, source_evaluation, paths):
    root.mkdir()
    gen = json.loads((source_evaluation / "generation_metrics.json").read_text())
    gen["groups"] = [g for g in gen["groups"] if g["method"] == "terminal_LSTM"]
    put(root, "GENERATION_METRICS.json", gen)
    put(root, "RUN_FACTS.json", {"scope": "completed AR-only descriptive feasibility", "attempts": 32})
    put(root, "SOURCE_BINDINGS.json", {"source_preservation_pass": True, "inputs": [
        v2.v1.record(Path(paths["terminal_LSTM"]) / "generated.csv"),
        {"path": "/never/open/train.csv", "sha256": v2.TRAIN_SHA},
        {"path": "/never/open/development.csv", "sha256": v2.DEV_SHA}]})
    put(root, "TEACHER_FORCED_NLL.json", [])
    put(root, "MANIFEST.json", {"files": []})


class V2ReadoutTests(unittest.TestCase):
    def test_missing_methods_have_no_fabricated_observations(self):
        with tempfile.TemporaryDirectory(prefix="h5_summary_v2_") as tmp:
            root = Path(tmp)
            paths = {m: str(root / m) for m in v2.METHODS}
            facts = v2.build(paths, [])
            self.assertEqual(facts["generation_observations"], [])
            self.assertEqual(facts["reconstruction_observations"], [])
            self.assertEqual(len(facts["method_availability"]), 4)
            self.assertTrue(all(a["state"] == "NOT_EVALUABLE" for a in facts["method_availability"]))
            self.assertEqual(facts["effective_G"], "NOT_EVALUABLE")
            self.assertIn("当前没有已计算的生成评价记录", v2.render(facts))

    def test_partial_cross_version_ce_ar_and_no_gate_splicing(self):
        with tempfile.TemporaryDirectory(prefix="h5_summary_v2_") as tmp:
            root = Path(tmp)
            paths, previous = setup(root)
            shutil.rmtree(paths["MSE21_DDPM"])
            shutil.rmtree(paths["scalar_DDPM"])
            ce, ar = root / "ce_eval", root / "ar_eval"
            standard_package(ce, previous, ["CE21_DDPM"], paths)
            ar_package(ar, previous, paths)
            facts = v2.build(paths, [ce, ar])
            self.assertEqual(facts["observation_counts"], {"generation_groups": 4, "reconstruction_groups": 2, "synthetic_missing_observations": 0})
            self.assertEqual({g["method"] for g in facts["generation_observations"]}, {"CE21_DDPM", "terminal_LSTM"})
            self.assertTrue(all(g["source_binding_pass"] for g in facts["generation_observations"]))
            self.assertEqual(facts["effective_G"], "NOT_EVALUABLE")
            self.assertEqual(facts["raw_G"], "NOT_EVALUABLE")
            self.assertEqual(facts["PROMOTE"], "HOLD")
            self.assertEqual(facts["operations"]["split_files_opened"], 0)

    def test_stale_v1_metric_binding_is_not_silently_attached_to_v2(self):
        with tempfile.TemporaryDirectory(prefix="h5_summary_v2_") as tmp:
            root = Path(tmp)
            paths, previous = setup(root)
            ce = root / "ce_eval"
            standard_package(ce, previous, ["CE21_DDPM"], paths)
            run = json.loads((ce / "run.json").read_text())
            run["sample_inputs"][0]["path"] = str(root / "wrong_v1/categorical_ce/generated.csv")
            put(ce, "run.json", run)
            facts = v2.build(paths, [ce])
            self.assertTrue(all(not g["source_binding_pass"] and g["interpretation_state"] == "NOT_EVALUABLE" for g in facts["generation_observations"]))
            self.assertEqual(len(facts["generation_observations"]), 2)

    def test_single_complete_bound_package_can_transcribe_raw_gates(self):
        with tempfile.TemporaryDirectory(prefix="h5_summary_v2_") as tmp:
            root = Path(tmp)
            paths, previous = setup(root)
            combined = root / "combined_eval"
            standard_package(combined, previous, v2.METHODS, paths)
            facts = v2.build(paths, [combined])
            self.assertEqual(facts["engineering_E"]["state"], "RECORDED_PASS_PENDING_INDEPENDENT_REVIEW")
            self.assertEqual(facts["effective_L"], "PASS")
            self.assertEqual(facts["effective_G"], "PASS")
            self.assertEqual(facts["AR_ADEQUACY"], "NOT_ESTABLISHED")
            self.assertEqual(facts["PROMOTE"], "HOLD")

    def test_duplicate_evaluation_group_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="h5_summary_v2_") as tmp:
            root = Path(tmp)
            paths, previous = setup(root)
            one, two = root / "first_eval", root / "second_eval"
            standard_package(one, previous, ["CE21_DDPM"], paths)
            standard_package(two, previous, ["CE21_DDPM"], paths)
            with self.assertRaises(ValueError):
                v2.build(paths, [one, two])

    def test_cli_new_output_and_source_hash_preservation(self):
        with tempfile.TemporaryDirectory(prefix="h5_summary_v2_") as tmp:
            root = Path(tmp)
            paths, _ = setup(root)
            output = root / "new_readout"
            command = [sys.executable, "-B", str(Path(v2.__file__))]
            for arm, method in zip(("ce", "mse", "scalar", "ar"), v2.METHODS):
                command += ["--" + arm + "-run", paths[method]]
            command += ["--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            facts = json.loads((output / "EXECUTION_FACTS.json").read_text())
            self.assertEqual(facts["observation_counts"]["generation_groups"], 0)
            self.assertTrue(facts["engineering_E"]["input_snapshot_stable"])
            self.assertTrue((output / "READOUT_MANIFEST.json").is_file())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(v2.v1.record(v2.V1)["sha256"], v2.EXPECTED_V1)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(V2ReadoutTests))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps({"scope": "synthetic cross-version readout contracts only; no AMP data/model/run reads",
            "tests_run": result.testsRun, "passed": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
            "source": v2.v1.record(v2.__file__), "tests": v2.v1.record(__file__), "frozen_dependency": v2.v1.record(v2.V1)}, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
