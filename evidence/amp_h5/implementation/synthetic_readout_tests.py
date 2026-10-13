#!/usr/bin/env python3
"""Small synthetic artifact tests; does not open any real AMP run/split."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import build_readout as b


def put(root, name, value):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value) + "\n")


def fixture(root):
    for directory, method in b.METHODS.items():
        config = {"purpose": "formal_pilot", "samples_per_terminal": 16, "training_seconds_cap": 100,
                  "generation_seconds_cap": 100, "vae_updates": 1, "ddpm_updates": 1, "updates": 1,
                  "train": "/never/read/sealed_test37.csv"}
        put(root, directory + "/config.json", config)
        put(root, directory + "/status.json", {"state": "completed", "test_read": False,
            "training_cap_exceeded": False, "training_seconds": 2, "process_seconds": 4})
        put(root, directory + "/manifest.json", {"files": []})
        rows = ["sample_id,method,terminal,sequence,status\n"]
        for t in b.TERMINALS:
            rows += [f"{method}_{t}_{i:03d},{method},{t},A,ok\n" for i in range(16)]
        (root / directory / "generated.csv").write_text("".join(rows))
        if method == "terminal_LSTM":
            put(root, directory + "/training/RUN.json", {"completed_steps": 1, "status": "COMPLETED_FIXED_STEPS",
                "total_pipeline_seconds_before_manifest": 2, "train_update_seconds": 1})
            put(root, directory + "/training/steps.jsonl", {"step": 1, "loss": 2})
            put(root, directory + "/training/MONITORS.json", [{"step": 1, "train": {"mean_nll": 2}, "development": {"mean_nll": 3}}])
            put(root, directory + "/generation/GENERATION.json", {"wall_seconds_before_manifest": 1})
            put(root, directory + "/train_nll.json", {"role": "train", "mean_nll": 2, "token_count": 20})
            put(root, directory + "/dev_nll.json", {"role": "development", "mean_nll": 3, "token_count": 10})
        else:
            for stage in ("vae", "ddpm"):
                put(root, directory + "/" + stage + "_training_summary.json", {"updates": 1, "fixed_endpoint_reached": True, "elapsed_seconds": 1})
            put(root, directory + "/generation_summary.json", {"elapsed_seconds": 1})
            put(root, directory + "/vae_initialization.json", {"tensor_state_sha256": "fake", "parameters": 1})
            put(root, directory + "/vae_train.jsonl", {"step": 1, "loss": 2})
            put(root, directory + "/ddpm_train.jsonl", {"step": 1, "mse": 2})
            put(root, directory + "/dev_monitor.jsonl", {"step": 1, "nonpad_token_accuracy": .1})
            for split in ("train", "dev"):
                (root / directory / (split + "_reconstruction.csv")).write_text("synthetic binding only\n")
    gen = [{"method": method, "terminal": t, "requests": 16, "valid_1_48_count": 16, "valid_5_48_count": 0,
            "valid_1_48": {"unique_count": 1}, "exact_copy_train_count_all_requests": 0,
            "exact_sequence_copy_any_train_terminal_count": 0,
            "valid_1_48_vs_train": {"mmd2_linear_biased_fixed_features": .25},
            "valid_1_48_vs_development": {"mmd2_linear_biased_fixed_features": .3}}
           for method in b.METHODS.values() for t in b.TERMINALS]
    rec = [{"method": method, "split": "development", "rows": 41, "rendered_nonpad_micro": .2,
            "rendered_nonpad_macro": .1, "length_mae": 2, "whole_rendered_exact_count": 0}
           for method in list(b.METHODS.values())[:3]]
    put(root, "evaluation/generation_metrics.json", {"groups": gen})
    put(root, "evaluation/reconstruction_metrics.json", {"groups": rec})
    put(root, "evaluation/exploratory_gates.json", {"reconstruction_ce_vs_matched_mse": {"status": "EVALUATED", "pass": True, "checks": {}},
                                                  "generation_ce_vs_terminal_ar": {"status": "EVALUATED", "pass": True, "checks": {}}})
    put(root, "evaluation/run.json", {"status": "COMPLETE", "wall_seconds": 1})
    put(root, "evaluation/MANIFEST.json", {"files": []})
    put(root, "evaluation/ar_teacher_forced_nll.json", [])


class ReadoutTests(unittest.TestCase):
    def test_empty_root_has_full_missing_rows_and_hold(self):
        with tempfile.TemporaryDirectory(prefix="h5_readout_synthetic_") as temp:
            result = b.build(temp)
            self.assertEqual(len(result["evaluation"]["generation_rows"]), 8)
            self.assertEqual(len(result["evaluation"]["development_reconstruction_rows"]), 3)
            self.assertEqual(result["engineering_E"]["state"], "NOT_EVALUABLE")
            self.assertEqual(result["PROMOTE"], "HOLD")
            self.assertTrue(result["missing_or_unreadable_artifacts"])
            self.assertIn("NOT_EVALUABLE", b.render(result))

    def test_full_fixture_only_transcribes_and_never_opens_config_split(self):
        with tempfile.TemporaryDirectory(prefix="h5_readout_synthetic_") as temp:
            root = Path(temp)
            fixture(root)
            result = b.build(root)
            self.assertEqual(result["engineering_E"]["state"], "RECORDED_PASS_PENDING_INDEPENDENT_REVIEW")
            self.assertEqual((result["raw_L"], result["raw_G"]), ("PASS", "PASS"))
            self.assertEqual(result["PROMOTE"], "HOLD")
            self.assertEqual(result["AR_ADEQUACY"], "NOT_ESTABLISHED")
            self.assertEqual(result["operations"]["split_files_opened"], 0)
            self.assertEqual(result["cost_summary"]["known_method_process_seconds_sum"], 16)
            self.assertEqual(result["evaluation"]["generation_rows"][0]["development_mmd2"], .3)
            self.assertTrue(all(str(root) in x["path"] for x in result["consumed_artifacts"]))

    def test_failed_method_keeps_raw_values_but_invalidates_effective_gates(self):
        with tempfile.TemporaryDirectory(prefix="h5_readout_synthetic_") as temp:
            root = Path(temp)
            fixture(root)
            put(root, "raw_onehot_mse/status.json", {"state": "failed", "error": "synthetic failure"})
            result = b.build(root)
            self.assertEqual(result["raw_G"], "PASS")
            self.assertEqual(result["effective_G"], "NOT_EVALUABLE")
            self.assertEqual(result["effective_L"], "NOT_EVALUABLE")
            self.assertEqual(result["evaluation"]["generation_rows"][0]["development_mmd2"], .3)
            self.assertIn("synthetic failure", b.render(result))
            (root / "evaluation/MANIFEST.json").unlink()
            self.assertFalse(b.build(root)["engineering_E"]["recorded_evaluation_complete"])

    def test_missing_reconstruction_and_nonfinite_log_are_explicit(self):
        with tempfile.TemporaryDirectory(prefix="h5_readout_synthetic_") as temp:
            root = Path(temp)
            fixture(root)
            (root / "categorical_ce/dev_reconstruction.csv").unlink()
            (root / "scalar_legacy/ddpm_train.jsonl").write_text('{"step":1,"mse":NaN}\n')
            result = b.build(root)
            self.assertEqual(result["engineering_E"]["state"], "NOT_EVALUABLE")
            paths = {p["path"] for p in result["missing_or_unreadable_artifacts"]}
            self.assertIn("categorical_ce/dev_reconstruction.csv", paths)
            self.assertIn("scalar_legacy/ddpm_train.jsonl", paths)

    def test_cli_hash_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="h5_readout_synthetic_") as temp:
            root = Path(temp)
            fixture(root)
            cmd = [sys.executable, "-B", str(Path(b.__file__)), "--run-root", temp]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            manifest = json.loads((root / "READOUT_MANIFEST.json").read_text())
            for entry in manifest["outputs"]:
                self.assertEqual(entry["sha256"], hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest())
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReadoutTests))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps({"scope": "synthetic existing-artifact summary only; no natural AMP run read",
            "tests_run": result.testsRun, "passed": result.wasSuccessful(), "errors": len(result.errors), "failures": len(result.failures),
            "source": b.record(b.__file__), "tests": b.record(__file__)}, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
