#!/usr/bin/env python3
"""Synthetic evaluator checks only; no natural AMP reference or model reads."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import evaluate_h5 as h


def molecule(sequence, terminal, index, role="train"):
    return {"sequence": sequence, "terminal_state": terminal, "class": str(h.TERMINALS.index(terminal)),
            "length": str(len(sequence)), "component_id": "synthetic_" + role + "_" + str(index), "split": role,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "molecule_sha256": hashlib.sha256(("AMP_MOLECULE_V1\n" + sequence + "\n" + terminal).encode()).hexdigest()}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generated(method, terminal, sequence, i, status="ok"):
    return {"method": method, "terminal": terminal, "sequence": sequence, "sample_id": f"{method}_{terminal}_{i:03d}",
            "status": status, "seed": str(422 + h.TERMINALS.index(terminal)), "length": str(len(sequence))}


class SyntheticEvaluator(unittest.TestCase):
    def setUp(self):
        self.train = [molecule("AAAAA", "FREE", 0), molecule("ACDEF", "FREE", 1),
                      molecule("CCCCC", "AMD", 2), molecule("DEFGH", "AMD", 3)]
        self.dev = [molecule("AAAAC", "FREE", 0, "development"), molecule("CCCCD", "AMD", 1, "development")]

    def test_frozen_hash_and_explicit_421_formula(self):
        self.assertEqual(h.base.sha256(h.METRICS), h.METRICS_SHA256)
        a = h.base.feature_vector("A")
        self.assertEqual(len(a), 421)
        self.assertAlmostEqual(a[0], 1 / (48 * math.sqrt(3)))
        self.assertAlmostEqual(a[1], 1 / math.sqrt(3))
        self.assertEqual(sum(a[21:]), 0)
        comparison = h.base.compare_summaries(h.base.sequence_summary(["A"]), h.base.sequence_summary(["C"]))
        self.assertAlmostEqual(comparison["mmd2_linear_biased_fixed_features"], 2 / 3)
        self.assertAlmostEqual(comparison["aa_jensen_shannon_bits"], 1)
        self.assertEqual(comparison["aa_l1_distance"], 2)
        self.assertIsNone(comparison["dipeptide_jensen_shannon_bits"])

    def test_global_alignment_includes_terminal_gaps(self):
        exact = h.base.global_identity("ACD", "ACD")
        self.assertEqual((exact["matches"], exact["aligned_columns"], exact["identity"]), (3, 3, 1))
        gap = h.base.global_identity("A", "AA")
        self.assertEqual((gap["matches"], gap["aligned_columns"], gap["identity"]), (1, 2, .5))

    def test_invalid_empty_failure_and_any_terminal_copy(self):
        groups = {}
        for t in h.TERMINALS:
            groups["CE21_DDPM", t] = [generated("CE21_DDPM", t, s, i, status)
               for i, (s, status) in enumerate([("", "ok"), ("CCCCC", "error:Synthetic"), ("A?", "ok"), ("A" * 49, "ok")])]
        metrics, audit = h.evaluate_generation(groups, self.train, self.dev)
        free = next(g for g in metrics["groups"] if g["terminal"] == "FREE")
        self.assertEqual(free["requests"], 4)
        self.assertEqual(free["valid_1_48_count"], 1)
        self.assertEqual(free["valid_5_48_count"], 1)
        self.assertEqual(free["empty_count"], 1)
        self.assertEqual(free["invalid_alphabet_count"], 1)
        self.assertEqual(free["length_over_48_count"], 1)
        self.assertEqual(free["exact_copy_train_count_all_requests"], 0)
        self.assertEqual(free["exact_sequence_copy_any_train_terminal_count"], 1)
        self.assertEqual(free["valid_unique_count_per_request"], .25)
        copied = next(r for r in audit if r["terminal"] == "FREE" and r["sequence"] == "CCCCC")
        self.assertEqual(copied["nearest_any_train_terminal"]["identity"], 1)
        self.assertFalse(copied["status_ok"])
        self.assertTrue(copied["valid_1_48"])
        self.assertEqual(len(audit), 8)

    def test_all_empty_metrics_remain_undefined(self):
        groups = {(m, t): [generated(m, t, "", i) for i in range(2)] for m in h.METHODS for t in h.TERMINALS}
        metrics, audit = h.evaluate_generation(groups, self.train, self.dev)
        for g in metrics["groups"]:
            self.assertEqual(g["empty_count"], 2)
            self.assertEqual(g["all_output_unique_count_including_empty_and_failed"], 1)
            self.assertEqual(g["valid_1_48"]["unique_count"], 0)
            self.assertIsNone(g["valid_1_48_vs_development"]["mmd2_linear_biased_fixed_features"])
            self.assertIsNone(g["pooled_aa_entropy_bits_absolute_gap_to_train"])
        self.assertFalse(h.exploratory_gates(metrics, [])["both_exploratory_screens_pass"])
        self.assertEqual(len(audit), 16)

    def test_rendered_reconstruction_denominators_macro_and_exact(self):
        truth = [molecule("AACD", "FREE", 0), molecule("EF", "AMD", 1), molecule("AGH", "FREE", 2)]
        rows = [{"row_id": str(i), "terminal": t["terminal_state"], "truth_sequence": t["sequence"],
                 "prediction_sequence": p, "status": "ok"}
                for i, (t, p) in enumerate(zip(truth, ["A", "", "AGHI"]))]
        m, audit = h.reconstruction_metrics(rows, truth)
        self.assertEqual(m["target_residues"], 9)
        self.assertEqual(m["rendered_aligned_correct"], 4)
        self.assertAlmostEqual(m["rendered_nonpad_micro"], 4 / 9)
        self.assertAlmostEqual(m["rendered_nonpad_macro"], ((2 / 3) + 1 + 1) / 7)
        self.assertAlmostEqual(m["per_sequence_mean_nonpad_accuracy"], (1 / 4 + 0 + 1) / 3)
        self.assertEqual(m["length_mae"], 2)
        self.assertEqual(m["whole_rendered_exact_count"], 0)
        rows[0]["prediction_sequence"] = "AACD"
        m, _ = h.reconstruction_metrics(rows, truth)
        self.assertEqual(m["whole_rendered_exact_count"], 1)
        self.assertEqual(len(audit), 3)
        rows[0]["truth_sequence"] = "C"
        with self.assertRaises(ValueError):
            h.reconstruction_metrics(rows, truth)

    def test_no_missing_truth_after_empty_prediction(self):
        rows = [{"row_id": i, "terminal": t["terminal_state"], "truth_sequence": t["sequence"],
                 "prediction_sequence": "", "status": "error:Synthetic"} for i, t in enumerate(self.train)]
        m, _ = h.reconstruction_metrics(rows, self.train)
        self.assertEqual(m["target_residues"], 20)
        self.assertEqual(m["rendered_nonpad_micro"], 0)
        self.assertEqual(m["rendered_nonpad_macro"], 0)
        self.assertEqual(m["length_mae"], 5)

    def test_exact_generation_count_seed_duplicate_and_metadata_guards(self):
        with tempfile.TemporaryDirectory(prefix="h5_synthetic_") as temporary:
            p = Path(temporary) / "samples.csv"
            rows = [generated(m, t, "", i) for m in h.METHODS for t in h.TERMINALS for i in range(2)]
            schedule = {m: {t: 422 + h.TERMINALS.index(t) for t in h.TERMINALS} for m in h.METHODS}
            write_csv(p, rows)
            self.assertEqual(len(h.sample_groups([p], h.METHODS, 2, schedule)), 8)
            write_csv(p, rows[:-1])
            with self.assertRaises(ValueError): h.sample_groups([p], h.METHODS, 2, schedule)
            changed = copy.deepcopy(rows)
            changed[1]["sample_id"] = changed[0]["sample_id"]
            write_csv(p, changed)
            with self.assertRaises(ValueError): h.sample_groups([p], h.METHODS, 2, schedule)
            changed = copy.deepcopy(rows)
            changed[0]["seed"] = "999"
            write_csv(p, changed)
            with self.assertRaises(ValueError): h.sample_groups([p], h.METHODS, 2, schedule)
            changed = copy.deepcopy(rows)
            changed[0]["sample_id"] = "substituted_slot"
            write_csv(p, changed)
            with self.assertRaises(ValueError): h.sample_groups([p], h.METHODS, 2, schedule)
            p.write_text("method,method,sequence\nx,x,\n")
            with self.assertRaises(ValueError): h.read_rows(p)
            write_csv(p, [{"sequence": "A\nC"}])
            self.assertEqual(h.read_rows(p)[0]["sequence"], "A\nC")

    def test_reference_hash_role_and_overlap_guards(self):
        with tempfile.TemporaryDirectory(prefix="h5_synthetic_") as temporary:
            p = Path(temporary) / "train.csv"
            write_csv(p, self.train)
            rows = h.dataset(p, "train", h.base.sha256(p), 4)
            self.assertEqual(len(rows), 4)
            with self.assertRaises(ValueError): h.dataset(p, "train", "0" * 64, 4)
            with self.assertRaises(ValueError): h.dataset(p, "development", h.base.sha256(p), 4)
            h.split_isolation(self.train, self.dev)
            with self.assertRaises(ValueError): h.split_isolation(self.train, self.train)
            prohibited = Path(temporary) / "test.csv"
            with self.assertRaises(ValueError): h.guarded_read(prohibited)
            link = Path(temporary) / "innocent.csv"
            link.symlink_to(prohibited)
            with self.assertRaises(ValueError): h.guarded_read(link)

    def test_gates_require_both_terminals_strict_mmd_and_entropy_distance(self):
        groups = {(m, t): [generated(m, t, "ACDEF", i) for i in range(2)] for m in h.METHODS for t in h.TERMINALS}
        metrics, _ = h.evaluate_generation(groups, self.train, self.dev)
        for g in metrics["groups"]:
            g["valid_1_48_vs_development"]["mmd2_linear_biased_fixed_features"] = .1 if g["method"] == "CE21_DDPM" else .2
            g["pooled_aa_entropy_bits_absolute_gap_to_train"] = .1
            g["pooled_dipeptide_entropy_bits_absolute_gap_to_train"] = .1
        rec = [{"method": m, "split": "development", "rendered_nonpad_micro": .5 if m == "CE21_DDPM" else .4,
                "rendered_nonpad_macro": .4 if m == "CE21_DDPM" else .3,
                "length_mae": 1, "whole_rendered_exact_count": 0, "rows": 2} for m in ("CE21_DDPM", "MSE21_DDPM")]
        self.assertTrue(h.exploratory_gates(metrics, rec)["both_exploratory_screens_pass"])
        ce_amd = next(g for g in metrics["groups"] if g["method"] == "CE21_DDPM" and g["terminal"] == "AMD")
        ce_amd["pooled_dipeptide_entropy_bits_absolute_gap_to_train"] = .2
        self.assertFalse(h.exploratory_gates(metrics, rec)["generation_ce_vs_terminal_ar"]["pass"])
        ce_amd["pooled_dipeptide_entropy_bits_absolute_gap_to_train"] = .1
        ce_amd["valid_1_48_vs_development"]["mmd2_linear_biased_fixed_features"] = .2
        self.assertFalse(h.exploratory_gates(metrics, rec)["generation_ce_vs_terminal_ar"]["pass"])
        ce_amd["valid_1_48_vs_development"]["mmd2_linear_biased_fixed_features"] = .1
        ce_amd["execution_completed_count"] = 1
        gate = h.exploratory_gates(metrics, rec)
        self.assertFalse(gate["generation_ce_vs_terminal_ar"]["pass"])
        self.assertEqual(gate["generation_ce_vs_terminal_ar"]["status"], "NOT_EVALUABLE_FAILED_GENERATION")

    def test_ar_nll_nested_report_separate_and_checked(self):
        with tempfile.TemporaryDirectory(prefix="h5_synthetic_") as temporary:
            p = Path(temporary) / "nll.json"
            h.dump(p, {"train": {"role": "train", "nll_sum": 10., "token_count": 5},
                       "development": {"role": "development", "nll_sum": 9., "token_count": 3}})
            result = h.ar_nll_diagnostics([p])
            self.assertEqual([x["nll_per_target_token_nats"] for x in result], [2., 3.])
            h.dump(p, {"role": "train", "nll_sum": 2., "token_count": 0})
            with self.assertRaises(ValueError): h.ar_nll_diagnostics([p])

    def test_synthetic_cli_artifacts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="h5_synthetic_") as temporary:
            root = Path(temporary)
            train, dev, samples, seeds = [root / p for p in ("train.csv", "development.csv", "samples.csv", "seeds.json")]
            write_csv(train, self.train)
            write_csv(dev, self.dev)
            write_csv(samples, [generated(m, t, "A" if i else "", i) for m in h.METHODS for t in h.TERMINALS for i in range(2)])
            h.dump(seeds, {m: {t: 422 + h.TERMINALS.index(t) for t in h.TERMINALS} for m in h.METHODS})
            command = [sys.executable, "-B", str(h.HERE / "evaluate_h5.py"), "--train", str(train), "--development", str(dev),
                       "--train-sha256", h.base.sha256(train), "--development-sha256", h.base.sha256(dev),
                       "--train-rows", "4", "--development-rows", "2", "--samples", str(samples),
                       "--requests", "2", "--sampling-seeds-json", str(seeds), "--output", str(root / "evaluation")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads((root / "evaluation" / "run.json").read_text())
            self.assertEqual(status["rows_retained"], 16)
            self.assertFalse(status["test_accessed"])
            self.assertEqual(status["model_calls"], 0)
            self.assertEqual(status["seed_check"], "prelocked_schedule_checked")
            self.assertTrue((root / "evaluation" / "MANIFEST.json").is_file())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticEvaluator)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {"scope": "synthetic evaluator checks only; no natural AMP data or model reads",
               "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "passed": result.wasSuccessful(), "evaluator": h.base.record(h.HERE / "evaluate_h5.py"),
               "tests": h.base.record(__file__), "dependency": h.base.record(h.METRICS)}
    if len(sys.argv) > 1:
        h.dump(sys.argv[1], payload)
    sys.exit(0 if result.wasSuccessful() else 1)
