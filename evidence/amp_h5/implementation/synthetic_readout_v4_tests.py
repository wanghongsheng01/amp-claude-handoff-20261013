#!/usr/bin/env python3
"""Phase-split readout fixtures only; never open natural AMP runs or datasets."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import build_readout_v4 as v3
import build_readout_v3 as frozen_v3
from synthetic_readout_v2_tests import setup, standard_package, ar_package
from synthetic_readout_tests import put


def seal_manifest(root, relative=False):
    records = [v3.v1.record(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "MANIFEST.json"]
    if relative:
        records = [{**r, "path": Path(r["path"]).name} for r in records]
    put(root, "MANIFEST.json", {"files": records})


def contract_file(root, paths=None, recovery=None, evaluations=(), receipt=None):
    paths = paths or {m: str(root / "unavailable" / m) for m in v3.METHODS}
    recovery = recovery or root / "recovery"
    content = {"schema_version": "AMP_H5_PHASE_SPLIT_READOUT_V4", "base_directory": str(root),
       "methods": {m: {"training_source_root": p, "generation_attempt_root": str(recovery) if m == v3.CE else p,
                       "training_instance_id": "synthetic_" + m,
                       "generation_attempt_id": v3.ATTEMPT if m == v3.CE else "synthetic_" + m}
                   for m, p in paths.items()},
       "evaluation_directories": [str(p) for p in evaluations], "recovery_postflight_receipt": receipt,
       "context_evidence": []}
    put(root, "readout_contract.json", content)
    return root / "readout_contract.json"


def ready_fixture(root, include_receipt=True, methods=None):
    paths, previous = setup(root)
    source = Path(paths[v3.CE])
    config = json.loads((source / "config.json").read_text())
    config.update(vae_updates=1000, ddpm_updates=1000)
    put(source, "config.json", config)
    put(source, "status.json", {"state": "running", "test_read": False})
    digests = {"vae_state": "a" * 64, "ddpm_ema_state": "b" * 64}
    for name, stage in (("vae_state", "vae"), ("ddpm_ema_state", "ddpm")):
        put(source, name + ".json", {"updates": 1000, "requested_updates": 1000,
           "fixed_endpoint_reached": True, "sha256": ("c" if stage == "vae" else "d") * 64,
           "tensor_state_sha256": digests[name]})
        put(source, stage + "_training_summary.json", {"updates": 1000, "fixed_endpoint_reached": True, "elapsed_seconds": 1})
        field = "loss" if stage == "vae" else "mse"
        (source / (stage + "_train.jsonl")).write_text("".join(json.dumps({"step": i, field: 1}) + "\n" for i in range(1, 1001)))
    put(source, "train_feature_scaler.json", {"fit_scope": "train only", "synthetic": True})
    recovery = root / "recovery"
    recovery.mkdir()
    for name in ("generated.csv", "train_reconstruction.csv", "dev_reconstruction.csv", "config.json"):
        shutil.copyfile(source / name, recovery / name)
    put(recovery, "status.json", {"state": "recovery_completed", "attempt_id": v3.ATTEMPT,
       "scope_id": "AMP_H5_CE_CHECKPOINT_RECOVERY_V1", "training_updates_this_attempt": 0,
       "original_v2_engineering_gate": "NOT_EVALUABLE_INTERRUPTION", "model_states_unchanged": True,
       "original_inputs_unchanged": True, "generation_cap_exceeded": False, "outer_cap_exceeded": False,
       "process_seconds_before_final_status_io": 4, "setup_seconds": 1, "export_seconds": 1, "test_read": False})
    put(recovery, "generation_summary.json", {"elapsed_seconds": 2, "requested": 32, "successful_execution": 32})
    put(recovery, "model_identity_after.json", {"before": digests, "after": digests, "expected": digests,
       "both_loaded_verified_and_unchanged": True})
    put(recovery, "loaded_state_identity.json", {"before": digests, "expected": digests})
    put(recovery, "input_integrity_after.json", {"original_errors": [], "core_errors": []})
    put(recovery, "restored_scaler_identity.json", {"source": str(source / "train_feature_scaler.json"),
       "sha256": v3.v1.record(source / "train_feature_scaler.json")["sha256"], "fit_called": False})
    put(recovery, "phase_provenance.json", {"new_training_updates": 0,
       "training": {"source_directory": str(source), "copied_or_linked": False},
       "generation_and_reconstruction": {"source_directory": str(recovery)}})
    put(recovery, "manifest.json", {"files": []})
    put(recovery, "recovery_contract.json", {"synthetic": True})
    mapped = {**paths, v3.CE: str(recovery)}
    evaluation = root / "evaluation"
    standard_package(evaluation, previous, methods or v3.METHODS, mapped)
    binding = None
    if include_receipt:
        sources = [v3.v1.record(recovery / n) for n in ("status.json", "generated.csv")]
        sources += [{"path": str(source / (name + ".pt")),
                    "sha256": json.loads((source / (name + ".json")).read_text())["sha256"]} for name in digests]
        put(root, "postflight.json", {"status": "PASS", "scope": "RECOVERY_ENGINEERING_ONLY",
           "generation_attempt_id": v3.ATTEMPT, "generation_attempt_root": str(recovery),
           "training_source_root": str(source), "source_bindings": sources,
           "original_v2_engineering_gate": "NOT_EVALUABLE_INTERRUPTION",
           "original_protocol_E_L_G": "NOT_EVALUABLE", "PROMOTE": "HOLD"})
        binding = v3.v1.record(root / "postflight.json")
    seal_manifest(evaluation)
    path = contract_file(root, paths, recovery, [evaluation], binding)
    return path, source, recovery, evaluation


def build(path):
    return v3.build(v3.load_contract(path))


class PhaseSplitTests(unittest.TestCase):
    v3_failure_reproduction = None

    def assert_manifest_blocked(self, facts):
        self.assertEqual(facts["evaluation_manifest_integrity"]["state"], "NOT_EVALUABLE")
        self.assertTrue(all(row["interpretation_state"] == "NOT_EVALUABLE" for row in facts["generation_observations"] + facts["reconstruction_observations"]))
        self.assertEqual([facts["recovery_descriptive_screens"][k]["state"] for k in "LG"], ["NOT_EVALUABLE"] * 2)
        self.assertEqual([facts["original_protocol"][k] for k in "ELG"], ["NOT_EVALUABLE"] * 3)

    def test_absent_methods_never_create_observations(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            facts = build(contract_file(Path(tmp).resolve()))
            self.assertEqual(facts["generation_observations"], [])
            self.assertEqual(facts["reconstruction_observations"], [])
            self.assertEqual([facts["original_protocol"][k] for k in "ELG"], ["NOT_EVALUABLE"] * 3)
            self.assertIsNone(facts["original_protocol"]["original_total_wall_seconds"])
            self.assertEqual(facts["synthetic_missing_observations"], 0)

    def test_bound_postflight_releases_only_recovery_screens(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            path, _, _, _ = ready_fixture(Path(tmp).resolve())
            facts = build(path)
            self.assertTrue(facts["CE_recovery_attempt"]["recorded_completion"], facts["CE_recovery_attempt"]["checks"])
            self.assertEqual(facts["recovery_engineering_postflight"]["state"], "RECOVERY_ENGINEERING_PASS")
            self.assertEqual([facts["recovery_descriptive_screens"][k]["state"] for k in "LG"], ["PASS"] * 2)
            self.assertEqual([facts["original_protocol"][k] for k in "ELG"], ["NOT_EVALUABLE"] * 3)
            self.assertEqual(facts["PROMOTE"], "HOLD")
            self.assertEqual(facts["AR_ADEQUACY"], "NOT_ESTABLISHED")
            self.assertEqual(facts["phase_costs"]["original_CE_amendment_record"]["interrupted_generation"]["seconds"], 451.935743)

    def test_no_postflight_retains_raw_metrics_but_cannot_release(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            path, _, _, _ = ready_fixture(Path(tmp).resolve(), include_receipt=False)
            facts = build(path)
            self.assertEqual(len(facts["generation_observations"]), 8)
            self.assertEqual(facts["recovery_descriptive_screens"]["G"]["state"], "NOT_EVALUABLE")
            self.assertEqual(facts["recovery_descriptive_screens"]["G"]["records"][0]["raw_state"], "PASS")

    def test_preflight_scope_is_not_postflight(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            root = Path(tmp).resolve()
            path, _, _, _ = ready_fixture(root)
            receipt = json.loads((root / "postflight.json").read_text())
            receipt["scope"] = "RECOVERY_PREFLIGHT"
            put(root, "postflight.json", receipt)
            contract = json.loads(path.read_text())
            contract["recovery_postflight_receipt"] = v3.v1.record(root / "postflight.json")
            put(root, path.name, contract)
            facts = build(path)
            self.assertFalse(facts["recovery_engineering_postflight"]["pass"])
            self.assertEqual(facts["recovery_descriptive_screens"]["L"]["state"], "NOT_EVALUABLE")

    def test_copied_training_log_blocks_recovery(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            path, source, recovery, _ = ready_fixture(Path(tmp).resolve())
            shutil.copyfile(source / "vae_train.jsonl", recovery / "vae_train.jsonl")
            facts = build(path)
            self.assertFalse(facts["CE_recovery_attempt"]["checks"]["no_training_artifact_copies"])
            self.assertEqual(facts["recovery_descriptive_screens"]["G"]["state"], "NOT_EVALUABLE")

    def test_original_root_cannot_supply_new_generation_binding(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            path, source, _, evaluation = ready_fixture(Path(tmp).resolve())
            run = json.loads((evaluation / "run.json").read_text())
            run["sample_inputs"][0]["path"] = str(source / "generated.csv")
            put(evaluation, "run.json", run)
            facts = build(path)
            self.assertTrue(all(not r["source_binding_pass"] for r in facts["generation_observations"] if r["method"] == v3.CE))
            self.assertEqual(facts["recovery_descriptive_screens"]["G"]["state"], "NOT_EVALUABLE")

    def test_missing_mse_scalar_partial_has_no_fake_groups(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            root = Path(tmp).resolve()
            path, _, _, _ = ready_fixture(root, methods=[v3.CE, "terminal_LSTM"])
            shutil.rmtree(root / "formal_v2/raw_onehot_mse")
            shutil.rmtree(root / "formal_v2/scalar_legacy")
            facts = build(path)
            self.assertEqual(len(facts["generation_observations"]), 4)
            self.assertEqual(len(facts["reconstruction_observations"]), 2)
            self.assertEqual(facts["recovery_descriptive_screens"]["L"]["state"], "NOT_EVALUABLE")
            self.assertEqual(facts["recovery_descriptive_screens"]["G"]["state"], "PASS")

    def test_does_not_open_split_checkpoint_or_array(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            path, _, _, _ = ready_fixture(Path(tmp).resolve())
            original = Path.open
            def guarded(p, *a, **kw):
                if p.suffix in (".pt", ".npz") or "test37" in str(p) or str(p).startswith("/never/"):
                    raise AssertionError("forbidden file opened: " + str(p))
                return original(p, *a, **kw)
            with patch.object(Path, "open", guarded):
                facts = build(path)
            self.assertTrue(facts["input_snapshot_stable"])
            self.assertEqual(set(facts["operations"].values()), {0})

    def test_contract_rejects_sealed_path_and_binary_evidence(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            root = Path(tmp).resolve()
            path = contract_file(root)
            raw = json.loads(path.read_text())
            raw["evaluation_directories"] = ["sealed_test37"]
            put(root, path.name, raw)
            with self.assertRaises(ValueError):
                v3.load_contract(path)
            raw["evaluation_directories"] = []
            raw["context_evidence"] = [{"path": "weights.pt", "sha256": "a" * 64}]
            put(root, path.name, raw)
            with self.assertRaises(ValueError):
                v3.load_contract(path)

    def test_cli_fresh_output_and_frozen_sources_preserved(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            root = Path(tmp).resolve()
            path, source, _, _ = ready_fixture(root)
            before = v3.v1.record(source / "status.json")
            command = [sys.executable, "-B", v3.__file__, "--contract", str(path), "--output", str(root / "readout")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(before, v3.v1.record(source / "status.json"))
            self.assertTrue((root / "readout/READOUT_MANIFEST.json").is_file())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(v3.v1.record(v3.V2)["sha256"], v3.EXPECTED_V2)
            self.assertEqual(v3.v1.record(v3.v2.V1)["sha256"], v3.v2.EXPECTED_V1)

    def test_every_consumed_standard_json_corruption_blocks_interpretation(self):
        names = ("generation_metrics.json", "reconstruction_metrics.json", "exploratory_gates.json",
                 "run.json", "ar_teacher_forced_nll.json")
        for name in names:
            with self.subTest(member=name), tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
                path, _, _, evaluation = ready_fixture(Path(tmp).resolve())
                put(evaluation, "ar_teacher_forced_nll.json", [])
                seal_manifest(evaluation)
                value = json.loads((evaluation / name).read_text())
                if isinstance(value, dict):
                    value["synthetic_corruption"] = True
                else:
                    value.append({"synthetic_corruption": True})
                put(evaluation, name, value)
                facts = build(path)
                self.assert_manifest_blocked(facts)
                self.assertEqual(len(facts["generation_observations"]), 8)
                self.assertTrue(facts["input_snapshot_stable"])
                failing = facts["evaluation_packages"][0]["manifest_integrity"]["checks"]
                self.assertTrue(any(Path(c["path"]).name == name and not c["pass"] for c in failing))

    def test_ar_only_every_consumed_json_is_covered(self):
        for name in ("GENERATION_METRICS.json", "RUN_FACTS.json", "SOURCE_BINDINGS.json", "TEACHER_FORCED_NLL.json"):
            with self.subTest(member=name), tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
                root = Path(tmp).resolve()
                path, _, _, _ = ready_fixture(root)
                raw = json.loads(path.read_text())
                paths = {m: d["generation_attempt_root"] for m, d in raw["methods"].items()}
                ar = root / "ar_only_evaluation"
                ar_package(ar, root / "fixture/evaluation", paths)
                seal_manifest(ar)
                raw["evaluation_directories"] = [str(ar)]
                put(root, path.name, raw)
                good = build(path)
                self.assertEqual(good["evaluation_manifest_integrity"]["state"], "VERIFIED_CONSUMED_JSON")
                self.assertEqual(len(good["generation_observations"]), 2)
                value = json.loads((ar / name).read_text())
                if isinstance(value, dict):
                    value["synthetic_corruption"] = True
                else:
                    value.append({"synthetic_corruption": True})
                put(ar, name, value)
                self.assert_manifest_blocked(build(path))

    def test_manifest_missing_duplicate_size_and_sha_are_rejected(self):
        for mutation in ("missing_manifest", "empty_members", "missing_member", "duplicate_member", "case_alias", "wrong_bytes", "wrong_sha", "malformed"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
                path, _, _, evaluation = ready_fixture(Path(tmp).resolve())
                manifest = json.loads((evaluation / "MANIFEST.json").read_text())
                member = next(r for r in manifest["files"] if Path(r["path"]).name == "generation_metrics.json")
                if mutation == "missing_manifest":
                    (evaluation / "MANIFEST.json").unlink()
                elif mutation == "malformed":
                    (evaluation / "MANIFEST.json").write_text("{invalid\n")
                else:
                    if mutation == "empty_members": manifest["files"] = []
                    if mutation == "missing_member": manifest["files"].remove(member)
                    if mutation == "duplicate_member": manifest["files"].append(dict(member))
                    if mutation == "case_alias": manifest["files"].append({**member, "path": "GENERATION_METRICS.JSON"})
                    if mutation == "wrong_bytes": member["bytes"] += 1
                    if mutation == "wrong_sha": member["sha256"] = "0" * 64
                    put(evaluation, "MANIFEST.json", manifest)
                self.assert_manifest_blocked(build(path))

    def test_manifest_entries_never_trigger_external_or_binary_reads(self):
        for member_path, expected_pass in (("unread_checkpoint.pt", True), ("unread_cache.npz", True),
                                            ("/never/read/sealed_test37.csv", False), ("../external.json", False)):
            with self.subTest(member_path=member_path), tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
                path, _, _, evaluation = ready_fixture(Path(tmp).resolve())
                manifest = json.loads((evaluation / "MANIFEST.json").read_text())
                manifest["files"].append({"path": member_path, "bytes": 9, "sha256": "1" * 64})
                put(evaluation, "MANIFEST.json", manifest)
                original = Path.open
                def guarded(p, *a, **kw):
                    if p.suffix in (".pt", ".npz") or "test37" in str(p) or str(p).startswith("/never/"):
                        raise AssertionError("manifest caused forbidden read: " + str(p))
                    return original(p, *a, **kw)
                with patch.object(Path, "open", guarded):
                    facts = build(path)
                self.assertEqual(facts["evaluation_packages"][0]["manifest_integrity"]["pass"], expected_pass)
                self.assertEqual(facts["evaluation_packages"][0]["manifest_integrity"]["manifest_listed_paths_opened"], 0)
                if not expected_pass:
                    self.assert_manifest_blocked(facts)

    def test_relative_manifest_members_are_supported(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            path, _, _, evaluation = ready_fixture(Path(tmp).resolve())
            seal_manifest(evaluation, relative=True)
            facts = build(path)
            self.assertEqual(facts["evaluation_manifest_integrity"]["state"], "VERIFIED_CONSUMED_JSON")
            self.assertEqual(facts["recovery_descriptive_screens"]["G"]["state"], "PASS")

    def test_contract_runtime_enforces_schema_fields(self):
        for mutation in ("missing_context", "extra_key", "extra_method_key", "missing_method_key", "bad_bytes", "duplicate_directory"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
                root = Path(tmp).resolve()
                path = contract_file(root)
                raw = json.loads(path.read_text())
                if mutation == "missing_context": del raw["context_evidence"]
                if mutation == "extra_key": raw["unrecognized_override"] = "PASS"
                if mutation == "extra_method_key": raw["methods"][v3.CE]["override"] = True
                if mutation == "missing_method_key": del raw["methods"][v3.CE]["training_instance_id"]
                if mutation == "bad_bytes": raw["context_evidence"] = [{"path": "dummy.json", "sha256": "a" * 64, "bytes": True}]
                if mutation == "duplicate_directory": raw["evaluation_directories"] = ["eval", "eval"]
                put(root, path.name, raw)
                with self.assertRaises(ValueError):
                    v3.load_contract(path)

    def test_preserved_v3_failure_is_blocked_by_v4(self):
        with tempfile.TemporaryDirectory(prefix="h5_phase_synthetic_") as tmp:
            root = Path(tmp).resolve()
            path, _, _, evaluation = ready_fixture(root)
            before = v3.v1.record(evaluation / "generation_metrics.json")
            metrics = json.loads((evaluation / "generation_metrics.json").read_text())
            metrics["groups"][0]["valid_1_48_count"] = 0
            put(evaluation, "generation_metrics.json", metrics)
            after = v3.v1.record(evaluation / "generation_metrics.json")
            raw = json.loads(path.read_text())
            raw["schema_version"] = "AMP_H5_PHASE_SPLIT_READOUT_V3"
            put(root, "old_contract.json", raw)
            old = frozen_v3.build(frozen_v3.load_contract(root / "old_contract.json"))
            new = build(path)
            self.assertEqual(old["recovery_descriptive_screens"]["G"]["state"], "PASS")
            self.assertTrue(old["input_snapshot_stable"])
            self.assert_manifest_blocked(new)
            PhaseSplitTests.v3_failure_reproduction = {"scope": "synthetic only; natural AMP artifacts not read",
                "frozen_v3_source": v3.v1.record(frozen_v3.__file__), "before_metric_sha256": before["sha256"],
                "after_metric_sha256": after["sha256"], "mutation": "first generation group valid_1_48_count changed to zero; MANIFEST unchanged",
                "v3_effective_G": old["recovery_descriptive_screens"]["G"]["state"],
                "v3_temporal_snapshot_stable": old["input_snapshot_stable"],
                "v4_effective_G": new["recovery_descriptive_screens"]["G"]["state"],
                "v4_manifest_integrity": new["evaluation_manifest_integrity"]["state"]}


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PhaseSplitTests))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps({"scope": "synthetic phase-split readout only; no natural AMP reads or model calls",
            "tests_run": result.testsRun, "passed": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
            "source": v3.v1.record(v3.__file__), "tests": v3.v1.record(__file__),
            "preserved_v3_failure_reproduction": PhaseSplitTests.v3_failure_reproduction,
            "frozen_dependencies": [v3.v1.record(v3.V2), v3.v1.record(v3.v2.V1)]}, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
