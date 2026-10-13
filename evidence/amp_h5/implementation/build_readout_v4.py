#!/usr/bin/env python3
"""Manifest-verified phase-split AMP H5 saved-artifact readout; never imports model/evaluator.

The interrupted original protocol E/L/G are permanently NOT_EVALUABLE.
Checkpoint recovery has its own attempt, phase costs and descriptive screens.
No data split, model checkpoint or posterior array is opened by this script.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V2 = HERE / "build_readout_v2.py"
EXPECTED_V2 = "ee585532c76a6be3adfd38480bdd304b3fbd316704362c69d4064dd18d181bca"
if hashlib.sha256(V2.read_bytes()).hexdigest() != EXPECTED_V2:
    raise ValueError("frozen phase-path summary dependency drift")
spec = importlib.util.spec_from_file_location("readout_v2_frozen", V2)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
v1 = v2.v1
METHODS = v2.METHODS
CE = "CE21_DDPM"
ATTEMPT = "ce_checkpoint_recovery_v1"
ORIGINAL_COSTS = {
    "vae_training": {"seconds": 1179.031922, "bound": "reported_complete_stage"},
    "ddpm_training": {"seconds": 564.346170, "bound": "reported_complete_stage"},
    "interrupted_generation": {"seconds": 451.935743, "bound": "lower_bound_only", "complete_duration_known": False},
    "original_process_total": {"seconds": None, "bound": "unknown"},
}


def resolved_under(base, path):
    result = (base / path).resolve()
    if result == base or base not in result.parents:
        raise ValueError("source path must be a child of the explicit base directory")
    if any(re.search(r"(?:^|[^a-z])test[0-9]*(?:[^a-z]|$)", part.lower()) for part in result.parts):
        raise ValueError("sealed test path is forbidden in readout sources")
    return result


def text_binding(base, entry):
    path = resolved_under(base, entry["path"])
    if path.suffix.lower() not in (".json", ".md") or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
        raise ValueError("evidence/receipt must be a hash-bound JSON or Markdown artifact")
    return {**entry, "path": str(path)}


def validate_contract_structure(raw):
    """Enforce the published contract shape using stdlib only."""
    expected = {"schema_version", "base_directory", "methods", "evaluation_directories", "recovery_postflight_receipt", "context_evidence"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("contract fields must exactly match the published schema")
    if not isinstance(raw["base_directory"], str) or not raw["base_directory"]:
        raise ValueError("base_directory must be a nonempty path string")
    if not isinstance(raw["methods"], dict) or set(raw["methods"]) != set(METHODS):
        raise ValueError("all four method descriptors are required")
    fields = {"training_source_root", "generation_attempt_root", "training_instance_id", "generation_attempt_id"}
    for entry in raw["methods"].values():
        if not isinstance(entry, dict) or set(entry) != fields or any(not isinstance(v, str) or not v for v in entry.values()):
            raise ValueError("method descriptor must contain only four nonempty strings")
    directories = raw["evaluation_directories"]
    if not isinstance(directories, list) or any(not isinstance(p, str) or not p for p in directories) or len(set(directories)) != len(directories):
        raise ValueError("evaluation_directories must be distinct nonempty path strings")
    if not isinstance(raw["context_evidence"], list):
        raise ValueError("context_evidence must be an array, including when empty")
    bindings = raw["context_evidence"] + ([raw["recovery_postflight_receipt"]] if raw["recovery_postflight_receipt"] is not None else [])
    for entry in bindings:
        if not isinstance(entry, dict) or not {"path", "sha256"} <= set(entry) <= {"path", "sha256", "bytes"}:
            raise ValueError("evidence binding fields differ from schema")
        if not isinstance(entry["path"], str) or not re.search(r"\.(json|md)$", entry["path"]):
            raise ValueError("evidence path must name JSON or Markdown")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("evidence SHA256 must be lowercase hexadecimal")
        if "bytes" in entry and (type(entry["bytes"]) is not int or entry["bytes"] < 0):
            raise ValueError("evidence bytes must be a nonnegative integer")


def load_contract(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(), parse_constant=v1.reject_nonfinite_json)
    validate_contract_structure(raw)
    if raw.get("schema_version") != "AMP_H5_PHASE_SPLIT_READOUT_V4":
        raise ValueError("unsupported readout contract version")
    base = (path.parent / raw["base_directory"]).resolve()
    if set(raw["methods"]) != set(METHODS):
        raise ValueError("all four method source descriptors required; absent observations remain absent")
    methods = {}
    for method, entry in raw["methods"].items():
        if any(not isinstance(entry.get(k), str) or not entry[k] for k in ("training_instance_id", "generation_attempt_id")):
            raise ValueError("explicit nonempty training and generation instance identities required")
        train = resolved_under(base, entry["training_source_root"])
        generation = resolved_under(base, entry["generation_attempt_root"])
        if method == CE:
            if train == generation or entry.get("generation_attempt_id") != ATTEMPT:
                raise ValueError("CE recovery must have a distinct generation root and fixed attempt identity")
        elif train != generation:
            raise ValueError("only the explicitly amended CE attempt may split phases in this schema")
        methods[method] = {**entry, "training_source_root": str(train), "generation_attempt_root": str(generation)}
    if len({m["generation_attempt_root"] for m in methods.values()}) != 4:
        raise ValueError("generation roots must be distinct")
    receipt = raw.get("recovery_postflight_receipt")
    if receipt is not None:
        receipt = text_binding(base, receipt)
    evidence = []
    for item in raw.get("context_evidence", []):
        evidence.append(text_binding(base, item))
    return {"source": v1.record(path), "base_directory": str(base), "methods": methods,
            "evaluation_directories": [str(resolved_under(base, p)) for p in raw.get("evaluation_directories", [])],
            "recovery_postflight_receipt": receipt, "context_evidence": evidence}


def manifest_member_key(root, value):
    """Lexically identify package members, without opening/following manifest paths."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("invalid manifest member path")
    member = Path(value)
    if ".." in member.parts:
        raise ValueError("manifest traversal is forbidden")
    normalized = Path(os.path.normpath(str(member if member.is_absolute() else root / member)))
    if normalized == root or root not in normalized.parents:
        raise ValueError("manifest member must remain in its evaluation package")
    # Conservatively reject aliases/case collisions on the task's macOS host.
    return str(normalized).casefold()


def verified_evaluation_artifacts(directory, method_inputs):
    """Wrap frozen v2 reads; verify only consumed JSON bytes against local MANIFEST.

    No filename discovered in a manifest is ever opened. The manifest itself is
    the inventory, not a self-hashed member. Checks establish internal integrity,
    not independent provenance or the correctness of the frozen metric formulas.
    """
    package = v2.evaluate_artifacts(directory, method_inputs)
    root = Path(directory).resolve()
    manifest_path = root / "MANIFEST.json"
    consumed = [r for r in package["records"] if Path(r["path"]).suffix.lower() == ".json" and Path(r["path"]).name != "MANIFEST.json"]
    manifest_records = [r for r in package["records"] if Path(r["path"]) == manifest_path]
    problems, checks = [], []
    inventory = {}
    try:
        if len(manifest_records) != 1:
            raise ValueError("exactly one consumed MANIFEST record required")
        payload = manifest_path.read_bytes()
        if len(payload) != manifest_records[0]["bytes"] or hashlib.sha256(payload).hexdigest() != manifest_records[0]["sha256"]:
            raise ValueError("MANIFEST changed between package read and verification")
        manifest = json.loads(payload.decode("utf-8-sig"), parse_constant=v1.reject_nonfinite_json)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
            raise ValueError("MANIFEST files must be an array")
        for index, entry in enumerate(manifest["files"]):
            if not isinstance(entry, dict):
                problems.append({"member_index": index, "problem": "INVALID_MEMBER_OBJECT"})
                continue
            try:
                key = manifest_member_key(root, entry.get("path"))
            except ValueError as exc:
                problems.append({"member_index": index, "problem": "INVALID_MEMBER_PATH", "detail": str(exc)})
                continue
            inventory.setdefault(key, []).append(entry)
            if (type(entry.get("bytes")) is not int or entry["bytes"] < 0 or
                not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
                problems.append({"member_index": index, "problem": "INVALID_MEMBER_DIGEST_OR_SIZE"})
        for key, entries in inventory.items():
            if len(entries) != 1:
                problems.append({"path": key, "problem": "DUPLICATE_MANIFEST_MEMBER", "count": len(entries)})
    except (ValueError, OSError, UnicodeError) as exc:
        problems.append({"path": str(manifest_path), "problem": "MANIFEST_UNAVAILABLE_OR_INVALID", "detail": str(exc)})
    for actual in consumed:
        key = manifest_member_key(root, actual["path"])
        entries = inventory.get(key, [])
        unique = len(entries) == 1
        size_matches = unique and type(entries[0].get("bytes")) is int and entries[0]["bytes"] == actual["bytes"]
        sha_matches = unique and entries[0].get("sha256") == actual["sha256"]
        check = {"path": actual["path"], "member_count": len(entries), "bytes_match": size_matches,
                 "sha256_match": sha_matches, "pass": unique and size_matches and sha_matches}
        checks.append(check)
        if not check["pass"]:
            problems.append({"path": actual["path"], "problem": "CONSUMED_JSON_MANIFEST_MISMATCH"})
    verified = bool(consumed) and not problems and all(c["pass"] for c in checks)
    package["manifest_integrity"] = {"state": "VERIFIED_CONSUMED_JSON" if verified else "NOT_EVALUABLE",
        "pass": verified, "consumed_json_count": len(consumed), "checks": checks, "problems": problems,
        "manifest_listed_paths_opened": 0,
        "scope": "same-package unique-member bytes and SHA256 coverage of actually consumed JSON only; MANIFEST excluded from self-coverage"}
    if not verified:
        package["state"] = "NOT_EVALUABLE"
        package["problems"] += problems
    return package


def record_or_none(reader, name):
    reader.read(name, kind="binding_only")
    return reader.records.get(name)


def source_training(root):
    r = v1.Inputs(root)
    status, config = r.read("status.json") or {}, r.read("config.json") or {}
    vae, ddpm = r.read("vae_training_summary.json") or {}, r.read("ddpm_training_summary.json") or {}
    metas = {name: r.read(name + ".json") or {} for name in ("vae_state", "ddpm_ema_state")}
    initialization = r.read("vae_initialization.json") or {}
    record_or_none(r, "train_feature_scaler.json")
    logs = {"vae": r.read("vae_train.jsonl", kind="jsonl"), "ddpm": r.read("ddpm_train.jsonl", kind="jsonl"),
            "development": r.read("dev_monitor.jsonl", kind="jsonl")}
    r.read("generation_progress.jsonl", kind="jsonl", required=False)
    checks = {}
    for name, summary, meta in (("vae", vae, metas["vae_state"]), ("ddpm", ddpm, metas["ddpm_ema_state"])):
        requested = config.get(name + "_updates")
        checks[name + "_fixed1000_recorded"] = (requested == 1000 and summary.get("updates") == 1000 and
            summary.get("fixed_endpoint_reached") is True and meta.get("updates") == meta.get("requested_updates") == 1000
            and meta.get("fixed_endpoint_reached") is True)
        checks[name + "_log1000_recorded"] = bool(logs[name]) and [row.get("step") for row in logs[name]] == list(range(1, 1001))
        field = "loss" if name == "vae" else "mse"
        checks[name + "_finite_logged_losses"] = bool(logs[name]) and all(v1.finite(row.get(field)) for row in logs[name])
    checks["required_training_artifacts_present"] = not r.problems
    return {"root": str(Path(root).resolve()), "recorded_original_state": status.get("state", "MISSING"),
       "original_state_interpretation": "stale running after independently established interruption; never treated as completion",
       "training_record_state": "RECORDED_FINAL_CHECKPOINTS" if all(checks.values()) else "NOT_EVALUABLE",
       "checks": checks, "config": config, "checkpoint_metadata": metas, "initialization": initialization,
       "curves": {"vae_training": v1.log_endpoints(logs["vae"], ("loss", "reconstruction", "kl")),
                  "ddpm_training": v1.log_endpoints(logs["ddpm"], ("mse",)),
                  "development_monitor": v1.log_endpoints(logs["development"], ("nonpad_token_accuracy", "length_mae", "mean_reconstruction_objective"))},
       "recorded_training_stage_seconds": {"vae": vae.get("elapsed_seconds"), "ddpm": ddpm.get("elapsed_seconds")},
       "records": list(r.records.values()), "problems": r.problems}


def recovery_phase(root, training):
    r = v1.Inputs(root)
    status, config = r.read("status.json") or {}, r.read("config.json") or {}
    summary = r.read("generation_summary.json") or {}
    provenance = r.read("phase_provenance.json")
    contract = r.read("recovery_contract.json") or {}
    identity = r.read("model_identity_after.json") or {}
    loaded = r.read("loaded_state_identity.json") or {}
    integrity = r.read("input_integrity_after.json") or {}
    scaler = r.read("restored_scaler_identity.json") or {}
    manifest = r.read("manifest.json")
    rows = r.read("generated.csv", kind="csv")
    sample_inventory = v1.sample_inventory(rows, CE, 16)
    for file in ("train_reconstruction.csv", "dev_reconstruction.csv"):
        record_or_none(r, file)
    prohibited = [p for p in ("vae_train.jsonl", "ddpm_train.jsonl", "dev_monitor.jsonl", "vae_training_summary.json",
                    "ddpm_training_summary.json", "vae_state.pt", "ddpm_ema_state.pt", "train_feature_scaler.json") if (Path(root) / p).exists()]
    expected_digests = {name: meta.get("tensor_state_sha256") for name, meta in training["checkpoint_metadata"].items()}
    scaler_records = [r for r in training["records"] if Path(r["path"]).name == "train_feature_scaler.json"]
    stage_seconds, outer_seconds = summary.get("elapsed_seconds"), status.get("process_seconds_before_final_status_io")
    checks = {
       "recovery_terminal_completed": status.get("state") == "recovery_completed",
       "correct_attempt_identity": status.get("attempt_id") == "ce_checkpoint_recovery_v1" and status.get("scope_id") == "AMP_H5_CE_CHECKPOINT_RECOVERY_V1",
       "no_training_updates": status.get("training_updates_this_attempt") == 0,
       "original_engineering_not_relabelled": status.get("original_v2_engineering_gate") == "NOT_EVALUABLE_INTERRUPTION",
       "same_original_config": bool(config) and config == training["config"],
       "same_final_tensor_digests": bool(expected_digests) and all(expected_digests.values()) and identity.get("before") == identity.get("after") == identity.get("expected") == expected_digests,
       "loaded_state_record_matches": loaded.get("before") == loaded.get("expected") == expected_digests,
       "saved_model_unchanged": status.get("model_states_unchanged") is True and identity.get("both_loaded_verified_and_unchanged") is True,
       "original_inputs_preserved": status.get("original_inputs_unchanged") is True and integrity.get("original_errors") == [] and integrity.get("core_errors") == [],
       "no_training_artifact_copies": not prohibited,
       "scaler_not_refit": scaler.get("fit_called") is False,
       "scaler_source_bound": len(scaler_records) == 1 and scaler.get("source") == scaler_records[0]["path"] and scaler.get("sha256") == scaler_records[0]["sha256"],
       "all32_slots_and_complete": sample_inventory["all_slots_present"] and sample_inventory["all_executed"] and summary.get("successful_execution") == summary.get("requested") == 32,
       "generation_within1800": v1.finite(stage_seconds) and stage_seconds <= 1800 and status.get("generation_cap_exceeded") is False,
       "outer_within1980": v1.finite(outer_seconds) and outer_seconds <= 1980 and status.get("outer_cap_exceeded") is False,
       "test_read_false": status.get("test_read") is False,
       "manifest_present": isinstance(manifest, dict), "phase_provenance_present": isinstance(provenance, dict),
       "phase_training_source_matches": v1.dig(provenance, "training", "source_directory") == training["root"] and v1.dig(provenance, "training", "copied_or_linked") is False,
       "phase_generation_source_matches": v1.dig(provenance, "generation_and_reconstruction", "source_directory") == str(Path(root).resolve()) and v1.dig(provenance, "new_training_updates") == 0,
       "required_recovery_artifacts_present": not r.problems,
    }
    return {"root": str(Path(root).resolve()), "attempt_id": ATTEMPT, "recorded_state": status.get("state", "MISSING"),
       "recorded_completion": all(checks.values()), "checks": checks, "samples": sample_inventory,
       "status_error": status.get("error"), "status": status, "phase_provenance": provenance,
       "recovery_contract": contract, "checkpoint_identity": identity,
       "costs": {"setup_seconds": status.get("setup_seconds"), "generation_seconds": stage_seconds,
                 "reconstruction_export_seconds": status.get("export_seconds"), "process_seconds_before_final_status_io": outer_seconds,
                 "generation_cap_seconds": 1800, "outer_cap_seconds": 1980, "new_training_updates": 0},
       "prohibited_copied_artifacts": prohibited, "records": list(r.records.values()), "problems": r.problems}


def postflight(binding, training, recovery):
    result = {"state": "NOT_EVALUABLE_NO_BOUND_POSTFLIGHT", "pass": False, "record": None, "receipt": None}
    if binding is None:
        return result
    path = Path(binding["path"])
    if not path.is_file():
        return result
    result["record"] = v1.record(path)
    if result["record"]["sha256"] != binding["sha256"]:
        result["state"] = "NOT_EVALUABLE_RECEIPT_HASH_MISMATCH"
        return result
    try:
        receipt = json.loads(path.read_text(), parse_constant=v1.reject_nonfinite_json)
    except (ValueError, OSError, UnicodeError):
        result["state"] = "NOT_EVALUABLE_RECEIPT_UNREADABLE"
        return result
    result["receipt"] = receipt
    checks = {
       "postflight_scope_only": receipt.get("scope") == "RECOVERY_ENGINEERING_ONLY",
       "pass_recorded": receipt.get("status") == "PASS",
       "attempt_id_matches": receipt.get("generation_attempt_id") == ATTEMPT,
       "generation_root_matches": receipt.get("generation_attempt_root") == recovery["root"],
       "training_root_matches": receipt.get("training_source_root") == training["root"],
       "original_E_not_relabelled": receipt.get("original_v2_engineering_gate") == "NOT_EVALUABLE_INTERRUPTION",
       "original_protocol_not_relabelled": receipt.get("original_protocol_E_L_G") == "NOT_EVALUABLE",
       "promote_hold": receipt.get("PROMOTE") == "HOLD",
    }
    bindings = v2.normalize_records(receipt.get("source_bindings", []))
    required = [r for r in recovery["records"] if Path(r["path"]).name in ("status.json", "generated.csv")]
    required += [{"path": str(Path(training["root"]) / (name + ".pt")), "sha256": meta.get("sha256")}
                 for name, meta in training["checkpoint_metadata"].items()]
    checks["critical_source_bindings_match"] = len(required) == 4 and all(
        len([r for r in bindings if r["path"] == item["path"] and r["sha256"] == item["sha256"]]) == 1 for item in required)
    result.update(checks=checks, **{"pass": all(checks.values())})
    result["state"] = "RECOVERY_ENGINEERING_PASS" if result["pass"] else "NOT_EVALUABLE_POSTFLIGHT_HOLD_OR_SCOPE"
    return result


def build(contract):
    mapping = contract["methods"]
    inputs = v2.MappedInputs({m: d["generation_attempt_root"] for m, d in mapping.items()})
    unified = [v1.method_facts(inputs, m, m) for m in METHODS if m != CE]
    training = source_training(mapping[CE]["training_source_root"])
    recovery = recovery_phase(mapping[CE]["generation_attempt_root"], training)
    review = postflight(contract["recovery_postflight_receipt"], training, recovery)
    # The frozen evaluation reader needs only the new generated/reconstruction
    # binding records; it must not redirect these to the original training root.
    for r in recovery["records"]:
        name = Path(r["path"]).name
        if name in ("generated.csv", "train_reconstruction.csv", "dev_reconstruction.csv"):
            inputs.records[CE + "/" + name] = r
    packages = [verified_evaluation_artifacts(p, inputs) for p in contract["evaluation_directories"]]
    readiness = {m["method"]: m["engineering_record_state"] == "RECORDED_PASS_PENDING_INDEPENDENT_REVIEW" for m in unified}
    readiness[CE] = training["training_record_state"] == "RECORDED_FINAL_CHECKPOINTS" and recovery["recorded_completion"] and review["pass"]
    gen, rec, seen_gen, seen_rec = [], [], set(), set()
    for package in packages:
        for kind, seen, target, second in (("generation", seen_gen, gen, "terminal"), ("reconstruction", seen_rec, rec, "split")):
            for group in package[kind]:
                key = (group["method"], group[second])
                if key in seen:
                    raise ValueError("duplicate observation across evaluation packages: " + repr(key))
                seen.add(key)
                bound = package["method_bindings"].get(group["method"], {}).get("pass", False)
                target.append({"method": group["method"], second: group[second], "metrics": group,
                    "source_evaluation": package["directory"], "source_binding_pass": bound,
                    "evaluation_manifest_state": package["manifest_integrity"]["state"],
                    "interpretation_state": "AVAILABLE_RECOVERY_DESCRIPTIVE" if bound and readiness[group["method"]] and package["state"] == "RECORDED_COMPLETE" else "NOT_EVALUABLE"})
    screens = {}
    for label, key, operands in (("L", "reconstruction_ce_vs_matched_mse", (CE, "MSE21_DDPM")),
                                ("G", "generation_ce_vs_terminal_ar", (CE, "terminal_LSTM"))):
        candidates = []
        for p in packages:
            observed = ({(g["method"], g["split"]) for g in p["reconstruction"]} if label == "L"
                        else {(g["method"], g["terminal"]) for g in p["generation"]})
            required = {(m, "development") for m in operands} if label == "L" else {(m, t) for m in operands for t in v2.TERMINALS}
            if required <= observed and isinstance(p.get("raw_gates"), dict):
                bound = all(p["method_bindings"].get(m, {}).get("pass", False) for m in operands)
                raw = p["raw_gates"].get(key)
                raw_state = v1.gate_status(raw)
                eligible = bound and all(readiness[m] for m in operands) and p["state"] == "RECORDED_COMPLETE"
                candidates.append({"source_evaluation": p["directory"], "raw_state": raw_state, "raw_gate": raw,
                                   "evaluation_manifest_state": p["manifest_integrity"]["state"],
                                   "raw_gate_authority": "TRANSCRIBED_ONLY; interpretation requires verified manifest and ready operands",
                                   "operands_bound_and_ready": eligible,
                                   "state": raw_state if eligible else "NOT_EVALUABLE"})
        screens[label] = {"state": candidates[0]["state"] if len(candidates) == 1 else "NOT_EVALUABLE",
                          "records": candidates, "comparison_scope": "amended checkpoint-recovery descriptive screen only"}
    records = [contract["source"]] + list(inputs.records.values()) + training["records"] + recovery["records"] + [r for p in packages for r in p["records"]]
    if review["record"]:
        records.append(review["record"])
    evidence = []
    for item in contract["context_evidence"]:
        actual = v1.record(item["path"])
        if actual["sha256"] != item["sha256"]:
            raise ValueError("context evidence hash mismatch")
        records.append(actual); evidence.append(actual)
    # Unique file records, no follow-through into config dataset paths or .pt.
    records = list({r["path"]: r for r in records}.values())
    stable = all(Path(r["path"]).is_file() and v1.record(r["path"])["sha256"] == r["sha256"] for r in records)
    if not stable:
        for screen in screens.values():
            screen["state"] = "NOT_EVALUABLE"
        for observation in gen + rec:
            observation["interpretation_state"] = "NOT_EVALUABLE"
    available = {m: {"generation_groups": [g["terminal"] for g in gen if g["method"] == m],
                      "reconstruction_splits": [g["split"] for g in rec if g["method"] == m],
                      "phase_records_ready": readiness[m]} for m in METHODS}
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "schema_version": "AMP_H5_PHASE_SPLIT_READOUT_V4",
       "source": v1.record(__file__), "dependencies": [v1.record(V2), v1.record(v2.V1)], "contract": contract,
       "original_protocol": {"E": "NOT_EVALUABLE", "L": "NOT_EVALUABLE", "G": "NOT_EVALUABLE",
          "reason": "original CE v2 interruption; recovery never backfills original completion",
          "CE_original_outcome": "NOT_EVALUABLE_INTERRUPTION", "exit_cause": "UNKNOWN", "exit_code": None,
          "last_unrecorded_work": "UNKNOWN", "original_total_wall_seconds": None},
       "CE_training_source": training, "CE_recovery_attempt": recovery, "recovery_engineering_postflight": review,
       "recovery_engineering": {"state": "PASS" if readiness[CE] and stable else "NOT_EVALUABLE",
          "scope": "RECOVERY_ENGINEERING_ONLY", "postflight_state": review["state"],
          "recorded_phases_complete": readiness[CE], "input_snapshot_stable": stable},
       "other_unified_runs": unified, "method_availability": available,
       "generation_observations": gen, "reconstruction_observations": rec,
       "synthetic_missing_observations": 0, "recovery_descriptive_screens": screens, "evaluation_packages": packages,
       "evaluation_manifest_integrity": {"state": "VERIFIED_CONSUMED_JSON" if packages and stable and all(p["manifest_integrity"]["pass"] for p in packages) else "NOT_EVALUABLE",
          "packages": [{"directory": p["directory"], **p["manifest_integrity"]} for p in packages]},
       "phase_costs": {"original_CE_amendment_record": ORIGINAL_COSTS, "new_CE_recovery": recovery["costs"],
                      "other_methods": {m["method"]: m["costs"] for m in unified},
                      "original_request_slots": 32, "new_recovery_request_slots": 32,
                      "denominator_rule": "separate attempts; never 64 completed sequences or independent draws",
                      "unknown_total_not_summed": True,
                      "additional_research_costs": "smoke, maturity diagnostic, zero-update entry failure, review and staged execution remain separate"},
       "AR_ADEQUACY": "NOT_ESTABLISHED", "PROMOTE": "HOLD", "input_snapshot_stable": stable,
       "input_snapshot_stable_scope": "temporal stability during this read only; not MANIFEST authenticity/integrity, which is reported separately",
       "context_evidence": evidence, "consumed_artifacts": records,
       "operations": {"model_calls": 0, "metric_recomputations": 0, "split_files_opened": 0,
                      "checkpoint_files_loaded_or_opened": 0, "posterior_arrays_opened": 0, "test_files_opened": 0},
       "limits": ["original E/L/G are permanently NOT_EVALUABLE; amended attempt is separately identified and costed",
                  "original FREE generation cost is at least 451.935743 seconds; termination time and total original wall time are unknown",
                  "single fitted seed, reused development41, no significance or independent generalization evidence",
                  "fixed original final checkpoint and seeds; curves do not select an endpoint or establish AR adequacy",
                  "readout transcribes metrics and engineering records; only the bound postflight receipt supplies independent recovery engineering approval",
                  "unverified evaluation MANIFEST disables interpretation and descriptive screens while retaining raw values; snapshot stability alone is insufficient",
                  "no antimicrobial activity, safety, novelty, strong-baseline superiority or submission-readiness claim; old wetlab is not H5 validation",
                  "BACKUP_BLOCKED; local source hashes do not constitute remote backup"]}


def render(facts):
    fmt, dig = v1.fmt, v1.dig
    lines = ["# AMP H5：固定checkpoint恢复的分阶段读出 v4", "",
       "**原 formal protocol E/L/G：NOT_EVALUABLE / NOT_EVALUABLE / NOT_EVALUABLE，永久保留，不由恢复结果回填。**",
       f"恢复工程：**{facts['recovery_engineering']['state']}**（后审回执：{facts['recovery_engineering_postflight']['state']}）。恢复描述性 L/G：**{facts['recovery_descriptive_screens']['L']['state']} / {facts['recovery_descriptive_screens']['G']['state']}**。",
       "**AR_ADEQUACY = NOT_ESTABLISHED；PROMOTE = HOLD。**", "",
       f"评价包已消费JSON清单完整性：**{facts['evaluation_manifest_integrity']['state']}**。快照在读取期间未变不代表清单有效；失配包的原始值仅保留供核查，解释与描述性门均不可判。", "",
       "原训练、恢复生成/重建按不同phase列源；复制训练日志或新增训练更新会使恢复记录检查失败。缺失方法不创建观测行。", "",
       "| 方法 | training来源 | generation/reconstruction来源 | generation attempt |",
       "| --- | --- | --- | --- |"]
    for method, source in facts["contract"]["methods"].items():
        lines.append("| " + " | ".join(fmt(x) for x in (method, source["training_source_root"], source["generation_attempt_root"], source["generation_attempt_id"])) + " |")
    lines += ["", "| phase | 记录秒数 | 范围 |",
       "| --- | ---: | --- |", "| 原CE VAE训练 | 1179.031922 | 原阶段记录 |",
       "| 原CE DDPM训练 | 564.346170 | 原阶段记录 |",
       "| 原CE中断生成 | ≥451.935743（约≥451.936） | 下界；终止时刻、未记录工作量未知 |",
       "| 原CE总wall | UNKNOWN | 不用阶段相加代替未知总wall |"]
    for key, label in (("setup_seconds", "新恢复加载/setup"), ("generation_seconds", "新恢复生成"),
                       ("reconstruction_export_seconds", "新恢复重建导出"), ("process_seconds_before_final_status_io", "新恢复进程至final status前")):
        lines.append(f"| {label} | {fmt(facts['CE_recovery_attempt']['costs'].get(key))} | 新attempt记录，阶段可能含I/O差异 |")
    lines += ["", "原32槽位与新32槽位分别保留；不能写成64个已完成序列。恢复是额外attempt及no-retry条款的前瞻修订，不是t200无缝续跑。", "",
       "| 已有方法 | terminal | 请求 | 合法1–48 | 合法5–48 | unique合法 | copy同端基/任意train | train MMD² | dev MMD² | 解释状态 |",
       "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for row in facts["generation_observations"]:
        g = row["metrics"]
        lines.append("| " + " | ".join(fmt(x) for x in (row["method"], row["terminal"], g.get("requests"), g.get("valid_1_48_count"), g.get("valid_5_48_count"),
          dig(g, "valid_1_48", "unique_count"), fmt(g.get("exact_copy_train_count_all_requests"))+"/"+fmt(g.get("exact_sequence_copy_any_train_terminal_count")),
          dig(g, "valid_1_48_vs_train", "mmd2_linear_biased_fixed_features"), dig(g, "valid_1_48_vs_development", "mmd2_linear_biased_fixed_features"), row["interpretation_state"])) + " |")
    if not facts["generation_observations"]:
        lines += ["", "暂无既有生成评价记录：NOT_EVALUABLE。"]
    lines += ["", "| 已有VAE dev mean | 对象数 | micro | AA支持类macro | sequence mean | 长度MAE | 整串exact | 解释状态 |",
       "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in facts["reconstruction_observations"]:
        if row["split"] != "development":
            continue
        g = row["metrics"]
        lines.append("| " + " | ".join(fmt(x) for x in (row["method"], g.get("rows"), g.get("rendered_nonpad_micro"), g.get("rendered_nonpad_macro"),
          g.get("per_sequence_mean_nonpad_accuracy"), g.get("length_mae"), g.get("whole_rendered_exact_count"), row["interpretation_state"])) + " |")
    lines += ["", "MMD²/JS/L1/长度KS/entropy/最近邻等完整指标及raw恢复门见 [EXECUTION_FACTS.json](EXECUTION_FACTS.json)；读出源与产物哈希见 [READOUT_MANIFEST.json](READOUT_MANIFEST.json)。",
       "", "缺失状态：", ""]
    for method, a in facts["method_availability"].items():
        lines.append(f"- {method}：生成组 {','.join(a['generation_groups']) or '无'}；重建split {','.join(a['reconstruction_splits']) or '无'}；phase记录可用 {a['phase_records_ready']}。缺失观测不当作零。")
    lines += ["", "独立后审通过仅指恢复工程范围；原协议不重命名为通过。训练/监测曲线源只在原training root，AR teacher-forced NLL与全部phase日志摘要保留JSON，不与VAE目标横比。", ""]
    lines += ["- " + item for item in facts["limits"]]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    contract = load_contract(args.contract)
    output = Path(args.output).resolve()
    roots = [Path(d[k]) for d in contract["methods"].values() for k in ("training_source_root", "generation_attempt_root")]
    roots += [Path(path) for path in contract["evaluation_directories"]]
    if output.exists() or any(output == r or r in output.parents or output in r.parents for r in roots):
        raise ValueError("fresh output directory separate from all input trees required")
    facts = build(contract)
    output.mkdir(parents=True)
    (output / "AUTHOR_READOUT.zh.md").write_text(render(facts))
    (output / "EXECUTION_FACTS.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    manifest = {"source": v1.record(__file__), "contract": contract["source"], "dependencies": facts["dependencies"],
       "outputs": [v1.record(output / name) for name in ("AUTHOR_READOUT.zh.md", "EXECUTION_FACTS.json")],
       "inputs": facts["consumed_artifacts"], "original_protocol_E_L_G": "NOT_EVALUABLE", "PROMOTE": "HOLD",
       "evaluation_manifest_integrity": facts["evaluation_manifest_integrity"],
       "input_snapshot_stable": facts["input_snapshot_stable"]}
    (output / "READOUT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "original_protocol_E_L_G": "NOT_EVALUABLE",
       "recovery_engineering": facts["recovery_engineering"]["state"], "PROMOTE": "HOLD"}))


if __name__ == "__main__":
    main()
