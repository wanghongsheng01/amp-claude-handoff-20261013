#!/usr/bin/env python3
"""Conditional AMP H5 VAE-only wrapper. Import is standard-library-only.

No invocation is authorized by this file. A separate, hash-bound root receipt
and independent preaudit are required before runtime imports or natural reads.
The frozen train_vae loop, including its fixed monitor100, is reused verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def beta_for_step(step):
    """One-based update index; no extrapolation or schedule extension."""
    if type(step) is not int or not 1 <= step <= 1000:
        raise ValueError("step must be an integer in [1,1000]")
    return min(1.0, (step - 1) / 200.0)


def composed_loss(reconstruction, normalized_kl, step):
    # This is the only optimization intervention; normalization stays upstream.
    return reconstruction + beta_for_step(step) * normalized_kl


def validate_receipt(receipt, manifest_sha):
    require(receipt.get("schema") == "AMP_H5_WARMUP_ROOT_LAUNCH_V1", "root receipt schema")
    require(receipt.get("status") == "ROOT_AUTHORIZED", "root has not authorized launch")
    require(receipt.get("candidate_manifest_sha256") == manifest_sha, "receipt candidate binding")
    require(receipt.get("scope") == "VAE_ONLY_1000", "receipt scope")
    for key in ("independent_preaudit_pass", "baseline_full_rms_complete",
                "root_fresh_process_load_check_pass", "single_serial_cpu_launch",
                "no_other_active_amp_model_job", "baseline_core_predictions_finite"):
        require(receipt.get(key) is True, "receipt gate: " + key)
    require(receipt.get("train_rows") == 279 and receipt.get("dev_rows") == 41
            and receipt.get("K") == 8, "full RMS denominators")
    m = receipt.get("dev_M_equal_sequence_accuracy")
    s = receipt.get("dev_S_equal_sequence_eight_draw_mean_accuracy")
    for value in (m, s):
        require(type(value) in (int, float) and math.isfinite(value)
                and 0 <= value <= 1, "finite exact unrounded dev accuracy required")
    require(m - s >= 0.05, "dev entry gap below 0.05")
    for key in ("independent_preaudit", "baseline_rms_delivery", "baseline_epsilon_bank_manifest",
                "root_process_load_check"):
        ref = receipt.get(key, {})
        require(isinstance(ref.get("path"), str) and isinstance(ref.get("sha256"), str),
                "missing receipt binding: " + key)
        p = Path(ref["path"]).resolve()
        require("amp_h5" in p.parts and not any("test" in x.lower() for x in p.parts),
                "receipt evidence must stay within AMP H5 and exclude test")
        require(sha(p) == ref["sha256"], "receipt evidence hash: " + key)


def verify_preflight(contract_path, receipt_path, output):
    """Read metadata/code only. Natural input validation occurs after admission."""
    require(Path(contract_path).resolve() == HERE / "CONTRACT.json", "fixed contract path")
    require(Path(receipt_path).resolve() == HERE / "ROOT_LAUNCH_AUTHORIZATION.json", "fixed receipt path")
    require(Path(output).resolve() == HERE / "run_v1", "fixed output path")
    require(not Path(output).exists(), "single attempt: output already exists")
    manifest_path = HERE / "MANIFEST.json"
    manifest = read_json(manifest_path)
    for member in manifest["files"]:
        p = HERE / member["path"]
        require(p.parent == HERE and sha(p) == member["sha256"], "candidate member drift")
    contract = read_json(contract_path)
    receipt = read_json(receipt_path)
    validate_receipt(receipt, sha(manifest_path))
    for name, ref in contract["frozen_metadata_and_weights"].items():
        require(sha(ref["path"]) == ref["sha256"], "frozen binding drift: " + name)
    config = read_json(contract["frozen_metadata_and_weights"]["baseline_config"]["path"])
    require(config == contract["baseline_config"], "baseline config equality")
    require(config["arm"] == "categorical_ce" and config["vae_updates"] == 1000, "CE1000 only")
    require([config[k] for k in ("initialization_seed", "shuffle_seed", "vae_noise_seed")]
            == [222, 222, 222], "seed lock")
    require(config["vae_batch_size"] == 32 and config["monitor_interval"] == 100, "batch/monitor lock")
    require(str(Path(sys.executable).resolve()) == contract["python_resolved"], "Python executable identity")
    require(sys.version == contract["baseline_runtime"]["python"], "Python version equality")
    return contract, config, receipt


def runtime_code_only(module, contract):
    """Same runtime imports/threads, without frozen bootstrap's broad data hash loop.

    The original init_runtime iterates a manifest containing unrelated CSVs,
    including a test_dataset.csv. Only frozen source code/config/pyc is read here.
    Bytecode is bound too because disabling writes does not disable pyc reads.
    Excluded manifest members are explicitly NOT verified or accessed.
    """
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.dont_write_bytecode = True
    for entry in contract["source_code_only"]:
        p = Path(entry["path"])
        require(p.suffix in (".py", ".pyc", ".yaml", ".yml") and not any("test" in x.lower() for x in p.parts),
                "code-only bootstrap allowlist")
        require(sha(p) == entry["sha256"], "frozen source code drift: " + str(p))
    sys.path.insert(0, str(Path(contract["baseline_config"]["source_root"]) / "corrected_code" / "AMP_VAEflow"))
    import torch
    import numpy as np
    from modules.vae import VAEModel
    from tasks.dataset_loader.dataset_utils.amp_input_contract import validate_file
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        require(torch.get_num_interop_threads() == 1, "interop threads")
    module.torch, module.np = torch, np
    module.VAEModel, module.validate_file = VAEModel, validate_file
    baseline = contract["baseline_runtime"]
    require(torch.__version__ == baseline["torch"] and np.__version__ == baseline["numpy"],
            "torch/numpy version equality")
    return {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "device": "cpu", "code_only_bootstrap": True,
            "excluded_manifest_members_not_accessed": contract["source_manifest_excluded_count"]}


def install_pairing_hooks(module, contract, out):
    """No extra forward/RNG calls. The inherited loop itself remains unchanged."""
    expected = [json.loads(line) for line in Path(
        contract["frozen_metadata_and_weights"]["baseline_train_trace"]["path"]).read_text().splitlines()]
    require(len(expected) == 1000 and [x["step"] for x in expected] == list(range(1, 1001)),
            "baseline trace endpoint")
    original = {name: getattr(module, name) for name in ("objective", "make_model", "cycling_indices", "log")}
    state = {"forward_calls": 0, "updates_logged": 0, "monitors_logged": 0, "model": None}

    def paired_model(arm, seed):
        model = original["make_model"](arm, seed)
        state["model"] = model
        require(module.state_hash(model) == contract["initial_tensor_state_sha256"], "initial weights not paired")
        return model

    def paired_indices(n, batch_size, seed):
        require((n, batch_size, seed) == (279, 32, 222), "batch stream inputs")
        stream = original["cycling_indices"](n, batch_size, seed)
        for row in expected:
            idx = next(stream)
            require(idx == row["record_indices"] and len(idx) == row["rows"], "batch order not paired")
            yield idx
        raise RuntimeError("more than 1000 batches requested")

    def warm_objective(model, target, g, h, arm):
        step = state["forward_calls"] + 1
        require(step <= 1000 and arm == "categorical_ce", "objective call scope")
        row = expected[step - 1]
        rng = module.torch.get_rng_state()
        rng_sha = module.batch_hash(rng)
        require(rng_sha == row["forward_rng_sha256"], "preforward RNG mismatch at step " + str(step))
        require(len(target) == row["rows"], "objective row count mismatch")
        pending = {"step": step, "zero_based_update": step - 1, "beta": beta_for_step(step),
                   "record_indices": row["record_indices"], "rows": len(target),
                   "target_g_h_sha256": module.batch_hash(target, g, h),
                   "forward_rng_sha256": rng_sha,
                   "forward_rng_state_hex": rng.numpy().tobytes().hex(),
                   "expected_lr": row["lr"], "initialization_batch_rng_verified": True}
        # Persist the attempted slot before any forward; retain failures.
        with (out / "paired_forward_attempts.jsonl").open("a") as fh:
            fh.write(json.dumps(pending, allow_nan=False) + "\n")
        state["forward_calls"] = step
        result = original["objective"](model, target, g, h, arm)
        if step == 1:
            for name in ("reconstruction", "kl", "density_sum"):
                require(float(result[name].detach()) == row[name], "first-forward parity: " + name)
        result["loss"] = composed_loss(result["reconstruction"], result["kl"], step)
        state["pending"] = {"step": step, "beta": beta_for_step(step),
                            "weighted_kl": float((beta_for_step(step) * result["kl"]).detach()),
                            "unweighted_kl": float(result["kl"].detach()),
                            "postforward_rng_sha256": module.batch_hash(module.torch.get_rng_state())}
        return result

    def paired_log(path, row):
        if Path(path).name == "vae_train.jsonl":
            step = row["step"]
            exp = expected[step - 1]
            for key in ("step", "rows", "record_indices", "lr", "forward_rng_sha256"):
                require(row[key] == exp[key], "paired train trace mismatch: " + key)
            require(step == state["updates_logged"] + 1, "logged update continuity")
            original["log"](path, row)  # Preserve original schema and loss components.
            original["log"](out / "beta_loss_split.jsonl", {
                **state["pending"], "loss": row["loss"], "reconstruction": row["reconstruction"],
                "density_sum": row["density_sum"], "normalization_denominator": row["rows"] * 48,
                "lr": row["lr"], "paired_trace_fields_equal": True})
            state["updates_logged"] = step
        elif Path(path).name == "dev_monitor.jsonl":
            require(row["step"] == 100 * (state["monitors_logged"] + 1)
                    and row["rng_preserved"] is True and row["checkpoint_selection"] is False,
                    "fixed monitor scope")
            original["log"](path, row)
            state["monitors_logged"] += 1
        else:
            original["log"](path, row)

    module.make_model, module.cycling_indices = paired_model, paired_indices
    module.objective, module.log = warm_objective, paired_log
    return state


def run(contract_path, receipt_path, output):
    started = time.perf_counter()
    contract, config, receipt = verify_preflight(contract_path, receipt_path, output)
    out = Path(output).resolve()
    out.mkdir(exist_ok=False)  # An attempted run can never silently resume/retry.
    status = {"state": "running", "scope": "VAE_ONLY_1000",
              "ddpm_updates": 0, "generation_requests": 0, "posttraining_q_or_rms_calls": 0,
              "test_reads": 0, "root_receipt_sha256": sha(receipt_path)}
    dump(out / "status.json", status)
    module, state = None, None
    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline(signum, frame):
        raise TimeoutError("outer2100s deadline; preserve partial, no extension")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, 2100 - (time.perf_counter() - started)))
    try:
        spec = importlib.util.spec_from_file_location("amp_h5_frozen_vae", contract["frozen_metadata_and_weights"]["runner"]["path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dump(out / "runtime_and_source.json", runtime_code_only(module, contract))
        # No original run()/init_runtime()/train_ddpm()/generate()/export() call.
        config = dict(config)
        config["training_seconds_cap"] = 2100
        config["vae_seconds_cap"] = 1920
        dump(out / "effective_vae_config.json", config)
        dump(out / "root_launch_receipt.json", receipt)
        train = module.read_records(config["train"], config["train_sha256"], 279,
            config["train_cohort"], config["train_cohort_sha256"], 0)
        dev = module.read_records(config["dev"], config["dev_sha256"], 41,
            config["dev_cohort"], config["dev_cohort_sha256"], 279)
        require(not ({r.protein for r in train} & {r.protein for r in dev}), "train/dev overlap")
        scaler = module.FeatureScaler(train)
        require(scaler.stats == read_json(contract["frozen_metadata_and_weights"]["baseline_scaler"]["path"]),
                "scaler equality")
        module.dump(out / "train_feature_scaler.json", scaler.stats)
        require(sha(out / "train_feature_scaler.json") == contract["frozen_metadata_and_weights"]["baseline_scaler"]["sha256"],
                "scaler serialized hash equality")
        state = install_pairing_hooks(module, contract, out)
        _, summary = module.train_vae(config, train, scaler, out, started, dev)
        require(summary["fixed_endpoint_reached"] and summary["updates"] == 1000, "fixed endpoint incomplete")
        require(state["forward_calls"] == state["updates_logged"] == 1000
                and state["monitors_logged"] == 10, "call/update/monitor endpoint mismatch")
        require(summary["elapsed_seconds"] <= 1920 and time.perf_counter() - started <= 2100,
                "cap exceeded at endpoint")
        status.update(state="completed_vae_only_not_scientific_success", vae_updates=1000,
                      fixed_monitor_calls=10, coupling="all 1000 batch/RNG/LR trace fields equal",
                      checkpoint_selection=False)
    except BaseException as exc:
        status.update(state="NOT_EVALUABLE", error=repr(exc), traceback=traceback.format_exc())
        # No retry/extra forward. Preserve partial weights if an exception interrupted the inherited loop.
        if state is not None and state["model"] is not None and not (out / "vae_state.pt").exists():
            try:
                module.save_model(out, "vae_partial_failed_state", state["model"],
                                  {"completed_updates_logged": state["updates_logged"], "not_endpoint": True})
            except BaseException as save_error:
                status["partial_checkpoint_write_error"] = repr(save_error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        status["outer_elapsed_seconds"] = time.perf_counter() - started
        if state is not None:
            status.update(objective_forward_attempts=state["forward_calls"],
                          completed_updates_logged=state["updates_logged"], fixed_monitors_logged=state["monitors_logged"],
                          partial_forward_or_monitor_possible=status["state"] == "NOT_EVALUABLE")
        dump(out / "status.json", status)
        dump(out / "manifest.json", {"files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"]})
    return 0 if status["state"] == "completed_vae_only_not_scientific_success" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--root-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return run(args.contract, args.root_receipt, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
