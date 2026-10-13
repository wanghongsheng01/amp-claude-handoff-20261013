#!/usr/bin/env python3
"""Read closed H5 VAE artifacts only; never load model weights or source datasets.

This author report is not independent approval or a sequence-quality evaluation.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def json_file(path):
    return json.loads(path.read_text()) if path.exists() else None


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def moments(value):
    a = np.asarray(value)
    f = a[np.isfinite(a)].astype(np.float64)
    result = {"shape": list(a.shape), "dtype": str(a.dtype), "elements": int(a.size),
              "finite": int(f.size), "nan": int(np.isnan(a).sum()),
              "positive_inf": int(np.isposinf(a).sum()), "negative_inf": int(np.isneginf(a).sum()),
              "zeros": int((a == 0).sum()), "negative": int((a < 0).sum()),
              "all_finite": bool(np.isfinite(a).all())}
    result["finite_only"] = {"min": float(f.min()), "max": float(f.max()),
        "mean": float(f.mean()), "std_population": float(f.std()),
        "rms": float(np.sqrt(np.square(f).mean())),
        "quantiles_p01_p05_p50_p95_p99": np.quantile(f, [.01, .05, .5, .95, .99]).tolist()} if f.size else None
    return result


def parameter_summary(values):
    if values.ndim != 4 or values.shape[1:] != (2, 16, 48):
        raise ValueError("unexpected posterior coordinate layout")
    mu, logsigma = values[:, 0], values[:, 1]
    # Saved logsigma only: this is an explicitly labelled NumPy diagnostic,
    # not a claim of bitwise replay of the runtime torch.exp kernel.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        sigma = np.exp(logsigma)
    result = {"coordinate_system": "raw [mu,log_sigma], no standardization",
              "mu": moments(mu), "logsigma": moments(logsigma), "derived_sigma": moments(sigma),
              "derived_sigma_semantics": "numpy.exp(saved logsigma), original dtype; not torch bitwise replay"}
    if np.isfinite(mu).all() and np.isfinite(sigma).all():
        m = mu.astype(np.float64); s = sigma.astype(np.float64)
        second = float(np.mean(m*m+s*s)); mean = float(m.mean())
        result["analytic_uniform_q_mixture_marginal"] = {"mean": mean,
            "std_population": math.sqrt(max(0.0, second-mean*mean)), "rms": math.sqrt(second),
            "sampled": False}
    return result


def curves(records, fields, expected_updates):
    numeric = {key: [r[key] for r in records if key in r] for key in fields}
    compact = {}
    for key, values in numeric.items():
        if values:
            v = np.asarray(values, dtype=np.float64)
            compact[key] = {"first": float(v[0]), "last": float(v[-1]),
                "first20_mean": float(v[:20].mean()), "last20_mean": float(v[-20:].mean()),
                "all_finite": bool(np.isfinite(v).all())}
    steps = [r.get("step") for r in records]
    return {"logged_updates": len(records), "expected_updates": expected_updates,
            "contiguous_from_one": steps == list(range(1, len(records)+1)),
            "fixed_endpoint_logged": steps == list(range(1, expected_updates+1)),
            "record_exposures": sum(r.get("rows", 0) for r in records), "statistics": compact,
            "curve": [{k: r[k] for k in ["step", "rows", *fields] if k in r} for r in records]}


def compare_ranges(generated, reference):
    result = {}
    for key in ("mu", "logsigma", "derived_sigma"):
        gs, rs = generated[key], reference[key]
        g, r = gs["finite_only"], rs["finite_only"]
        result[key] = {"all_generated_finite": gs["all_finite"],
            "finite_range_inside_reference_extrema": bool(g and r and g["min"] >= r["min"] and g["max"] <= r["max"]),
            "rms_ratio": (g["rms"]/r["rms"] if g and r and r["rms"] else None),
            "mean_difference": (g["mean"]-r["mean"] if g and r else None)}
    return result


def readout(request_path, output):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    if sha(__file__) != request["readout_script_sha256"]:
        raise ValueError("readout script hash differs")
    run = Path(request["run_directory"]).resolve()
    supervisor_path = Path(request["supervisor_status_path"]).resolve()
    # Only explicitly named run artifacts are opened. Paths in original run config
    # are metadata and are never dereferenced for source train/dev/test data.
    supervisor = json_file(supervisor_path)
    supervisor_sha256 = sha(supervisor_path) if supervisor is not None else None
    if supervisor is None or supervisor.get("state") not in ("completed", "completed_with_failures", "failed"):
        raise ValueError("supervisor is not terminal; refusing a live readout")
    jobs = [j for j in supervisor.get("jobs", []) if j.get("id") == request["arm"]]
    if len(jobs) != 1 or jobs[0].get("state") not in ("completed", "failed", "timeout", "timed_out"):
        raise ValueError("selected supervisor job is not terminal")
    job = jobs[0]
    argv = job["argv"]
    if Path(argv[argv.index("--output")+1]).resolve() != run:
        raise ValueError("supervisor selected a different run directory")
    if supervisor.get("plan_sha256") != request["plan_sha256"]:
        raise ValueError("supervisor plan binding differs")
    config_path = run / "config.json"
    if sha(config_path) != request["config_sha256"]:
        raise ValueError("run config hash differs")
    config = json_file(config_path)
    if config["arm"] != request["arm"] or config["implementation_sha256"] != request["runner_sha256"]:
        raise ValueError("arm or implementation binding differs")
    status = json_file(run / "status.json")
    if status is None:
        raise ValueError("run status absent")
    blockers = []
    if status.get("state") == "running":
        blockers.append("runner status is stale running despite terminal supervisor; treat as interrupted, not completed")
    if status.get("state") != "completed" or job.get("exit_code") != 0:
        blockers.append("run did not complete successfully")
    # Hash verification reads .pt bytes only; it never deserializes checkpoint content.
    before = {p.name: sha(p) for p in sorted(run.iterdir()) if p.is_file()}
    run_manifest = json_file(run / "manifest.json")
    manifest_errors = []
    if run_manifest is None:
        manifest_errors.append("run manifest missing (possible interrupted execution)")
    else:
        for item in run_manifest.get("files", []):
            if Path(item["path"]).name != item["path"] or before.get(item["path"]) != item["sha256"]:
                manifest_errors.append(item["path"])
    if manifest_errors:
        blockers.append("closed run manifest missing or hash mismatch")
    vae = json_lines(run / "vae_train.jsonl"); ddpm = json_lines(run / "ddpm_train.jsonl")
    dev = json_lines(run / "dev_monitor.jsonl")
    training = {"vae": curves(vae, ["loss", "reconstruction", "kl", "density_sum", "lr", "gradient_norm_before_clip", "duration_seconds", "elapsed_stage_seconds"], config["vae_updates"]),
        "ddpm": curves(ddpm, ["mse", "lr", "duration_seconds", "elapsed_stage_seconds"], config["ddpm_updates"]),
        "dev_saved_curve": dev,
        "dev_expected_steps": list(range(config["monitor_interval"], config["vae_updates"]+1, config["monitor_interval"])),
        "dev_checkpoint_selection": False,
        "vae_forward_rng_hash_present_each_step": all(bool(r.get("forward_rng_sha256")) for r in vae) if vae else False}
    training["dev_all_rng_preserved"] = all(r.get("rng_preserved") is True for r in dev) if dev else False
    latent = {"train": None, "generated": {}, "failures": {}}
    p = run / "train_posterior_parameters.npz"
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            values, labels, ids = z["values"], z["labels"], z["row_ids"]
            if len(values) != config["train_rows"] or len(labels) != len(values) or ids.tolist() != list(range(config["train_rows"])):
                raise ValueError("saved train posterior row identity/count differs")
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("saved posterior terminal labels differ")
            latent["train"] = {"pooled": parameter_summary(values), "by_terminal": {
                terminal: {"rows": int((labels == label).sum()), **parameter_summary(values[labels == label])}
                for label, terminal in enumerate(("FREE", "AMD"))}}
    for terminal in ("FREE", "AMD"):
        p = run / f"generated_posterior_{terminal}.npz"
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                entry = parameter_summary(z["values"])
            success_path = run / f"generated_{terminal}.npz"
            if success_path.exists():
                with np.load(success_path, allow_pickle=False) as z:
                    entry["saved_z"] = moments(z["z"])
                    entry["saved_decoder_raw"] = moments(z["raw"])
            if latent["train"] is not None:
                entry["same_terminal_scale_comparison"] = compare_ranges(entry, latent["train"]["by_terminal"][terminal])
                entry["pooled_scale_comparison"] = compare_ranges(entry, latent["train"]["pooled"])
            latent["generated"][terminal] = entry
        failure_path = run / f"generation_failure_{terminal}.npz"
        if failure_path.exists():
            with np.load(failure_path, allow_pickle=False) as z:
                latent["failures"][terminal] = {"saved_phase": json_file(run / f"generation_failure_{terminal}.json"),
                    "available_tensors": {name: moments(z[name]) for name in z.files}}
    rows = []
    if (run / "generated.csv").exists():
        with (run / "generated.csv").open(newline="") as fh:
            rows = [{k: row[k] for k in ("sample_id", "method", "terminal", "status", "length", "seed", "terminal_sample_id")} for row in csv.DictReader(fh)]
    generation = {"allocated_expected": config["samples_per_terminal"]*2, "rows": rows,
        "statuses": dict(collections.Counter(r["status"] for r in rows)),
        "summary": json_file(run / "generation_summary.json"), "errors": json_lines(run / "generation_errors.jsonl"),
        "progress": json_lines(run / "generation_progress.jsonl")}
    expected_slots = {(terminal, str(i)) for terminal in ("FREE", "AMD") for i in range(config["samples_per_terminal"])}
    actual_slots = [(r["terminal"], r["terminal_sample_id"]) for r in rows]
    generation["complete_unique_slot_coverage"] = len(actual_slots) == len(expected_slots) and set(actual_slots) == expected_slots
    if not generation["complete_unique_slot_coverage"]:
        blockers.append("saved generation slots do not cover every preallocated request exactly once")
    vs = json_file(run / "vae_training_summary.json"); ds = json_file(run / "ddpm_training_summary.json")
    gs = generation["summary"]
    budget = {"caps": {k: config[k] for k in ("vae_seconds_cap", "ddpm_seconds_cap", "training_seconds_cap", "generation_seconds_cap")},
        "vae_seconds": vs.get("elapsed_seconds") if vs else None,
        "ddpm_seconds": ds.get("elapsed_seconds") if ds else None,
        "training_seconds": status.get("training_seconds"),
        "generation_seconds": gs.get("elapsed_seconds") if gs else None,
        "process_seconds": status.get("process_seconds"), "supervisor_job_seconds": job.get("seconds"),
        "vae_summary": vs, "ddpm_summary": ds,
        "extra_prior_engineering_costs": "Not folded into current run; disclose original smoke, maturity diagnostic, zero-update input failure and review costs separately."}
    budget["cap_exceeded"] = {name: value > config[cap] if value is not None else None for name, value, cap in [
        ("vae", budget["vae_seconds"], "vae_seconds_cap"), ("ddpm", budget["ddpm_seconds"], "ddpm_seconds_cap"),
        ("training", budget["training_seconds"], "training_seconds_cap"), ("generation", budget["generation_seconds"], "generation_seconds_cap")]}
    after = {p.name: sha(p) for p in sorted(run.iterdir()) if p.is_file()}
    if before != after or sha(supervisor_path) != supervisor_sha256:
        raise RuntimeError("input run changed during readout; no report emitted")
    report = {"report_role": "author engineering readout; independent review still required",
        "arm": request["arm"], "run_directory": str(run), "status": status, "supervisor_job": job,
        "blockers": blockers, "manifest_errors": manifest_errors, "training": training,
        "latent": latent, "generation": generation, "budget": budget,
        "input_run_files": before, "input_stable_during_read": True,
        "supervisor_status_sha256": supervisor_sha256,
        "request_sha256": sha(request_path), "script_sha256": sha(__file__),
        "limitations": ["No model calls, weight deserialization, source CSV reads, sequence-quality evaluation or test access.",
            "Saved development diagnostics do not select a checkpoint or tune the frozen next-arm config.",
            "Finite values or similar moments/ranges do not establish distributional fit, scientific advantage or biological activity.",
            "Coordinates are correlated; extrema and ratios are descriptive diagnostics with no inferential cutoff.",
            "Fresh finite VAE/DDPM results do not establish autoregressive baseline adequacy.",
            "This report cannot authorize or sign a receipt for any next arm."], "PROMOTE": "HOLD"}
    output = Path(output).resolve()
    if output == run or run in output.parents or output.exists():
        raise ValueError("author output must be a fresh directory outside the run")
    output.mkdir(parents=True)
    dump(output / "ENGINEERING_READOUT.json", report)
    dump(output / "SAVED_CURVES.json", training)
    lines = ["# H5 terminal author engineering readout", "", f"Arm: {request['arm']}; runner state: {status.get('state')}; supervisor exit: {job.get('exit_code')}.",
        f"Logged updates: VAE {len(vae)}/{config['vae_updates']}, DDPM {len(ddpm)}/{config['ddpm_updates']}.",
        f"Generation statuses across {len(rows)} saved slots: {generation['statuses']}.", "", "Budget seconds:", ""]
    lines += [f"- {k}: {budget[k]}" for k in ("vae_seconds", "ddpm_seconds", "training_seconds", "generation_seconds", "process_seconds", "supervisor_job_seconds")]
    lines += ["", "Numerical moments, saved dev curves, terminal-matched train comparisons, failure tensors and source hashes are in ENGINEERING_READOUT.json.",
        "This is an author readout, not independent clearance. Finite does not imply a reasonable fitted distribution. No sequence-quality result was computed.",
        "PROMOTE remains HOLD. Any next arm requires its own finalized plan and independent receipt."]
    (output / "READOUT.md").write_text("\n".join(lines)+"\n")
    dump(output / "MANIFEST.json", {"files": [{"path": p.name, "sha256": sha(p)} for p in sorted(output.iterdir()) if p.is_file()]})
    print(json.dumps({"output": str(output), "runner_state": status.get("state"), "blockers": blockers, "model_calls": 0}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", required=True); p.add_argument("--output", required=True)
    args = p.parse_args()
    readout(args.request, args.output)
