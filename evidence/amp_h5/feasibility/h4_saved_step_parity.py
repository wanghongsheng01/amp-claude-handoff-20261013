#!/usr/bin/env python3
"""Hash-bound replay of two SAVED H4 final DDPM steps; no new generation.

Run only after independent review of PARITY_CONFIG.json. No train/dev/test CSV,
encoder, optimizer, random sampling, 998-step chain, or new sequence request.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_bound(entry):
    path = Path(entry["path"]).resolve()
    if sha(path) != entry["sha256"]:
        raise ValueError("bound source changed: " + str(path))
    return path


def assignment_code(path, function_name, expected_count):
    """Use exact source AST assignments, not a separately rewritten reverse rule."""
    tree = ast.parse(Path(path).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    nodes = sorted([n for n in ast.walk(function) if isinstance(n, ast.Assign)
                    and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == "x" and isinstance(n.value, ast.BinOp)], key=lambda n: n.lineno)
    if len(nodes) != expected_count:
        raise ValueError("unexpected reverse-step expression count")
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec")


def packing_code(path, function_name):
    """Extract the exact posterior-target packing assignment from each runner."""
    tree = ast.parse(Path(path).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    nodes = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "value"]
    if len(nodes) != 1:
        raise ValueError("unexpected posterior-target packing expression count")
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec")


def array_check(np, actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    same_shape_dtype = a.shape == b.shape and a.dtype == b.dtype
    exact = same_shape_dtype and a.tobytes(order="C") == b.tobytes(order="C")
    return {"shape": list(a.shape), "dtype": str(a.dtype), "shape_dtype_equal": same_shape_dtype,
            "all_finite": finite, "bitwise_exact": exact,
            "max_abs_difference": float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
            if same_shape_dtype and a.size and finite else None}


def model_state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().numpy()
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["script_sha256"] != sha(__file__):
        raise ValueError("checker hash mismatch")
    if (config["mode"] != "two_saved_h4_t1_batches_only" or config["new_generation_requests"] != 0
            or config["training_updates"] != 0 or config["encoder_forwards"] != 0):
        raise ValueError("unsupported execution scope")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", config)
    started = time.perf_counter()
    status = {"state": "RUNNING", "new_generation_requests": 0, "training_updates": 0,
              "encoder_forwards": 0, "train_dev_test_reads": 0, "random_draws_after_construction": 0,
              "ddpm_forwards": 0, "decoder_forwards": 0, "independent_full_chain_replay": False}
    try:
        files = {name: read_bound(entry) for name, entry in config["inputs"].items()}
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = "1"
        sys.dont_write_bytecode = True
        import numpy as np
        import torch
        if (sys.version_info[:3] != (3, 12, 13) or np.__version__ != "1.26.4"
                or torch.__version__ != "2.2.2"):
            raise RuntimeError("exact H4 environment required; no tolerance relaxation")
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32); torch.use_deterministic_algorithms(True)
        source_root = Path(config["source_root"])
        source_manifest = json.loads(files["corrected_source_manifest"].read_text())
        for entry in source_manifest["files"]:
            if sha(source_root / "corrected_code" / entry["relative_path"]) != entry["corrected_sha256"]:
                raise ValueError("corrected source changed")
        sys.path.insert(0, str(source_root / "corrected_code/AMP_VAEflow"))
        from modules.vae import VAEModel
        from modules.diff import UNet_conditional
        spec = importlib.util.spec_from_file_location("h5_bound_parity_module", files["h5_runner"])
        h5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h5)
        h5.torch = torch; h5.np = np
        old_step = assignment_code(files["v3r2_runner"], "generate_one_class", 1)
        h5_step = assignment_code(files["h5_runner"], "generate", 2)
        old_packing = packing_code(files["v3r2_runner"], "encode")
        h5_packing = packing_code(files["h5_runner"], "posterior_parameters")
        vae = VAEModel(out_dim=1, con_dim=3, latent_size=16, sqz_prior=False,
                       cond_drop=False, use_prior_flow=True, device="cpu")
        ddpm = UNet_conditional(c_in=2, c_out=2, time_dim=256, num_class1=2, num_class2=2)
        vae.load_state_dict(torch.load(files["vae_weights"], map_location="cpu", weights_only=True), strict=True)
        ddpm.load_state_dict(torch.load(files["ddpm_ema_weights"], map_location="cpu", weights_only=True), strict=True)
        for model in (vae, ddpm):
            model.eval(); model.requires_grad_(False)
        def forbidden_encoder(*_args, **_kwargs):
            raise RuntimeError("encoder forbidden by replay contract")
        vae.vae.encoder.forward = forbidden_encoder
        before = {"vae": model_state_hash(vae), "ddpm": model_state_hash(ddpm)}
        h4_before = json.loads(files["h4_runtime"].read_text())["parameter_state_before"]
        if before != {"vae": h4_before["vae"], "ddpm": h4_before["ddpm_ema"]}:
            raise ValueError("H4 model state binding differs")
        rng_before = torch.get_rng_state().clone()
        beta, alpha, abar = h5.noise_schedule(999)
        results = {}
        with np.load(files["schedule"], allow_pickle=False) as archive:
            schedule_checks = {k: array_check(np, tensor.numpy(), archive[k])
                               for k, tensor in (("beta", beta), ("alpha", alpha), ("alpha_hat", abar))}
        if not all(v["bitwise_exact"] for v in schedule_checks.values()):
            raise ValueError("schedule differs from saved H4")
        with torch.no_grad():
            for terminal, label in (("FREE", 0), ("AMD", 1)):
                if time.perf_counter()-started >= config["seconds_cap"]:
                    raise TimeoutError("parity cap reached; no retry")
                with np.load(files[terminal], allow_pickle=False) as archive:
                    saved = {k: archive[k].copy() for k in archive.files}
                x = torch.from_numpy(saved["ddpm_last_input"].copy())
                if x.shape != (16, 2, 16, 48) or x.dtype != torch.float32:
                    raise ValueError("H4 saved batch shape/dtype")
                if saved["class"].tolist() != [label]*16 or not np.all(saved["ddpm_last_noise"] == 0):
                    raise ValueError("terminal or final t=1 zero-noise binding")
                labels = torch.full((16,), label, dtype=torch.long)
                t = torch.ones(16, dtype=torch.long)
                status["ddpm_forwards"] += 1
                pred = ddpm(x, t, labels)
                checks = {"checkpoint_prediction_vs_saved": array_check(np, pred.numpy(), saved["ddpm_last_prediction"])}
                # VAE infer returns B,T,C; its encoder returns B,C,T. Use the same
                # saved coordinate values to audit both exact packing expressions.
                saved_parameters = torch.from_numpy(saved["ddpm_final_parameters"].copy())
                saved_mu, saved_logs = saved_parameters[:, 0], saved_parameters[:, 1]
                old_pack_locals = {"mu": saved_mu.transpose(1, 2), "logs": saved_logs.transpose(1, 2)}
                h5_pack_locals = {"mu": saved_mu, "logs": saved_logs}
                exec(old_packing, {"__builtins__": {}, "torch": torch}, old_pack_locals)
                exec(h5_packing, {"__builtins__": {}, "torch": torch}, h5_pack_locals)
                checks["old_h5_posterior_coordinate_packing"] = array_check(
                    np, old_pack_locals["value"].numpy(), h5_pack_locals["value"].numpy())
                checks["h5_packing_vs_saved_coordinates"] = array_check(
                    np, h5_pack_locals["value"].numpy(), saved["ddpm_final_parameters"])
                noise = torch.from_numpy(saved["ddpm_last_noise"].copy())
                old_locals = {"x": x.clone(), "pred": pred, "i": 1, "alpha": alpha,
                              "alpha_hat": abar, "beta": beta, "noise": noise}
                new_locals = {"x": x.clone(), "prediction": pred, "timestep": 1,
                              "alpha": alpha, "abar": abar, "beta": beta, "noise": noise}
                exec(old_step, {"__builtins__": {}}, old_locals)
                exec(h5_step, {"__builtins__": {}}, new_locals)
                old_x, new_x = old_locals["x"], new_locals["x"]
                checks["old_step_vs_saved_parameters"] = array_check(np, old_x.numpy(), saved["ddpm_final_parameters"])
                checks["h5_step_vs_old_step"] = array_check(np, new_x.numpy(), old_x.numpy())
                checks["h5_step_vs_saved_parameters"] = array_check(np, new_x.numpy(), saved["ddpm_final_parameters"])
                mu, logsigma = new_x[:, 0], new_x[:, 1]
                sigma = logsigma.exp()
                checks["exp_logsigma_vs_saved_scale"] = array_check(np, sigma.numpy(), saved["posterior_scale"])
                epsilon = torch.from_numpy(saved["epsilon"].copy())
                z = mu+epsilon*sigma
                checks["reparameterization_vs_saved_latent"] = array_check(np, z.numpy(), saved["latent"])
                status["decoder_forwards"] += 1
                _, tokens, _, raw, _ = h5.decode(vae, z, "scalar_legacy")
                checks["h5_decoder_vs_saved_raw"] = array_check(np, raw.numpy(), saved["raw_replay"])
                checks["h5_rawtokens_vs_saved_rawtokens"] = array_check(np, tokens.numpy(), saved["original_scalar_rawtokens"])
                # A saved input replay is not a new sequence request; do not export sequence strings.
                np.savez_compressed(out / f"replayed_saved_t1_{terminal}.npz", prediction=pred.numpy(),
                    old_reverse_parameters=old_x.numpy(), h5_reverse_parameters=new_x.numpy(),
                    posterior_scale=sigma.numpy(), latent=z.numpy(), decoder_raw=raw.numpy())
                results[terminal] = {"saved_request_count": 16, "new_request_count": 0,
                    "checks": checks, "pass": all(v["bitwise_exact"] and v["all_finite"] for v in checks.values())}
                dump(out / "parity_checks.json", results)
        unchanged = before == {"vae": model_state_hash(vae), "ddpm": model_state_hash(ddpm)}
        rng_unchanged = bool(torch.equal(rng_before, torch.get_rng_state()))
        gradients = sum(p.grad is not None for model in (vae, ddpm) for p in model.parameters())
        status.update(state="PASS" if all(v["pass"] for v in results.values()) and unchanged and rng_unchanged
                      and gradients == 0 else "FAIL", schedule_checks=schedule_checks,
                      model_states_unchanged=unchanged, rng_unchanged_after_construction=rng_unchanged,
                      gradient_tensors=gradients, results=results,
                      limitations="Two stored final-step batches only; does not validate new H5 weights, all reverse timesteps, training adequacy or distribution quality.")
    except Exception as exc:
        status.update(state="FAIL", error=repr(exc), traceback=traceback.format_exc())
    finally:
        status["elapsed_seconds"] = time.perf_counter()-started
        dump(out / "status.json", status)
        dump(out / "manifest.json", {"files": [{"path": p.name, "sha256": sha(p)}
             for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"]})
        print(json.dumps(status, indent=2))
    return 0 if status["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
