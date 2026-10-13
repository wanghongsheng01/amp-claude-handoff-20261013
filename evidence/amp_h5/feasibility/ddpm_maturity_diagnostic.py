#!/usr/bin/env python3
"""Finite DDPM-only diagnostic on frozen CE20 train posterior parameters.

One fresh 1000-update trajectory; retain endpoints20/1000. No VAE, decoder,
development/test CSV, sequence quality, clipping, retry, or horizon selection.
Numerical sampling failures are retained diagnostic outcomes, not repaired.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def log(path, data):
    with Path(path).open("a") as f:
        f.write(json.dumps(data, allow_nan=False) + "\n")


def write_slots(out, slots):
    fields = ["attempt_id", "checkpoint_update", "terminal", "seed", "status", "error"]
    with (out / "ATTEMPTS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(slots)


def load_module(path):
    spec = importlib.util.spec_from_file_location("h5_maturity_bound", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def assert_config(config):
    if config["script_sha256"] != sha(__file__):
        raise ValueError("diagnostic script hash mismatch")
    if sha(config["h5_runner"]) != config["h5_runner_sha256"]:
        raise ValueError("bound H5 implementation changed")
    if sha(config["protocol"]) != config["protocol_sha256"]:
        raise ValueError("bound scientific protocol changed")
    fixed = {"train_rows": 279, "updates": 1000, "checkpoint_updates": [20, 1000],
             "ddpm_seed": 322, "sampling_seeds": [422, 423], "batch_size": 8,
             "noise_steps": 999, "ddpm_lr": 0.005, "samples_per_terminal": 1,
             "new_vae_updates": 0, "decoder_forwards": 0, "dev_test_reads": 0}
    for key, value in fixed.items():
        if config[key] != value:
            raise ValueError("prelocked diagnostic setting differs: " + key)
    for key in ("training_seconds_cap", "generation_seconds_cap"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("root must freeze a finite positive cap: " + key)
    if sha(config["posterior_npz"]) != config["posterior_npz_sha256"]:
        raise ValueError("frozen CE20 posterior file changed")


def tensor_stats(tensor, h5):
    """Report finite-value moments separately from nonfinite counts; never alter data."""
    np = h5.np
    base = h5.tensor_observation_summary(tensor)
    array = tensor.detach().cpu().numpy().astype(np.float64)
    finite = array[np.isfinite(array)]
    base.update(finite_mean=float(finite.mean()) if finite.size else None,
                finite_std=float(finite.std()) if finite.size else None,
                finite_rms=float(np.sqrt(np.mean(finite*finite))) if finite.size else None,
                finite_q01_q50_q99=np.quantile(finite, [.01, .5, .99]).tolist() if finite.size else None)
    return base


def save_checkpoint(out, update, model, h5):
    path = out / f"ddpm_ema_update_{update:04d}.pt"
    h5.torch.save(model.state_dict(), path)
    info = {"path": path.name, "sha256": sha(path), "tensor_state_sha256": h5.state_hash(model),
            "update": update, "fresh_total_OneCycle_horizon": 1000,
            "EMA_semantics": "original copy-current branch below2000; no averaging yet"}
    dump(path.with_suffix(".json"), info)
    return info


def train(config, out, h5):
    torch, np = h5.torch, h5.np
    started = time.perf_counter()
    with np.load(config["posterior_npz"], allow_pickle=False) as archive:
        values = torch.from_numpy(archive["values"].copy())
        labels = torch.from_numpy(archive["labels"].copy())
        row_ids = archive["row_ids"].copy()
    if values.shape != (279, 2, 16, 48) or values.dtype != torch.float32 or labels.dtype != torch.long:
        raise ValueError("frozen posterior schema mismatch")
    if labels.shape != (279,) or not np.array_equal(row_ids, np.arange(279)):
        raise ValueError("frozen posterior identity/order mismatch")
    if torch.bincount(labels, minlength=2).tolist() != [126, 153]:
        raise ValueError("terminal class counts mismatch")
    h5.finite(values, "frozen train posterior")
    train_mu, train_sigma = values[:, 0].double(), values[:, 1].exp().double()
    train_z_mean = float(train_mu.mean())
    train_z_second = float((train_mu.square()+train_sigma.square()).mean())
    dump(out / "target_summary.json", {"mu": tensor_stats(values[:, 0], h5),
         "logsigma": tensor_stats(values[:, 1], h5), "sigma": tensor_stats(values[:, 1].exp(), h5),
         "train_z_analytic_marginal": {"mean": train_z_mean,
             "std": math.sqrt(max(0.0, train_z_second-train_z_mean**2)), "rms": math.sqrt(train_z_second),
             "scope": "analytic moments of uniformly mixed train q(z); no z draws, no encoder"},
         "sha256": config["posterior_npz_sha256"], "fit_vae_here": False})
    torch.manual_seed(config["ddpm_seed"]); random.seed(config["ddpm_seed"])
    model = h5.UNet_conditional(c_in=2, c_out=2, time_dim=256, num_class1=2, num_class2=2)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    helper = h5.EMA(0.995)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, total_steps=1000)
    h5.save_model(out, "ddpm_initial_state", model, {"seed": 322, "fresh": True})
    _, _, abar = h5.noise_schedule(999)
    generator = torch.Generator().manual_seed(322)
    order, cursor, checkpoints, done = [], 0, {}, 0
    for step in range(1, 1001):
        if time.perf_counter()-started >= config["training_seconds_cap"]:
            raise TimeoutError("training cap reached before fixed1000 endpoint; no continuation")
        tic = time.perf_counter()
        if cursor >= len(order):
            order = torch.randperm(len(values), generator=generator).tolist(); cursor = 0
        idx = order[cursor:cursor+8]; cursor += len(idx)
        value, label = values[idx], labels[idx]
        rng_before = h5.batch_hash(torch.get_rng_state())
        t = torch.randint(1, 999, (len(idx),))
        noise = torch.randn_like(value)
        a = abar[t][:, None, None, None]
        xt = a.sqrt()*value+(1-a).sqrt()*noise
        dropped = random.random() < 0.1
        model.train(); optimizer.zero_grad(set_to_none=True)
        pred = model(xt, t, None if dropped else label)
        h5.finite(pred, "DDPM prediction")
        loss = torch.nn.functional.mse_loss(pred, noise)
        h5.finite(loss, "DDPM loss"); loss.backward(); h5.check_parameters(model)
        optimizer.step(); h5.check_parameters(model, optimizer)
        helper.step_ema(ema, model); scheduler.step(); done = step
        log(out / "ddpm_train.jsonl", {"step": step, "record_indices": idx, "rows": len(idx),
            "mse": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"], "label_dropped": dropped,
            "rng_before_noise_sha256": rng_before, "noise_timestep_sha256": h5.batch_hash(noise, t),
            "duration_seconds": time.perf_counter()-tic, "elapsed_training_seconds": time.perf_counter()-started})
        if step in (20, 1000):
            checkpoints[step] = save_checkpoint(out, step, ema, h5)
            print(json.dumps({"event": "fixed_checkpoint_saved", "step": step,
                              "training_seconds": time.perf_counter()-started}), flush=True)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "ema_step": helper.step, "completed_updates": done}, out / "training_final_state.pt")
    summary = {"updates": done, "training_seconds": time.perf_counter()-started,
               "seconds_cap": config["training_seconds_cap"], "checkpoints": checkpoints,
               "fixed_endpoint_reached": done == 1000, "no_sampling_during_training": True}
    dump(out / "training_summary.json", summary)
    return ema, checkpoints, summary


def noise_bank(config, out, h5):
    """Freeze exact H5 random-draw order once per terminal, shared by both endpoints."""
    torch, np = h5.torch, h5.np
    banks = {}
    for terminal, seed in zip(("FREE", "AMD"), (422, 423)):
        generator = torch.Generator().manual_seed(seed)
        initial = torch.randn((1, 2, 16, 48), generator=generator)
        noises = torch.stack([torch.randn((1, 2, 16, 48), generator=generator)
                              for _ in reversed(range(2, 999))])
        epsilon = torch.randn((1, 16, 48), generator=generator)
        banks[terminal] = (initial, noises, epsilon)
        np.savez_compressed(out / f"frozen_noise_{terminal}.npz", initial=initial.numpy(),
                            reverse_noise_t998_to2=noises.numpy(), posterior_epsilon=epsilon.numpy())
        dump(out / f"frozen_noise_{terminal}.json", {"seed": seed,
            "initial_sha256": h5.batch_hash(initial), "reverse_noise_sha256": h5.batch_hash(noises),
            "epsilon_sha256": h5.batch_hash(epsilon), "file_sha256": sha(out / f"frozen_noise_{terminal}.npz"),
            "random_order": "initial x; reverse t998..2 noises; t1 zeros; posterior epsilon",
            "paired_checkpoints": [20, 1000]})
    return banks


def generate_latents(config, out, h5, model, checkpoints, slots):
    torch, np = h5.torch, h5.np
    started = time.perf_counter()
    rng_before = h5.batch_hash(torch.get_rng_state())
    beta, alpha, abar = h5.noise_schedule(999)
    np.savez_compressed(out / "SCHEDULE.npz", beta=beta.numpy(), alpha=alpha.numpy(), abar=abar.numpy())
    banks = noise_bank(config, out, h5)
    model.eval(); model.requires_grad_(False)
    # Analytic null predictor: independent-coordinate variance with pred identically0.
    null_variance = 1.0
    for t in reversed(range(1, 999)):
        null_variance = null_variance/float(alpha[t])+(float(beta[t]) if t > 1 else 0.0)
    dump(out / "zero_predictor_reference.json", {"final_parameter_variance": null_variance,
        "final_parameter_std": math.sqrt(null_variance), "model_calls": 0,
        "scope": "analytic null recurrence under the frozen schedule; resemblance is not a root-cause proof"})
    progress_steps = {998, 900, 750, 500, 250, 100, 50, 10, 1}
    for slot in slots:
        if time.perf_counter()-started >= config["generation_seconds_cap"]:
            break
        step, terminal = slot["checkpoint_update"], slot["terminal"]
        label = 0 if terminal == "FREE" else 1
        checkpoint = out / checkpoints[step]["path"]
        if sha(checkpoint) != checkpoints[step]["sha256"]:
            raise ValueError("fresh fixed checkpoint changed")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        before_state = h5.state_hash(model)
        initial, noises, epsilon = banks[terminal]
        attempt = slot["attempt_id"]
        arrays, current_t, phase = {}, None, "initial"
        try:
            with torch.no_grad():
                x = initial.clone(); labels = torch.full((1,), label, dtype=torch.long)
                arrays.update(initial=initial, posterior_epsilon=epsilon)
                for current_t in reversed(range(1, 999)):
                    if time.perf_counter()-started >= config["generation_seconds_cap"]:
                        raise TimeoutError("total four-attempt generation cap; no retry")
                    phase = "reverse_ddpm"
                    t = torch.full((1,), current_t, dtype=torch.long)
                    arrays["reverse_input"] = x
                    h5.finite(x, "reverse input")
                    pred = model(x, t, labels); arrays["prediction"] = pred
                    h5.finite(pred, "reverse prediction")
                    noise = noises[998-current_t] if current_t > 1 else torch.zeros_like(x)
                    arrays["reverse_noise"] = noise
                    x = (x-(1-alpha[current_t])/(1-abar[current_t]).sqrt()*pred)/alpha[current_t].sqrt()
                    x = x+beta[current_t].sqrt()*noise
                    arrays["posterior_parameters"] = x; h5.finite(x, "reverse output")
                    log(out / "generation_noise_hashes.jsonl", {"attempt_id": attempt, "timestep": current_t,
                        "noise_sha256": h5.batch_hash(noise)})
                    if current_t in progress_steps:
                        log(out / "reverse_scale_checkpoints.jsonl", {"attempt_id": attempt,
                            "timestep": current_t, "x": tensor_stats(x, h5),
                            "mu": tensor_stats(x[:, 0], h5),
                            "logsigma": tensor_stats(x[:, 1], h5),
                            "prediction": tensor_stats(pred, h5),
                            "generation_seconds": time.perf_counter()-started})
                        np.savez_compressed(out / f"{attempt}_t{current_t:03d}.npz", x=x.numpy(), prediction=pred.numpy())
                phase = "posterior_reparameterization"
                mu, logsigma = x[:, 0], x[:, 1]
                sigma = logsigma.exp(); z = mu+epsilon*sigma
                arrays.update(mu=mu, logsigma=logsigma, sigma=sigma, latent=z)
                # Observe both nonfinite and underflow at once; do not hide zeros behind earlier Inf.
                dump(out / f"{attempt}_posterior_summary.json", {"mu": tensor_stats(mu, h5),
                    "logsigma": tensor_stats(logsigma, h5),
                    "sigma": tensor_stats(sigma, h5), "sigma_zero_count": int((sigma == 0).sum()),
                    "latent": tensor_stats(z, h5), "epsilon_sha256": h5.batch_hash(epsilon)})
                h5.finite(sigma, "exp(logsigma)")
                if not bool((sigma > 0).all()):
                    raise FloatingPointError("exp(logsigma) underflow; no repair")
                h5.finite(z, "reparameterized latent")
                slot["status"] = "finite_latent_no_decoder"
        except Exception as exc:
            slot.update(status="error:"+type(exc).__name__, error=repr(exc))
            dump(out / f"{attempt}_failure.json", {"phase": phase, "timestep": current_t,
                "error": repr(exc), "traceback": traceback.format_exc()})
        finally:
            np.savez_compressed(out / f"{attempt}_available_tensors.npz",
                                **{k: v.detach().cpu().numpy() for k, v in arrays.items()})
            if h5.state_hash(model) != before_state:
                raise RuntimeError("generation changed model weights")
            write_slots(out, slots)
            print(json.dumps({"event": "attempt_complete", **slot}), flush=True)
    rng_unchanged = rng_before == h5.batch_hash(torch.get_rng_state())
    if not rng_unchanged:
        raise RuntimeError("sampling changed global RNG despite fixed local noise bank")
    summary = {"seconds": time.perf_counter()-started, "global_rng_unchanged": rng_unchanged,
               "attempts": slots, "decoder_forwards": 0, "generated_sequence_strings": 0,
               "numerical_success_count": sum(s["status"] == "finite_latent_no_decoder" for s in slots),
               "failed_count": sum(s["status"].startswith("error:") for s in slots),
               "numerical_failure_count": sum(s["status"] == "error:FloatingPointError" for s in slots),
               "execution_failure_count": sum(s["status"].startswith("error:")
                   and s["status"] != "error:FloatingPointError" for s in slots),
               "not_executed_count": sum(s["status"] == "not_executed" for s in slots)}
    dump(out / "generation_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text()); assert_config(config)
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", config)
    slots = [{"attempt_id": f"UPDATE{step:04d}_{terminal}_000", "checkpoint_update": step,
              "terminal": terminal, "seed": seed, "status": "not_executed", "error": ""}
             for step in (20, 1000) for terminal, seed in zip(("FREE", "AMD"), (422, 423))]
    write_slots(out, slots)
    started = time.perf_counter()
    status = {"state": "RUNNING", "train_vae_updates": 0, "encoder_forwards": 0,
              "decoder_forwards": 0, "dev_test_reads": 0, "scientific_scope": "DDPM maturity on frozen CE20 representation only"}
    try:
        sys.dont_write_bytecode = True
        h5 = load_module(config["h5_runner"])
        dump(out / "runtime.json", h5.init_runtime(config["source_root"]))
        model, checkpoints, training = train(config, out, h5)
        generation = generate_latents(config, out, h5, model, checkpoints, slots)
        complete = (generation["not_executed_count"] == 0 and generation["execution_failure_count"] == 0
                    and training["fixed_endpoint_reached"])
        status.update(state="COMPLETE_WITH_NUMERICAL_FAILURES" if complete and generation["numerical_failure_count"]
                      else "COMPLETE" if complete else "INCOMPLETE", training=training, generation=generation)
    except Exception as exc:
        status.update(state="FAILED", error=repr(exc), traceback=traceback.format_exc())
    finally:
        status["seconds"] = time.perf_counter()-started
        status["frozen_posterior_unchanged"] = sha(config["posterior_npz"]) == config["posterior_npz_sha256"]
        dump(out / "status.json", status)
        dump(out / "manifest.json", {"files": [{"path": p.name, "sha256": sha(p)}
             for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"]})
        print(json.dumps(status, indent=2), flush=True)
    # Recorded numerical outcomes are expected diagnostic endpoints, never silently treated as finite.
    return 0 if status["state"] in ("COMPLETE", "COMPLETE_WITH_NUMERICAL_FAILURES") else 2


if __name__ == "__main__":
    raise SystemExit(main())
