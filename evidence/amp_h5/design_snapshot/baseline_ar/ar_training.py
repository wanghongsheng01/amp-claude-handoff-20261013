"""Bounded AR fit/evaluation library; caller must enforce the root START gate.

There is no executable entrypoint, no default input path, no model-selection
loop, and no implicit validation call. All fits retain initial/final weights,
optimizer/RNG state, step ledger, timing, and failed/capped status.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch

from ar_model import ARConfig, TerminalGRU, make_teacher_forcing_batch, sequence_nll


@dataclass(frozen=True)
class TrainPlan:
    updates: int
    wall_seconds: float
    initialization_seed: int
    batch_seed: int
    batch_size: int = 32
    learning_rate: float = 0.001
    gradient_clip_l2: float = 1.0
    sampling_order: str = "shuffle_each_epoch"

    def validate(self):
        if self.updates < 1 or self.wall_seconds <= 0 or self.batch_size < 1:
            raise ValueError("fixed positive update/time/batch budget required")
        if not math.isfinite(self.wall_seconds) or self.learning_rate != 0.001 or self.gradient_clip_l2 != 1.0:
            raise ValueError("unlocked optimizer change")
        if self.sampling_order not in ("shuffle_each_epoch", "fixed_batch_repeat"):
            raise ValueError("unknown data order")


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def state_digest(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialize(plan):
    """Seeded CPU initialization without changing the caller's default RNG."""
    plan.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan.initialization_seed)
        model = TerminalGRU().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=plan.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                                 foreach=False, fused=False)
    return model, optimizer


def fit_fixed(train_bundle, *, plan, output_dir, authorization, row_indices=None):
    """Fit one fixed arm; no dev read or checkpoint/temperature/seed selection.

    Root supplies a bound authorization dict (recorded verbatim), and must run
    its own START/hash gate before calling. A row subset is allowed only for a
    separately declared smoke, never as validation-based selection. Wall cap is
    checked at update boundaries. Final checkpoint/ledger IO can finish beyond
    the cap; that time is included in total_pipeline_seconds. No retry occurs.
    """
    started = time.perf_counter()
    plan.validate()
    if train_bundle.get("role") != "train":
        raise ValueError("fit accepts train role only")
    if not authorization or authorization.get("approved") is not True:
        raise ValueError("caller must supply root-approved bound authorization")
    rows = train_bundle["rows"]
    if not rows:
        raise ValueError("empty training data")
    if row_indices is None:
        row_indices = list(range(len(rows)))
    if not row_indices or len(set(row_indices)) != len(row_indices) or any(i < 0 or i >= len(rows) for i in row_indices):
        raise ValueError("invalid fixed training row subset")
    if plan.sampling_order == "fixed_batch_repeat" and len(row_indices) > plan.batch_size:
        raise ValueError("fixed-repeat smoke subset must fit a single batch")
    selected = [rows[i] for i in row_indices]
    batch = make_teacher_forcing_batch([r["sequence"] for r in selected], [int(r["class"]) for r in selected])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    model, optimizer = initialize(plan)
    batch_generator = torch.Generator(device="cpu").manual_seed(plan.batch_seed)
    ledger = []
    initial_digest = state_digest(model)
    protocol = {"architecture": asdict(ARConfig()), "plan": asdict(plan), "authorization": authorization,
                "train_path": train_bundle.get("path"), "train_sha256": train_bundle.get("sha256"),
                "selected_row_indices": row_indices,
                "selected_molecule_sha256": [r["molecule_sha256"] for r in selected],
                "train_objects": len(rows), "selected_objects": len(selected),
                "test_accessed": False, "development_accessed": False,
                "observed_token_objective": "mean AA plus one first EOS for length<48; no EOS49",
                "parameter_count": sum(p.numel() for p in model.parameters())}
    dump(out / "BOUND_TRAINING_INPUT.json", protocol)
    initial_path = out / "initial_state.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(ARConfig()), "plan": asdict(plan),
                "batch_generator_state": batch_generator.get_state(), "tensor_state_sha256": initial_digest}, initial_path)
    order = torch.arange(len(selected))
    cursor, epoch = len(selected), -1
    status, failure = "RUNNING", None
    cumulative_tokens = cumulative_aa = cumulative_eos = cumulative_examples = 0
    compute_started = time.perf_counter()
    attempted_updates, failed_positions = 0, []
    with (out / "steps.jsonl").open("x") as stream:
        try:
            model.train()
            for update in range(1, plan.updates + 1):
                if time.perf_counter() - started >= plan.wall_seconds:
                    status = "CAPPED_BEFORE_NEXT_UPDATE"
                    break
                if plan.sampling_order == "fixed_batch_repeat":
                    selected_positions = torch.arange(len(selected))
                    epoch = update - 1
                    cursor = len(selected)
                else:
                    if cursor >= len(selected):
                        order = torch.randperm(len(selected), generator=batch_generator)
                        cursor = 0
                        epoch += 1
                    selected_positions = order[cursor:cursor + plan.batch_size]
                    cursor += len(selected_positions)
                begun = time.perf_counter()
                attempted_updates = update
                failed_positions = [row_indices[i] for i in selected_positions.tolist()]
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(batch["inputs"][selected_positions], batch["labels"][selected_positions])
                terms = sequence_nll(logits, batch["targets"][selected_positions], batch["mask"][selected_positions])
                if not bool(torch.isfinite(terms["loss"])):
                    raise FloatingPointError("nonfinite train objective")
                terms["loss"].backward()
                preclip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan.gradient_clip_l2, error_if_nonfinite=True)
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError("nonfinite parameters after update")
                tokens, aa, eos = (int(terms[k]) for k in ("observed_tokens", "aa_tokens", "eos_tokens"))
                cumulative_tokens += tokens
                cumulative_aa += aa
                cumulative_eos += eos
                cumulative_examples += len(selected_positions)
                entry = {"update": update, "epoch_zero_based": epoch,
                         "original_train_row_indices": [row_indices[i] for i in selected_positions.tolist()],
                         "examples": len(selected_positions), "observed_tokens": tokens, "aa_tokens": aa, "eos_tokens": eos,
                         "preupdate_loss_mean": float(terms["loss"].detach()),
                         "preupdate_nll_sum": float(terms["nll_sum"].detach()),
                         "preupdate_aa_nll_sum": float(terms["aa_nll_sum"].detach()),
                         "preupdate_eos_nll_sum": float(terms["eos_nll_sum"].detach()),
                         "preupdate_correct_tokens": int(terms["correct_tokens"]),
                         "gradient_norm_before_clip": float(preclip_norm),
                         "step_seconds": time.perf_counter() - begun,
                         "cumulative_observed_tokens": cumulative_tokens,
                         "cumulative_examples": cumulative_examples,
                         "elapsed_pipeline_seconds": time.perf_counter() - started}
                ledger.append(entry)
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
            else:
                status = "COMPLETE_FIXED_UPDATES"
        except Exception as error:
            status = "FAILED"
            failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                       "attempted_update": attempted_updates, "original_train_row_indices": failed_positions,
                       "elapsed_pipeline_seconds": time.perf_counter() - started}
    compute_seconds = time.perf_counter() - compute_started
    checkpoint_path = out / "final_state.pt"
    final_digest = state_digest(model)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(ARConfig()), "plan": asdict(plan), "status": status,
                "completed_updates": len(ledger), "batch_generator_state": batch_generator.get_state(),
                "order": order, "cursor": cursor, "epoch_zero_based": epoch,
                "tensor_state_sha256": final_digest, "global_torch_rng_state": torch.get_rng_state()}, checkpoint_path)
    summary = {"status": status, "failure": failure, "completed_updates": len(ledger), "attempted_updates": attempted_updates,
               "requested_updates": plan.updates, "parameter_count": protocol["parameter_count"],
               "examples_presented": cumulative_examples, "observed_tokens_presented": cumulative_tokens,
               "aa_tokens_presented": cumulative_aa, "eos_tokens_presented": cumulative_eos,
               "train_loop_seconds": compute_seconds, "total_pipeline_seconds": time.perf_counter() - started,
               "updates_per_train_loop_second": len(ledger) / compute_seconds if compute_seconds else None,
               "observed_tokens_per_train_loop_second": cumulative_tokens / compute_seconds if compute_seconds else None,
               "initial_tensor_state_sha256": initial_digest, "final_tensor_state_sha256": final_digest,
               "initial_checkpoint": file_record(initial_path), "final_checkpoint": file_record(checkpoint_path),
               "steps": file_record(out / "steps.jsonl"), "inputs": file_record(out / "BOUND_TRAINING_INPUT.json"),
               "test_accessed": False, "development_accessed": False,
               "model_selection": "none; final fixed-budget or failed/capped state retained", "retries": 0}
    dump(out / "RUN.json", summary)
    return model, summary


@torch.no_grad()
def evaluate_teacher_forcing(model, bundle, *, batch_size=32):
    """Descriptive token NLL/accuracy; never used here to select a checkpoint."""
    if bundle.get("role") not in ("train", "development") or batch_size < 1:
        raise ValueError("only explicit train/development evaluation roles allowed")
    rows = bundle["rows"]
    if not rows:
        raise ValueError("empty evaluation split")
    was_training = model.training
    before = state_digest(model)
    model.eval()
    totals = {key: 0 for key in ("nll_sum", "aa_nll_sum", "eos_nll_sum", "observed_tokens", "aa_tokens", "eos_tokens", "correct_tokens")}
    begun = time.perf_counter()
    try:
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            batch = make_teacher_forcing_batch([r["sequence"] for r in part], [int(r["class"]) for r in part])
            logits, _ = model(batch["inputs"], batch["labels"])
            terms = sequence_nll(logits, batch["targets"], batch["mask"])
            for key in totals:
                totals[key] += float(terms[key]) if key.endswith("nll_sum") else int(terms[key])
    finally:
        model.train(was_training)
    after = state_digest(model)
    if before != after:
        raise AssertionError("evaluation changed model parameters")
    return {"role": bundle["role"], "objects": len(rows), **totals,
            "mean_observed_token_nll": totals["nll_sum"] / totals["observed_tokens"],
            "teacher_forced_token_accuracy": totals["correct_tokens"] / totals["observed_tokens"],
            "wall_seconds": time.perf_counter() - begun, "tensor_state_sha256": before,
            "test_accessed": False, "is_generation_metric": False}
