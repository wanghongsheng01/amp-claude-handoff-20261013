"""Synthetic-only AR contract tests. No empirical input, fitting, or sampling."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import sys

import torch
from torch import nn

from ar_pipeline import (ALPHABET, ARConfig, BOS, MAX_LENGTH, PAD, TerminalLSTM,
                      make_teacher_forcing_batch, sample_batch, sequence_nll,
                      tokens_to_sequence, assert_split_isolation, read_split,
                      run_train, generate, load_checkpoint, state_digest)


class ScriptedModel(nn.Module):
    """Synthetic finite token script, used to test the actual sampling routine."""
    def __init__(self, lengths):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.lengths = lengths
        self.position = 0

    def forward(self, inputs, labels, hidden=None):
        assert inputs.shape == (len(self.lengths), 1)
        logits = torch.full((len(self.lengths), 1, 21), -1000.0)
        for row, length in enumerate(self.lengths):
            token = 0 if self.position < length else PAD
            logits[row, 0, token] = 0.0
        self.position += 1
        return logits, None


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7101)  # Synthetic-unit seed; never a training/selection candidate.
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(name, function):
        try:
            function()
        except ValueError:
            checks.append(name)
            return
        raise AssertionError(name)

    batch = make_teacher_forcing_batch(["A", "RN", "D" * 48, ""], [0, 1, 0, 1], allow_empty=True)
    check("BOS is input only", bool((batch["inputs"][:, 0] == BOS).all()) and bool((batch["targets"] < BOS).all()))
    check("teacher forcing is one-step shifted", torch.equal(batch["inputs"][:, 1:], batch["targets"][:, :-1]))
    check("observed lengths are L+1 capped at 48", batch["mask"].sum(1).tolist() == [2, 3, 48, 1])
    check("48-AA has no observed EOS", not bool(((batch["targets"][2] == PAD) & batch["mask"][2]).any()))
    check("short sequences each have exactly one observed EOS", ((batch["targets"] == PAD) & batch["mask"]).sum(1).tolist() == [1, 1, 0, 1])
    rejects("empirical empty sequence rejected", lambda: make_teacher_forcing_batch([""], [0]))
    rejects("noncanonical target rejected", lambda: make_teacher_forcing_batch(["AX"], [0]))
    rejects("overlong target rejected", lambda: make_teacher_forcing_batch(["A" * 49], [0]))
    rejects("invalid terminal rejected", lambda: make_teacher_forcing_batch(["A"], [2]))
    rejects("terminal shape mismatch rejected", lambda: make_teacher_forcing_batch(["A"], [0, 1]))
    model = TerminalLSTM()
    check("static and actual parameter count = 62501", ARConfig().parameter_count() == 62501 == sum(p.numel() for p in model.parameters()))
    with torch.no_grad():
        logits, _ = model(batch["inputs"], batch["labels"])
        check("21-class output shape", logits.shape == (4, 48, 21))
        forward_step, hidden = [], None
        for position in range(48):
            score, hidden = model(batch["inputs"][:, position:position + 1], batch["labels"], hidden)
            forward_step.append(score)
        check("parallel teacher forcing agrees with incremental path", torch.allclose(logits, torch.cat(forward_step, 1), atol=2e-7, rtol=2e-6))
        future_changed = batch["inputs"].clone()
        future_changed[:, 10:] = 0
        future_logits, _ = model(future_changed, batch["labels"])
        check("future tokens cannot alter prefix scores", torch.equal(logits[:, :10], future_logits[:, :10]))
    terms = sequence_nll(logits, batch["targets"], batch["mask"])
    modified = logits.clone()
    modified[~batch["mask"]] = torch.linspace(-100, 100, 21)
    modified_terms = sequence_nll(modified, batch["targets"], batch["mask"])
    check("unobserved PAD suffix has no loss effect", torch.equal(terms["loss"], modified_terms["loss"]))
    check("AA and EOS accounting closes", int(terms["aa_tokens"] + terms["eos_tokens"]) == int(terms["observed_tokens"]) == 54)
    check("AA and EOS NLL sum closes", torch.allclose(terms["aa_nll_sum"] + terms["eos_nll_sum"], terms["nll_sum"]))
    check("uniform-token CE is log21", abs(float(sequence_nll(torch.zeros_like(logits), batch["targets"], batch["mask"])["loss"]) - __import__("math").log(21)) < 1e-6)
    gradient_logits = logits.detach().clone().requires_grad_(True)
    sequence_nll(gradient_logits, batch["targets"], batch["mask"])["loss"].backward()
    check("masked suffix gradient is exactly zero", int(torch.count_nonzero(gradient_logits.grad[~batch["mask"]])) == 0)
    scripted = ScriptedModel([0, 3, 48])
    sampled = sample_batch(scripted, [0, 1, 0], seed=7111, requests=3)
    check("empty short and cap outputs all retained", sampled["sequences"] == ["", "AAA", "A" * 48])
    check("first EOS is absorbing", bool((sampled["tokens"][0] == PAD).all()) and bool((sampled["tokens"][1, 3:] == PAD).all()))
    check("exact requests and steps", sampled["requests"] == 3 and sampled["steps"] == 48)
    check("termination versus cap separated", sampled["terminated"] == [True, True, False] and sampled["max_length_hit"] == [False, False, True])
    check("sampling active-token accounting", sampled["active_before"].sum(1).tolist() == [1, 4, 48])
    check("empty output status explicit", sampled["status"] == ["empty_sequence", "ok", "ok"])
    check("sampler restores model train/eval state", scripted.training)
    all_empty = sample_batch(ScriptedModel([0, 0]), [0, 1], seed=7112, requests=2)
    check("all-empty finishes at first step", all_empty["steps"] == 1 and all_empty["terminated"] == [True, True])
    check("48 without PAD legal decode", tokens_to_sequence([19] * 48) == (ALPHABET[19] * 48, False))
    rejects("invalid output token rejected", lambda: tokens_to_sequence([21]))
    rejects("request cardinality is explicit", lambda: sample_batch(ScriptedModel([0]), [0], seed=1, requests=2))
    initial_rng = torch.get_rng_state().clone()
    sample_batch(ScriptedModel([0]), [0], seed=7113, requests=1)
    check("sampling generator does not consume global RNG", torch.equal(initial_rng, torch.get_rng_state()))
    check("raw probabilities normalize at every evaluated time", torch.allclose(sampled["probabilities"].sum(-1), torch.ones(3, 48)))
    zero_cap = sample_batch(ScriptedModel([4, 2]), [0, 1], seed=19, requests=2, cap_seconds=0)
    check("cap retains not-executed slots without forward", zero_cap["status"] == ["not_executed"] * 2 and zero_cap["steps"] == 0)

    class Broken(ScriptedModel):
        def forward(self, inputs, labels, hidden=None):
            if self.position == 2:
                raise FloatingPointError("synthetic failure")
            return super().forward(inputs, labels, hidden)
    broken = sample_batch(Broken([0, 48]), [0, 1], seed=20, requests=2)
    check("failure retains completed and failed slots", broken["status"] == ["empty_sequence", "error"])
    check("failure retains prior raw probabilities", broken["steps"] == 2 and broken["failure"]["type"] == "FloatingPointError" and int(broken["active_before"].sum()) == 3)
    # Synthetic row fixtures test input isolation without opening any actual split.
    train = {"role": "train", "rows": [{"sequence": "A", "molecule_sha256": "a", "component_id": "c1"}]}
    dev = {"role": "development", "rows": [{"sequence": "R", "molecule_sha256": "b", "component_id": "c2"}]}
    check("disjoint train/dev accepted", assert_split_isolation(train, dev))
    for field in ("sequence", "molecule_sha256", "component_id"):
        changed = {"role": "development", "rows": [dict(dev["rows"][0], **{field: train["rows"][0][field]})]}
        rejects("train/dev overlap rejected: " + field, lambda changed=changed: assert_split_isolation(train, changed))
    rejects("test role rejected before read", lambda: read_split("/does-not-exist", role="test", expected_sha256="0", expected_rows=1))
    rejects("test path rejected before read", lambda: read_split("/test37/sealed.csv", role="train", expected_sha256="0", expected_rows=1))
    with tempfile.TemporaryDirectory(prefix="h5_ar_synthetic_") as directory:
        p = Path(directory) / "fixture.csv"
        p.write_text("sequence\nA\n")
        rejects("hash mismatch rejected", lambda: read_split(p, role="train", expected_sha256="0" * 64, expected_rows=1))
        # Two synthetic optimizer updates validate the actual run/checkpoint path.
        # No empirical sequence, train/dev file, or model-selection loop is used.
        fixture = {"role": "train", "path": "SYNTHETIC_ONLY", "sha256": "SYNTHETIC_ONLY",
                   "rows": [{"sequence": "ARN", "class": "0", "molecule_sha256": "synthetic0"},
                            {"sequence": "KLL", "class": "1", "molecule_sha256": "synthetic1"}]}
        fitted, run = run_train(30.0, 2, 7201, Path(directory) / "fit", train_bundle=fixture)
        check("synthetic training reaches fixed final step", run["status"] == "COMPLETED_FIXED_STEPS" and run["completed_steps"] == 2)
        check("synthetic training changes parameters", run["initial_tensor_state_sha256"] != run["final_tensor_state_sha256"])
        check("train-only exposure accounting", run["examples_presented"] == 4 and run["observed_tokens_presented"] == 16 and not run["development_accessed"])
        restored = load_checkpoint(Path(directory) / "fit" / "final.pt")
        check("checkpoint roundtrip exact", state_digest(restored) == state_digest(fitted))
        generated, generation = generate(restored, {"FREE": 7222, "AMD": 7223}, Path(directory) / "generate_smoke", smoke=True)
        check("explicit smoke creates only two slots", len(generated) == 2 and generation["terminal_counts"] == {"FREE": 1, "AMD": 1})
        check("per-terminal seeds retained", [r["seed"] for r in generated] == [7222, 7223])
        raw = torch.load(Path(directory) / "generate_smoke" / "raw_sampling.pt", weights_only=True)
        check("raw outputs probabilities and two RNG states", raw["probabilities"].shape == (2, 48, 21) and raw["generator_states"].shape[0] == 2)
        capped, caprun = run_train(1e-12, 2, 7201, Path(directory) / "fit_capped", train_bundle=fixture)
        check("setup-only cap retains zero-step checkpoint", caprun["status"] == "CAPPED_BEFORE_NEXT_UPDATE" and caprun["completed_steps"] == 0 and caprun["initial_tensor_state_sha256"] == caprun["final_tensor_state_sha256"])
        for index, row in enumerate(fixture["rows"]):
            row["component_id"] = f"synthetic_train_{index}"
        synthetic_dev = {"role": "development", "path": "SYNTHETIC_DEV_ONLY", "sha256": "SYNTHETIC_DEV_ONLY",
                         "rows": [{"sequence": "DCE", "class": "0", "molecule_sha256": "synthetic_dev0", "component_id": "synthetic_dev0"}]}
        monitored, monitorrun = run_train(30.0, 100, 7211, Path(directory) / "fit_monitor",
                                         train_bundle=fixture, dev_bundle=synthetic_dev, monitor_every=100)
        monitoring = json.loads((Path(directory) / "fit_monitor" / "MONITORS.json").read_text())
        check("optional synthetic dev monitor only at exact100 boundary", monitorrun["monitor_count"] == 1 and monitoring[0]["step"] == 100)
        check("monitor cannot select checkpoint", not monitoring[0]["used_for_selection"] and monitorrun["checkpoint_selection"] == "final only" and monitorrun["completed_steps"] == 100)
    report = {"status": "PASS", "kind": "synthetic_only_unit_contract", "checks": checks,
              "checks_passed": len(checks), "wall_seconds": time.perf_counter() - started,
              "real_training": False, "real_data_read": False, "real_generation": False,
              "test_accessed": False, "torch": torch.__version__, "parameter_count": 62501,
              "synthetic_optimizer_steps": 102, "synthetic_sampling_only": True,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
