"""Synthetic-only AR contract tests. No empirical input, fitting, or sampling."""
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch
from torch import nn

from ar_model import (ALPHABET, ARConfig, BOS, MAX_LENGTH, PAD, TerminalGRU,
                      make_teacher_forcing_batch, sample_batch, sequence_nll,
                      tokens_to_sequence)
from data_contract import assert_split_isolation, read_split


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
    model = TerminalGRU()
    check("static and actual parameter count = 68709", ARConfig().parameter_count() == 68709 == sum(p.numel() for p in model.parameters()))
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
    report = {"status": "PASS", "kind": "synthetic_only_unit_contract", "checks": checks,
              "checks_passed": len(checks), "wall_seconds": time.perf_counter() - started,
              "real_training": False, "real_data_read": False, "real_generation": False,
              "test_accessed": False, "torch": torch.__version__, "parameter_count": 68709,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
