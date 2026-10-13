"""Fixed small terminal-conditional autoregressive reference model for H5.

No file is read and no training starts on import. The only generation condition
is terminal class. This is a local canonical AR implementation, not a claimed
reproduction of a named paper. Root protocol owns budgets, splits, and seeds.
"""
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

ALPHABET = "ARNDCEQGHILKMFPSTWYV"
PAD = 20  # PAD is also the first-stop/EOS outcome.
BOS = 21  # Input only, never an output class.
MAX_LENGTH = 48
TERMINAL_CLASS = {"FREE": 0, "AMD": 1}


@dataclass(frozen=True)
class ARConfig:
    token_embedding_dim: int = 32
    terminal_embedding_dim: int = 8
    hidden_dim: int = 128
    layers: int = 1
    max_length: int = MAX_LENGTH

    def validate(self):
        if asdict(self) != asdict(ARConfig()):
            raise ValueError("H5 AR architecture is fixed; a change needs a new prelock")

    def parameter_count(self):
        e, c, h = self.token_embedding_dim, self.terminal_embedding_dim, self.hidden_dim
        return 22 * e + 2 * c + 3 * h * (e + c) + 3 * h * h + 6 * h + 21 * h + 21


def check_labels(labels, count):
    if labels.ndim != 1 or labels.shape[0] != count or labels.dtype != torch.long:
        raise ValueError("terminal labels must be a one-dimensional long tensor")
    if not bool(torch.all((labels == 0) | (labels == 1))):
        raise ValueError("terminal labels must be FREE=0 or AMD=1")


def make_teacher_forcing_batch(sequences: Sequence[str], labels, *, allow_empty=False):
    """Return BOS/shifted inputs, targets, and loss mask, never oracle length input.

    A sequence shorter than 48 contributes its residues and exactly its first
    PAD/EOS target. Later PAD positions are excluded. A 48-residue sequence has
    no observed EOS inside this finite observation window (48 AA targets).
    """
    if not sequences:
        raise ValueError("empty batch")
    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    check_labels(labels, len(sequences))
    targets = torch.full((len(sequences), MAX_LENGTH), PAD, dtype=torch.long)
    inputs = torch.full_like(targets, PAD)
    inputs[:, 0] = BOS
    mask = torch.zeros_like(targets, dtype=torch.bool)
    lengths = []
    for row, sequence in enumerate(sequences):
        if not isinstance(sequence, str) or not set(sequence) <= set(ALPHABET):
            raise ValueError("noncanonical sequence")
        if not (0 if allow_empty else 1) <= len(sequence) <= MAX_LENGTH:
            raise ValueError("sequence length outside contract")
        lengths.append(len(sequence))
        if sequence:
            targets[row, :len(sequence)] = torch.tensor([ALPHABET.index(a) for a in sequence])
        mask[row, :min(len(sequence) + 1, MAX_LENGTH)] = True
    inputs[:, 1:] = targets[:, :-1]
    return {"inputs": inputs, "targets": targets, "mask": mask,
            "labels": labels, "lengths_for_accounting_only": torch.tensor(lengths)}


class TerminalGRU(nn.Module):
    def __init__(self, config=ARConfig()):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(22, config.token_embedding_dim)
        self.terminal_embedding = nn.Embedding(2, config.terminal_embedding_dim)
        self.gru = nn.GRU(config.token_embedding_dim + config.terminal_embedding_dim,
                          config.hidden_dim, num_layers=1, batch_first=True, dropout=0.0)
        self.output = nn.Linear(config.hidden_dim, 21)
        if sum(p.numel() for p in self.parameters()) != config.parameter_count():
            raise AssertionError("parameter accounting mismatch")

    def forward(self, inputs, labels, hidden=None):
        if inputs.ndim != 2 or inputs.dtype != torch.long:
            raise ValueError("inputs must be batch by time long tokens")
        if not bool(torch.all((inputs >= 0) & (inputs <= BOS))):
            raise ValueError("input token outside vocabulary")
        check_labels(labels, inputs.shape[0])
        terminal = self.terminal_embedding(labels).unsqueeze(1).expand(-1, inputs.shape[1], -1)
        embedded = torch.cat((self.token_embedding(inputs), terminal), dim=-1)
        states, hidden = self.gru(embedded, hidden)
        return self.output(states), hidden


def sequence_nll(logits, targets, mask):
    """Mean observed-token CE; AA and first-EOS sums are logged separately."""
    if logits.shape != (*targets.shape, 21) or targets.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("logits/targets/mask shape mismatch")
    if not bool(torch.any(mask)) or not bool(torch.all((targets >= 0) & (targets <= PAD))):
        raise ValueError("empty observation mask or out-of-vocabulary target")
    token_loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    observed = token_loss.masked_select(mask)
    aa_mask, eos_mask = mask & (targets != PAD), mask & (targets == PAD)
    return {"loss": observed.mean(), "nll_sum": observed.sum(), "observed_tokens": mask.sum(),
            "aa_nll_sum": token_loss.masked_select(aa_mask).sum(), "aa_tokens": aa_mask.sum(),
            "eos_nll_sum": token_loss.masked_select(eos_mask).sum(), "eos_tokens": eos_mask.sum(),
            "correct_tokens": ((logits.argmax(-1) == targets) & mask).sum()}


def tokens_to_sequence(tokens):
    sequence = []
    stopped = False
    for value in tokens:
        token = int(value)
        if not 0 <= token <= PAD:
            raise ValueError("output token outside vocabulary")
        if token == PAD:
            stopped = True
            break
        sequence.append(ALPHABET[token])
    if len(sequence) > MAX_LENGTH:
        raise ValueError("output exceeds fixed maximum")
    return "".join(sequence), stopped


@torch.no_grad()
def sample_batch(model, labels, *, seed, requests):
    """Exactly the requested attempts, temperature 1, no length/sample rejection.

    The seed owns an isolated CPU torch.Generator. Draws are batched at each
    position, including absorbing finished rows; this RNG rule is fixed. Saved
    active-before masks distinguish real next-token scores from finished rows.
    There is no reference-sequence argument or length condition.
    """
    if not isinstance(requests, int) or requests < 1:
        raise ValueError("an explicit positive request count is mandatory")
    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    check_labels(labels, requests)
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("H5 pilot is CPU only")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    was_training = model.training
    model.eval()
    inputs = torch.full((requests, 1), BOS, dtype=torch.long)
    finished = torch.zeros(requests, dtype=torch.bool)
    tokens = torch.full((requests, MAX_LENGTH), PAD, dtype=torch.long)
    draws = torch.full_like(tokens, PAD)
    scores = torch.zeros((requests, MAX_LENGTH, 21), dtype=torch.float32)
    active_before = torch.zeros((requests, MAX_LENGTH), dtype=torch.bool)
    hidden = None
    steps = 0
    try:
        for position in range(MAX_LENGTH):
            logits, hidden = model(inputs, labels, hidden)
            logits = logits[:, 0, :]
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("nonfinite AR sampling logits")
            proposal = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).squeeze(1)
            active_before[:, position] = ~finished
            scores[:, position, :] = logits
            draws[:, position] = proposal
            actual = torch.where(finished, PAD, proposal)
            tokens[:, position] = actual
            finished = finished | (actual == PAD)
            inputs = actual.unsqueeze(1)
            steps += 1
            if bool(finished.all()):
                break
        sequences, stopped = zip(*(tokens_to_sequence(row.tolist()) for row in tokens))
        # A PAD beyond the evaluated positions is padding, not a sampled EOS.
        terminated = [(len(s) < steps) for s in sequences]
        return {"sequences": list(sequences), "tokens": tokens, "raw_draws": draws,
                "logits": scores, "active_before": active_before,
                "terminated": terminated, "max_length_hit": [len(s) == MAX_LENGTH for s in sequences],
                "status": ["empty_sequence" if not s else "ok" for s in sequences],
                "steps": steps, "seed": int(seed), "requests": requests,
                "observed_sampling_tokens": int(active_before.sum()), "generator_state": generator.get_state()}
    finally:
        model.train(was_training)
