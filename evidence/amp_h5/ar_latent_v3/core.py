"""Frozen v3 AR mechanism pilot. Import has no RNG draw or model execution."""
import copy
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

AA = 'ARNDCEQGHILKMFPSTWYV'
EOS, BOS, POSITIONS = 20, 21, 49

def digest(tensor):
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode() + b'\0' + str(tuple(t.shape)).encode() + b'\0')
    h.update(t.numpy().tobytes())
    return h.hexdigest()

def state_digest(state):
    if isinstance(state, nn.Module): state = state.state_dict()
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        h.update(name.encode() + b'\0' + digest(t).encode())
    return h.hexdigest()

def configure_cpu():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def teacher_batch(sequences, labels):
    n = len(sequences)
    labels = torch.as_tensor(labels, dtype=torch.long)
    assert n > 0 and labels.shape == (n,) and bool(((labels == 0) | (labels == 1)).all())
    targets = torch.full((n, 49), EOS, dtype=torch.long)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for i, s in enumerate(sequences):
        assert 1 <= len(s) <= 48 and set(s) <= set(AA)
        targets[i, :len(s)] = torch.tensor([AA.index(a) for a in s])
        mask[i, :len(s)+1] = True
    inputs = torch.full_like(targets, EOS)
    inputs[:, 0], inputs[:, 1:] = BOS, targets[:, :-1]
    assert int(mask.sum()) == sum(len(s)+1 for s in sequences)
    return {'inputs': inputs, 'targets': targets, 'mask': mask, 'labels': labels}

class AR(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.latent = bool(latent)
        self.token_embedding = nn.Embedding(22, 32)
        self.terminal_embedding = nn.Embedding(2, 8)
        self.lstm = nn.LSTM(56 if latent else 40, 64, 2, batch_first=True, dropout=0)
        self.output = nn.Linear(64, 21)
        assert sum(p.numel() for p in self.parameters()) == (66597 if latent else 62501)

    def forward(self, inputs, labels, z=None, hidden=None):
        assert inputs.ndim == 2 and inputs.dtype == torch.long
        assert labels.shape == (inputs.shape[0],)
        x = [self.token_embedding(inputs), self.terminal_embedding(labels)[:, None].expand(-1, inputs.shape[1], -1)]
        if self.latent:
            assert z is not None and z.shape == (inputs.shape[0], inputs.shape[1], 16)
            x.append(z)
        else: assert z is None
        y, hidden = self.lstm(torch.cat(x, -1), hidden)
        return self.output(y), hidden

def initial_models():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(222)
        c = AR(False)
        a = AR(True)
    cs, ast = c.state_dict(), a.state_dict()
    extra_gen = torch.Generator(device='cpu').manual_seed(7222)
    extra_initial = extra_gen.get_state().clone()
    for key in cs:
        if key == 'lstm.weight_ih_l0':
            ast[key][:, :40].copy_(cs[key])
            ast[key][:, 40:].uniform_(-1/8, 1/8, generator=extra_gen)
        else: ast[key].copy_(cs[key])
    a.load_state_dict(ast)
    b = copy.deepcopy(a)
    common = {k: (v[:, :40] if k == 'lstm.weight_ih_l0' else v) for k, v in a.state_dict().items()}
    assert state_digest(common) == state_digest(c)
    assert state_digest(a) == state_digest(b)
    return {'A': a, 'B': b, 'C': c}, {'common_sha256': state_digest(c), 'A_B_sha256': state_digest(a),
        'extra_generator_initial': extra_initial, 'extra_generator_final': extra_gen.get_state()}

def extend_z(z):
    assert z.ndim == 3 and z.shape[1:] == (16, 48)
    return torch.cat((z.transpose(1, 2), torch.zeros((len(z), 1, 16), dtype=z.dtype)), 1)

def derangements(labels, count, seed):
    labels = torch.as_tensor(labels, dtype=torch.long)
    g = torch.Generator(device='cpu').manual_seed(seed)
    before = g.get_state().clone()
    groups = [torch.where(labels == k)[0] for k in (0, 1)]
    assert all(len(v) > 1 for v in groups)
    maps = []
    for _ in range(count):
        mapping = torch.empty(len(labels), dtype=torch.long)
        for group in groups:
            ordered = group[torch.randperm(len(group), generator=g)]
            mapping[ordered] = ordered.roll(1)
        validate_mapping(mapping, labels)
        maps.append(mapping)
    return torch.stack(maps), before, g.get_state()

def validate_mapping(mapping, labels):
    ids = torch.arange(len(labels))
    assert mapping.shape == labels.shape
    assert torch.equal(mapping.sort().values, ids)
    assert bool((mapping != ids).all()) and torch.equal(labels[mapping], labels)

def loss_parts(logits, targets, mask):
    assert logits.shape == (*targets.shape, 21) and targets.shape == mask.shape
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
    aa, eos = mask & (targets != EOS), mask & (targets == EOS)
    sums = {'nll_sum': ce[mask].sum(), 'observed_tokens': mask.sum(),
        'aa_nll_sum': ce[aa].sum(), 'aa_tokens': aa.sum(), 'eos_nll_sum': ce[eos].sum(), 'eos_tokens': eos.sum(),
        'correct_tokens': ((logits.argmax(-1) == targets) & mask).sum()}
    sums['loss'] = sums['nll_sum'] / sums['observed_tokens']
    return sums, ce

def uniform_bank():
    banks, states = [], {}
    for terminal, seed in [('FREE', 3422), ('AMD', 3423)]:
        g = torch.Generator(device='cpu').manual_seed(seed)
        initial = g.get_state().clone()
        banks.append(torch.rand((16, 49), generator=g, dtype=torch.float32))
        states[terminal] = {'initial': initial, 'final': g.get_state()}
    return torch.cat(banks), states

@torch.no_grad()
def decode(model, labels, *, z=None, uniforms=None, before_forward=lambda: None):
    """Only terminal and cached z enter generation. uniforms=None means argmax.

    Every position has a fixed uniform cell, including absorbing stopped rows.
    Returned evidence includes unexecuted masks and raw all-21-class proposals.
    """
    n = len(labels)
    if uniforms is not None: assert uniforms.shape == (n, 49)
    zz = extend_z(z) if model.latent else None
    if not model.latent: assert z is None
    inputs = torch.full((n, 1), BOS, dtype=torch.long)
    tokens = torch.full((n, 49), EOS, dtype=torch.long)
    raw = torch.full_like(tokens, -1)
    logits_out = torch.zeros((n, 49, 21))
    probs_out = torch.zeros_like(logits_out)
    active_out = torch.zeros((n, 49), dtype=torch.bool)
    executed = torch.zeros(49, dtype=torch.bool)
    done = torch.zeros(n, dtype=torch.bool)
    hidden = None
    steps, failure = 0, None
    try:
        for p in range(49):
            before_forward()
            logits, hidden = model(inputs, labels, None if zz is None else zz[:, p:p+1], hidden)
            logits = logits[:, 0]
            assert bool(torch.isfinite(logits).all()), 'nonfinite decode logits'
            probs = logits.softmax(-1)
            if uniforms is None: draw = logits.argmax(-1)
            else:
                cdf = probs.cumsum(-1)
                cdf[:, -1] = 1.0
                draw = (uniforms[:, p, None] >= cdf).sum(-1)
                assert bool((draw <= EOS).all())
            active_out[:, p], executed[p] = ~done, True
            logits_out[:, p], probs_out[:, p], raw[:, p] = logits, probs, draw
            actual = torch.where(done, EOS, draw)
            tokens[:, p] = actual
            done |= actual == EOS
            inputs = actual[:, None]
            steps = p+1
            if bool(done.all()): break
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    sequences, statuses = [], []
    for i in range(n):
        values = tokens[i, :steps].tolist()
        first_eos = values.index(EOS) if EOS in values else None
        end = min(first_eos if first_eos is not None else steps, 48)
        seq = ''.join(AA[v] for v in values[:end])
        complete = bool(done[i]) or steps == 49
        status = ('empty_sequence' if seq == '' else ('eos' if bool(done[i]) else 'cap_nonEOS')) if complete else 'error'
        sequences.append(seq if complete else None)
        statuses.append(status)
    return {'sequences': sequences, 'status': statuses, 'steps': steps, 'failure': failure,
        'tokens': tokens, 'raw_tokens': raw, 'logits': logits_out, 'probabilities': probs_out,
        'active_before': active_out, 'executed_positions': executed, 'terminated': done,
        'uniforms': uniforms}
