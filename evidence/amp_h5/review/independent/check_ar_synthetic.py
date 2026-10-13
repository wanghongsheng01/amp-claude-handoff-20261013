"""Independent synthetic checks only. No natural split/model checkpoint reads."""
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
CODE = Path(__file__).resolve().parents[2] / 'implementation/ar_pipeline.py'
spec = importlib.util.spec_from_file_location('h5_ar_under_review', CODE)
ar = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ar
spec.loader.exec_module(ar)
checks = []


def check(name, condition):
    assert bool(condition), name
    checks.append(name)


seqs = ['', 'A', 'AR', 'A' * 47, 'R' * 48]
batch = ar.make_teacher_forcing_batch(seqs, [0, 0, 1, 1, 0], allow_empty=True)
check('BOS input only', torch.all(batch['inputs'][:, 0] == 21) and int(batch['targets'].max()) <= 20)
check('strictly shifted targets', torch.equal(batch['inputs'][:, 1:], batch['targets'][:, :-1]))
check('first EOS observed / full48 no EOS49', batch['mask'].sum(1).tolist() == [1, 2, 3, 48, 48])
check('exact observed EOS counts', ((batch['targets'] == 20) & batch['mask']).sum(1).tolist() == [1, 1, 1, 1, 0])
zero = torch.zeros((5, 48, 21), requires_grad=True)
terms = ar.sequence_nll(zero, batch['targets'], batch['mask'])
check('zero logits NLL=log21', math.isclose(float(terms['loss']), math.log(21), abs_tol=2e-6))
terms['loss'].backward()
check('masked PAD gradient is zero', torch.all(zero.grad[~batch['mask']] == 0))
check('observed target gradients nonzero', torch.all(zero.grad[batch['mask']].abs().sum(-1) > 0))
torch.manual_seed(88)
model = ar.TerminalLSTM()
check('actual two-layer LSTM', isinstance(model.lstm, torch.nn.LSTM) and model.lstm.num_layers == 2 and model.lstm.hidden_size == 64)
check('independent parameter count 62501', sum(p.numel() for p in model.parameters()) == 22*32+2*8+4*64*(40+64+2)+4*64*(64+64+2)+21*64+21 == 62501)
model.eval()
with torch.no_grad():
    baseline, _ = model(batch['inputs'], batch['labels'])
    changed = batch['inputs'].clone()
    changed[:, 20:] = (changed[:, 20:] + 3) % 21
    altered, _ = model(changed, batch['labels'])
check('future inputs cannot change past logits', torch.equal(baseline[:, :20], altered[:, :20]))
state_before = torch.get_rng_state().clone()
sample1 = ar.sample_batch(model, [0, 1], seed=422, requests=2)
sample2 = ar.sample_batch(model, [0, 1], seed=422, requests=2)
check('sample isolated from global torch RNG', torch.equal(state_before, torch.get_rng_state()))
check('repeat seed exact synthetic outputs', torch.equal(sample1['tokens'], sample2['tokens']))
check('all synthetic attempts retained', len(sample1['sequences']) == 2)
check('length range includes empty/full', all(0 <= len(s) <= 48 for s in sample1['sequences']))
active = sample1['active_before']
check('sampling active probabilities sum1', torch.allclose(sample1['probabilities'][active].sum(-1), torch.ones(int(active.sum())), atol=1e-6))
check('no source sequence or oracle generation args', set(inspect.signature(ar.sample_batch).parameters) == {'model','labels','seed','requests','cap_seconds'})
with torch.no_grad():
    model.output.weight.zero_()
    model.output.bias.fill_(-1000)
    model.output.bias[20] = 1000
empty = ar.sample_batch(model, [0, 1], seed=1, requests=2)
check('forced immediate EOS keeps two empties', empty['sequences'] == ['', ''] and empty['steps'] == 1)
check('empty slots finite/raw retained', empty['status'] == ['empty_sequence','empty_sequence'] and empty['observed_sampling_tokens'] == 2)
with torch.no_grad():
    model.output.bias.fill_(-1000)
    model.output.bias[0] = 1000
full = ar.sample_batch(model, [0, 1], seed=1, requests=2)
check('forced all AA yields full48', full['sequences'] == ['A'*48,'A'*48] and full['steps'] == 48)
check('full48 has no synthetic extra EOS', full['terminated'] == [False,False] and full['max_length_hit'] == [True,True])
zero_cap = ar.sample_batch(model, [0,1], seed=1, requests=2, cap_seconds=0)
check('zero cap executes no token decisions', zero_cap['steps'] == 0 and not bool(zero_cap['active_before'].any()))
check('zero cap retains unexecuted slots', zero_cap['status'] == ['not_executed','not_executed'] and zero_cap['sequences'] == ['',''])
clock = ar.time.perf_counter
moments = iter([0.0,0.0,2.0])
try:
    ar.time.perf_counter = lambda: next(moments)
    partial = ar.sample_batch(model, [0,1], seed=1, requests=2, cap_seconds=1)
finally:
    ar.time.perf_counter = clock
check('mid-generation cap retains partial prefix/status', partial['steps'] == 1 and partial['sequences'] == ['A','A'] and partial['status'] == ['partial_capped','partial_capped'])
class Broken(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.))
    def forward(self, inputs, labels, hidden=None):
        return torch.full((len(inputs),1,21), float('nan')), hidden
broken = ar.sample_batch(Broken(), [0,1], seed=1, requests=2)
check('nonfinite generation retains error slots', broken['status'] == ['error','error'] and broken['steps'] == 0 and broken['failure']['type'] == 'FloatingPointError')
result = {'status':'PASS','checks':checks,'check_count':len(checks),'source':str(CODE),'source_sha256':hashlib.sha256(CODE.read_bytes()).hexdigest(),'natural_records_read':0,'test_reads':0,'train_updates':0,'synthetic_forward_backward_only':True,'torch':torch.__version__}
out = Path(__file__).with_name('AR_SYNTHETIC_RESULT.json')
out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2))
