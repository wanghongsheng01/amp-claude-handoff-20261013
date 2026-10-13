"""Independent synthetic VAE algebra/gradient/decoder checks; no dataset reads."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[2] / 'implementation/vae_pipeline.py'
spec = importlib.util.spec_from_file_location('h5_vae_under_review', CODE)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
runtime = v.init_runtime(v.DEFAULT_SOURCE)
torch = v.torch
checks = []


def check(name, condition):
    assert bool(condition), name
    checks.append(name)


ce = v.make_model('categorical_ce', 222)
mse = v.make_model('raw_onehot_mse', 222)
check('CE and MSE full initial state identical', v.state_hash(ce) == v.state_hash(mse))
check('21-head has 357 parameters', sum(x.numel() for x in ce.vae.decoder.out_proj.parameters()) == 357)
check('21-head whole model has 1681549 parameters', sum(x.numel() for x in ce.parameters()) == 1681549)
target = torch.tensor([[0, 1, 2] + [20] * 45, [19] * 48], dtype=torch.long)
g = torch.zeros(2, 3, 48)
g[0, :, :3] = .3
g[1, :, :] = -.2
h = torch.arange(18).reshape(2, 9).double() / 19
torch.manual_seed(377)
ce_terms = v.objective(ce, target, g, h, 'categorical_ce')
torch.manual_seed(377)
mse_terms = v.objective(mse, target, g, h, 'raw_onehot_mse')
check('coupled epsilon gives identical initial raw outputs', torch.equal(ce_terms['raw'], mse_terms['raw']))
check('same initial density difference', torch.equal(ce_terms['density_sum'], mse_terms['density_sum']))
raw = ce_terms['raw']
gathered = torch.gather(raw, 1, target[:, None]).squeeze(1)
manual_ce = (torch.logsumexp(raw, 1) - gathered).sum() / 96
manual_mse = sum((raw[:, k, :] - (target == k).to(raw)).square().sum() for k in range(21)) * .5 / 96
check('CE all48 position denominator', torch.allclose(ce_terms['reconstruction'], manual_ce))
check('raw onehot MSE half sum no divide21', torch.allclose(mse_terms['reconstruction'], manual_mse))
check('new KL density/(B48)', torch.equal(ce_terms['kl'], ce_terms['density_sum']/96))
check('total includes beta1 no clipped KL', torch.equal(ce_terms['loss'], ce_terms['reconstruction']+ce_terms['kl']))
ce_terms['loss'].backward()
check('finite gradients all reached parameters', all(torch.isfinite(p.grad).all() for p in ce.parameters() if p.grad is not None))
check('encoder receives nonzero gradient', any(p.grad is not None and p.grad.abs().sum()>0 for p in ce.vae.encoder.parameters()))
check('decoder receives nonzero gradient', any(p.grad is not None and p.grad.abs().sum()>0 for p in ce.vae.decoder.parameters()))
check('flow receives nonzero gradient', any(p.grad is not None and p.grad.abs().sum()>0 for p in ce.vae.prior_flow.parameters()))
scalar = v.make_model('scalar_legacy', 222)
check('scalar head retains17 parameters', sum(x.numel() for x in scalar.vae.decoder.out_proj.parameters()) == 17)
torch.manual_seed(378)
new_legacy = v.objective(scalar, target, g, h, 'scalar_legacy')
torch.manual_seed(378)
original_raw, original_kl, _, _, _ = scalar.vae(x=target.T, g=g.permute(2,1,0), h=h, infer=False)
check('scalar forward raw matches inherited native route', torch.equal(new_legacy['raw'], original_raw))
check('scalar KL matches inherited native route', torch.equal(new_legacy['kl'], original_kl))
check('scalar MSE uses full48 numerator/nonPAD denominator', torch.equal(new_legacy['reconstruction'], (original_raw-target).square().sum()/(target!=20).sum()))
score = torch.zeros(5,21,48)
for row, length in enumerate([0,1,47,48]):
    score[row,20,:] = 1
    score[row,0,:length] = 2
tokens, lengths, j = v.joint_decode(score)
check('joint MAP length boundaries0/1/47/48', lengths[:4].tolist() == [0,1,47,48])
check('joint MAP exact tie chooses shortest0', int(lengths[4]) == 0)
check('joint MAP AA exact tie chooses lowest0', torch.all(tokens[3] == 0))
for row in range(5):
    independent = [sum(float(score[row,:20,t].max()) if t<L else float(score[row,20,t]) for t in range(48)) for L in range(49)]
    check('joint scores direct-sum oracle row'+str(row), j[row].tolist() == independent)
    nearest_square = []
    for L in range(49):
        onehot = torch.zeros(21,48)
        for t in range(48):
            onehot[int(score[row,:20,t].argmax()) if t<L else 20,t] = 1
        nearest_square.append(float((score[row]-onehot).square().sum()))
    check('J equals nearest legal raw-onehot square row'+str(row), nearest_square.index(min(nearest_square)) == int(lengths[row]))
result = {'status':'PASS','checks':checks,'check_count':len(checks),'source':str(CODE),'source_sha256':hashlib.sha256(CODE.read_bytes()).hexdigest(),'natural_records_read':0,'test_reads':0,'optimizer_updates':0,'synthetic_model_calls_only':True,'torch':runtime['torch'],'frozen_source_manifest_sha256':runtime['source_manifest_sha256']}
Path(__file__).with_name('VAE_SYNTHETIC_RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2))
