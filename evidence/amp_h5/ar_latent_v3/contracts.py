"""Hash admission, cached inputs, sealed random schedules; no model forwards."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from core import AA, digest, derangements, uniform_bank

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def sha(path):
    p = Path(path).resolve()
    assert not any('test' in part.lower() for part in p.parts), 'forbidden input'
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(path, obj):
    p = Path(path)
    temporary = p.with_name(p.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)

def record(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}

def load_json(path): return json.loads(Path(path).read_text())

def verify_bindings(config):
    for b in config['bindings']:
        assert sha(b['path']) == b['sha256'], 'binding drift: ' + b['path']

def admission(config_path, receipt_path, receipt_sha):
    assert sha(receipt_path) == receipt_sha, 'receipt hash mismatch'
    c, r = load_json(config_path), load_json(receipt_path)
    assert r['status'] == 'PASS' and r['natural_execution_authorized'] is True
    assert r['independent_author'] != c['implementation_author']
    assert r['config_sha256'] == sha(config_path)
    expected = {str(HERE / n): sha(HERE / n) for n in ('core.py', 'contracts.py', 'runner.py', 'scoring.py')}
    assert r['code_sha256'] == expected
    assert r['synthetic_checks'] == 'PASS' and r['throughput_checks'] == 'PASS'
    verify_bindings(c)
    return c

def csv_rows(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))

def array(npz, key, shape, dtype):
    value = npz[key].copy()
    assert value.shape == shape and value.dtype == np.dtype(dtype), (key, value.shape, value.dtype)
    assert bool(np.isfinite(value).all())
    return torch.from_numpy(value)

def prepare(c, out):
    """Reads only explicit hash-bound train/dev caches and historical D."""
    verify_bindings(c)
    paths = c['inputs']
    refs = csv_rows(paths['reference_rows'])
    assert len(refs) == 320
    train, dev = csv_rows(paths['train']), csv_rows(paths['development'])
    assert len(train) == 279 and len(dev) == 41
    for offset, rows, split in [(0, train, 'train'), (279, dev, 'development')]:
        for i, r in enumerate(rows):
            ref = refs[offset+i]
            assert ref['split'] == split and int(ref['row_id']) == i
            assert int(ref['source_row_id']) == offset+i
            assert ref['truth_sequence'] == r['sequence'] and ref['terminal'] == r['terminal_state']
            assert int(ref['class']) == int(r['class']) and int(ref['target_length']) == len(r['sequence'])
            assert 1 <= len(r['sequence']) < 48 and set(r['sequence']) <= set(AA)
            assert hashlib.sha256(r['sequence'].encode()).hexdigest() == r['sequence_sha256']
    for field in ['sequence', 'component_id', 'molecule_sha256']:
        assert not {r[field] for r in train} & {r[field] for r in dev}
    labels = torch.tensor([int(r['class']) for r in train+dev])
    assert [int((labels[:279] == k).sum()) for k in (0,1)] == [126,153]
    assert [int((labels[279:] == k).sum()) for k in (0,1)] == [18,23]
    with np.load(paths['posterior'], allow_pickle=False) as f:
        q = array(f, 'values', (320,2,16,48), 'float32')
        assert torch.equal(array(f,'labels',(320,),'int64'), labels)
        assert torch.equal(array(f,'source_row_ids',(320,),'int64'), torch.arange(320))
    eps = {}
    for seed in range(1422,1430):
        with np.load(paths['epsilon_'+str(seed)], allow_pickle=False) as f:
            eps[seed] = array(f,'epsilon',(320,16,48),'float32')[279:]
            assert f['generator_initial_state'].dtype == np.uint8 and f['generator_final_state'].dtype == np.uint8
    identity = load_json(paths['loaded_state_identity'])
    assert identity['before'] == identity['expected'] == c['expected_original_CE_state']
    generation_z = []
    for t in ('FREE','AMD'):
        with np.load(paths['generation_'+t], allow_pickle=False) as f:
            generation_z.append(array(f,'z',(16,16,48),'float32'))
            for key, shape, dtype in [('posterior_parameters',(16,2,16,48),'float32'),
                    ('raw',(16,21,48),'float32'),('tokens',(16,48),'int64'),
                    ('lengths',(16,),'int64'),('length_scores',(16,49),'float32')]:
                array(f,key,shape,dtype)
    historical = csv_rows(paths['historical_D'])
    assert len(historical) == 32
    drows = []
    for t in ('FREE','AMD'):
        rr = sorted([r for r in historical if r['terminal']==t], key=lambda r:int(r['terminal_sample_id']))
        assert len(rr)==16 and [int(r['terminal_sample_id']) for r in rr]==list(range(16))
        assert all(r['method']=='CE21_DDPM' and int(r['length'])==len(r['sequence']) for r in rr)
        for slot,r in enumerate(rr):
            drows.append({'arm':'D','terminal':t,'slot':slot,'sequence':r['sequence'] if r['status'] in ('ok','empty_sequence','no_eos_at_limit') else None,
                'status':'historical_ok' if r['status'] in ('ok','empty_sequence','no_eos_at_limit') else r['status'], 'historical_status':r['status'],
                'historical_sample_id':r['sample_id'], 'historical_length_J':int(r['length'])})
    traces = [json.loads(line) for line in Path(paths['batch_trace']).read_text().splitlines()]
    assert len(traces)==1000
    g = torch.Generator(device='cpu').manual_seed(222)
    batches, cursor, order = [], 279, None
    for step,r in enumerate(traces,1):
        if cursor >= 279: order, cursor = torch.randperm(279,generator=g), 0
        batch=order[cursor:cursor+32];cursor += len(batch)
        assert r['step']==step and r['row_indices']==batch.tolist()
        batches.append(batch)
    maps, map_initial, map_final = derangements(labels[:279],1000,2422)
    devmaps, dev_initial, dev_final = derangements(labels[279:],8,6422)
    uniforms, ustates = uniform_bank()
    tensors = {'train_mappings':maps,'dev_mappings':devmaps,'uniforms':uniforms,
        'map_rng_initial':map_initial,'map_rng_final':map_final,'dev_map_rng_initial':dev_initial,'dev_map_rng_final':dev_final,
        'batch_rng_final':g.get_state()}
    for t,v in ustates.items():
        for k,state in v.items(): tensors['uniform_'+t+'_'+k]=state
    np.savez(out/'RANDOM_BANKS.npz', **{k:v.numpy() for k,v in tensors.items()})
    dump(out/'BATCHES.json', [b.tolist() for b in batches])
    dump(out/'PREPARATION_SEAL.json', {'inputs':c['bindings'],'tensor_sha256':{k:digest(v) for k,v in tensors.items()},
        'batches_sha256':sha(out/'BATCHES.json'),'random_banks_sha256':sha(out/'RANDOM_BANKS.npz'),
        'posterior_tensor_sha256':digest(q),'generation_z_tensor_sha256':[digest(z) for z in generation_z],
        'new_encoder_calls':0,'new_DDPM_calls':0,'test_reads':0})
    return {'train':train,'dev':dev,'q':q,'eps':eps,'labels':labels,'batches':batches,'maps':maps,
        'devmaps':devmaps,'uniforms':uniforms,'generation_z':torch.cat(generation_z),'D':drows}
