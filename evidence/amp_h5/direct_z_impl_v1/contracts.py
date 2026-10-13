"""Explicit hash-bound train/dev/cache admission. No model/random execution at import."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import torch
from mechanics import finite,positive_sigma,normalization,tensor_digest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CODE_NAMES=('architecture.py','mechanics.py','contracts.py','runner.py','score.py')

def sha(path):
    p=Path(path).resolve()
    assert not any('test' in x.lower() for x in p.parts),'forbidden test path'
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def dump(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def record(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}

def code_hashes():return {str(HERE/n):sha(HERE/n) for n in CODE_NAMES}

def verify(c):
    for b in c['bindings']:assert sha(b['path'])==b['sha256'],'hash drift: '+b['path']

def verify_lock():
    fd=int(os.environ['H5_LOCK_FD']);parent=int(os.environ['H5_SUPERVISOR_PID'])
    assert os.getppid()==parent and parent>1
    actual=os.fstat(fd);expected=os.stat(ROOT.parent/'H5_PROCESS.lock')
    assert (actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino),'inherited H5 lock fd mismatch'
    return {'parent_pid':parent,'lock_fd':fd,'lock_device':actual.st_dev,'lock_inode':actual.st_ino,
        'host_process_scan_completed':False,'scope':'inherited shared H5 lock and root managed-jobs receipt'}

def admit(config,receipt,receipt_sha):
    assert sha(receipt)==receipt_sha
    c,r=read(config),read(receipt)
    assert r['status']=='PASS_ENGINEERING' and r['authorized_natural'] is True
    assert r['independent_author']!=c['implementation_author']
    assert r['config_sha256']==sha(config) and r['code_sha256']==code_hashes()
    assert r['plan_sha256']==c['plan_sha256']
    assert r['synthetic_checks'] is True and r['throughput_checks'] is True
    assert r['runner_sha256']==sha(HERE/'runner.py')
    verify(c);return c,r

def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def load_ar_core(c):
    p=c['inputs']['A_core'];spec=importlib.util.spec_from_file_location('direct_z_frozen_ar_core',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def load_inputs(c):
    p=c['inputs'];train,dev=rows(p['train']),rows(p['development'])
    assert len(train)==279 and len(dev)==41
    for r in train+dev:
        assert 1<=len(r['sequence'])<=48 and set(r['sequence'])<=set('ARNDCEQGHILKMFPSTWYV')
        assert hashlib.sha256(r['sequence'].encode()).hexdigest()==r['sequence_sha256']
        assert r['terminal_state']==('FREE','AMD')[int(r['class'])]
    for field in ('sequence','component_id','molecule_sha256'):
        assert not {r[field] for r in train}&{r[field] for r in dev}
    labels=torch.tensor([int(r['class']) for r in train],dtype=torch.long)
    assert [int((labels==k).sum()) for k in (0,1)]==[126,153]
    assert [sum(int(r['class'])==k for r in dev) for k in (0,1)]==[18,23]
    with np.load(p['posterior279'],allow_pickle=False) as f:
        a=f['values'].copy();lab=f['labels'].copy();ids=f['row_ids'].copy()
    assert a.shape==(279,2,16,48) and a.dtype==np.float32
    assert lab.dtype==ids.dtype==np.int64 and np.array_equal(lab,labels.numpy()) and np.array_equal(ids,np.arange(279))
    with np.load(p['A_posterior320'],allow_pickle=False) as f:
        assert a.tobytes()==f['values'][:279].tobytes(),'posterior not byte-identical to A training cache'
        assert np.array_equal(lab,f['labels'][:279]) and np.array_equal(ids,f['source_row_ids'][:279])
    q=finite(torch.from_numpy(a),'q279');positive_sigma(q[:,1])
    stats=normalization(q);frozen=read(c['plan_path'])['normalization']
    for key in ('R_mean','R_std'):
        assert np.allclose(stats[key],frozen[key],rtol=0,atol=1e-12),key
    assert abs(stats['Z_mean']-frozen['Z_mean_analytic'])<=1e-12
    assert abs(stats['Z_std']-frozen['Z_std_analytic'])<=1e-12
    with np.load(p['uniform_bank'],allow_pickle=False) as f:uniforms=torch.from_numpy(f['uniforms'].copy())
    assert uniforms.shape==(32,49) and uniforms.dtype==torch.float32
    finite(uniforms,'uniforms');assert bool(((uniforms>=0)&(uniforms<1)).all())
    history=read(p['historical_slots']);cs=[dict(r) for r in history if r['arm']=='C']
    assert len(cs)==32
    for terminal in ('FREE','AMD'):
        rr=[r for r in cs if r['terminal']==terminal]
        assert len(rr)==16 and sorted(r['slot'] for r in rr)==list(range(16))
    # Saved C decoder evidence must use precisely the same 32x49 uniforms.
    with np.load(p['C_generation_evidence'],allow_pickle=False) as f:
        assert np.array_equal(f['uniforms'],uniforms.numpy())
        ctokens=f['tokens'];terminated=f['terminated'];active=f['active_before']
        assert ctokens.shape==(32,49) and active.shape==(32,49) and terminated.shape==(32,)
    bykey={(r['terminal'],r['slot']):r for r in cs};aa='ARNDCEQGHILKMFPSTWYV'
    for i in range(32):
        rr=bykey[('FREE' if i<16 else 'AMD',i%16)]
        if rr['sequence'] is not None:
            values=ctokens[i].tolist();end=values.index(20) if 20 in values else 49
            assert rr['sequence']==''.join(aa[v] for v in values[:min(end,48)])
    return {'train':train,'dev':dev,'q':q,'labels':labels,'normalization':stats,'uniforms':uniforms,'C':cs}
