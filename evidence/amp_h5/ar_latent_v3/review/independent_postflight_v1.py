#!/usr/bin/env python3
"""Saved-output audit only. No model construction, forward, optimizer or RNG."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import collections
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import signal
import time

ROOT=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/ar_latent_v3')
RUN=ROOT/'runs/natural_v1'
OUT=ROOT/'review/INDEPENDENT_POSTFLIGHT_v1.json'
START=time.monotonic()

def expired(signum,frame): raise TimeoutError('Saved-output audit30s exhausted')
signal.signal(signal.SIGALRM,expired)
signal.setitimer(signal.ITIMER_REAL,30)
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-10):
    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<=tol), (a,b)
def td(a):
    h=hashlib.sha256();h.update(str(a.dtype).replace('int64','torch.int64').replace('float32','torch.float32').encode()+b'\0'+str(tuple(a.shape)).encode()+b'\0');h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
AA='ARNDCEQGHILKMFPSTWYV'
METRIC_AA='ACDEFGHIKLMNPQRSTVWY'

def feature(s):
    c=collections.Counter(s);p=collections.Counter(s[i:i+2] for i in range(len(s)-1));scale=math.sqrt(3)
    return [len(s)/48/scale]+[c[a]/len(s)/scale for a in METRIC_AA]+[p[a+b]/max(1,len(s)-1)/scale for a in METRIC_AA for b in METRIC_AA]
def pooled_pairs(seq):
    c=collections.Counter(s[i:i+2] for s in seq for i in range(len(s)-1));n=sum(c.values())
    return {a+b:c[a+b]/n for a in METRIC_AA for b in METRIC_AA} if n else None
def metric(seq,ref):
    p,q=pooled_pairs(seq),pooled_pairs(ref)
    f=np.mean([feature(s) for s in seq],axis=0);g=np.mean([feature(s) for s in ref],axis=0)
    div=None
    if p is not None and q is not None:
        div=0.
        for key in p:
            mid=(p[key]+q[key])/2
            if p[key]:div+=p[key]*math.log2(p[key]/mid)/2
            if q[key]:div+=q[key]*math.log2(q[key]/mid)/2
    return float(np.sum((f-g)**2)),div
def lev(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

report={'schema':'AMP_H5_AR_LATENT_V3_INDEPENDENT_POSTFLIGHT_V1','independent_author':'/root/ar_admission_independent',
        'model_calls':0,'new_rng_draws':0,'new_optimizer_updates':0,'status':'RUNNING','scientific_retries':0,'test37_reads':0}
checks=[]
try:
    plan=js(ROOT/'ROOT_PLAN.json');receipt=js(ROOT/'review/INDEPENDENT_PREFLIGHT_RECEIPT_v1.json')
    execution=js(ROOT/'execution_v1/RESULT.json');summary=js(RUN/'SUMMARY.json');config=js(ROOT/'CANDIDATE_CONFIG.json')
    assert receipt['plan_sha256']==sha(ROOT/'ROOT_PLAN.json')
    assert execution['receipt_sha256']==sha(ROOT/'review/INDEPENDENT_PREFLIGHT_RECEIPT_v1.json')
    assert execution['state']=='completed' and execution['exit_code']==0 and not execution['timed_out']
    assert execution['wall_seconds']<=660 and execution['all_bindings_unchanged']
    for b in plan['bindings']:assert sha(b['path'])==b['sha256']
    manifest=js(RUN/'MANIFEST.json')
    for f in manifest['files']:assert sha(f['path'])==f['sha256']
    assert summary['status']=='COMPLETE' and summary['failure'] is None
    assert js(RUN/'INDEPENDENT_PREPARATION_PARITY.json')==receipt['deterministic_preparation']
    assert js(RUN/'INITIAL_SEAL.json')['states']==receipt['deterministic_preparation']['initial_state_sha256']
    checks.append('40 exact input/code/plan bindings and complete saved manifest; pre-forward preparation parity')

    train,dev=rows(config['inputs']['train']),rows(config['inputs']['development'])
    assert len(train)==279 and len(dev)==41
    batches=js(RUN/'BATCHES.json');assert len(batches)==1000
    with np.load(RUN/'RANDOM_BANKS.npz',allow_pickle=False) as f:
        maps=f['train_mappings'];uniforms=f['uniforms']
    traces={a:[json.loads(x) for x in (RUN/a/'steps.jsonl').read_text().splitlines()] for a in 'ABC'}
    for arm in 'ABC':
        ar=js(RUN/arm/'TRAIN_RESULT.json');assert ar['status']=='COMPLETE' and ar['completed_updates']==1000
        assert ar['seconds']<=120 and not ar['inner120_exceeded'] and not ar['outer150_exceeded']
        assert len(traces[arm])==1000
        for i,r in enumerate(traces[arm]):
            assert r['step']==i+1 and r['batch_indices']==batches[i]
            assert r['observed_tokens']==sum(len(train[j]['sequence'])+1 for j in batches[i])
            assert r['eos_tokens']==len(batches[i])
            close(r['loss'],r['nll_sum']/r['observed_tokens'],1e-6)
    for i,(a,b,c) in enumerate(zip(traces['A'],traces['B'],traces['C'])):
        assert a['zbank_digest']==b['zbank_digest']
        assert a['train_rng_before_digest']==b['train_rng_before_digest'] and a['train_rng_after_digest']==b['train_rng_after_digest']
        assert a['batch_digest']==b['batch_digest']==c['batch_digest']
        assert b['mapping_digest']==td(maps[i]) and a['mapping_digest'] is None and c['zbank_digest'] is None
    checks.append('ABC1000 updates; exact batches/observed-token counts and all1000 A/B full sampled-bank/RNG digests')

    conditions=js(RUN/'DEVELOPMENT.json');assert len(conditions)==28
    total_free_calls=0;reconstruction={};nll={}
    for name,item in conditions.items():
        assert item['status']=='COMPLETE' and item['failure'] is None
        with np.load(RUN/'evaluation'/(name+'_teacher.npz'),allow_pickle=False) as f:
            logits=f['logits'].astype(np.float64);target=f['target_tokens'];mask=f['observed_mask'];inp=f['inputs']
        assert logits.shape==(41,49,21) and bool(np.isfinite(logits).all())
        assert np.all(inp[:,0]==21) and np.array_equal(inp[:,1:],target[:,:-1])
        assert mask.sum()==sum(len(r['sequence'])+1 for r in dev)
        for i,r in enumerate(dev):
            length=len(r['sequence']);assert mask[i].tolist()==([True]*(length+1)+[False]*(48-length))
            assert target[i,:length].tolist()==[AA.index(x) for x in r['sequence']] and target[i,length]==20
        m=logits.max(-1);lse=m+np.log(np.exp(logits-m[...,None]).sum(-1))
        ce=lse-np.take_along_axis(logits,target[...,None],axis=-1)[...,0]
        close(float(ce[mask].sum()),item['NLL']['nll_sum'],.003)
        close(float(ce[mask].mean()),item['NLL']['loss'],.000003)
        nll[name]=float(ce[mask].mean())
        with np.load(RUN/'evaluation'/(name+'_free.npz'),allow_pickle=False) as f:
            tokens=f['tokens'];raw=f['raw_tokens'];logits=f['logits'];executed=f['executed_positions'];active=f['active_before']
        steps=int(executed.sum());assert steps==item['free_steps'] and steps<=49
        assert np.array_equal(raw[:,executed],logits[:,executed].argmax(-1))
        total_free_calls+=steps;acc=[]
        assert len(item['reconstruction']['rows'])==41
        for i,r in enumerate(dev):
            vals=tokens[i,:steps].tolist();first=vals.index(20) if 20 in vals else steps
            pred=''.join(AA[x] for x in vals[:min(first,48)])
            saved=item['reconstruction']['rows'][i];assert saved['prediction']==pred and saved['truth']==r['sequence']
            matches=sum(j<len(pred) and ch==pred[j] for j,ch in enumerate(r['sequence']))
            assert saved['matched_true_AA']==matches and saved['true_AA']==len(r['sequence'])
            acc.append(Fraction(matches,len(r['sequence'])))
            assert active[i,:steps].tolist()==[p<=first for p in range(steps)]
        reconstruction[name]=sum(acc,Fraction(0))/41
        close(float(reconstruction[name]),item['reconstruction']['accuracy_equal_sequence'])
    ma=lambda prefix:sum((reconstruction[prefix+str(s)] for s in range(1422,1430)),Fraction(0))/8
    A,B,W=ma('A_S'),ma('B_S'),ma('A_mismatch_S')
    ig='PASS' if A-B>=Fraction(1,20) and A-W>=Fraction(1,20) else 'FAIL'
    assert ig==summary['information_gate']['status']
    for key,value in [('A_S8',A),('B_S8',B),('A_mismatch_S8',W)]:close(float(value),summary['information_gate'][key])
    report['information_gate']={'status':ig,'A_S8':float(A),'B_S8':float(B),'A_mismatch_S8':float(W),
        'A_minus_B':float(A-B),'A_minus_mismatch':float(A-W),'A_mu':float(reconstruction['A_mu']),
        'C_plain':float(reconstruction['C_plain']),'interpretation':'Target-informed reconstruction information only; not prior generation gain.'}
    checks.append('All28x41 saved free reconstructions and teacher-forced NLL; exact rational0.05 information gate')

    slots=js(RUN/'GENERATION_SLOTS.json');gm=js(RUN/'GENERATION_METRICS.json');assert len(slots)==128
    gen_calls=0
    for arm in 'ABC':
        with np.load(RUN/'evaluation'/(arm+'_generation.npz'),allow_pickle=False) as f:
            tokens=f['tokens'];raw=f['raw_tokens'];probs=f['probabilities'];executed=f['executed_positions'];u=f['uniforms'];terminated=f['terminated']
        assert np.array_equal(uniforms,u)
        steps=int(executed.sum());gen_calls+=steps
        cdf=np.cumsum(probs[:,executed],axis=-1);cdf[:,:,-1]=1.
        predicted=(u[:,executed,None]>=cdf).sum(-1);assert np.array_equal(predicted,raw[:,executed])
        for i in range(32):
            vals=tokens[i,:steps].tolist();first=vals.index(20) if 20 in vals else steps
            seq=''.join(AA[x] for x in vals[:min(first,48)])
            r=next(x for x in slots if x['arm']==arm and x['terminal']==('FREE' if i<16 else 'AMD') and x['slot']==i%16)
            assert r['sequence']==seq and r['terminated']==bool(terminated[i]) and len(seq)<=48
            expected='empty_sequence' if seq=='' else ('eos' if bool(terminated[i]) else 'cap_nonEOS')
            assert r['status']==expected
    counts=summary['counts'];assert counts['training_forward']==counts['updates']==3000
    assert counts['dev_teacher_forced_forward']==28 and counts['dev_one_token_forward']==total_free_calls
    assert counts['generation_forward']==gen_calls and counts['new_generation_requests']==96
    assert sum(counts[k] for k in ('training_forward','dev_teacher_forced_forward','dev_one_token_forward','generation_forward'))<=4547
    assert all(counts[k]==0 for k in ('encoder_forward','DDPM_forward','D_forward'))
    report['call_counts']=counts
    checks.append('Raw generation inverseCDF/shared uniforms, EOS statuses and128 retained slots; exact call counts')

    table={};nearest={};trainseq=[r['sequence'] for r in train];trainset=set(trainseq)
    for arm in 'ABCD':
        table[arm]={}
        for terminal in ('FREE','AMD'):
            rr=[r for r in slots if r['arm']==arm and r['terminal']==terminal]
            assert len(rr)==16 and sorted(r['slot'] for r in rr)==list(range(16))
            assert all(r['sequence'] is not None and 1<=len(r['sequence'])<=48 and set(r['sequence'])<=set(AA) for r in rr)
            seq=[r['sequence'] for r in rr];ref=[r['sequence'] for r in dev if r['terminal_state']==terminal]
            mm,j=metric(seq,ref);sm=gm['summary'][arm][terminal]
            close(mm,sm['conditional_valid_vs_dev']['mmd2_linear_biased_fixed_features'])
            close(j,sm['conditional_valid_vs_dev']['dipeptide_jensen_shannon_bits'])
            length=sum(map(len,seq))/16;gap=abs(length-sum(map(len,ref))/len(ref));legal=sum(5<=len(s)<=48 for s in seq)
            dup=16-len(set(seq));copy=sum(s in trainset for s in seq)
            for key,value in [('length_mean_including_completed_empty',length),('length_mean_difference_absolute',gap),('legal5_48_n',legal),('duplicates_n',dup),('copy_train_n',copy)]:close(value,sm[key])
            for r in rr:
                s=r['sequence']
                if s not in nearest:nearest[s]=min(lev(s,t)/max(len(s),len(t)) for t in trainseq)
                close(nearest[s],r['nearest_train_edit']['normalized_Levenshtein'])
            table[arm][terminal]={'MMD2':mm,'dipeptide_JS_bits':j,'mean_length':length,'absolute_length_mean_gap':gap,'legal5_48':legal,'copy':copy,'duplicate':dup}
    failures=[]
    for terminal in ('FREE','AMD'):
        a=table['A'][terminal]
        for arm in 'BCD':
            b=table[arm][terminal]
            failed=[]
            for key in ('MMD2','dipeptide_JS_bits'):
                if not a[key]<b[key]:failed.append(key)
            for key in ('absolute_length_mean_gap','copy','duplicate'):
                if not a[key]<=b[key]:failed.append(key)
            if a['legal5_48']<b['legal5_48']:failed.append('legal5_48')
            failures.append({'terminal':terminal,'control':arm,'failed_metrics':failed,'pass':not failed})
    gg='PASS' if all(r['pass'] for r in failures) else 'FAIL'
    assert gg==gm['status']==summary['generation_gate']
    report.update(status='PASS',audit_checks=checks,generation_gate={'status':gg,'comparisons':failures,'metrics':table},
        execution={'wall_seconds':execution['wall_seconds'],'overrun_seconds':execution['overrun_seconds'],'training':summary['training']},
        whole_experiment_claim='Reconstruction information PASS; legal conditional-DDPM free-generation gain is not established. No expansion, no retry.',
        limitations=['One training seed and reused dev41; no generalization or biological efficacy claim.','D remains historical convolution/J; A/C are not capacity-matched.'],
        final_action='STOP_ROUTE_KEEP_ALL_RESULTS',PROMOTE='HOLD',BACKUP='BACKUP_BLOCKED')
except BaseException as exc:
    import traceback
    report.update(status='AUDIT_FAILED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),completed_checks=checks,rerun_authorized=False)
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
    report['audit_wall_seconds']=time.monotonic()-START
    report['audit_script_sha256']=sha(__file__)
    report['bindings']=[{'path':str(p),'sha256':sha(p)} for p in (ROOT/'ROOT_PLAN.json',ROOT/'review/INDEPENDENT_PREFLIGHT_RECEIPT_v1.json',ROOT/'execution_v1/RESULT.json',RUN/'SUMMARY.json',RUN/'MANIFEST.json') if p.exists()]
    with OUT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':report['status'],'audit_wall_seconds':report['audit_wall_seconds'],'path':str(OUT)}))
raise SystemExit(0 if report['status']=='PASS' else 2)
