"""Independent saved-artifact audit. No fitting, generation, original split reads.

Reevaluates frozen model predictions and numerical summaries. Runs only after the
natural runner ends, with a 195-second signal cap including imports in main.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import hashlib
import importlib.util
import json
import math
import pathlib
import signal
import time
import traceback
from collections import Counter

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'natural_run'
DEST=ROOT/'independent/RESULT_AUDIT.json'
AA='ACDEFGHIKLMNPQRSTVWY'
TERMINALS=('FREE','AMD')
PRETTY={'FREE':'Free','AMD':'Amide'}

def read(path): return json.loads(pathlib.Path(path).read_text())
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def close(a,b,atol=1e-7):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol,(a,b)
def recursive_compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(a.keys(),b.keys())
        for k in a: recursive_compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): recursive_compare(x,y)
    elif isinstance(a,float): close(a,b)
    else: assert a==b,(a,b)

def summary(records):
    by={}
    for terminal in TERMINALS:
        rows=[r for r in records if r['terminal']==terminal]
        by[terminal]={'n':len(rows),'sequence_nll':sum(r['nll'] for r in rows)/len(rows),
                      'token_nll':sum(r['nll'] for r in rows)/sum(r['observed_tokens'] for r in rows),
                      'aa_nll':sum(r['aa_nll'] for r in rows),'eos_nll':sum(r['eos_nll'] for r in rows)}
    return {'by_terminal':by,'equal_terminal_sequence_nll':sum(by[t]['sequence_nll'] for t in TERMINALS)/2}

def context(prefix,depth): return tuple(reversed(prefix[-depth:]))+('<BOS>',)*max(0,depth-len(prefix)) if depth else ()

def probabilities(payload,prefix,terminal):
    config=payload['config']
    if payload['kind']=='independent_grouped_context_eb':
        node=payload['tree']; ctx=context(prefix,config['max_depth'])
        while node['children']:
            children=[ch for ch in node['children'] if ctx[node['depth']] in ch['symbols']]
            assert len(children)==1
            node=children[0]
        counts=node['counts']
    else:
        counts=None
        for depth in range(config['max_depth'],-1,-1):
            candidate=payload['tables'].get('|'.join(context(prefix,depth)))
            if candidate is None: continue
            support=sum(candidate[terminal]) if config['mode']=='separate' else sum(sum(v) for v in candidate.values())
            if support or depth==0:
                counts=candidate;break
        assert counts is not None
    pooled=[counts['Free'][i]+counts['Amide'][i] for i in range(21)]
    alpha=config['alpha']; total=sum(pooled)
    global_p=[(n+alpha)/(total+21*alpha) for n in pooled]
    if config['mode']=='pooled': probs=global_p
    else:
        local=counts[terminal]; lam=config['shrinkage'] if config['mode']=='partial' else 0
        probs=[(n+lam*g+alpha)/(sum(local)+lam+21*alpha) for n,g in zip(local,global_p)]
    close(sum(probs),1,1e-12)
    assert min(probs)>0
    return probs

def tree_counts_check(payload,rows):
    pairs=sorted((r['sequence'],PRETTY[r['terminal_state']]) for r in rows)
    expected=hashlib.sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest()
    assert payload['fit_summary']['fit_rows_sha256']==expected
    assert payload['fit_summary']['sequences']==len(rows)
    assert payload['fit_summary']['transitions']==sum(min(len(s)+1,48) for s,t in pairs)
    # Independently recount the root table; descendant partition conservation is
    # checked structurally and was separately tested on hand-generated examples.
    counts={t:[0]*21 for t in ('Free','Amide')}
    for sequence,terminal in pairs:
        for a in sequence: counts[terminal][AA.index(a)]+=1
        if len(sequence)<48: counts[terminal][20]+=1
    root=payload['tree']['counts'] if payload['kind']=='independent_grouped_context_eb' else payload['tables']['']
    assert counts==root
    if payload['kind']=='independent_grouped_context_eb':
        def visit(node):
            if not node['children']:return
            assert sorted(s for c in node['children'] for s in c['symbols'])==sorted(list(AA)+['<BOS>'])
            for t in ('Free','Amide'):
                assert node['counts'][t]==[sum(c['counts'][t][i] for c in node['children']) for i in range(21)]
            for child in node['children']:visit(child)
        visit(payload['tree'])

def score_saved(path,rows,is_ar,R):
    if is_ar:
        saved=R.torch.load(path,map_location='cpu',weights_only=True)
        model=R.ar.TerminalLSTM();model.load_state_dict(saved['model']);model.eval()
        inputs=R.torch.full((len(rows),48),20,dtype=R.torch.long)
        targets=inputs.clone();inputs[:,0]=21
        labels=R.torch.tensor([int(r['class']) for r in rows])
        for i,r in enumerate(rows):
            targets[i,:len(r['sequence'])]=R.torch.tensor([R.ar.ALPHABET.index(a) for a in r['sequence']])
        inputs[:,1:]=targets[:,:-1]
        with R.torch.no_grad():
            logits,_=model(inputs,labels)
            logp=R.torch.log_softmax(logits,-1)
            losses=-logp.gather(2,targets.unsqueeze(-1)).squeeze(-1).double()
        values=[losses[i,:min(len(r['sequence'])+1,48)].tolist() for i,r in enumerate(rows)]
    else:
        payload=read(path);values=[]
        for r in rows:
            seq=r['sequence'];terminal=PRETTY[r['terminal_state']]
            vals=[-math.log(probabilities(payload,seq[:i],terminal)[AA.index(a)]) for i,a in enumerate(seq)]
            if len(seq)<48: vals.append(-math.log(probabilities(payload,seq,terminal)[20]))
            values.append(vals)
    records=[]
    for r,vals in zip(rows,values):
        L=len(r['sequence'])
        records.append(dict(molecule_sha256=r['molecule_sha256'],sequence_sha256=r['sequence_sha256'],terminal=r['terminal_state'],
                            length=L,observed_tokens=len(vals),token_nlls=vals,aa_nll=sum(vals[:L]),eos_nll=sum(vals[L:]),nll=sum(vals)))
    return records

def compare_records(actual,saved):
    lookup={r['molecule_sha256']:r for r in saved}
    assert len(lookup)==len(saved)==len(actual)
    for r in actual:
        previous=lookup[r['molecule_sha256']]
        for k,v in r.items(): recursive_compare(v,previous[k])

def main():
    started=time.monotonic()
    report={'status':'RUNNING','no_training':True,'no_sampling':True,'original_split_reads':False,'checks':[]}
    def persist():
        report['elapsed_seconds']=time.monotonic()-started
        DEST.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def timeout(*_):raise TimeoutError('195-second independent saved-output audit cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(195)
    try:
        freeze=read(ROOT/'FREEZE.json')
        for path,h in freeze['files'].items():assert sha(path)==h,path
        assert read(ROOT/'independent/PREFLIGHT.json')['freeze_sha256']==sha(ROOT/'FREEZE.json')
        report['checks'].append('freeze and independent preflight bindings unchanged');persist()
        if (OUT/'FAILURE.json').exists():
            report['natural_failure']=read(OUT/'FAILURE.json')
            report['status']='BLOCKED_NATURAL_RUN_NO_RETRY';persist();return
        result=read(OUT/'RESULT.json');train=read(OUT/'TRAIN_SNAPSHOT.json');dev=read(OUT/'DEVELOPMENT_SNAPSHOT.json')
        protocol=read(ROOT/'PROTOCOL.json');groups=read(OUT/'GROUPING.json')
        assert len(train)==279 and len(dev)==41
        assert not groups['blockers'] and len(groups['assignments'])==279
        assert sorted(i for g in groups['components'] for i in g['indices'])==list(range(279))
        assert all(groups['assignments'][a]==groups['assignments'][b] for a,b in groups['edges'])
        for field in ('sequence','molecule_sha256','component_id'):assert not ({r[field] for r in train}&{r[field] for r in dev})
        for r in train+dev:
            assert hashlib.sha256(r['sequence'].encode()).hexdigest()==r['sequence_sha256']
            assert hashlib.sha256(('AMP_MOLECULE_V1\n'+r['sequence']+'\n'+r['terminal_state']).encode()).hexdigest()==r['molecule_sha256']
        report['checks'].append('snapshot identities, split isolation, grouping coverage and edge isolation');persist()
        spec=importlib.util.spec_from_file_location('audit_readonly_runner_dependencies',ROOT/'run.py')
        R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R);R.ar.configure_cpu()
        saved_cv=read(OUT/'CV_SCORES.json');saved_summaries=read(OUT/'CV_SUMMARIES.json');selection=read(OUT/'SELECTION.json')
        recomputed={method:[[] for _ in configs] for method,configs in protocol['grids'].items()}
        checkpoints=0
        for fold in range(3):
            tr=[r for i,r in enumerate(train) if groups['assignments'][i]!=fold]
            va=[r for i,r in enumerate(train) if groups['assignments'][i]==fold]
            ids=read(OUT/'cv_models'/f'fold{fold}_rows.json')
            assert ids=={'training':[r['molecule_sha256'] for r in tr],'heldout':[r['molecule_sha256'] for r in va]}
            assert not set(ids['training'])&set(ids['heldout'])
            for method,configs in protocol['grids'].items():
                for i,cfg in enumerate(configs):
                    is_ar=method=='ar_cv'
                    path=OUT/'cv_models'/f'{method}_fold{fold}_config{i}.{"pt" if is_ar else "json"}'
                    if not is_ar:tree_counts_check(read(path),tr)
                    scores=score_saved(path,va,is_ar,R)
                    compare_records(scores,[r for r in saved_cv[method][i] if r['fold']==fold])
                    recomputed[method][i].extend(scores);checkpoints+=1
            report['checks'].append(f'fold {fold} frozen models independently forward-scored; EB fit identities/root counts conserved');persist()
        summaries={m:[summary(rows) for rows in configs] for m,configs in recomputed.items()}
        recursive_compare(summaries,saved_summaries)
        for m,vals in summaries.items():
            chosen=min(range(9),key=lambda i:(vals[i]['equal_terminal_sequence_nll'],i))
            assert chosen==selection[m]['index']
            assert selection[m]['config']==protocol['grids'][m][chosen]
        report['cv_checkpoints_recomputed']=checkpoints
        report['selected_cv_balanced_nll']={m:vals[selection[m]['index']]['equal_terminal_sequence_nll'] for m,vals in summaries.items()}
        report['checks'].append('all 45 train-CV configurations independently aggregated; selected grid minima/ties reproduced');persist()
        models=read(OUT/'MODELS_FROZEN_BEFORE_DEV.json')['models'];saved_dev=read(OUT/'DEV_SCORES.json');dev_recomputed={}
        for m,record in models.items():
            assert sha(record['path'])==record['sha256']
            if not m.startswith('ar_'):tree_counts_check(read(record['path']),train)
            scores=score_saved(record['path'],dev,m.startswith('ar_'),R)
            compare_records(scores,saved_dev[m]);dev_recomputed[m]=summary(scores)
        recursive_compare(dev_recomputed,read(OUT/'DEV_SUMMARIES.json'))
        report['development_summary']=dev_recomputed
        report['checks'].append('all six dev predictive models independently forward-scored, including full BOS/EOS/cap48');persist()
        generated=read(OUT/'GENERATIONS.json')
        assert len(generated)==384 and len({r['sample_id'] for r in generated})==384
        train_sequences={r['sequence'] for r in train};guards={}
        for m in models:
            guards[m]={}
            for t in TERMINALS:
                rows=[r for r in generated if r['method']==m and r['terminal']==t]
                assert len(rows)==32
                assert {r['sample_id'] for r in rows}=={f'{m}_{t}_{i:03d}' for i in range(32)}
                for r in rows:
                    assert set(r['sequence'])<=set(AA) and len(r['sequence'])<=48
                    assert r['length']==len(r['sequence']) and r['ended_with_eos']==(len(r['sequence'])<48)
                    assert r['seed']==protocol['generation_seeds'][t]
                guards[m][t]={'empty':sum(not r['sequence'] for r in rows),'short_below5':sum(len(r['sequence'])<5 for r in rows),
                              'cap48':sum(len(r['sequence'])==48 for r in rows),'copies':sum(r['sequence'] in train_sequences for r in rows)}
        assert guards==result['generation_guards']
        failures=[]
        stages=[('grouped_train_cv',{m:summaries[m][selection[m]['index']] for m in summaries}),('development',dev_recomputed)]
        for stage,vals in stages:
            p=vals['partial']
            for m,s in vals.items():
                if m=='partial':continue
                if p['equal_terminal_sequence_nll']>=s['equal_terminal_sequence_nll']-1e-9:failures.append(f'{stage}: partial does not improve balanced sequence NLL over {m}')
                for t in TERMINALS:
                    if p['by_terminal'][t]['sequence_nll']>s['by_terminal'][t]['sequence_nll']+1e-9:failures.append(f'{stage}: partial worsens {t} sequence NLL versus {m}')
        for m in ('pooled','separate','ar_cv','ar_historical'):
            for t in TERMINALS:
                for k in ('empty','short_below5','cap48','copies'):
                    if guards['partial'][t][k]>guards[m][t][k]:failures.append(f'generation: partial {t} {k} exceeds {m}')
        assert sorted(failures)==sorted(result['failure_reasons'])
        assert result['status']==('NOT_SUPPORTED' if failures else 'DEVELOPMENT_SUPPORT_ONLY')
        report['recomputed_decision']=result['status'];report['recomputed_failure_reasons']=failures
        report['checks'].append('384 raw attempts, full per-terminal degeneration/copy guards, stop decision reproduced');persist()
        # Shared frozen legacy metric formulas are rerun for consistency. This
        # is not an independently implemented nearest-neighbor/MMD algorithm.
        metric_groups={(m,t):[r for r in generated if r['method']==m and r['terminal']==t] for m in models for t in TERMINALS}
        metrics,detail=R.ev.evaluate_generation(metric_groups,train,dev)
        metrics=json.loads(json.dumps(metrics));detail=json.loads(json.dumps(detail))
        original=read(OUT/'GENERATION_METRICS.json')
        recursive_compare(metrics['groups'],original['groups']);recursive_compare(metrics['references_by_terminal'],original['references_by_terminal'])
        recursive_compare(detail,read(OUT/'GENERATION_AUDIT.json'))
        report['checks'].append('saved raw samples reproduced frozen legacy MMD/JS/length/duplication/nearest metrics; shared-formula consistency check')
        report['runner_seconds']=result['elapsed_seconds']
        report['prior_auditor_attempt_seconds']=sum(read(ROOT/'independent'/p)['elapsed_seconds'] for p in ('INITIAL_AUDITOR_VOCAB_FAILURE.json','SECOND_AUDITOR_JSON_KEY_FAILURE.json'))
        report['auditor_repairs']=['Corrected auditor AR alphabet to frozen AR vocabulary; natural runner was correct.','Normalized metric histogram keys through JSON before comparing saved JSON; no metric formula change.']
        report['total_natural_plus_audit_seconds']=result['elapsed_seconds']+time.monotonic()-started+report['prior_auditor_attempt_seconds']
        assert time.monotonic()-started+report['prior_auditor_attempt_seconds']<=200
        assert report['total_natural_plus_audit_seconds']<=1200
        report['status']='PASS_RECOMPUTED';persist()
    except BaseException as exc:
        report['status']='AUDIT_BLOCKED'
        report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        persist();raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
