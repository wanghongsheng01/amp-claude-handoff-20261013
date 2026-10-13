#!/usr/bin/env python3
"""One frozen, bounded, serial development experiment. No run on import."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import time
import traceback

HERE = Path(__file__).resolve().parent
H5 = HERE.parent
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

eb = load('eb_model', HERE/'implementation/model.py')
grouping = load('eb_grouping', HERE/'grouping.py')
ar = load('eb_ar', H5/'implementation/ar_pipeline.py')
ev = load('eb_metrics', H5/'implementation/evaluate_h5.py')
torch = ar.torch
TERMINALS = ('FREE','AMD')
PRETTY = {'FREE':'Free','AMD':'Amide'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')

def aggregate(scores):
    by = {}
    for terminal in TERMINALS:
        s = [r for r in scores if r['terminal']==terminal]
        assert s
        by[terminal] = dict(n=len(s), sequence_nll=sum(r['nll'] for r in s)/len(s),
                            token_nll=sum(r['nll'] for r in s)/sum(r['observed_tokens'] for r in s),
                            aa_nll=sum(r['aa_nll'] for r in s), eos_nll=sum(r['eos_nll'] for r in s))
    return dict(by_terminal=by, equal_terminal_sequence_nll=sum(by[t]['sequence_nll'] for t in TERMINALS)/2)

def score(model, rows, is_ar=False):
    if is_ar:
        batch = ar.make_teacher_forcing_batch([r['sequence'] for r in rows], [int(r['class']) for r in rows])
        model.eval()
        with torch.no_grad():
            logits,_ = model(batch['inputs'], batch['labels'])
            logp = torch.log_softmax(logits, -1)
            selected = logp.gather(2,batch['targets'].unsqueeze(-1)).squeeze(-1).double()
        losses = [(-selected[i][batch['mask'][i]]).tolist() for i in range(len(rows))]
    else:
        losses = []
        for row in rows:
            sequence, terminal = row['sequence'], PRETTY[row['terminal_state']]
            vals = [-math.log(model.probs(sequence[:i],terminal)[eb.INDEX[a]]) for i,a in enumerate(sequence)]
            if len(sequence)<48:
                vals.append(-math.log(model.probs(sequence,terminal)[-1]))
            losses.append(vals)
    result = []
    for row, vals in zip(rows,losses):
        length = len(row['sequence'])
        assert len(vals)==min(length+1,48) and all(math.isfinite(v) and v>=0 for v in vals)
        result.append(dict(molecule_sha256=row['molecule_sha256'], sequence_sha256=row['sequence_sha256'],
                           terminal=row['terminal_state'], length=length, observed_tokens=len(vals),
                           token_nlls=vals, aa_nll=sum(vals[:length]), eos_nll=sum(vals[length:]), nll=sum(vals)))
    return result

def fit_rows(rows):
    return [dict(sequence=r['sequence'],terminal=PRETTY[r['terminal_state']]) for r in rows]

def train_ar(rows, config, seed, on_step=None):
    torch.manual_seed(seed)
    model=ar.TerminalLSTM()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],betas=(.9,.999),eps=1e-8)
    batch=ar.make_teacher_forcing_batch([r['sequence'] for r in rows],[int(r['class']) for r in rows])
    generator=torch.Generator(device='cpu').manual_seed(seed)
    order=torch.randperm(len(rows),generator=generator)
    cursor=0
    trace=[]
    for step in range(1,config['steps']+1):
        if cursor>=len(rows):
            order=torch.randperm(len(rows),generator=generator); cursor=0
        idx=order[cursor:cursor+32]; cursor+=len(idx)
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits,_=model(batch['inputs'][idx],batch['labels'][idx])
        loss=ar.sequence_nll(logits,batch['targets'][idx],batch['mask'][idx])['loss']
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite training loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('nonfinite gradient norm')
        optimizer.step()
        if step in (250,500,1000):
            trace.append(dict(step=step,train_batch_token_nll=float(loss.detach()),tensor_sha=ar.state_digest(model)))
            if on_step is not None:
                on_step(step,model)
    return model,trace

def generation(model, method, config, is_ar=False):
    result=[]
    for terminal in TERMINALS:
        seed=config['generation_seeds'][terminal]
        if is_ar:
            raw=ar.sample_batch(model,[TERMINALS.index(terminal)]*config['requests_per_terminal'],
                                seed=seed,requests=config['requests_per_terminal'])
            if raw['failure'] or raw['capped']:
                raise RuntimeError('AR generation execution failure: '+str(raw['failure']))
            sequences=raw['sequences']
        else:
            sequences=model.sample(seed,PRETTY[terminal],config['requests_per_terminal'])
        for i,sequence in enumerate(sequences):
            result.append(dict(method=method,terminal=terminal,seed=seed,sample_id=f'{method}_{terminal}_{i:03d}',
                               sequence=sequence,length=len(sequence),ended_with_eos=len(sequence)<48,
                               status=('empty_sequence' if not sequence else 'no_eos_at_limit' if len(sequence)==48 else 'ok')))
    return result

def decision(cv_summaries, dev_summaries, generations, train):
    failures=[]
    for stage, summaries in [('grouped_train_cv',cv_summaries),('development',dev_summaries)]:
        partial=summaries['partial']
        for method,summary in summaries.items():
            if method=='partial': continue
            if partial['equal_terminal_sequence_nll'] >= summary['equal_terminal_sequence_nll'] - 1e-9:
                failures.append(f'{stage}: partial does not improve balanced sequence NLL over {method}')
            for terminal in TERMINALS:
                if partial['by_terminal'][terminal]['sequence_nll'] > summary['by_terminal'][terminal]['sequence_nll']+1e-9:
                    failures.append(f'{stage}: partial worsens {terminal} sequence NLL versus {method}')
    train_seqs={r['sequence'] for r in train}
    guard={}
    for method in dev_summaries:
        guard[method]={}
        for terminal in TERMINALS:
            rows=[r for r in generations if r['method']==method and r['terminal']==terminal]
            guard[method][terminal]=dict(empty=sum(not r['sequence'] for r in rows),short_below5=sum(len(r['sequence'])<5 for r in rows),
                                         cap48=sum(len(r['sequence'])==48 for r in rows),copies=sum(r['sequence'] in train_seqs for r in rows))
    for baseline in ('pooled','separate','ar_cv','ar_historical'):
        for terminal in TERMINALS:
            for metric in ('empty','short_below5','cap48','copies'):
                if guard['partial'][terminal][metric]>guard[baseline][terminal][metric]:
                    failures.append(f'generation: partial {terminal} {metric} exceeds {baseline}')
    return dict(status='NOT_SUPPORTED' if failures else 'DEVELOPMENT_SUPPORT_ONLY',failure_reasons=failures,generation_guards=guard,
                further_tuning_authorized=False,claims='One reused development split; no efficacy, novelty, significance, or submission-readiness evidence.')

def main():
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    freeze=json.loads((HERE/'FREEZE.json').read_text())
    for path,digest in freeze['files'].items():
        if sha(path)!=digest: raise RuntimeError('frozen input changed: '+path)
    gate=json.loads((HERE/'independent/PREFLIGHT.json').read_text())
    if gate['status']!='PASS' or gate['freeze_sha256']!=sha(HERE/'FREEZE.json'):
        raise RuntimeError('independent preflight not bound to this freeze')
    out=HERE/'natural_run'
    out.mkdir(exist_ok=False)
    started=time.monotonic()
    def timeout(*_): raise TimeoutError('1000 second natural-run hard cap; remaining 200 seconds reserved for independent recomputation')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(1000)
    ledger=[]
    def event(stage,**kwargs):
        item=dict(stage=stage,elapsed_seconds=time.monotonic()-started,**kwargs)
        ledger.append(item); dump(out/'EVENTS.json',ledger); print(json.dumps(item),flush=True)
    try:
        ar.configure_cpu()
        event('STARTED',unix_time=time.time(),freeze_sha256=sha(HERE/'FREEZE.json'),threads=torch.get_num_threads())
        train_spec=protocol['data']['train']
        train=ar.read_split(train_spec['path'],role='train',expected_sha256=train_spec['sha256'],expected_rows=279)['rows']
        dump(out/'TRAIN_SNAPSHOT.json',train)
        try:
            grouping_result=grouping.grouped_folds(train)
        except grouping.GroupingBlocked as exc:
            dump(out/'GROUPING.json',exc.report); raise
        dump(out/'GROUPING.json',grouping_result)
        event('GROUPING_ADMITTED',components=grouping_result['component_count'],folds=grouping_result['folds'])
        grids=protocol['grids']
        scores={method:[[] for _ in configs] for method,configs in grids.items()}
        training_traces=[]
        cv_models=out/'cv_models'; cv_models.mkdir()
        for fold in range(3):
            tr=[r for i,r in enumerate(train) if grouping_result['assignments'][i]!=fold]
            va=[r for i,r in enumerate(train) if grouping_result['assignments'][i]==fold]
            trees={d:eb.fit(fit_rows(tr),dict(protocol['structure'],max_depth=d)) for d in (1,2,3)}
            dump(cv_models/f'fold{fold}_rows.json',dict(training=[r['molecule_sha256'] for r in tr],heldout=[r['molecule_sha256'] for r in va]))
            for method in ('pooled','separate','partial','ngram'):
                for idx,config in enumerate(grids[method]):
                    model=(eb.fit_ngram(fit_rows(tr),config) if method=='ngram' else
                           trees[config['max_depth']].with_emission(method,alpha=config['alpha'],shrinkage=config.get('shrinkage')))
                    dump(cv_models/f'{method}_fold{fold}_config{idx}.json',model.to_dict())
                    records=score(model,va)
                    for row in records: row.update(fold=fold,config_index=idx,method=method)
                    scores[method][idx].extend(records)
            dump(out/'CV_SCORES_PARTIAL.json',scores)
            event('CONTEXT_CV_FOLD',fold=fold)
            for lr in protocol['ar_learning_rates']:
                def on_step(step,model):
                    idx=next(i for i,c in enumerate(grids['ar_cv']) if c==dict(lr=lr,steps=step))
                    torch.save(dict(model=model.state_dict(),config=grids['ar_cv'][idx]),cv_models/f'ar_cv_fold{fold}_config{idx}.pt')
                    records=score(model,va,True)
                    for row in records: row.update(fold=fold,config_index=idx,method='ar_cv')
                    scores['ar_cv'][idx].extend(records)
                _,trace=train_ar(tr,dict(lr=lr,steps=1000),protocol['training_seed'],on_step)
                training_traces.append(dict(fold=fold,lr=lr,trace=trace))
                dump(out/'CV_SCORES_PARTIAL.json',scores)
                dump(out/'AR_TRAINING_TRACES.json',training_traces)
                event('AR_CV_TRAJECTORY',fold=fold,lr=lr)
        summaries={m:[aggregate(s) for s in records] for m,records in scores.items()}
        selection={m:min(range(len(configs)),key=lambda i:(summaries[m][i]['equal_terminal_sequence_nll'],i)) for m,configs in grids.items()}
        dump(out/'CV_SCORES.json',scores); dump(out/'CV_SUMMARIES.json',summaries)
        dump(out/'SELECTION.json',{m:dict(index=i,config=grids[m][i],score=summaries[m][i]) for m,i in selection.items()})
        models={}; model_paths={}; traces={}
        final_trees={}
        for method,idx in selection.items():
            config=grids[method][idx]
            if method=='ar_cv':
                models[method],traces[method]=train_ar(train,config,protocol['training_seed'])
                path=out/(method+'.pt'); torch.save(dict(model=models[method].state_dict(),config=config),path)
            else:
                if method=='ngram': models[method]=eb.fit_ngram(fit_rows(train),config)
                else:
                    depth=config['max_depth']
                    if depth not in final_trees:
                        final_trees[depth]=eb.fit(fit_rows(train),dict(protocol['structure'],max_depth=depth))
                    models[method]=final_trees[depth].with_emission(method,alpha=config['alpha'],shrinkage=config.get('shrinkage'))
                path=out/(method+'.json'); dump(path,models[method].to_dict())
            model_paths[method]=dict(path=str(path),sha256=sha(path))
        models['ar_historical']=ar.load_checkpoint(protocol['historical_ar_path'])
        model_paths['ar_historical']=dict(path=protocol['historical_ar_path'],sha256=sha(protocol['historical_ar_path']))
        dump(out/'FINAL_TRAINING_TRACES.json',traces)
        dump(out/'MODELS_FROZEN_BEFORE_DEV.json',dict(models=model_paths,selected_configs={m:grids[m][i] for m,i in selection.items()},
                                                   elapsed_seconds=time.monotonic()-started,development_opened=False))
        event('MODELS_FROZEN_BEFORE_DEV')
        generations=[]
        for method,model in models.items():
            generations.extend(generation(model,method,protocol,method.startswith('ar_')))
            dump(out/'GENERATIONS.json',generations)
        event('GENERATION_COMPLETE_BEFORE_DEV',attempts=len(generations))
        dev_spec=protocol['data']['development']
        dev=ar.read_split(dev_spec['path'],role='development',expected_sha256=dev_spec['sha256'],expected_rows=41)['rows']
        ev.split_isolation(train,dev)
        dump(out/'DEVELOPMENT_SNAPSHOT.json',dev)
        event('DEVELOPMENT_OPENED_ONCE',rows=len(dev))
        dev_scores={m:score(model,dev,m.startswith('ar_')) for m,model in models.items()}
        dev_summaries={m:aggregate(s) for m,s in dev_scores.items()}
        dump(out/'DEV_SCORES.json',dev_scores); dump(out/'DEV_SUMMARIES.json',dev_summaries)
        groups={(m,t):[r for r in generations if r['method']==m and r['terminal']==t] for m in models for t in TERMINALS}
        metrics,detail=ev.evaluate_generation(groups,train,dev)
        metrics['policy']=dict(protocol['metrics_policy'],inherited_metric_formulas=ev.POLICY)
        dump(out/'GENERATION_METRICS.json',metrics); dump(out/'GENERATION_AUDIT.json',detail)
        result=decision({m:summaries[m][i] for m,i in selection.items()},dev_summaries,generations,train)
        result.update(elapsed_seconds=time.monotonic()-started,models=model_paths,test_accessed=False,backup='BACKUP_BLOCKED',
                      attempts=len(generations),natural_run_retries=0)
        dump(out/'RESULT.json',result)
        event('COMPLETE',status=result['status'])
    except BaseException as exc:
        dump(out/'FAILURE.json',dict(status='BLOCKED',type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),
                                    elapsed_seconds=time.monotonic()-started,retry_allowed=False))
        print(traceback.format_exc(),flush=True)
        raise
    finally:
        signal.alarm(0)

if __name__=='__main__':
    main()
