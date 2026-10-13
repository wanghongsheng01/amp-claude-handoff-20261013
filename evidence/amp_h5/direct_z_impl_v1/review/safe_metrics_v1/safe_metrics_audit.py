#!/usr/bin/env python3
"""Independent saved-sequence metrics audit using only Python's standard library.

No candidate/source modules are imported. Only the frozen JSON/CSV/text whitelist
is read. The runner metric JSON is parsed only after independent results are saved.
No checkpoint, pickle, NumPy archive, model, random generator or scientific retry.
"""
import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_SET = frozenset(AA)
PAIRS = tuple(a+b for a in AA for b in AA)
ARMS, TERMINALS = ('R', 'Z', 'C'), ('FREE', 'AMD')
COMPLETE_STATUSES = frozenset(('eos', 'cap_nonEOS', 'empty_sequence', 'historical_ok'))
HERE = Path(__file__).resolve().parent


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def ratio(a, b):
    return a/b if b else None


def entropy(counts):
    total = sum(counts)
    return -math.fsum((n/total)*math.log2(n/total) for n in counts if n) if total else None


def describe(sequences):
    n = len(sequences)
    lengths = [len(s) for s in sequences]
    residues, pairs, copies = Counter(), Counter(), Counter(sequences)
    feature_rows = []
    for seq in sequences:
        assert 1 <= len(seq) <= 48 and set(seq) <= AA_SET
        r, d = Counter(seq), Counter(seq[i:i+2] for i in range(len(seq)-1))
        residues.update(r); pairs.update(d)
        # Each complete block has weight1/3 in the squared Euclidean kernel.
        scale = math.sqrt(3.)
        feature_rows.append([len(seq)/(48*scale)] + [r[a]/(len(seq)*scale) for a in AA]
                            + [d[p]/(max(1,len(seq)-1)*scale) for p in PAIRS])
    mean = [math.fsum(row[j] for row in feature_rows)/n for j in range(421)] if n else None
    ordered = sorted(lengths)
    length = {'n':n, 'mean':ratio(sum(lengths),n), 'min':min(lengths) if n else None,
              'max':max(lengths) if n else None,
              'median':(ordered[(n-1)//2]+ordered[n//2])/2 if n else None}
    return {'n':n, 'unique_count':len(copies), 'unique_fraction':ratio(len(copies),n),
            'duplicate_fraction':ratio(n-len(copies),n), 'top_mode_fraction':ratio(max(copies.values(),default=0),n),
            'pair_collision_probability':ratio(sum(c*(c-1) for c in copies.values()),n*(n-1)),
            'sequence_entropy_bits':entropy(list(copies.values())),
            'pooled_aa_entropy_bits':entropy(list(residues.values())),
            'pooled_dipeptide_entropy_bits':entropy(list(pairs.values())),
            'mean_per_sequence_aa_entropy_bits':ratio(math.fsum(entropy(list(Counter(s).values())) for s in sequences),n),
            'length':length, 'length_counts':{str(k):v for k,v in sorted(Counter(lengths).items())},
            'aa_counts':{a:residues[a] for a in AA},
            'aa_frequency':{a:ratio(residues[a],sum(residues.values())) for a in AA},
            'dipeptide_counts':{p:pairs[p] for p in PAIRS},
            'dipeptide_frequency':{p:ratio(pairs[p],sum(pairs.values())) for p in PAIRS},
            'fixed_feature_mean':mean}


def js_bits(p, q):
    if any(v is None for v in list(p.values())+list(q.values())):
        return None
    terms = []
    for key in p:
        middle = (p[key]+q[key])/2
        for value in (p[key],q[key]):
            if value > 0: terms.append(value*math.log2(value/middle)/2)
    return math.fsum(terms)


def compare(sample, reference):
    result = {}
    for name in ('aa','dipeptide'):
        p,q = sample[name+'_frequency'],reference[name+'_frequency']
        result[name+'_jensen_shannon_bits'] = js_bits(p,q)
        result[name+'_l1_distance'] = math.fsum(abs(p[k]-q[k]) for k in p) if all(
            v is not None for v in list(p.values())+list(q.values())) else None
    x,y = sample['fixed_feature_mean'],reference['fixed_feature_mean']
    result['mmd2_linear_biased_fixed_features'] = math.fsum((a-b)**2 for a,b in zip(x,y)) if x is not None and y is not None else None
    p,q = sample['length_counts'],reference['length_counts']
    if sample['n'] and reference['n']:
        result['length_ks_statistic_no_pvalue'] = max(abs(
            sum(v for k,v in p.items() if int(k)<=edge)/sample['n']-
            sum(v for k,v in q.items() if int(k)<=edge)/reference['n'])
            for edge in sorted({int(k) for k in p}|{int(k) for k in q}))
    else: result['length_ks_statistic_no_pvalue'] = None
    return result


def recompute(slots, train, dev):
    trainset = {r['sequence'] for r in train}
    trainpairs = {(r['sequence'],r['terminal_state']) for r in train}
    summaries, extra, per_slot = {}, {}, []
    for arm in ARMS:
        summaries[arm],extra[arm] = {},{}
        for terminal in TERMINALS:
            rows = sorted((s for s in slots if s['arm']==arm and s['terminal']==terminal),key=lambda s:s['slot'])
            assert [s['slot'] for s in rows] == list(range(16))
            completed = [s for s in rows if isinstance(s.get('sequence'),str) and s['status'] in COMPLETE_STATUSES]
            seq = [s['sequence'] for s in completed]
            valid = [s for s in seq if 1<=len(s)<=48 and set(s)<=AA_SET]
            ref = [r['sequence'] for r in dev if r['terminal_state']==terminal]
            ss, rs = describe(valid), describe(ref)
            length_mean = ratio(sum(map(len,seq)),len(seq))
            copies = sum(s in trainset for s in seq)
            duplicate = len(seq)-len(set(seq))
            legal5 = sum(5<=len(s)<=48 and set(s)<=AA_SET for s in seq)
            summary = {'requests':16,'completed_n':len(seq),'valid_nonempty_n':len(valid),
                'main_complete':len(completed)==16 and len(valid)==16,
                'status_counts':dict(Counter(s['status'] for s in rows)),
                'empty_n':sum(s=='' for s in seq),
                'length_mean_including_completed_empty':length_mean,
                'length_mean_difference_absolute':abs(length_mean-rs['length']['mean']) if length_mean is not None else None,
                'legal5_48_n':legal5,'legal5_48_rate':legal5/16,
                'copy_train_n':copies,'copy_train_rate':copies/16,
                'copy_train_terminal_n':sum((s,terminal) in trainpairs for s in seq),
                'duplicates_n':duplicate,'duplicates_rate':duplicate/16,
                'cap_nonEOS_n':sum(s['status']=='cap_nonEOS' for s in rows),
                'cap_nonEOS_rate':sum(s['status']=='cap_nonEOS' for s in rows)/16,
                'conditional_valid_summary':ss,'conditional_valid_vs_dev':compare(ss,rs)}
            summaries[arm][terminal] = summary
            extra[arm][terminal] = {'legal1_48_n':len(valid),'legal1_48_rate_all_requested':len(valid)/16,
                'incomplete_or_failed_n':16-len(completed),
                'not_executed_n':sum(s['status']=='not_executed' for s in rows),
                'invalid_nonempty_completed_n':sum(bool(s) and not (1<=len(s)<=48 and set(s)<=AA_SET) for s in seq),
                'reference_n':len(ref),'reference_length_mean':rs['length']['mean']}
            for row in completed:
                per_slot.append({'arm':arm,'terminal':terminal,'slot':row['slot'],
                    'exact_copy_train':row['sequence'] in trainset,
                    'exact_copy_train_same_terminal':(row['sequence'],terminal) in trainpairs})
    candidate_ok = all(summaries['Z'][t]['main_complete'] for t in TERMINALS)
    controls_ok = all(summaries[a][t]['main_complete'] for a in ('R','C') for t in TERMINALS)
    comparisons, inequalities = {}, []
    for control in ('R','C'):
        comparisons[control] = []
        for terminal in TERMINALS:
            z,b = summaries['Z'][terminal],summaries[control][terminal]
            x,y = z['conditional_valid_vs_dev']['mmd2_linear_biased_fixed_features'],b['conditional_valid_vs_dev']['mmd2_linear_biased_fixed_features']
            strict = x is not None and y is not None and x<y
            relative = x is not None and y is not None and x<=.90*y
            checks = {'MMD_strict_and_at_least_10percent':strict and relative}
            rows = [{'metric':'MMD_strict','lhs':x,'operator':'<','rhs':y,'pass':strict},
                    {'metric':'MMD_10percent','lhs':x,'operator':'<=','rhs':.90*y if y is not None else None,'pass':relative}]
            x,y = z['conditional_valid_vs_dev']['dipeptide_jensen_shannon_bits'],b['conditional_valid_vs_dev']['dipeptide_jensen_shannon_bits']
            checks['JS_strict'] = x is not None and y is not None and x<y
            rows.append({'metric':'JS_strict','lhs':x,'operator':'<','rhs':y,'pass':checks['JS_strict']})
            for metric in ('length_mean_difference_absolute','copy_train_n','duplicates_rate'):
                x,y=z[metric],b[metric]
                checks[metric]=x is not None and y is not None and x<=y
                rows.append({'metric':metric,'lhs':x,'operator':'<=','rhs':y,'pass':checks[metric]})
            checks['legal5_48_rate']=z['legal5_48_rate']>=b['legal5_48_rate']
            rows.append({'metric':'legal5_48_rate','lhs':z['legal5_48_rate'],'operator':'>=',
                         'rhs':b['legal5_48_rate'],'pass':checks['legal5_48_rate']})
            comparisons[control].append({'terminal':terminal,'checks':checks,'pass':all(checks.values())})
            inequalities.append({'control':control,'terminal':terminal,'inequalities':rows})
    mechanism_pass=all(row['pass'] for row in comparisons['R'])
    value_pass=all(row['pass'] for row in comparisons['C'])
    status=('FAIL_CANDIDATE' if not candidate_ok else 'NOT_EVALUABLE' if not controls_ok else
            'FAIL_MECHANISM_SCREEN' if not mechanism_pass else 'FAIL_VALUE_SCREEN' if not value_pass else 'SCREEN_POSITIVE_HOLD')
    mechanism=('PASS' if candidate_ok and controls_ok and mechanism_pass else
               'NOT_EVALUABLE' if not controls_ok and candidate_ok else 'FAIL')
    value=('PASS' if status=='SCREEN_POSITIVE_HOLD' else 'FAIL') if candidate_ok and controls_ok and mechanism_pass else 'NOT_INTERPRETED'
    return {'status':status,'summary':summaries,'comparisons':comparisons,'mechanism_screen':mechanism,
            'value_screen':value,'additional_denominators':extra,'per_slot_copy_flags':per_slot,
            'individual_inequalities':inequalities,
            'C_comparisons_descriptive_only':not (candidate_ok and controls_ok and mechanism_pass)}


def worker(args):
    started=time.monotonic();out=Path(args.out)
    report={'status':'RUNNING','model_forward':0,'new_rng':0,'test_reads':0,'checkpoint_reads':0,
            'independent_author':'/root/ar_admission_independent','checks':[],
            'scope':'Saved-sequence metrics only; checkpoint, latent and end-to-end audit remain incomplete.'}
    plan=json.loads(Path(args.plan).read_text())
    inputs={i['role']:i for i in plan['inputs']}
    read_roles=[]
    def guarded(role, parse=True):
        item=inputs[role];p=Path(item['path']).resolve()
        assert p.suffix in ('.json','.csv','.py')
        assert not any('test' in x.lower() for x in p.parts)
        data=p.read_bytes();assert sha_bytes(data)==item['sha256'],role
        read_roles.append(role)
        if not parse:return data
        if p.suffix=='.json':return json.loads(data)
        if p.suffix=='.csv':return list(csv.DictReader(data.decode('utf-8-sig').splitlines()))
        return data.decode()
    def passed(name, detail=None):
        report['checks'].append({'check':name,'status':'PASS','detail':detail})
    try:
        assert sha_bytes(Path(args.plan).read_bytes())==args.plan_sha256
        assert sha_bytes(Path(__file__).read_bytes())==plan['audit_script_sha256']
        assert plan['outer_seconds']==30 and plan['numeric_tolerance']=={'absolute':1e-12,'relative':1e-10}
        # Hash all whitelist inputs now; do not parse runner results before recomputation.
        for role in inputs:guarded(role,False)
        config=guarded('config');train=guarded('train');dev=guarded('development')
        assert len(train)==279 and len(dev)==41
        bindingmap={b['path']:b['sha256'] for b in config['bindings']}
        for role in ('train','development','historical_slots'):
            assert inputs[role]['path']==config['inputs'][role]
            assert inputs[role]['sha256']==bindingmap[inputs[role]['path']]
        for collection,expected in ((train,{'FREE':126,'AMD':153}),(dev,{'FREE':18,'AMD':23})):
            assert dict(Counter(r['terminal_state'] for r in collection))==expected
            for row in collection:
                seq=row['sequence']
                assert 1<=len(seq)<=48 and set(seq)<=AA_SET
                assert row['sequence_sha256']==sha_bytes(seq.encode())
        slots=guarded('generation_slots');historical=guarded('historical_slots')
        keys=lambda rows:{(s['arm'],s['terminal'],s['slot']) for s in rows}
        assert len(slots)==96 and keys(slots)=={(a,t,i) for a in ARMS for t in TERMINALS for i in range(16)}
        oldc={ (s['terminal'],s['slot']):s for s in historical if s['arm']=='C'}
        newc={ (s['terminal'],s['slot']):s for s in slots if s['arm']=='C'}
        assert len(oldc)==len(newc)==32 and oldc==newc,'C full JSON row drift'
        passed('exact_whitelist_hashes_train_dev_identity_96_slots_and_C32_full_row_equality')
        independent=recompute(slots,train,dev)
        ownpath=out/'INDEPENDENT_METRICS.json';write_json(ownpath,independent)
        ownsha=sha_bytes(ownpath.read_bytes())
        passed('independent_recalculation_saved_before_runner_result_parse',{'sha256':ownsha})
        # Only now expose runner metrics for item-by-item comparison.
        runner=guarded('runner_metrics')
        errors=[];comparisons=[];max_abs=0.;max_scaled=0.
        def compare_value(x,y,path):
            nonlocal max_abs,max_scaled
            if isinstance(x,dict):
                if not isinstance(y,dict) or set(x)!=set(y):errors.append({'path':path,'reason':'dictionary keys/type mismatch'});return
                for k in x:compare_value(x[k],y[k],path+'/'+str(k))
            elif isinstance(x,list):
                if not isinstance(y,list) or len(x)!=len(y):errors.append({'path':path,'reason':'list length/type mismatch'});return
                for i,(a,b) in enumerate(zip(x,y)):compare_value(a,b,path+'/'+str(i))
            elif isinstance(x,bool) or x is None or isinstance(x,(str,int)):
                ok=type(x)==type(y) and x==y
                comparisons.append({'path':path,'equal':ok,'comparison':'exact'})
                if not ok:errors.append({'path':path,'independent':x,'runner':y})
            else:
                numeric=isinstance(y,(float,int)) and not isinstance(y,bool)
                delta=abs(x-y) if numeric else None
                tolerance=1e-12+1e-10*max(abs(x),abs(y)) if numeric else None
                ok=numeric and math.isfinite(x) and math.isfinite(y) and delta<=tolerance
                if numeric:
                    max_abs=max(max_abs,delta);max_scaled=max(max_scaled,delta/tolerance)
                comparisons.append({'path':path,'equal':ok,'comparison':'numeric','absolute_error':delta,'tolerance':tolerance})
                if not ok:errors.append({'path':path,'independent':x,'runner':y})
        for key in ('status','summary','comparisons','mechanism_screen','value_screen'):
            compare_value(independent[key],runner[key],'/'+key)
        # Separate signed inequalities use exact Boolean agreement, no tolerance relaxation.
        for row in independent['per_slot_copy_flags']:
            saved=next(s for s in slots if (s['arm'],s['terminal'],s['slot'])==(row['arm'],row['terminal'],row['slot']))
            for key in ('exact_copy_train','exact_copy_train_same_terminal'):
                compare_value(row[key],saved[key],'/slot_copy/'+row['arm']+'/'+row['terminal']+'/'+str(row['slot'])+'/'+key)
        assert runner['slots']==slots,'Runner embedded raw slots differ from saved generation JSON'
        write_json(out/'COMPARISON_ITEMS.json',{'comparisons':comparisons,'errors':errors,
            'maximum_absolute_numeric_error':max_abs,'maximum_fraction_of_allowed_tolerance':max_scaled})
        assert not errors, 'Independent versus runner comparison failed'
        passed('all_summary_features_metrics_denominators_copy_flags_and_frozen_gates_agree',
               {'scalar_checks':len(comparisons),'maximum_absolute_numeric_error':max_abs})
        oldaudit=guarded('old_audit');oldproc=guarded('old_audit_process')
        assert oldaudit['status']=='AUDIT_FAILED' and oldproc['status']=='FAILED'
        assert 'UnpicklingError' in oldaudit['error']
        for role in inputs:guarded(role,False)
        core=[]
        for arm in ARMS:
            for terminal in TERMINALS:
                s=independent['summary'][arm][terminal];e=independent['additional_denominators'][arm][terminal]
                core.append({'arm':arm,'terminal':terminal,'requests':16,'completed_n':s['completed_n'],
                    'valid1_48_n':s['valid_nonempty_n'],'legal1_48_rate':e['legal1_48_rate_all_requested'],
                    'legal5_48_n':s['legal5_48_n'],'legal5_48_rate':s['legal5_48_rate'],
                    'empty_n':s['empty_n'],'failed_or_incomplete_n':e['incomplete_or_failed_n'],
                    'cap_nonEOS_n':s['cap_nonEOS_n'],'copy_train_n':s['copy_train_n'],
                    'copy_train_terminal_n':s['copy_train_terminal_n'],'duplicates_n':s['duplicates_n'],
                    'length_mean':s['length_mean_including_completed_empty'],
                    'reference_n':e['reference_n'],'reference_length_mean':e['reference_length_mean'],
                    'length_gap_abs':s['length_mean_difference_absolute'],
                    'mmd2':s['conditional_valid_vs_dev']['mmd2_linear_biased_fixed_features'],
                    'dipeptide_JS_bits':s['conditional_valid_vs_dev']['dipeptide_jensen_shannon_bits']})
        report.update(status='PASS_SAVED_SEQUENCE_METRICS',scientific_outcome=independent['status'],
                      mechanism_screen=independent['mechanism_screen'],value_screen=independent['value_screen'],
                      all_six_groups=core,individual_inequalities=independent['individual_inequalities'],
                      scalar_comparisons=len(comparisons),maximum_absolute_numeric_error=max_abs,
                      maximum_fraction_of_allowed_tolerance=max_scaled,
                      independent_metrics_sha256=ownsha,checkpoint_audit='INCOMPLETE_ORIGINAL_FAILURE_RETAINED',
                      original_postflight_status='AUDIT_FAILED',original_postflight_wall_seconds=oldproc['wall_seconds'],
                      promotion='HOLD',new_training_or_generation_authorized=False)
    except Exception as exc:
        report.update(status='FAILED_SAFE_METRICS_AUDIT',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        report['worker_wall_seconds']=time.monotonic()-started
        report['read_roles']=sorted(set(read_roles))
        report['plan_sha256']=args.plan_sha256
        report['audit_script_sha256']=sha_bytes(Path(__file__).read_bytes())
        write_json(out/'AUDIT_RESULT.json',report)
    return 0 if report['status']=='PASS_SAVED_SEQUENCE_METRICS' else 2


def main():
    start=time.monotonic()
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--out',required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:return worker(args)
    planpath=Path(args.plan).resolve();out=Path(args.out).resolve()
    assert planpath.parent==HERE and out==HERE/'run_v1'
    raw=planpath.read_bytes();assert sha_bytes(raw)==args.plan_sha256
    plan=json.loads(raw);assert plan['outer_seconds']==30
    assert plan['audit_script_sha256']==sha_bytes(Path(__file__).read_bytes())
    with (HERE/'ATTEMPT_v1.json').open('x') as f:
        json.dump({'plan_sha256':args.plan_sha256,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'scope':'safe_text_saved_sequence_metrics_only','retry_allowed':False},f)
    out.mkdir(exist_ok=False)
    argv=[sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    timeout=False;rc=None
    with (out/'console.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=child.wait(timeout=max(0.,30-(time.monotonic()-start)))
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
    wall=time.monotonic()-start
    process={'status':'PASS' if rc==0 and not timeout and wall<=30 else 'FAILED','exit_code':rc,
        'timed_out':timeout,'wall_seconds':wall,'outer_cap_seconds':30,'overrun_seconds':max(0.,wall-30),
        'plan_sha256':args.plan_sha256,'audit_script_sha256':sha_bytes(Path(__file__).read_bytes()),
        'model_forward':0,'new_rng':0,'checkpoint_reads':0,'test_reads':0,
        'wall_scope':'Main entry through child exit; final receipt serialization excluded, no computation grace.'}
    write_json(out/'PROCESS_RESULT.json',process)
    return 0 if process['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
