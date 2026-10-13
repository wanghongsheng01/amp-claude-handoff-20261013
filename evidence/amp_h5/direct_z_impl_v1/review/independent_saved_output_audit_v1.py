#!/usr/bin/env python3
"""One <=60s saved-output audit; never instantiates a model or draws RNG.

Outer invocation supervises only its own worker. On engineering failure the
audit reports a blocked natural run and all planned, unlaunched denominators.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def canonical(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def audit(args):
    start = time.monotonic()
    out = Path(args.out)
    report = {'status': 'RUNNING', 'model_forward': 0, 'new_rng': 0, 'checks': [],
              'natural_execution_authorized': False, 'script_sha256': sha(__file__)}
    def check():
        if time.monotonic() - start >= 60:
            raise TimeoutError('Saved-output audit60s cap')
    def passed(name, detail=None):
        check(); report['checks'].append({'check': name, 'status': 'PASS', 'detail': detail})
    try:
        engineering = read(args.engineering_process)
        ereport = read(args.engineering_worker) if Path(args.engineering_worker).exists() else None
        report['engineering_process_sha256'] = sha(args.engineering_process)
        report['engineering_wall_seconds'] = engineering['wall_seconds']
        assert engineering['wall_seconds'] > 0
        if engineering['status'] != 'PASS' or ereport is None or ereport['status'] != 'PASS':
            assert not args.natural_manifest
            report.update(status='PASS_AUDIT_ENGINEERING_BLOCKED', natural_started=False,
                          engineering_failure=engineering,
                          planned_slots={'R':32,'Z':32,'C_historical_reuse':32},
                          newly_generated=0, completed_natural_updates=0,
                          outcome='BLOCKED_NO_NATURAL_ATTEMPT', all64_new_slots='not_launched')
            return report
        assert engineering['wall_seconds'] <= 180 and engineering['exit_code'] == 0
        assert engineering['timed_out'] is False and ereport['actual_model_forward'] <= 132
        assert ereport['natural_forward'] == 0
        assert args.natural_manifest and args.receipt
        manifest, receipt = read(args.natural_manifest), read(args.receipt)
        assert sha(args.natural_manifest) == receipt['manifest_sha256']
        assert sha(args.engineering_process) == manifest['engineering_result_sha256']
        for entry in manifest['bindings']:
            assert sha(entry['path']) == entry['sha256'], entry['path']
        result = read(Path(manifest['execution']) / 'RESULT.json')
        root = Path(manifest['output'])
        report['natural_process'] = result
        report['natural_started'] = result.get('child_pid') is not None
        assert result['wall_seconds'] + engineering['wall_seconds'] + 60 <= 4800
        slots_path = root / 'GENERATION_SLOTS.json'
        slots = read(slots_path) if slots_path.exists() else read(Path(manifest['execution']) / 'REQUESTED_SLOTS.json')
        assert len(slots) == 96
        expected = {(a,t,i) for a in ('R','Z','C') for t in ('FREE','AMD') for i in range(16)}
        assert {(s['arm'],s['terminal'],s['slot']) for s in slots} == expected
        passed('exact_input_bindings_and_complete96_slot_denominator')
        if result['state'] != 'completed':
            report.update(status='PASS_AUDIT_NATURAL_FAILURE_RETAINED', outcome='STOP_NO_RETRY',
                          slot_statuses={a:{str(s.get('status')):sum(r['arm']==a and r.get('status')==s.get('status') for r in slots)
                                            for s in slots if s['arm']==a} for a in ('R','Z','C')})
            return report
        assert result['exit_code'] == 0 and result['timed_out'] is False and result['wall_seconds'] <= 4560
        summary, counts = read(root/'SUMMARY.json'), read(root/'COUNTS.json')
        assert summary['status'] == 'COMPLETE'
        c = counts['counts']
        assert counts['updates_by_arm'] == {'R':1000,'Z':1000}
        assert c['updates'] == c['denoiser_train_forward'] == 2000
        assert c['denoiser_generation_forward'] == 3992 and 0 < c['A_decoder_forward'] <= 98
        assert c['new_requests'] == 64
        assert c['encoder_forward'] == c['old_DDPM_forward'] == c['C_forward'] == 0
        assert c['denoiser_train_forward'] + c['denoiser_generation_forward'] + c['A_decoder_forward'] <= 6090
        timings = read(root/'PHASE_TIMINGS.json')
        caps = {'prepare':60,'R_training':720,'Z_training':720,'R_generation':1440,'Z_generation':1440,'decode_score':120,'closure':60}
        assert set(timings) == set(caps)
        for phase, cap in caps.items():
            assert timings[phase]['cap_seconds'] == cap and timings[phase]['wall_seconds'] <= cap
            assert timings[phase]['exceeded'] is False
        assert read(root/'PHASE.json')['phase'] == 'completed'
        for entry in read(root/'MANIFEST.json')['files']:
            assert sha(entry['path']) == entry['sha256'], entry['path']
        report['final_control_bindings'] = {str(root/n):sha(root/n) for n in ('PHASE.json','PHASE_TIMINGS.json','MANIFEST.json')}
        passed('actual_phase_caps_counts_and_saved_artifact_hashes', c)

        # Tensor files are read only. No candidate mechanics or architecture import.
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        rng_before = torch.get_rng_state().clone()
        def td(t):
            a=t.detach().cpu().contiguous()
            h=hashlib.sha256(str(a.dtype).encode()+b'\0'+str(tuple(a.shape)).encode()+b'\0')
            h.update(a.numpy().tobytes()); return h.hexdigest()
        def sd(state):
            h=hashlib.sha256()
            for key,value in sorted(state.items()):h.update(key.encode()+b'\0'+td(value).encode())
            return h.hexdigest()
        prepared = read(root/'DETERMINISTIC_PREPARATION.json')
        assert prepared == receipt['deterministic_preparation'] == ereport['deterministic_preparation']
        initial = {a:torch.load(root/(a+'_initial.pt'),map_location='cpu',weights_only=True) for a in ('R','Z')}
        for key,z in initial['Z'].items():
            r=initial['R'][key]
            expected_tensor=(r[:,:1]*math.sqrt(2) if key=='inc.double_conv.0.weight'
                             else r[:1] if key in ('outc.weight','outc.bias') else r)
            assert torch.equal(z,expected_tensor),key
        for a in ('R','Z'):assert sd(initial[a]) == prepared[a+'_initial_sha256']
        del initial
        seal=read(root/'RANDOM_SEAL.json')
        assert canonical(seal)==prepared['random_seal_sha256']
        schedule=read(root/'SCHEDULER_SEQUENCE.json')
        assert hashlib.sha256(json.dumps(schedule,separators=(',',':')).encode()).hexdigest()==prepared['scheduler_sequence_sha256']
        for arm in ('R','Z'):
            rows=[json.loads(line) for line in (root/arm/'training_steps.jsonl').read_text().splitlines()]
            assert len(rows)==1000 and sum(len(r['ids']) for r in rows)==7972
            for i,row in enumerate(rows):
                assert row['step']==i+1
                for key,value in seal['training'][arm][i].items():assert row[key]==value
                assert row['elements']==len(row['ids'])*(2 if arm=='R' else 1)*16*48
                assert row['lr_used']==schedule[i]['lr'] and row['beta1_used']==schedule[i]['beta1']
                assert row['gradients_finite'] and row['parameters_finite'] and math.isfinite(row['mse'])
            ckpt=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=True)
            assert ckpt['updates']==ckpt['logged_complete_steps']==1000
            assert sd(ckpt['model'])==ckpt['state_sha256']==ckpt['ema_state_sha256']
            for k,state in ckpt['rng_states'].items():
                assert td(state)==seal['generator_states_sha256'][f'{arm}_train_final_{k}']
            del ckpt
            for terminal in ('FREE','AMD'):
                reverse=[json.loads(line) for line in (root/arm/('reverse_'+terminal+'.jsonl')).read_text().splitlines()]
                assert [r['t'] for r in reverse]==list(range(998,0,-1))
                for row in reverse:
                    assert row['finite'] is True
                    assert row['full_noise_sha256']==(seal['generation'][terminal]['reverse_noise_sha256'][str(row['t'])] if row['t']>1 else None)
                with np.load(root/arm/('generated_latent_'+terminal+'.npz'),allow_pickle=False) as data:
                    u,target,z=data['u'],data['target'],data['decoder_z']
                    assert all(np.isfinite(x).all() for x in (u,target,z)) and z.shape==(16,16,48)
                    stats=prepared['normalization']
                    if arm=='R':
                        expected_target=np.asarray(stats['R_mean'],dtype=np.float32)[None,:,None,None]+np.asarray(stats['R_std'],dtype=np.float32)[None,:,None,None]*u
                        sigma=data['sigma'];eps=data['reparameterization_epsilon']
                        assert np.isfinite(sigma).all() and (sigma>0).all()
                        assert td(torch.from_numpy(eps))==seal['reparameterization'][terminal]
                        assert np.allclose(sigma,np.exp(target[:,1]),rtol=2e-6,atol=0)
                        assert np.allclose(z,target[:,0]+sigma*eps,rtol=2e-6,atol=2e-6)
                    else:
                        expected_target=stats['Z_mean']+stats['Z_std']*u
                        assert np.array_equal(z,target[:,0])
                    assert np.allclose(target,expected_target,rtol=2e-6,atol=2e-6)
        passed('saved_initialization_private_seal_training_and_reverse_replay_contract')

        config=read(manifest['config'])
        with np.load(config['inputs']['uniform_bank'],allow_pickle=False) as data:uniforms=data['uniforms']
        aa='ARNDCEQGHILKMFPSTWYV'
        decoder_calls=0
        for arm in ('R','Z'):
            with np.load(root/(arm+'_decoder_evidence.npz'),allow_pickle=False) as d:
                assert np.array_equal(d['uniforms'],uniforms)
                executed=d['executed_positions'];steps=int(executed.sum());decoder_calls+=steps
                assert np.array_equal(executed,np.arange(49)<steps)
                probs=d['probabilities'][:,:steps];cdf=torch.from_numpy(probs).cumsum(-1).numpy();cdf[:,:,-1]=1
                drawn=(uniforms[:,:steps,None]>=cdf).sum(-1)
                assert np.array_equal(drawn,d['raw_tokens'][:,:steps])
                for i in range(32):
                    token=d['tokens'][i,:steps].tolist();end=token.index(20) if 20 in token else steps
                    sequence=''.join(aa[x] for x in token[:min(end,48)])
                    slot=next(s for s in slots if s['arm']==arm and s['terminal']==('FREE' if i<16 else 'AMD') and s['slot']==i%16)
                    assert slot['sequence']==sequence
        assert decoder_calls==c['A_decoder_forward']
        historic=read(config['inputs']['historical_slots'])
        for old in (s for s in historic if s['arm']=='C'):
            new=next(s for s in slots if (s['arm'],s['terminal'],s['slot'])==(old['arm'],old['terminal'],old['slot']))
            assert all(new[k]==old[k] for k in ('sequence','status'))
        passed('fixed_uniform_inverse_CDF_raw_outputs_and_C_reuse')

        # Independent formulas; load only the frozen feature definition.
        fp=Path(manifest['config']).parent.parent/'implementation/frozen_metrics_v1.py'
        spec=importlib.util.spec_from_file_location('postflight_frozen_metrics',fp)
        frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        def csv_rows(path):
            with Path(path).open(newline='') as handle:return list(csv.DictReader(handle))
        train,dev=csv_rows(config['inputs']['train']),csv_rows(config['inputs']['development'])
        trainset={r['sequence'] for r in train}
        saved=read(root/'METRICS.json');metrics={}
        for arm in ('R','Z','C'):
            metrics[arm]={}
            for terminal in ('FREE','AMD'):
                seq=[s['sequence'] for s in slots if s['arm']==arm and s['terminal']==terminal]
                complete=all(isinstance(x,str) and 1<=len(x)<=48 and set(x)<=set(aa) for x in seq)
                if not complete:
                    metrics[arm][terminal]={'complete':False};continue
                ref=[r['sequence'] for r in dev if r['terminal_state']==terminal]
                xs=np.asarray([frozen.feature_vector(s) for s in seq],dtype=np.float64)
                ys=np.asarray([frozen.feature_vector(s) for s in ref],dtype=np.float64)
                mmd=float(np.square(xs.mean(0)-ys.mean(0)).sum())
                def dip(strings):
                    v=np.zeros(400,dtype=np.float64)
                    for s in strings:
                        for a,b in zip(s,s[1:]):v[aa.index(a)*20+aa.index(b)]+=1
                    return v/v.sum() if v.sum() else None
                p,q=dip(seq),dip(ref)
                if p is None or q is None:js=None
                else:
                    mid=(p+q)/2
                    js=.5*sum(float((v[v>0]*np.log2(v[v>0]/mid[v>0])).sum()) for v in (p,q))
                values={'complete':True,'mmd':mmd,'js':js,
                        'length_gap':abs(np.mean(list(map(len,seq)))-np.mean(list(map(len,ref)))),
                        'legal':sum(5<=len(s)<=48 for s in seq)/16,
                        'copy':sum(s in trainset for s in seq),'dup':(16-len(set(seq)))/16}
                metrics[arm][terminal]=values
                old=saved['summary'][arm][terminal]
                for observed,expected_value in [(old['conditional_valid_vs_dev']['mmd2_linear_biased_fixed_features'],mmd),
                     (old['conditional_valid_vs_dev']['dipeptide_jensen_shannon_bits'],js),
                     (old['length_mean_difference_absolute'],values['length_gap']),
                     (old['legal5_48_rate'],values['legal']),(old['copy_train_n'],values['copy']),(old['duplicates_rate'],values['dup'])]:
                    assert observed is None if expected_value is None else math.isclose(observed,expected_value,rel_tol=1e-9,abs_tol=1e-12)
        candidate_ok=all(metrics['Z'][t]['complete'] for t in ('FREE','AMD'))
        controls_ok=all(metrics[a][t]['complete'] for a in ('R','C') for t in ('FREE','AMD'))
        comparisons=[]
        for control in ('R','C'):
            for terminal in ('FREE','AMD'):
                z,b=metrics['Z'][terminal],metrics[control][terminal]
                checks={} if not z['complete'] or not b['complete'] else {
                    'MMD_strict_and_at_least_10percent':z['mmd']<b['mmd'] and z['mmd']<=.9*b['mmd'],
                    'JS_strict':z['js'] is not None and b['js'] is not None and z['js']<b['js'],'length_mean_difference_absolute':z['length_gap']<=b['length_gap'],
                    'copy_train_n':z['copy']<=b['copy'],'duplicates_rate':z['dup']<=b['dup'],
                    'legal5_48_rate':z['legal']>=b['legal']}
                comparisons.append({'control':control,'terminal':terminal,'checks':checks,'pass':bool(checks) and all(checks.values())})
        mechanism=all(x['pass'] for x in comparisons if x['control']=='R')
        value=all(x['pass'] for x in comparisons if x['control']=='C')
        status=('FAIL_CANDIDATE' if not candidate_ok else 'NOT_EVALUABLE' if not controls_ok else
                'FAIL_MECHANISM_SCREEN' if not mechanism else 'FAIL_VALUE_SCREEN' if not value else 'SCREEN_POSITIVE_HOLD')
        assert saved['status']==summary['screen_status']==status
        assert torch.equal(rng_before,torch.get_rng_state())
        report.update(status='PASS_AUDIT', outcome=status, metrics=metrics, comparisons=comparisons,
                      natural_counts=c, phase_timings=timings, scientific_retry_allowed=False)
        passed('independent_fixed_formula_and_two_gate_recalculation')
    except Exception as exc:
        report.update(status='AUDIT_FAILED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        report['worker_wall_seconds']=time.monotonic()-start
        dump(out/'AUDIT.json',report)
    return report


def main():
    start=time.monotonic()
    p=argparse.ArgumentParser()
    p.add_argument('--engineering-process',required=True);p.add_argument('--engineering-worker',required=True)
    p.add_argument('--natural-manifest');p.add_argument('--receipt');p.add_argument('--out',required=True)
    p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:
        return 0 if audit(args)['status'].startswith('PASS_') else 2
    out=Path(args.out).resolve()
    allowed=Path(__file__).resolve().parent
    assert allowed in out.parents
    out.mkdir(parents=False,exist_ok=False)
    argv=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    env=dict(os.environ)
    env.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    env['PYTHONDONTWRITEBYTECODE']='1'
    timed_out=False
    with (out/'console.log').open('x') as log:
        process=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=process.wait(timeout=max(0.,60-(time.monotonic()-start)))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGKILL);rc=process.wait()
    wall=time.monotonic()-start
    dump(out/'PROCESS_RESULT.json',{'status':'PASS' if rc==0 and not timed_out and wall<=60 else 'FAILED',
         'exit_code':rc,'timed_out':timed_out,'wall_seconds':wall,'outer_cap_seconds':60,
         'overrun_seconds':max(0.,wall-60),'model_forward':0,'new_rng':0,'script_sha256':sha(__file__)})
    return 0 if rc==0 and not timed_out and wall<=60 else 2


if __name__=='__main__':
    raise SystemExit(main())
