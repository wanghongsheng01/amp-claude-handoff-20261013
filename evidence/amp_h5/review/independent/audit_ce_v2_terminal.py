#!/usr/bin/env python3
"""Read-only terminal artifact audit; never imports a model or calls a forward.
Prepared while CE is running. Execute only after parent confirms terminal state.
Writes new reports exclusively; will not overwrite prior evidence.
"""
from pathlib import Path
import argparse, collections, csv, datetime, hashlib, json, math, os, pickle, random, types
W = Path('/Users/loctek/Documents/Codex/2026-10-11/task-4')
H = W / 'amp_h5'
B = H / 'runs/formal_v2/categorical_ce'
O = H / 'review/independent'
PLAN_SHA = 'ce193e133e86d995c21e4596412ce08ee61bb2be3cb47b2549d4f11f5c86f2d6'
RUNNER_SHA = 'e97d2a766774461793cba1860455e4cb4c8d689656b60becc326a9f11297ca9b'
CONFIG_SHA = 'd44631a12bffc0ac85eecc2ca6d361419a03e4bf6cbfc2479092bd53b52be467'
AA = 'ARNDCEQGHILKMFPSTWYV'
SUCCESS = {'ok', 'empty_sequence', 'no_eos_at_limit'}
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    return json.loads(Path(p).read_text())
def lines(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]
def numeric_finite(x):
    if isinstance(x, (int, float)): return math.isfinite(x)
    if isinstance(x, dict): return all(numeric_finite(v) for v in x.values())
    if isinstance(x, (list, tuple)): return all(numeric_finite(v) for v in x)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terminal-confirmed', action='store_true', required=True)
    args = parser.parse_args()
    dest = O / 'CE_V2_TERMINAL_AUDIT.json'
    if dest.exists(): raise FileExistsError('New audit report already exists; preserve it.')
    supervisor = read(H / 'runs/single_ce_execution_v2/SUPERVISOR_STATUS.json')
    status = read(B / 'status.json')
    if supervisor['state'] == 'running' or status['state'] == 'running':
        raise RuntimeError('Not terminal. No saved-model or live-log audit performed.')
    if any(j.get('state') == 'running' for j in supervisor.get('jobs', [])):
        raise RuntimeError('Supervisor job still running.')
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[k] = '1'
    global np, torch
    import numpy as np
    import torch
    torch.set_num_threads(1)
    checks, section_errors = [], []
    facts = {'recorded_status': status, 'supervisor': supervisor}
    def ck(name, condition, detail=None):
        checks.append({'name': name, 'pass': bool(condition), **({'detail':detail} if detail is not None else {})})
    def section(name, fn):
        try: fn()
        except Exception as exc:
            ck(name + '_section_completed', False, repr(exc)); section_errors.append({'section':name,'error':repr(exc)})
    def tensor_digest(state):
        h = hashlib.sha256()
        for name, t in sorted(state.items()):
            a=t.detach().cpu().contiguous().numpy()
            h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        return h.hexdigest()
    def tensors(x):
        if torch.is_tensor(x): yield x
        elif isinstance(x, dict):
            for v in x.values(): yield from tensors(v)
        elif isinstance(x, (list,tuple)):
            for v in x: yield from tensors(v)
    def summary(x):
        a=np.asarray(x,dtype=np.float64); f=a[np.isfinite(a)]
        return {'shape':list(a.shape),'elements':int(a.size),'finite_elements':int(f.size),
                'nan':int(np.isnan(a).sum()),'positive_inf':int(np.isposinf(a).sum()),'negative_inf':int(np.isneginf(a).sum()),
                'min':float(f.min()) if f.size else None,'max':float(f.max()) if f.size else None,
                'mean':float(f.mean()) if f.size else None,'std':float(f.std()) if f.size else None,
                'rms':float(np.sqrt(np.mean(f*f))) if f.size else None}
    cfg = read(H/'protocol/formal_configs_v2/categorical_ce.json')
    plan = read(H/'protocol/SINGLE_CE_V2_PLAN.json')
    def bindings():
        ck('exact_plan_runner_config_hashes', sha(H/'protocol/SINGLE_CE_V2_PLAN.json')==PLAN_SHA and
           sha(H/'implementation/vae_pipeline_formal_v2.py')==RUNNER_SHA and
           sha(H/'protocol/formal_configs_v2/categorical_ce.json')==CONFIG_SHA)
        ck('all_plan_bindings_unchanged', all(sha(e['path'])==e['sha256'] for e in plan['bindings']))
        ck('output_config_value_identical', read(B/'config.json')==cfg)
        ck('recorded_hashes_bound',status['config_sha256']==CONFIG_SHA and status['runner_sha256']==RUNNER_SHA)
        ck('exact_single_supervisor_job', supervisor['plan_sha256']==PLAN_SHA and len(supervisor['jobs'])==1 and
           supervisor['jobs'][0]['argv']==plan['jobs'][0]['argv'])
        ck('supervisor_success',supervisor['state']=='completed' and supervisor['jobs'][0].get('exit_code')==0)
        files=read(B/'manifest.json')['files']
        ck('manifest_relative_direct_files',all(Path(e['path']).name==e['path'] for e in files))
        ck('complete_manifest_inventory', {e['path'] for e in files}=={p.name for p in B.iterdir() if p.is_file() and p.name!='manifest.json'})
        ck('manifest_size_and_hashes',all((B/e['path']).stat().st_size==e['bytes'] and sha(B/e['path'])==e['sha256'] for e in files))
        runtime=read(B/'runtime_and_source.json')
        ck('cpu_singlethread_source_bound',runtime['device']=='cpu' and runtime['threads']==1 and runtime['interop_threads']==1 and
           all(sha(e['path'])==e['sha256'] for e in runtime['source_files']))
        facts['manifest_files']=len(files); facts['runtime']=runtime
    section('bindings',bindings)
    def data_identity():
        allowed=[cfg[k] for k in ('train','train_cohort','dev','dev_cohort')]
        ck('known_train_dev_hashes', all(sha(cfg[k])==cfg[k+'_sha256'] for k in ('train','train_cohort','dev','dev_cohort')))
        ck('only_bound_train_dev_declared', all('test' not in p.lower() for p in allowed) and cfg['train_rows']==279 and cfg['dev_rows']==41 and
           cfg['train_idx_start']==0 and cfg['dev_idx_start']==279 and status['test_read'] is False and
           all(status[k] is True for k in ('train_read_attempted','train_read_validated','development_read','development_read_attempted','development_read_validated')))
        # Two exact hash-bound cohort files only, never descriptor/test content.
        for k in ('train_cohort','dev_cohort'):
            if sha(cfg[k])!=cfg[k+'_sha256']: raise ValueError('Refuse unbound cohort read')
        cohorts={role:list(csv.DictReader(open(cfg[role+'_cohort'],newline=''))) for role in ('train','dev')}
        ck('cohort_counts_disjoint',len(cohorts['train'])==279 and len(cohorts['dev'])==41 and not
           ({r['sequence'] for r in cohorts['train']} & {r['sequence'] for r in cohorts['dev']}))
        facts['cohort_read_paths']=[cfg['train_cohort'],cfg['dev_cohort']]
        for role in ('train','dev'):
            rows=list(csv.DictReader((B/(role+'_reconstruction.csv')).open(newline='')))
            ck(role+'_export_identity_and_rendering',len(rows)==len(cohorts[role]) and all(
                int(r['row_id'])==i and r['truth_sequence']==c['sequence'] and r['terminal']==c['terminal_state'] and r['status']=='ok' and
                len(r['prediction_sequence'])<=48 and all(a in AA for a in r['prediction_sequence']) and
                json.loads(r['prediction_tokens48'])==[AA.index(a) for a in r['prediction_sequence']]+[20]*(48-len(r['prediction_sequence'])) and
                len(json.loads(r['raw_prediction_tokens48']))==48 and all(type(v) is int and 0<=v<=20 for v in json.loads(r['raw_prediction_tokens48']))
                for i,(r,c) in enumerate(zip(rows,cohorts[role]))))
        scaler=read(B/'train_feature_scaler.json')
        ck('scaler_train_only_finite',scaler['fit_scope']=='train only' and scaler['fit_record_ids']==list(range(279)) and
           scaler['global']['n']==279 and numeric_finite(scaler) and all(x>0 for typ in ('global','residue') for x in scaler[typ]['scale']))
    section('data_identity',data_identity)
    def steps_and_caps():
        logs={stage:lines(B/(stage+'_train.jsonl')) for stage in ('vae','ddpm')}
        summaries={stage:read(B/(stage+'_training_summary.json')) for stage in ('vae','ddpm')}
        ck('recorded_complete1000_1000',status['state']=='completed' and status.get('vae_updates')==1000 and status.get('ddpm_updates')==1000 and
           all(s['updates']==s['requested_updates']==1000 and s['fixed_endpoint_reached'] for s in summaries.values()))
        for stage, rows in logs.items():
            ck(stage+'_contiguous1000_finite', [r['step'] for r in rows]==list(range(1,1001)) and numeric_finite(rows))
            cap=cfg[stage+'_seconds_cap']
            ck(stage+'_update_start_cap_respected',all(r['elapsed_stage_seconds']-r['duration_seconds'] < cap for r in rows))
        # Local Python shuffle reproduces original VAE batches, without tensors/model calls.
        rng=random.Random(222); order=[]; cursor=279; batch_ok=[]
        for r in logs['vae']:
            if cursor>=279: order=list(range(279)); rng.shuffle(order); cursor=0
            ids=order[cursor:cursor+32]; cursor+=len(ids)
            batch_ok.append(r['record_indices']==ids and r['rows']==len(ids))
        ck('vae_seed222_all_batches',all(batch_ok))
        ck('vae_warmup_exact',all(r['lr']==max(.0005*max(0,r['step']-2)/1000,1e-7) for r in logs['vae']))
        ck('vae_objective_accounting',all(abs(r['loss']-(r['reconstruction']+r['kl']))<=2e-6*max(1,abs(r['reconstruction'])+abs(r['kl'])) and
           math.isclose(r['kl'],r['density_sum']/(r['rows']*48),rel_tol=2e-6,abs_tol=2e-6) for r in logs['vae']))
        old=lines(H/'runs/smoke_v1/categorical_ce/vae_train.jsonl')
        ck('vae_first20_seed_batch_forward_rng_match_smoke',len(old)==20 and all(all(a[k]==b[k] for k in
           ('step','record_indices','rows','forward_rng_sha256')) for a,b in zip(logs['vae'][:20],old)))
        prior=lines(H/'feasibility/ddpm_maturity_v1/ddpm_train.jsonl')
        ck('ddpm_all1000_batch_noise_drop_lr_match_fixed1000_diagnostic',len(prior)==len(logs['ddpm'])==1000 and
           all(all(a[k]==b[k] for k in ('step','record_indices','rows','label_dropped','noise_timestep_sha256','lr')) for a,b in zip(logs['ddpm'],prior)))
        mon=lines(B/'dev_monitor.jsonl')
        ck('ten_prelocked_rng_preserving_monitors',[m['step'] for m in mon]==list(range(100,1001,100)) and all(
           m['rows']==41 and m['rng_preserved'] and m['checkpoint_selection'] is False and len(m['rng_sha256'])==64 for m in mon) and numeric_finite(mon))
        gen=read(B/'generation_summary.json')
        ck('shared_training_generation_outer_caps', status.get('training_cap_exceeded') is False and status['training_seconds']<=2940 and
           gen['elapsed_seconds']<=1800 and supervisor['jobs'][0]['seconds']<=4920)
        facts['budgets']={stage:summaries[stage]['elapsed_seconds'] for stage in ('vae','ddpm')}
        facts['budgets'].update(training=status.get('training_seconds'),generation=gen['elapsed_seconds'],process=status['process_seconds'],outer=supervisor['jobs'][0]['seconds'])
        facts['monitors']={'count':len(mon),'used_for_selection':False,'metrics_not_used_for_engineering_gate':True}
        facts['updates']={stage:len(rows) for stage,rows in logs.items()}
    section('steps_and_caps',steps_and_caps)
    def model_states():
        states={}
        for name in ('vae_initial_state','vae_state','ddpm_state','ddpm_ema_state'):
            p=B/(name+'.pt'); meta=read(B/(name+'.json'))
            st=torch.load(p,map_location='cpu',weights_only=True); states[name]=st
            ck(name+'_hash_finite',sha(p)==meta['sha256'] and tensor_digest(st)==meta['tensor_state_sha256'] and
               all(bool(torch.isfinite(t).all()) for t in tensors(st)))
        ck('fresh_vae_initialization_matches_prelocked_smoke',tensor_digest(states['vae_initial_state'])==read(H/'runs/smoke_v1/categorical_ce/vae_initialization.json')['tensor_state_sha256'])
        ck('vae_weights_changed',tensor_digest(states['vae_initial_state'])!=tensor_digest(states['vae_state']))
        ck('inherited_ema_exact_copy_at1000',tensor_digest(states['ddpm_state'])==tensor_digest(states['ddpm_ema_state']))
        ck('ddpm_fresh_seed322_initial_digest',read(B/'ddpm_training_summary.json')['initial_tensor_state_sha256']==read(H/'feasibility/ddpm_maturity_v1/ddpm_initial_state.json')['tensor_state_sha256'])
        # Fixed local OneCycle metadata has a serialized getattr; replace only this exact callable with an inert string.
        def inert(cls,name):
            if cls is torch.optim.lr_scheduler.OneCycleLR and name=='_annealing_cos': return 'INERT_ONECYCLE_ANNEALING_COS_REFERENCE'
            raise pickle.UnpicklingError('unexpected serialized attribute')
        class Restricted(pickle.Unpickler):
            def find_class(self,module,name):
                allowed={('torch._utils','_rebuild_tensor_v2'):torch._utils._rebuild_tensor_v2,('torch','FloatStorage'):torch.FloatStorage,
                         ('torch','ByteStorage'):torch.ByteStorage,('collections','OrderedDict'):collections.OrderedDict,
                         ('__builtin__','getattr'):inert,('torch.optim.lr_scheduler','OneCycleLR'):torch.optim.lr_scheduler.OneCycleLR}
                if (module,name) not in allowed: raise pickle.UnpicklingError('unapproved '+module+'.'+name)
                return allowed[(module,name)]
        restricted=types.SimpleNamespace(__name__='restricted_local_state',Unpickler=Restricted)
        vae=torch.load(B/'vae_training_state.pt',map_location='cpu',weights_only=True)
        ddpm=torch.load(B/'ddpm_training_state.pt',map_location='cpu',pickle_module=restricted,weights_only=False)
        ck('saved_optimizers_finite1000',all(x['completed_updates']==1000 and all(bool(torch.isfinite(t).all()) for t in tensors(x)) and
           all(float(s['step'])==1000 for s in x['optimizer']['state'].values()) for x in (vae,ddpm)))
        ck('ddpm_scheduler_and_ema1000',ddpm['scheduler']['last_epoch']==1000 and ddpm['ema_step']==1000)
        mon=lines(B/'dev_monitor.jsonl')
        ck('last_monitor_restored_final_vae_rng',hashlib.sha256(vae['torch_rng'].numpy().tobytes()).hexdigest()==mon[-1]['rng_sha256'])
    section('model_states',model_states)
    def generation_and_scales():
        rows=list(csv.DictReader((B/'generated.csv').open(newline=''))); gen=read(B/'generation_summary.json')
        expected=[f'CE21_DDPM_{t}_{i:03d}' for t in ('FREE','AMD') for i in range(16)]
        ck('exact32_preallocated_ids_seeds',[r['sample_id'] for r in rows]==expected and all(r['method']=='CE21_DDPM' and
           r['terminal']==('FREE' if i<16 else 'AMD') and int(r['terminal_sample_id'])==i%16 and int(r['seed'])==(422 if i<16 else 423) for i,r in enumerate(rows)))
        ck('32_successful_execution_slots',gen['requested']==gen['successful_execution']==32 and gen['failed_execution']==gen['not_executed']==0 and all(r['status'] in SUCCESS for r in rows))
        ck('generation_terminal_only_no_retry',gen['condition_inputs']==['terminal class only'] and gen['no_retry_or_filter'] and status.get('generation')==gen)
        progress=lines(B/'generation_progress.jsonl'); expect_ts=[900,800,700,600,500,400,300,200,100,1]
        ck('both_chains_reached_t1',all([r['timestep'] for r in progress if r['terminal']==t]==expect_ts for t in ('FREE','AMD')))
        with np.load(B/'train_posterior_parameters.npz',allow_pickle=False) as a:
            values=a['values'].copy(); labels=a['labels'].copy(); ids=a['row_ids'].copy()
        ck('train_posterior_shape_identity_finite',values.shape==(279,2,16,48) and values.dtype==np.float32 and np.isfinite(values).all() and
           np.array_equal(ids,np.arange(279)) and np.bincount(labels,minlength=2).tolist()==[126,153])
        def target_scale(v):
            mu=v[:,0].astype(np.float64); sig=torch.from_numpy(v[:,1].copy()).exp().numpy().astype(np.float64)
            q2=float(np.mean(mu*mu+sig*sig)); mean=float(mu.mean())
            return {'mu':summary(mu),'logsigma':summary(v[:,1]),'sigma':summary(sig),'analytic_qz_mean':mean,
                    'analytic_qz_rms':math.sqrt(q2),'analytic_qz_std':math.sqrt(max(0,q2-mean*mean))}
        pooled=target_scale(values); ck('train_sigma_finite_positive',pooled['sigma']['finite_elements']==pooled['sigma']['elements'] and pooled['sigma']['min']>0)
        scales={'pooled_train':pooled,'terminal_train':{},'generated':{}}
        for label,t in enumerate(('FREE','AMD')):
            scales['terminal_train'][t]=target_scale(values[labels==label])
            with np.load(B/f'generated_{t}.npz',allow_pickle=False) as a: data={k:a[k].copy() for k in a.files}
            with np.load(B/f'generated_posterior_{t}.npz',allow_pickle=False) as a: posterior=a['values'].copy()
            ck(t+'_all_saved_generated_tensors_finite',all(np.isfinite(a).all() for a in data.values()))
            ck(t+'_generated_shape_identity',posterior.shape==(16,2,16,48) and data['raw'].shape==(16,21,48) and
               data['z'].shape==(16,16,48) and data['tokens'].shape==(16,48) and np.array_equal(posterior,data['posterior_parameters']))
            mu=torch.from_numpy(posterior[:,0].copy()); logsigma=torch.from_numpy(posterior[:,1].copy()); sigma=logsigma.exp()
            ck(t+'_strict_finite_positive_sigma',bool(torch.isfinite(sigma).all()) and bool((sigma>0).all()))
            # Reproduce RNG draws only. No DDPM/decoder call and no new sequence request.
            rng=torch.Generator().manual_seed(422+label)
            for unused in range(998): torch.randn((16,2,16,48),generator=rng)
            epsilon=torch.randn((16,16,48),generator=rng); z=mu+epsilon*sigma
            ck(t+'_reparameterization_seed_bitwise',z.numpy().tobytes()==data['z'].tobytes())
            # Independent raw-logit legal-prefix computation; no softmax needed (per-position additive constants cancel).
            raw=torch.from_numpy(data['raw']); aa_values,aa=raw[:,:20,:].max(1); pad=raw[:,20,:]
            lengthscores=pad.sum(1,keepdim=True)+torch.cat((torch.zeros((16,1)),(aa_values-pad).cumsum(1)),dim=1)
            lengths=lengthscores.argmax(1); toks=torch.where(torch.arange(48)[None,:]<lengths[:,None],aa,torch.full_like(aa,20))
            ck(t+'_joint_decode_from_raw_exact',np.array_equal(toks.numpy(),data['tokens']) and np.array_equal(lengths.numpy(),data['lengths']) and np.array_equal(lengthscores.numpy(),data['length_scores']))
            selected=[r for r in rows if r['terminal']==t]
            sequences=[''.join(AA[x] for x in tok[:n]) for tok,n in zip(toks.tolist(),lengths.tolist())]
            ck(t+'_rendered_slots_exact',all(r['sequence']==s and int(r['length'])==len(s)==n and
               r['status']==('empty_sequence' if n==0 else ('ok' if n<48 else 'no_eos_at_limit')) for r,s,n in zip(selected,sequences,lengths.tolist())))
            scale={'mu':summary(mu.numpy()),'logsigma':summary(logsigma.numpy()),'sigma':summary(sigma.numpy()),'z':summary(data['z']),
                   'decoder_raw':summary(data['raw'])}
            scale['z_rms_ratio_to_pooled_train']=scale['z']['rms']/pooled['analytic_qz_rms']
            scale['z_rms_ratio_to_terminal_train']=scale['z']['rms']/scales['terminal_train'][t]['analytic_qz_rms']
            scale['per_request_z_rms']=np.sqrt(np.mean(data['z'].astype(np.float64)**2,axis=(1,2))).tolist()
            scales['generated'][t]=scale
        facts['scales']=scales
        facts['slot_status_counts']=dict(collections.Counter(r['status'] for r in rows))
        facts['requested_slots']=len(rows)
        facts['generation_quality_metrics_computed']=False
    section('generation_and_scales',generation_and_scales)
    def available_failure_evidence():
        evidence={}
        for terminal in ('FREE','AMD'):
            p=B/f'generation_failure_{terminal}.json'
            if p.exists():
                item={'recorded_failure':read(p)}
                archive=B/f'generation_failure_{terminal}.npz'
                if archive.exists():
                    with np.load(archive,allow_pickle=False) as a:
                        item['independent_saved_tensor_summaries']={k:summary(a[k]) for k in a.files}
                evidence[terminal]=item
        facts['available_failure_evidence']=evidence
        facts['slot_inventory_even_if_failed']=dict(collections.Counter(r['status'] for r in csv.DictReader((B/'generated.csv').open(newline=''))))
    section('available_failure_evidence',available_failure_evidence)
    passed=all(c['pass'] for c in checks)
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status':'PASS_ENGINEERING_CHECKS_PENDING_INDEPENDENT_SCALE_JUDGMENT' if passed else 'ENGINEERING_FINDINGS_HOLD',
            'checks':checks,'failed_checks':[c for c in checks if not c['pass']],'section_errors':section_errors,'facts':facts,
            'audit_script_sha256':sha(__file__),'manifest_sha256':sha(B/'manifest.json') if (B/'manifest.json').exists() else None,
            'further_arms_authorized':False,'PROMOTE':'HOLD',
            'reviewer_actions':{'model_constructions':0,'model_forwards':0,'training_updates':0,'new_generation_requests':0,'test_reads':0,
                               'actions':'Read fixed saved artifacts, hash four exact train/dev inputs, read only two cohort files; tensor arithmetic and local RNG draw reproduction without a model'},
            'boundaries':['No OS syscall audit; no-test evidence comes from exact command/config/source, runtime flags and reviewed paths.',
                          'Finite engineering checks are not generation quality, model novelty, AMP activity or strong-baseline adequacy.',
                          'No automatic scale threshold added; an independent descriptive scale assessment is still required.',
                          'No quality metrics used to authorize or block MSE/scalar. Any subsequent arm needs a separate exact plan and receipt.',
                          'Only seed222/322 and 16 requests per terminal; adaptive feasibility and pretraining repair attempts remain disclosed.']}
    with dest.open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':report['status'],'checks':len(checks),'failed_checks':report['failed_checks'],'report':str(dest)},ensure_ascii=False))
if __name__=='__main__': main()
