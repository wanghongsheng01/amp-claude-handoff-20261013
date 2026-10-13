"""Read-only bookkeeping of completed AMP H5 records. No model/data loads.

Deliberately never traverses or reads formal_v2 or its live supervisor. Pending
status is the delegated snapshot instruction, not a newly observed runtime fact.
The output directory is a new append-only ledger version, not an old run path.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {}


def payload(rel):
    assert 'formal_v2/' not in rel and 'single_ce_execution_v2/' not in rel
    path = ROOT / rel
    data = path.read_bytes()
    SOURCES[rel] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return data.decode('utf-8')


def read(rel):
    return json.loads(payload(rel))


def rows(rel):
    return list(csv.DictReader(payload(rel).splitlines()))


def logs(rel):
    return [json.loads(line) for line in payload(rel).splitlines() if line]


def job(rel, name):
    return next(x for x in read(rel)['jobs'] if x['id'] == name)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def base(event_id, method, category, supervisor, run_status):
    return {'event_id': event_id, 'method': method, 'category': category,
            'state': run_status['state'], 'started_utc': supervisor['started_utc'],
            'timing': {'supervisor_job_wall_seconds': supervisor['seconds'],
                       'method_process_wall_seconds': run_status.get('process_seconds', run_status.get('seconds', run_status.get('elapsed_seconds'))),
                       'cpu_process_user_system_seconds': None,
                       'measurement_device': 'CPU, one torch thread',
                       'scope_note': 'Nested timings overlap. Supervisor is outer elapsed wall, not CPU process utilization time.'},
            'updates': {'AR': 0, 'VAE': 0, 'DDPM': 0},
            'attempts': {'sequence_allocated_slots': 0, 'sequence_sampling_attempts': 0,
                         'latent_only_attempts': 0, 'completed_sequence_executions': 0,
                         'failed_sequence_executions': 0, 'not_executed_sequence_slots': 0,
                         'reused_historical_rows': 0},
            'train_only_development_stage': False, 'development_access': False,
            'test_access': False, 'training_warm_start_from_prior_run': False}


events = []
for phase in ('smoke_v1', 'formal_v1'):
    rel = f'runs/{phase}/terminal_lstm'
    s, r, g = read(rel+'/status.json'), read(rel+'/training/RUN.json'), read(rel+'/generation/GENERATION.json')
    sup = job('runs/smoke_v1/SUPERVISOR_STATUS.json' if phase == 'smoke_v1' else 'runs/ar_formal_execution_v1/SUPERVISOR_STATUS.json', 'terminal_lstm')
    e = base('AR_'+phase, 'terminal_LSTM', 'smoke' if phase == 'smoke_v1' else 'formal', sup, s)
    log = logs(rel+'/training/steps.jsonl')
    assert len(log) == r['completed_steps'] and [x['step'] for x in log] == list(range(1, len(log)+1))
    sample = rows(rel+'/generated.csv')
    statuses = {status: sum(row['status'] == status for row in sample) for status in sorted({row['status'] for row in sample})}
    e['updates']['AR'] = r['completed_steps']
    e['timing'].update(training_pipeline_wall_seconds=r['total_pipeline_seconds_before_manifest'],
                       AR_update_wall_seconds=r['train_update_seconds'], monitor_wall_seconds=r['monitoring_seconds'],
                       generation_wall_seconds=g['wall_seconds_before_manifest'])
    e['attempts'].update(sequence_allocated_slots=g['requests'], sequence_sampling_attempts=g['requests'],
                         completed_sequence_executions=sum(row['status'] in ('ok','empty_sequence') for row in sample),
                         status_counts=statuses, nonempty_sequences=sum(bool(row['sequence']) for row in sample))
    e.update(train_only_development_stage=phase == 'smoke_v1', development_access=s['development_read'],
             monitor_points=r['monitor_count'], fixed_endpoint=True, training_seed=222, sampling_seeds=g['seeds'],
             budget_cap_seconds=r['cap_seconds'], checkpoint_selection='fixed final',
             note='Fresh initialization for each run. Repeated prefix work remains charged; smoke is not free formal training.')
    events.append(e)

for arm in ('categorical_ce', 'raw_onehot_mse', 'scalar_legacy'):
    rel = f'runs/smoke_v1/{arm}'
    s, v, d, g = (read(rel+'/'+name) for name in ('status.json','vae_training_summary.json','ddpm_training_summary.json','generation_summary.json'))
    sup = job('runs/smoke_v1/SUPERVISOR_STATUS.json', arm)
    e = base(arm+'_smoke_v1', arm, 'smoke', sup, s)
    vl, dl = logs(rel+'/vae_train.jsonl'), logs(rel+'/ddpm_train.jsonl')
    assert len(vl) == v['updates'] == 20 and len(dl) == d['updates'] == 20
    e['updates'].update(VAE=v['updates'], DDPM=d['updates'])
    e['timing'].update(training_pipeline_wall_seconds=s['training_seconds'],
                       VAE_stage_wall_seconds=v['elapsed_seconds'], DDPM_stage_wall_seconds=d['elapsed_seconds'],
                       VAE_update_wall_seconds=sum(x['duration_seconds'] for x in vl),
                       DDPM_update_wall_seconds=sum(x['duration_seconds'] for x in dl),
                       DDPM_encoding_and_setup_wall_seconds=d['posterior_encoding_and_setup_seconds'],
                       generation_wall_seconds=g['elapsed_seconds'])
    sample = rows(rel+'/generated.csv')
    assert len(sample) == g['requested'] == 2
    e['attempts'].update(sequence_allocated_slots=2, sequence_sampling_attempts=2,
                         failed_sequence_executions=g['failed_execution'], status_counts={x:sum(y['status']==x for y in sample) for x in sorted({y['status'] for y in sample})})
    e.update(train_only_development_stage=True, monitor_points=0, training_seed=222, DDPM_seed=322,
             budget_cap_seconds=180, checkpoint_selection='fixed 20/20 endpoint',
             note='Both requests failed exp(logsigma); all attempted requests charged. No formal quality evidence. DDPM stage includes encoding/setup; do not add its nested setup time twice.')
    events.append(e)

p = 'feasibility/h4_parity_v1'
s = read(p+'/status.json')
e = base('H4_saved_t1_parity', 'shared_VAE_path', 'additional_diagnostic', job('runs/h4_parity_execution_v1/SUPERVISOR_STATUS.json','h4_saved_parity'), s)
e['attempts']['reused_historical_rows'] = sum(x['saved_request_count'] for x in s['results'].values())
e.update(ddpm_forwards=s['ddpm_forwards'], decoder_forwards=s['decoder_forwards'],
         comparison_checks=sum(len(x['checks']) for x in s['results'].values()),
         note='Two stored 16-row batches, saved t=1 only. Zero new samples or training; not a full 998-step chain replay.')
events.append(e)

p = 'feasibility/ddpm_maturity_v1'
s, t, g = read(p+'/status.json'), read(p+'/training_summary.json'), read(p+'/generation_summary.json')
e = base('DDPM_maturity_1000_4', 'categorical_ce20_frozen_representation', 'additional_diagnostic', job('runs/ddpm_maturity_execution_v1/SUPERVISOR_STATUS.json','ddpm_maturity'), s)
dl = logs(p+'/ddpm_train.jsonl')
assert len(dl) == t['updates'] == 1000
attempts = rows(p+'/ATTEMPTS.csv')
assert len(attempts) == 4
e['updates']['DDPM'] = 1000
e['timing'].update(training_pipeline_wall_seconds=t['training_seconds'], DDPM_update_wall_seconds=sum(x['duration_seconds'] for x in dl), generation_wall_seconds=g['seconds'])
e['attempts'].update(latent_only_attempts=4, latent_finite_successes=g['numerical_success_count'], latent_numerical_failures=g['numerical_failure_count'])
e.update(train_only_development_stage=True, checkpoint_observation_points=[20,1000], training_seed=322,
         frozen_VAE_training_updates_charged_here=0, budget_cap_seconds=t['seconds_cap'],
         note='One fresh 1000-update trajectory, not 20+1000 independent training updates. Four latent-only requests; no decoder or sequence strings. CE20 posterior reused from smoke, no new VAE training. Horizon1000 schedule differs from original smoke horizon20. Provides extra engineering information before CE formal; not matched extra diagnostics for all arms.')
events.append(e)

p='runs/formal_v1/categorical_ce'
s=read(p+'/status.json'); sample=rows(p+'/generated.csv')
assert len(sample)==32 and all(x['status']=='not_executed' for x in sample)
e=base('CE_formal_v1_entry_failure','categorical_ce','failed_formal_entry',job('runs/single_ce_execution_v1/SUPERVISOR_STATUS.json','categorical_ce'),s)
e['attempts'].update(sequence_allocated_slots=32,not_executed_sequence_slots=32)
repair=payload('feasibility/formal_v2_contract_repair/REPAIR.zh.md')
assert 'development_read=false' in repair
e.update(development_access=True, raw_status_development_read=s['development_read'],
         development_access_correction='Raw false flag was written only after successful validation; dev file was already opened. Corrected interpretation from frozen repair report, no back-edit.',
         note='Input idx guard failed before model creation. Zero updates and sampling attempts, but real setup/failed-entry cost retained. One explicit contract repair follows; not a successful formal run.')
events.append(e)

# Do not read the live v2 path, logs, checkpoint or supervisor in this ledger.
events.append({'event_id':'CE_formal_v2_pending','method':'categorical_ce','category':'formal_pending',
               'state':'PENDING_NOT_INSPECTED','timing':None,'updates':{'AR':None,'VAE':None,'DDPM':None},
               'attempts':None,'development_access':None,'test_access':False,
               'planned_updates':{'VAE':1000,'DDPM':1000},'planned_sequence_slots':32,
               'planned_training_cap_seconds':2940,
               'snapshot_basis':'Root delegation: CE v2 running; mark pending and do not inspect current outputs.',
               'note':'Null actual costs/counts are unknown, never zero. Merge only a new ledger version after a terminal record is independently available.'})

formal=read('protocol/FORMAL_PROTOCOL.json')
monitors=read('runs/formal_v1/terminal_lstm/training/MONITORS.json')
extra=read('feasibility/ar_formal_readout/RUN_FACTS.json')
coordinate=read('feasibility/coordinate_audit/SAVED_NUMERIC_AUDIT.json')
analytic=read('feasibility/coordinate_audit/ZERO_PREDICTOR_ANALYTIC.json')
contract=read('review/independent/FORMAL_V2_CONTRACT_REVIEW.json')
assert all(x['used_for_selection'] is False for x in monitors)
complete=[e for e in events if e['timing'] is not None]
totals={'completed_or_failed_terminal_jobs':len(complete),
        'outer_supervisor_job_wall_seconds_sum':sum(e['timing']['supervisor_job_wall_seconds'] for e in complete),
        'inner_method_process_wall_seconds_sum':sum(e['timing']['method_process_wall_seconds'] for e in complete),
        'updates_by_family':{kind:sum(e['updates'][kind] for e in complete) for kind in ('AR','VAE','DDPM')},
        'sequence_allocated_slots':sum(e['attempts']['sequence_allocated_slots'] for e in complete),
        'sequence_sampling_attempts':sum(e['attempts']['sequence_sampling_attempts'] for e in complete),
        'latent_only_attempts':sum(e['attempts']['latent_only_attempts'] for e in complete),
        'not_executed_sequence_slots':sum(e['attempts']['not_executed_sequence_slots'] for e in complete),
        'historical_rows_reused_by_parity':32,
        'pending_actual_costs_included':False,
        'scope':'Sum only the eight terminal supervisor jobs. Excludes unrecorded author/review work, pauses, source H0-H4 training, analytical bookkeeping, and pending CEv2. Nested timing columns must not be added to this total.'}
opportunities={'prelocked_formal_policy':formal['train_selection_opportunities'],
               'actual_compute_equal':False,'formal_common_training_cap_seconds':2940,
               'actual_opportunity_accounting':[
                   'Four real smoke runs: AR100 and three VAE20+DDPM20; none promoted as formal weights.',
                   'One completed AR formal initialization/seed, 10 descriptive train/dev monitors and one final checkpoint. Dev trajectory reveals overfit; no best checkpoint selection.',
                   'Shared saved-t1 parity plus coordinate/zero-predictor analyses were added after numerical smoke failures.',
                   'CE20-specific frozen-representation diagnostic adds fresh DDPM1000 and two checkpoint observations with four paired latent requests. Counts as extra development information and cost.',
                   'One failed CE formal entry opened dev before failing; one input-contract repair version changes idx guards/status accounting, with no model/training/seed/loss change documented.',
                   'CEv2 is a separate pending execution, and later formal-arm starts are staged by engineering feasibility. This is adaptive execution order, not an untouched simultaneous one-shot experiment.',
                   'The declared zero hyperparameter search is a narrow formal-policy field; it does not mean zero additional diagnosis, implementation revision, development exposure or execution opportunity.'
               ],
               'known_additional_readonly_work':[
                   {'work':'AR final descriptor/nearest-neighbor/NLL author readout','recorded_wall_seconds':extra['costs']['evaluation_seconds_before_output'],'model_calls':0,'train_and_dev_rows_read':[279,41],'included_in_supervisor_total':False},
                   {'work':'Saved coordinate audit','recorded_wall_seconds':None,'no_model_loads':coordinate['no_model_loads'],'no_generation':coordinate['no_generation'],'included_in_supervisor_total':False},
                   {'work':'Analytic zero-predictor reference','recorded_wall_seconds':None,'model_calls':0,'included_in_supervisor_total':False},
                   {'work':'Contract repair independent check','recorded_wall_seconds':None,'checks':len(contract['checks']),'model_calls':0,'train_and_dev_read':True,'included_in_supervisor_total':False}
               ],
               'AR_adequacy':{'status':'NOT_ESTABLISHED','train100_NLL':monitors[0]['train']['mean_nll'],
                              'train1000_NLL':monitors[-1]['train']['mean_nll'],'dev100_NLL':monitors[0]['development']['mean_nll'],
                              'dev1000_NLL':monitors[-1]['development']['mean_nll'],
                              'diagnosis':'training NLL drops at all recorded intervals while dev rises after200; overfitting evidence; fixed final retained. Neither tuning adequacy nor strong-baseline status established.'},
               'promotion':'HOLD','four_arm_comparison':'NOT_COMPLETE'}
ledger={'schema':'AMP_H5_COST_OPPORTUNITY_LEDGER_V1','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Append-only ledger of completed old evidence; CEv2 deliberately pending.',
        'all_seconds_are_elapsed_wall_unless_explicitly_named':True,
        'cpu_utilization_time_not_recorded':True,'no_model_imports_or_execution':True,'no_test_file_reads':True,
        'pending_runtime_outputs_not_read':True,'events':events,'totals':totals,'development_opportunities':opportunities,
        'future_merge':'Create cost_opportunity_ledger_v2 or later; preserve this snapshot and old bindings.'}
dump('COST_OPPORTUNITY_LEDGER.json',ledger)
flat=[]
for e in events:
    t=e.get('timing') or {}; a=e.get('attempts') or {}; u=e['updates']
    flat.append({'event_id':e['event_id'],'category':e['category'],'state':e['state'],
                 'outer_wall_seconds':t.get('supervisor_job_wall_seconds'),
                 'inner_wall_seconds':t.get('method_process_wall_seconds'),
                 'train_pipeline_seconds':t.get('training_pipeline_wall_seconds'),
                 'generation_seconds':t.get('generation_wall_seconds'),
                 'AR_updates':u['AR'],'VAE_updates':u['VAE'],'DDPM_updates':u['DDPM'],
                 'sequence_slots':a.get('sequence_allocated_slots'),
                 'sequence_attempts':a.get('sequence_sampling_attempts'),
                 'latent_only_attempts':a.get('latent_only_attempts'),
                 'not_executed_slots':a.get('not_executed_sequence_slots')})
with (OUT/'COSTS.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
dump('SOURCE_BINDINGS.json',{'read_files':list(SOURCES.values()),'live_v2_files_read':0,'test_files_read':0})
print(json.dumps({'totals':totals,'sources':len(SOURCES),'output':str(OUT)},indent=2))
