"""Read-only standard-library rollup: immutable v2 plus terminal MSE.

Scalar launch is recorded, its live work remains excluded. No model execution,
weight deserialization, source cohort parsing, or original CE status mutation.
"""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCES={}

def get(rel,mode='json'):
    assert not rel.startswith(('runs/formal_v2/scalar_legacy/','runs/single_scalar_legacy_execution_v2/','runs/formal_v2/categorical_ce/','runs/single_ce_execution_v2/'))
    p=ROOT/rel;b=p.read_bytes();SOURCES[rel]={'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    if mode=='json':return json.loads(b)
    if mode=='csv':return list(csv.DictReader(b.decode().splitlines()))
    if mode=='jsonl':return [json.loads(x) for x in b.decode().splitlines() if x]
    return b

def dump(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

previous='feasibility/cost_opportunity_ledger_v2/'
pm=get(previous+'MANIFEST.json')
for row in pm['files']:
    assert hashlib.sha256(get(previous+row['path'],'bytes')).hexdigest()==row['sha256']
v2=get(previous+'COST_OPPORTUNITY_LEDGER_V2.json')
mse='runs/formal_v2/raw_onehot_mse/'
status=get(mse+'status.json');vs=get(mse+'vae_training_summary.json');ds=get(mse+'ddpm_training_summary.json');gs=get(mse+'generation_summary.json')
sup=get('runs/single_raw_onehot_mse_execution_v2/SUPERVISOR_STATUS.json');job=sup['jobs'][0]
post=get('review/independent/MSE_V2_POSTFLIGHT_v1.json')
launch=get('runs/single_scalar_v2_launch_v1/LAUNCH.json')
vl=get(mse+'vae_train.jsonl','jsonl');dl=get(mse+'ddpm_train.jsonl','jsonl');mon=get(mse+'dev_monitor.jsonl','jsonl')
assert status['state']==job['state']=='completed' and job['exit_code']==0
assert job['seconds']==2877.830811874941
assert vs['updates']==ds['updates']==1000
assert [x['step'] for x in vl]==[x['step'] for x in dl]==list(range(1,1001))
assert len(mon)==10 and all(x['checkpoint_selection'] is False for x in mon)
assert post['status']=='PASS' and post['scope']=='SINGLE_MSE_ENGINEERING_ONLY'
assert gs['requested']==gs['successful_execution']==32 and gs['failed_execution']==gs['not_executed']==0

completed_exports={
 'AR_smoke_v1':'runs/smoke_v1/terminal_lstm/generated.csv',
 'categorical_ce_smoke_v1':'runs/smoke_v1/categorical_ce/generated.csv',
 'raw_onehot_mse_smoke_v1':'runs/smoke_v1/raw_onehot_mse/generated.csv',
 'scalar_legacy_smoke_v1':'runs/smoke_v1/scalar_legacy/generated.csv',
 'AR_formal_v1':'runs/formal_v1/terminal_lstm/generated.csv',
 'CE_checkpoint_recovery_v1':'runs/ce_checkpoint_recovery_v1/generated.csv',
 'raw_onehot_mse_formal_v2':mse+'generated.csv'}
denominators={};alphabet=set('ARNDCEQGHILKMFPSTWYV')
for event,rel in completed_exports.items():
    rs=get(rel,'csv');done=[x for x in rs if x['status'] in ('ok','empty_sequence','no_eos_at_limit')]
    valid=lambda s,lo:lo<=len(s)<=48 and set(s)<=alphabet
    d={'allocated_slots':len(rs),'started_sampling_attempts':len(rs),
       'completed_sequence_executions':len(done),
       'failed_sequence_executions':sum(x['status'].startswith('error') for x in rs),
       'status_counts':{s:sum(x['status']==s for x in rs) for s in sorted({x['status'] for x in rs})},
       'completed_empty_sequences':sum(not x['sequence'] for x in done),
       'completed_legal1_48':sum(valid(x['sequence'],1) for x in done),
       'completed_legal5_48':sum(valid(x['sequence'],5) for x in done),
       'latent_only_attempts':0,'source':rel,
       'interpretation':'Literal alphabet/length accounting on completed exported rows; no biology or quality inference.'}
    denominators[event]=d

events=[copy.deepcopy(e) for e in v2['events'] if e['event_id'] not in ('raw_onehot_mse_formal_not_started','scalar_legacy_formal_not_started')]
for e in events:
    if e['event_id'] in denominators:e['denominator_accounting_v3']=denominators[e['event_id']]
events.append({'event_id':'raw_onehot_mse_formal_v2','method':'raw_onehot_mse','category':'formal_completed',
 'state':status['state'],'started_utc':job['started_utc'],'completed_utc':sup['completed_utc'],
 'exit_code':0,'independent_postflight':'PASS_SINGLE_MSE_ENGINEERING_ONLY',
 'timing':{'supervisor_job_wall_seconds':job['seconds'],'method_process_wall_seconds':status['process_seconds'],
  'training_pipeline_wall_seconds':status['training_seconds'],'VAE_stage_wall_seconds':vs['elapsed_seconds'],
  'DDPM_stage_wall_seconds':ds['elapsed_seconds'],'VAE_update_wall_seconds':sum(x['duration_seconds'] for x in vl),
  'DDPM_update_wall_seconds':sum(x['duration_seconds'] for x in dl),
  'DDPM_encoding_and_setup_wall_seconds':ds['posterior_encoding_and_setup_seconds'],
  'dev_monitor_wall_seconds':sum(x['seconds'] for x in mon),'generation_wall_seconds':gs['elapsed_seconds'],
  'scope_note':'All timings nested within supervisor. Use outer once, not sum of outer and phases.'},
 'updates':{'AR':0,'VAE':1000,'DDPM':1000},'attempts':denominators['raw_onehot_mse_formal_v2'],
 'denominator_accounting_v3':denominators['raw_onehot_mse_formal_v2'],
 'development_access':True,'monitor_points':10,'checkpoint_selection':'fixed final1000; none selected by dev',
 'test_access':False,'formal_common_training_cap_seconds':2940,'actual_equal_compute':False,
 'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD'})
events.append({'event_id':'scalar_legacy_formal_v2_live_excluded','method':'scalar_legacy','category':'live_not_in_terminal_ledger',
 'state':'LAUNCHED_LIVE_AT_DELEGATED_SNAPSHOT','launched_utc':launch['launched_utc'],
 'supervisor_pid_at_launch':launch['supervisor_pid'],'launch_record':'runs/single_scalar_v2_launch_v1/LAUNCH.json',
 'timing':None,'updates':{'AR':None,'VAE':None,'DDPM':None},'attempts':None,
 'planned_updates':{'VAE':1000,'DDPM':1000},'planned_sequence_slots':32,
 'current_actual_costs_and_denominators_included':False,
 'note':'Scalar was launched2026-10-11 19:10:17 UTC. It is not not-started. Only immutable launch read; no live logs/model outputs inspected. Merge actual terminal totals in newv4.'})

old=v2['totals'];accounted=old['all_accounted_disjoint_wall_cost_lower_bound_seconds']+job['seconds']
totals=copy.deepcopy(old)
totals.update({
 'previous_v2_wall_lower_bound_seconds':old['all_accounted_disjoint_wall_cost_lower_bound_seconds'],
 'new_MSE_supervisor_outer_wall_seconds':job['seconds'],
 'ten_terminal_job_supervisor_wall_seconds_sum':old['nine_terminal_job_supervisor_wall_seconds_sum']+job['seconds'],
 'all_accounted_disjoint_wall_cost_lower_bound_seconds':accounted,
 'overall_exact_wall_cost_seconds':None,
 'updates_by_family':{'AR':1100,'VAE':2060,'DDPM':3060},
 'sequence_allocated_slots':old['sequence_allocated_slots']+32,
 'sequence_sampling_attempts_observed_lower_bound':old['sequence_sampling_attempts_observed_lower_bound']+32,
 'sequence_sampling_attempts_exact':None,
 'final_sequence_exports_with_completed_execution':sum(x['completed_sequence_executions'] for x in denominators.values()),
 'completed_legal1_48_sequences':sum(x['completed_legal1_48'] for x in denominators.values()),
 'completed_legal5_48_sequences':sum(x['completed_legal5_48'] for x in denominators.values()),
 'completed_empty_sequences':sum(x['completed_empty_sequences'] for x in denominators.values()),
 'completed_length1_4_sequences':sum(x['completed_legal1_48']-x['completed_legal5_48'] for x in denominators.values()),
 'recorded_failed_sequence_executions':sum(x['failed_sequence_executions'] for x in denominators.values()),
 'live_scalar_actual_costs_or_slots_included':False,
 'scope':'Ten terminal job outer times plus interrupted originalCE disjoint stage/partial-generation lower bound. Live scalar actuals, unlogged originalCE work/setup gaps and untimed reviews excluded.',
 'slot_partition_check':'168 = 98 completed exports + 6 numerical failures + 16 partial originalFREE + 32 CEv1 not_executed + 16 originalAMD no-start-evidence; live scalar excluded.',
 'legal_count_warning':'Counts pool historical smoke and completed methods solely for bookkeeping. Do not use pooled legal rates or claim biological quality; see per-event counts.'})
totals.pop('formal_MSE_scalar_new_training_or_generation',None)
assert totals['sequence_allocated_slots']==168 and totals['sequence_sampling_attempts_observed_lower_bound']==120
assert totals['final_sequence_exports_with_completed_execution']==98
assert totals['completed_legal1_48_sequences']==97 and totals['completed_legal5_48_sequences']==88
op=copy.deepcopy(v2['development_opportunities'])
op['v2_additions_retained_as_historical_snapshot']=op.pop('current_snapshot_additions')
op.update({'v3_additions':[
 'MSE formalVAE1000/DDPM1000 completed; 32 attempts all executed, 32 legal1-48 and26 legal5-48. Added2877.830811874941s supervisor wall, not a free comparison arm.',
 'MSE ten dev monitor points and fixed final train/dev exports are additional reused-dev observation opportunities; no checkpoint selection recorded.',
 'Scalar launched19:10:17UTC. Its live work is excluded from these terminal totals, not treated as zero actualwork or notstarted.',
 'CE original E/L/G remains not evaluable after interruption; CE recovery is distinct extraattempt. MSE engineeringPASS does not backfill originalCE.',
 'One training seed per formal method, reused dev41, AR overfit/adequacy not established, unequal actualcompute and staged engineering-based starts remain. No combined superiority/promotion verdict.'
 ],'MSE_engineering':'PASS_SCOPED_ENGINEERING_ONLY','scalar_status':'LAUNCHED_LIVE_EXCLUDED_PENDING_TERMINAL',
 'four_arm_comparison':'NOT_COMPLETE_SCALAR_LIVE','actual_compute_equal':False,'promotion':'HOLD'})
ledger={'schema':'AMP_H5_COST_OPPORTUNITY_LEDGER_V3','created_utc':datetime.now(timezone.utc).isoformat(),
 'snapshot_scope':'Includes MSE terminal evidence; scalar launched19:10:17UTC but live outputs excluded; v1/v2 untouched.',
 'prior_manifest_sha256':SOURCES[previous+'MANIFEST.json']['sha256'],
 'events':events,'totals':totals,'per_completed_export_denominators':denominators,'development_opportunities':op,
 'all_seconds_elapsed_wall_not_CPU_user_system_time':True,
 'no_models_weights_or_source_cohort_reads':True,'original_CE_status_policy':v2['original_status_policy'],
 'PROMOTE':'HOLD','original_four_arm_E_L_G':'NOT_EVALUABLE_INTERRUPTION',
 'next_version':'Wait for scalar terminal evidence; create newv4, do not alterv3 or older ledgers.'}
dump('COST_OPPORTUNITY_LEDGER_V3.json',ledger)
with (OUT/'DENOMINATORS_V3.csv').open('w',newline='') as f:
 fields=['event_id','allocated_slots','started_sampling_attempts','completed_sequence_executions','failed_sequence_executions','completed_empty_sequences','completed_legal1_48','completed_legal5_48','latent_only_attempts']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
 for k,v in denominators.items():w.writerow({'event_id':k,**v})
flat=[]
for e in events:
 t=e.get('timing') or {};a=e.get('denominator_accounting_v3') or e.get('attempts') or {}
 flat.append({'event_id':e['event_id'],'state':e['state'],'outer_exact_wall_seconds':t.get('supervisor_job_wall_seconds'),
  'original_stage_and_partial_wall_lower_bound_seconds':t.get('recorded_disjoint_work_wall_seconds_lower_bound'),
  'AR_updates':e['updates']['AR'],'VAE_updates':e['updates']['VAE'],'DDPM_updates':e['updates']['DDPM'],
  'allocated_slots':a.get('allocated_slots',a.get('sequence_allocated_slots')),
  'started_attempts_or_lower_bound':a.get('started_sampling_attempts',a.get('sequence_sampling_attempts',a.get('sequence_sampling_attempts_lower_bound'))),
  'completed_sequence_executions':a.get('completed_sequence_executions',a.get('completed_sequence_exports')),
  'legal1_48':a.get('completed_legal1_48'),'legal5_48':a.get('completed_legal5_48'),
  'latent_only_attempts':a.get('latent_only_attempts'),'live_excluded':e['category']=='live_not_in_terminal_ledger'})
with (OUT/'COSTS_V3.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
dump('SOURCE_BINDINGS.json',{'direct_read_sources':list(SOURCES.values()),'prior_v1_v2_sources_transitively_bound':True,
 'source_cohort_csv_reads':0,'model_calls':0,'checkpoint_loads':0,'live_scalar_status_or_log_reads':0,
 'immutable_scalar_launch_record_read':True})
print(json.dumps({'lower_bound_seconds':accounted,'updates':totals['updates_by_family'],'slots':totals['sequence_allocated_slots'],
 'started_at_least':totals['sequence_sampling_attempts_observed_lower_bound'],'completed':totals['final_sequence_exports_with_completed_execution'],
 'legal1_48':totals['completed_legal1_48_sequences'],'legal5_48':totals['completed_legal5_48_sequences'],
 'latent_only':totals['latent_only_attempts'],'scalar':'LIVE_EXCLUDED'},indent=2))
