"""Standard-library-only cost rollup from v1 + frozen interruption/recovery evidence.

No live-original status polling, raw train/dev/test reads, checkpoint loading,
model imports or model execution. The stale original status is never rewritten.
"""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCES = {}


def get(rel, json_file=True):
    # Original process evidence comes solely from archived forensic snapshots.
    assert not rel.startswith(('runs/formal_v2/', 'runs/single_ce_execution_v2/'))
    path=ROOT/rel; data=path.read_bytes()
    SOURCES[rel]={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    return json.loads(data) if json_file else data.decode('utf-8')


def dump(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


v1root='feasibility/cost_opportunity_ledger_v1/'
v1manifest=get(v1root+'MANIFEST.json')
for item in v1manifest['files']:
    data=get(v1root+item['path'],False)
    assert hashlib.sha256(data.encode()).hexdigest()==item['sha256']
v1=get(v1root+'COST_OPPORTUNITY_LEDGER.json')
forensic=get('review/independent/CE_V2_INTERRUPTION_FORENSIC_v2.json')
author=get('feasibility/ce_v2_interruption_author_v1/EVIDENCE.json')
root_snapshot=get('feasibility/ce_v2_interruption_root_v1/INTERRUPTED_ARTIFACT_SNAPSHOT.json')
process_snapshot=get('feasibility/ce_v2_interruption_root_v1/PROCESS_SNAPSHOT.json')
postflight=get('review/independent/CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json')
recovery=get('runs/ce_checkpoint_recovery_v1/status.json')
generation=get('runs/ce_checkpoint_recovery_v1/generation_summary.json')
supervisor=get('runs/ce_checkpoint_recovery_execution_v1/SUPERVISOR_STATUS.json')
sample=list(csv.DictReader(get('runs/ce_checkpoint_recovery_v1/generated.csv',False).splitlines()))
amendment=get('protocol/CE_CHECKPOINT_RECOVERY_AMENDMENT_v1.zh.md',False)
assert root_snapshot['classification']=='INTERRUPTED_WITH_STALE_RUNNING_STATUS'
assert forensic['original_v2_PASS'] is False and root_snapshot['exit_cause']=='UNKNOWN'
assert postflight['status']=='PASS' and postflight['scope']=='RECOVERY_ENGINEERING_ONLY'
assert recovery['training_updates_this_attempt']==0 and len(sample)==generation['requested']==32
original_supervisor=author['original_supervisor_status']
assert original_supervisor['state']=='running' and original_supervisor['jobs'][0]['state']=='running'
train=forensic['training_summary']; originalgen=forensic['generation_observation']
vae_sec=train['vae']['elapsed_seconds']; ddpm_sec=train['ddpm']['elapsed_seconds']
partial_sec=originalgen['last_recorded_elapsed_seconds']
assert train['vae']['updates']==train['ddpm']['updates']==1000
assert partial_sec==451.9357427917421
job=supervisor['jobs'][0]
assert job['state']=='completed' and job['seconds']==1124.2520439173095
events=[copy.deepcopy(e) for e in v1['events'] if e['event_id']!='CE_formal_v2_pending']
original={
 'event_id':'CE_formal_v2_original_interrupted','method':'categorical_ce','category':'formal_interrupted',
 'external_forensic_state':'INTERRUPTED_WITH_STALE_RUNNING_STATUS','state':'INTERRUPTED_NO_FINAL_STATUS',
 'original_runner_saved_state':author['original_runner_status']['state'],
 'original_supervisor_saved_state':original_supervisor['state'],
 'original_started_utc':original_supervisor['jobs'][0]['started_utc'],
 'exit_cause':'UNKNOWN','exit_code':None,'exit_time':None,'original_exact_total_wall_seconds':None,
 'timing':{'supervisor_job_wall_seconds':None,'method_process_wall_seconds':None,
           'VAE_stage_wall_seconds':vae_sec,'DDPM_stage_wall_seconds':ddpm_sec,
           'training_stage_subtotal_wall_seconds':vae_sec+ddpm_sec,
           'partial_generation_wall_seconds_lower_bound':partial_sec,
           'recorded_disjoint_work_wall_seconds_lower_bound':vae_sec+ddpm_sec+partial_sec,
           'scope_note':'Sequential stage times plus partial FREE generation; setup/interstage gaps and unlogged work remain unknown. Not a measured original whole-process time.'},
 'updates':{'AR':0,'VAE':1000,'DDPM':1000},
 'attempts':{'sequence_allocated_slots':32,'sequence_sampling_attempts_lower_bound':16,
             'sequence_sampling_attempts_exact':None,'partial_interrupted_slots':16,
             'FREE':{'slots':16,'state':'partial_interrupted','reverse_steps_each_at_least':799},
             'AMD':{'slots':16,'state':'no_start_evidence'},
             'completed_sequence_exports':0,'raw_csv_status_counts':{'not_executed':32},
             'latent_only_attempts':0,
             'interpretation':'Raw CSV is preallocated. FREE progress proves real partial work; AMD has no start evidence. Unknown later unlogged work prevents an exact original attempt/time total.'},
 'development_access':True,'monitor_points':len(author['saved_dev_curve']),
 'test_access':False,'checkpoint_selection':'fixed final1000, not dev selected',
 'original_E_L_G':'NOT_EVALUABLE_INTERRUPTION',
 'note':'No final generation/export/manifest in original attempt; checkpoint recovery never backfills its status or cost.'}
events.append(original)
status_counts={x:sum(r['status']==x for r in sample) for x in sorted({r['status'] for r in sample})}
assert status_counts==postflight['execution_facts']['slot_status_counts']
events.append({
 'event_id':'CE_checkpoint_recovery_v1','method':'categorical_ce_checkpoint_recovery',
 'category':'additional_generation_restart','state':recovery['state'],'started_utc':job['started_utc'],
 'timing':{'supervisor_job_wall_seconds':job['seconds'],
           'method_process_wall_seconds':recovery['process_seconds_before_final_status_io'],
           'setup_wall_seconds':recovery['setup_seconds'],
           'generation_wall_seconds':generation['elapsed_seconds'],
           'export_wall_seconds':recovery['export_seconds'],
           'scope_note':'Use supervisor1124.252... once. Setup/generation/export are nested and not added again.'},
 'updates':{'AR':0,'VAE':0,'DDPM':0},
 'attempts':{'sequence_allocated_slots':32,'sequence_sampling_attempts':32,
             'completed_sequence_executions':32,'failed_sequence_executions':0,
             'not_executed_sequence_slots':0,'latent_only_attempts':0,
             'status_counts':status_counts},
 'development_access':True,'source_train_dev_rows_exported':[279,41],'test_access':False,
 'generation_seeds':{'FREE':422,'AMD':423},'training_seed_replicates_added':0,
 'training_weights_from':'original CEv2 fixed1000 VAE and DDPM EMA checkpoint',
 'protocol_deviation':'One separately authorized full32 generation restart violates original no-retry rule; not t200 continuation. Count added generation/cost; do not count as an independent training seed or new matched arm.',
 'recovery_engineering':postflight['status'],'original_E_L_G':'NOT_EVALUABLE_INTERRUPTION',
 'PROMOTE':'HOLD'})
for method in ('raw_onehot_mse','scalar_legacy'):
    events.append({'event_id':method+'_formal_not_started','method':method,'category':'formal_not_started',
      'state':'NOT_STARTED_PER_DELEGATED_SNAPSHOT','timing':None,
      'updates':{'AR':0,'VAE':0,'DDPM':0},
      'attempts':{'sequence_allocated_slots':0,'sequence_sampling_attempts':0,'latent_only_attempts':0},
      'planned_sequence_slots':32,'planned_updates':{'VAE':1000,'DDPM':1000},
      'planned_training_cap_seconds':2940,
      'note':'Snapshot only, no inferred launch from prepared configuration. Smoke remains separately charged.'})

old=v1['totals']; ce_lower=vae_sec+ddpm_sec+partial_sec+job['seconds']
totals={
 'historical_v1_eight_job_outer_wall_seconds':old['outer_supervisor_job_wall_seconds_sum'],
 'recovery_outer_wall_seconds':job['seconds'],
 'nine_terminal_job_supervisor_wall_seconds_sum':old['outer_supervisor_job_wall_seconds_sum']+job['seconds'],
 'original_CEv2_recorded_training_stage_subtotal_seconds':vae_sec+ddpm_sec,
 'original_CEv2_partial_generation_lower_bound_seconds':partial_sec,
 'original_CEv2_exact_total_seconds':None,
 'CEv2_plus_recovery_disjoint_wall_cost_lower_bound_seconds':ce_lower,
 'all_accounted_disjoint_wall_cost_lower_bound_seconds':old['outer_supervisor_job_wall_seconds_sum']+ce_lower,
 'overall_exact_wall_cost_seconds':None,
 'scope':'Nine terminal job outer wall times plus original interrupted CE recorded disjoint training stages and partial generation lower bound. Excludes unknown original work/setup gaps, unclocked diagnostics/reviews, and all future work. Not complete project CPU time.',
 'updates_by_family':{k:sum(e['updates'][k] for e in events) for k in ('AR','VAE','DDPM')},
 'sequence_allocated_slots':old['sequence_allocated_slots']+32+32,
 'sequence_sampling_attempts_observed_lower_bound':old['sequence_sampling_attempts']+16+32,
 'sequence_sampling_attempts_exact':None,
 'latent_only_attempts':old['latent_only_attempts'],
 'final_sequence_exports_with_completed_execution':34+32,
 'recorded_failed_sequence_executions':6,
 'original_CEv2_partial_FREE_slots':16,
 'CEv1_zero_update_failure_not_executed_slots':32,
 'original_CEv2_AMD_no_start_evidence_slots':16,
 'H4_reused_historical_rows_not_new_attempts':32,
 'formal_MSE_scalar_new_training_or_generation':0,
 'slot_partition_check':'136 = 66 completed exports + 6 numerical failures + 16 observed partial FREE + 32 CEv1 not_executed + 16 original AMD no-start-evidence. This is an evidence-state partition, not an exact unknown-exit runtime census.'}
assert totals['updates_by_family']=={'AR':1100,'VAE':1060,'DDPM':2060}
assert totals['sequence_allocated_slots']==136 and totals['sequence_sampling_attempts_observed_lower_bound']==88
op=copy.deepcopy(v1['development_opportunities'])
op['v1_accounting_retained_as_history']=op.pop('actual_opportunity_accounting')
op.update({
 'current_snapshot_additions':[
  'CEv2 completed original VAE1000/DDPM1000 and ten recorded dev monitoring points; did not complete original generation/export or obtain terminal process status.',
  'Saved FREE16 progress at t200 adds at least451.9357427917421s; original raw not_executed32 does not erase this cost.',
  'One new explicitly authorized checkpoint recovery reran all32 original seeds/chains, with zero training and1124.2520439173095s outer wall. This is additional development/execution opportunity and a documented no-retry deviation.',
  'Recovery read train279/dev41 for originally planned mean reconstructions and exported both. Same reused dev is not independent confirmation.',
  'CE remains one training seed. Recovery same seeds are a restart, not independent stochastic or training replication. Formal MSE/scalar are not started in this snapshot.',
  'Common2940s training cap remains a planned resource envelope. Actual cost is unequal, and CE total includes extra generation, failed entry, smoke and targeted DDPM diagnostic.',
  'AR1000 retained; evidence of overfit and AR_ADEQUACY=NOT_ESTABLISHED remain. No checkpoint/model/seed selection was added by this ledger.'
 ],
 'original_E':'NOT_EVALUABLE_INTERRUPTION','original_L':'NOT_EVALUABLE_INTERRUPTION',
 'original_G':'NOT_EVALUABLE_INTERRUPTION','recovery_engineering':'PASS_SCOPED_ENGINEERING_ONLY',
 'actual_compute_equal':False,'training_seed_replication':'one per executed formal method; not established across seeds',
 'development_set':'reused41, not an untouched holdout','four_arm_comparison':'NOT_COMPLETE',
 'promotion':'HOLD'})
ledger={'schema':'AMP_H5_COST_OPPORTUNITY_LEDGER_V2','created_utc':datetime.now(timezone.utc).isoformat(),
 'supersedes_as_snapshot_only':'cost_opportunity_ledger_v1 remains unchanged',
 'prior_manifest_sha256':SOURCES[v1root+'MANIFEST.json']['sha256'],
 'original_status_policy':'Preserve stale running bytes; classify via frozen forensic evidence, never invent exitcause/time/code or terminal status.',
 'events':events,'totals':totals,'development_opportunities':op,
 'no_models_or_source_train_dev_test_csv_read':True,'no_original_live_status_polling':True,
 'all_times_elapsed_wall_not_CPU_user_system_time':True,
 'PROMOTE':'HOLD','original_four_arm_E_L_G':'NOT_EVALUABLE_INTERRUPTION'}
dump('COST_OPPORTUNITY_LEDGER_V2.json',ledger)
flat=[]
for e in events:
 t=e.get('timing') or {};a=e.get('attempts') or {}
 flat.append({'event_id':e['event_id'],'category':e['category'],'state':e['state'],
  'outer_exact_wall_seconds':t.get('supervisor_job_wall_seconds'),
  'original_stage_and_partial_wall_lower_bound_seconds':t.get('recorded_disjoint_work_wall_seconds_lower_bound'),
  'AR_updates':e['updates']['AR'],'VAE_updates':e['updates']['VAE'],'DDPM_updates':e['updates']['DDPM'],
  'sequence_allocated_slots':a.get('sequence_allocated_slots'),
  'observed_sequence_attempts_or_lower_bound':a.get('sequence_sampling_attempts',a.get('sequence_sampling_attempts_lower_bound')),
  'latent_only_attempts':a.get('latent_only_attempts'),
  'attempt_count_is_lower_bound':e['event_id']=='CE_formal_v2_original_interrupted'})
with (OUT/'COSTS_V2.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
dump('SOURCE_BINDINGS.json',{'direct_read_sources':list(SOURCES.values()),
 'prior_v1_51_sources_transitively_bound_via_v1_manifest':True,
 'raw_original_live_files_read':0,'source_train_dev_test_csv_reads':0,'checkpoint_loads':0})
print(json.dumps(totals,indent=2))
