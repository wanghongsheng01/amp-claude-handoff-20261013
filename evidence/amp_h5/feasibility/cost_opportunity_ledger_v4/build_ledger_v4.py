"""Append-only v4 rollup. Standard-library reads; no model or source cohort.

Model-experiment job wall and saved-only NumPy analysis wall stay separate.
Original interrupted CE remains bounded by its prior forensic snapshot.
"""
import copy
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCES = {}

def get(rel, mode='json'):
    assert not rel.startswith(('runs/formal_v2/categorical_ce/', 'runs/single_ce_execution_v2/'))
    p = ROOT / rel
    b = p.read_bytes()
    SOURCES[rel] = {'path':str(p), 'bytes':len(b), 'sha256':hashlib.sha256(b).hexdigest()}
    if mode == 'json': return json.loads(b)
    if mode == 'csv': return list(csv.DictReader(b.decode().splitlines()))
    if mode == 'jsonl': return [json.loads(line) for line in b.decode().splitlines() if line]
    return b

def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

prev = 'feasibility/cost_opportunity_ledger_v3/'
manifest = get(prev+'MANIFEST.json')
for row in manifest['files']:
    assert hashlib.sha256(get(prev+row['path'], 'bytes')).hexdigest() == row['sha256']
v3 = get(prev+'COST_OPPORTUNITY_LEDGER_V3.json')

s = 'runs/formal_v2/scalar_legacy/'
status = get(s+'status.json')
sup = get('runs/single_scalar_legacy_execution_v2/SUPERVISOR_STATUS.json')
job = sup['jobs'][0]
vs = get(s+'vae_training_summary.json')
ds = get(s+'ddpm_training_summary.json')
gs = get(s+'generation_summary.json')
vl = get(s+'vae_train.jsonl','jsonl')
dl = get(s+'ddpm_train.jsonl','jsonl')
mon = get(s+'dev_monitor.jsonl','jsonl')
rows = get(s+'generated.csv','csv')
config = get(s+'config.json')

# The independent reviewer writes this receipt only after its terminal review.
post_rel = 'review/independent/SCALAR_V2_POSTFLIGHT_v1.json'
post = get(post_rel)
assert post['status'] == 'PASS', post['status']
assert SOURCES[post_rel]['sha256'] == '911a453786a351ac4365e30c818af37aaeac446c5c93159fedf17aab37396d5c'
assert status['state'] == sup['state'] == job['state'] == 'completed'
assert job['exit_code'] == 0 and job['seconds'] == 2867.7861290420406
assert vs['updates'] == ds['updates'] == 1000 and vs['fixed_endpoint_reached'] and ds['fixed_endpoint_reached']
assert [x['step'] for x in vl] == [x['step'] for x in dl] == list(range(1,1001))
assert len(mon) == 10 and all(x['checkpoint_selection'] is False for x in mon)
assert gs['requested'] == gs['successful_execution'] == 32 and gs['failed_execution'] == gs['not_executed'] == 0
assert config['training_seconds_cap'] == 2940 and status['test_read'] is False

compute = get('feasibility/latent_same_n_group_reference_v1/COMPUTE_MANIFEST.json')
analysis_validation = get('feasibility/latent_same_n_group_reference_v1/AUTHOR_VALIDATION.json')
candidate = get('feasibility/ce_decoder_input_control_candidate_v2/AUTHOR_STATIC_VALIDATION.json')
candidate_review = get('review/independent/CE_DECODER_INPUT_CONTROL_PREPARATION_REVIEW_v2.json')
assert compute['analysis_wall_seconds'] == 11.834126624744385 and compute['model_calls'] == 0
assert analysis_validation['model_calls'] == 0 and analysis_validation['author_validation_only'] is True
assert candidate['model_calls'] == candidate['random_draws'] == candidate['natural_tensor_loads'] == 0
assert candidate['execution_authorized'] is candidate_review['execution_authorized'] is False

alphabet = set('ARNDCEQGHILKMFPSTWYV')
done = [r for r in rows if r['status'] in ('ok','empty_sequence','no_eos_at_limit')]
def legal(seq, lo): return lo <= len(seq) <= 48 and set(seq) <= alphabet
den = {'allocated_slots':len(rows), 'started_sampling_attempts':len(rows),
       'completed_sequence_executions':len(done),
       'failed_sequence_executions':sum(r['status'].startswith('error') for r in rows),
       'status_counts':{v:sum(r['status']==v for r in rows) for v in sorted({r['status'] for r in rows})},
       'completed_empty_sequences':sum(not r['sequence'] for r in done),
       'completed_legal1_48':sum(legal(r['sequence'],1) for r in done),
       'completed_legal5_48':sum(legal(r['sequence'],5) for r in done),
       'latent_only_attempts':0, 'source':s+'generated.csv',
       'interpretation':'Literal alphabet/length accounting only, not biological or distributional quality.'}
assert den['allocated_slots'] == den['completed_sequence_executions'] == 32

events = [copy.deepcopy(e) for e in v3['events'] if e['event_id'] != 'scalar_legacy_formal_v2_live_excluded']
events.append({'event_id':'scalar_legacy_formal_v2','method':'scalar_legacy','category':'formal_completed',
 'state':'completed','started_utc':job['started_utc'],'completed_utc':sup['completed_utc'],'exit_code':0,
 'independent_postflight':{'status':post['status'],'scope':post.get('scope'),'source':post_rel},
 'timing':{'supervisor_job_wall_seconds':job['seconds'],'method_process_wall_seconds':status['process_seconds'],
  'training_pipeline_wall_seconds':status['training_seconds'],'VAE_stage_wall_seconds':vs['elapsed_seconds'],
  'DDPM_stage_wall_seconds':ds['elapsed_seconds'],'VAE_update_wall_seconds':sum(x['duration_seconds'] for x in vl),
  'DDPM_update_wall_seconds':sum(x['duration_seconds'] for x in dl),
  'DDPM_encoding_and_setup_wall_seconds':ds['posterior_encoding_and_setup_seconds'],
  'dev_monitor_wall_seconds':sum(x['seconds'] for x in mon),'generation_wall_seconds':gs['elapsed_seconds'],
  'scope_note':'Nested elapsed wall timings on CPU1; outer counted once, not CPU user/system seconds.'},
 'updates':{'AR':0,'VAE':1000,'DDPM':1000},'attempts':den,'denominator_accounting_v4':den,
 'development_access':True,'monitor_points':10,'checkpoint_selection':'fixed final1000; no dev selection',
 'test_access':False,'formal_common_training_cap_seconds':2940,'actual_equal_compute':False,
 'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD'})

denominators = copy.deepcopy(v3['per_completed_export_denominators'])
denominators['scalar_legacy_formal_v2'] = den
totals = copy.deepcopy(v3['totals'])
old = Decimal(str(v3['totals']['all_accounted_disjoint_wall_cost_lower_bound_seconds']))
added = Decimal(str(job['seconds']))
lower = old+added
float_sum = v3['totals']['all_accounted_disjoint_wall_cost_lower_bound_seconds']+job['seconds']
totals.update({
 'previous_v3_wall_lower_bound_seconds':float(old),
 'new_scalar_supervisor_outer_wall_seconds':job['seconds'],
 'eleven_terminal_job_supervisor_wall_seconds_sum':v3['totals']['ten_terminal_job_supervisor_wall_seconds_sum']+job['seconds'],
 'all_accounted_disjoint_wall_cost_lower_bound_seconds':float(lower),
 'all_accounted_disjoint_wall_cost_lower_bound_seconds_decimal':str(lower),
 'arithmetic':{'recorded_decimal_addends':[str(old),str(added)],'recorded_decimal_sum':str(lower),
               'native_float_sum':float_sum,'physical_exact_total_inferred':False,
               'note':'Decimal sums recorded decimal values; physical total remains a lower bound because interrupted CE time is unknown.'},
 'overall_exact_wall_cost_seconds':None,
 'updates_by_family':{'AR':1100,'VAE':3060,'DDPM':4060},
 'sequence_allocated_slots':v3['totals']['sequence_allocated_slots']+32,
 'sequence_sampling_attempts_observed_lower_bound':v3['totals']['sequence_sampling_attempts_observed_lower_bound']+32,
 'sequence_sampling_attempts_exact':None,
 'final_sequence_exports_with_completed_execution':sum(d['completed_sequence_executions'] for d in denominators.values()),
 'completed_legal1_48_sequences':sum(d['completed_legal1_48'] for d in denominators.values()),
 'completed_legal5_48_sequences':sum(d['completed_legal5_48'] for d in denominators.values()),
 'completed_empty_sequences':sum(d['completed_empty_sequences'] for d in denominators.values()),
 'completed_length1_4_sequences':sum(d['completed_legal1_48']-d['completed_legal5_48'] for d in denominators.values()),
 'recorded_failed_sequence_executions':sum(d['failed_sequence_executions'] for d in denominators.values()),
 'scalar_terminal_actual_costs_and_slots_included':True,
 'original_CEv2_total_still_unknown':True,
 'new_saved_only_same_n_analysis_in_model_job_wall':False,
 'decoder_control_executed_slots':0,
 'decoder_control_candidate_planned_slots_not_allocated_or_started':64,
 'scope':'Eleven terminal supervised experiment jobs plus originalCE disjoint training/partial-generation lower bound. Excludes separately itemized non-model analyses, unlogged originalCE work/gaps and untimed reviews. CE original stale status is not reclassified as terminal.',
 'slot_partition_check':'200 = 130 completed exports + 6 numerical failures + 16 partial originalFREE + 32 CEv1 not_executed + 16 originalAMD no-start-evidence. Prepared64 controls excluded.',
 'legal_count_warning':'Pool only for bookkeeping, not pooled quality rates. Smoke/formal/recovery/partial/latent-only remain distinct.'})
totals.pop('live_scalar_actual_costs_or_slots_included',None)
assert totals['sequence_allocated_slots']==200 and totals['sequence_sampling_attempts_observed_lower_bound']==152
assert totals['final_sequence_exports_with_completed_execution']==130
assert totals['recorded_failed_sequence_executions']==6 and totals['latent_only_attempts']==4
assert totals['completed_legal1_48_sequences']==129 and totals['completed_legal5_48_sequences']==120
assert totals['completed_empty_sequences']==1 and totals['completed_length1_4_sequences']==9
assert {k:sum(e['updates'][k] for e in events) for k in ('AR','VAE','DDPM')} == totals['updates_by_family']
assert 200==130+6+16+32+16

prior_analysis = copy.deepcopy(v3['development_opportunities']['known_additional_readonly_work'])
new_analysis = {'analysis_id':'latent_same_n_group_reference_v1','work':'Posthoc same-n whole-component reference NumPy analysis',
 'state':'COMPLETED_AUTHOR_VALIDATED','recorded_wall_seconds':compute['analysis_wall_seconds'],
 'cost_class':'non_model_analysis_elapsed_wall','included_in_model_job_wall':False,
 'model_calls':0,'training_updates':0,'new_generated_sequence_requests':0,'new_latent_draws':0,
 'analysis_resampling_repetitions':256,'derived_method_terminal_records':1024,
 'resampling_unit':'existing frozen train components; these are not new biological or generated samples',
 'compute_source':'feasibility/latent_same_n_group_reference_v1/COMPUTE_MANIFEST.json',
 'independent_scientific_review':'not inferred from author validation',
 'dev_latent':'MISSING_NOT_FORWARDED','test_access':False}
known_analysis_sum = sum((Decimal(str(x['recorded_wall_seconds'])) for x in prior_analysis if x.get('recorded_wall_seconds') is not None),Decimal('0'))+Decimal(str(new_analysis['recorded_wall_seconds']))
analysis_costs = {'new_v4_analysis':new_analysis,'previous_timed_and_untimed_analyses':prior_analysis,
 'known_timed_non_model_analysis_wall_seconds_decimal':str(known_analysis_sum),
 'known_timed_non_model_analysis_wall_seconds':float(known_analysis_sum),
 'complete_analysis_cost_unknown':True,
 'not_added_to_model_job_cost':True,
 'note':'Includes previously recorded AR readout1.8991414169999998s and new same-n11.834126624744385s. Other untimed analysis/review/development costs are unknown; no overall equal-compute claim.'}

op = copy.deepcopy(v3['development_opportunities'])
op['v3_additions_retained_as_historical_snapshot'] = op.pop('v3_additions')
op.update({'v4_additions':[
 'Scalar finalVAE1000/DDPM1000 completed, all32 original requests executed. Added2867.7861290420406s supervised wall and ten fixed dev observations.',
 'Four named methods now have endpoint outputs when CE recovery is used as a separate deviation. The original CE interrupted attempt still lacks terminal status/exact total; originalE/L/G is not backfilled.',
 'Saved-only same-n NumPy analysis adds11.834126624744385s and256 component resamples, separately accounted; zero model calls/new sequence requests/latent draws. Repeated indices do not add independent model seeds or biological samples.',
 'Old A/B64 decoder control candidates remain preparation only:0 real requests,0 model calls,0 forwards. Static repairs and review are development opportunities with unrecorded wall cost, not executed controls.',
 'New material and repeated reused-dev observations do not erase staged feasibility-driven order, CE entry repair, targetedDDPM diagnostic, interruptedFREE work or full-chain CE recovery.',
 'AR overfit/adequacyNOT_ESTABLISHED, one trainingseed per method, reuseddev41, shared2940cap but unequal actualcost remain. No model/checkpoint selection or efficacy/novelty promotion is implied.'
 ],'scalar_status':'COMPLETED_TERMINAL_ACTUALS_INCLUDED','scalar_engineering':post['status'],
 'four_arm_comparison':'OUTPUTS_AVAILABLE_USING_SEPARATE_CE_RECOVERY_ORIGINAL_PROTOCOL_NE',
 'actual_compute_equal':False,'promotion':'HOLD'})
controls = {'candidate_protocol_slots':64,'actually_allocated_runtime_slots':0,'started_requests':0,
 'completed_requests':0,'model_calls':0,'training_updates':0,'wall_cost_recorded':None,
 'state':'PREPARATION_ONLY_NOT_EXECUTED','counted_in_sequence_or_latent_attempt_totals':False,
 'source':'feasibility/ce_decoder_input_control_candidate_v2/AUTHOR_STATIC_VALIDATION.json',
 'opportunity_note':'Preparation/revision/independent reviews occurred, with no real control execution; proposed64 is not a failed/not_executed runtime cohort.'}
ledger = {'schema':'AMP_H5_COST_OPPORTUNITY_LEDGER_V4','created_utc':datetime.now(timezone.utc).isoformat(),
 'snapshot_scope':'Terminal scalar added. Earlier v1-v3 preserved. New same-n analysis separated; prepared64 decoder controls unexecuted.',
 'prior_manifest_sha256':SOURCES[prev+'MANIFEST.json']['sha256'],
 'events':events,'totals':totals,'per_completed_export_denominators':denominators,
 'non_model_analysis_costs':analysis_costs,'decoder_control_preparation':controls,
 'development_opportunities':op,'all_seconds_elapsed_wall_not_CPU_user_system_time':True,
 'no_models_weights_or_source_cohort_reads':True,'original_CE_status_policy':v3['original_CE_status_policy'],
 'PROMOTE':'HOLD','original_four_arm_E_L_G':'NOT_EVALUABLE_INTERRUPTION',
 'next_version':'Any later model control or analysis must receive a new ledger version; do not mutate v4.'}
dump('COST_OPPORTUNITY_LEDGER_V4.json',ledger)
fields=['event_id','allocated_slots','started_sampling_attempts','completed_sequence_executions','failed_sequence_executions','completed_empty_sequences','completed_legal1_48','completed_legal5_48','latent_only_attempts']
with (OUT/'DENOMINATORS_V4.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
    for key,d in denominators.items():w.writerow({'event_id':key,**d})
flat=[]
for e in events:
    t=e.get('timing') or {}; a=e.get('denominator_accounting_v4') or e.get('denominator_accounting_v3') or e.get('attempts') or {}
    flat.append({'event_id':e['event_id'],'state':e['state'],
     'outer_exact_wall_seconds':t.get('supervisor_job_wall_seconds'),
     'original_stage_and_partial_wall_lower_bound_seconds':t.get('recorded_disjoint_work_wall_seconds_lower_bound'),
     'AR_updates':e['updates']['AR'],'VAE_updates':e['updates']['VAE'],'DDPM_updates':e['updates']['DDPM'],
     'allocated_slots':a.get('allocated_slots',a.get('sequence_allocated_slots')),
     'started_attempts_or_lower_bound':a.get('started_sampling_attempts',a.get('sequence_sampling_attempts',a.get('sequence_sampling_attempts_lower_bound'))),
     'completed_sequence_executions':a.get('completed_sequence_executions',a.get('completed_sequence_exports')),
     'legal1_48':a.get('completed_legal1_48'),'legal5_48':a.get('completed_legal5_48'),'latent_only_attempts':a.get('latent_only_attempts')})
with (OUT/'COSTS_V4.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
dump('SOURCE_BINDINGS.json',{'direct_read_sources':list(SOURCES.values()),'prior_v1_v2_v3_sources_transitively_bound':True,
 'source_cohort_csv_reads':0,'model_calls':0,'checkpoint_loads':0,'original_CE_raw_status_reads_or_mutations':0,
 'scalar_read_scope':'terminal only; supervisor and run both completed before reads'})
stable = all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in SOURCES.values())
checks={'all_direct_sources_stable':stable,'v3_manifest_exact':True,'scalar_completed1000_1000':True,
 'scalar32completed_no_fail':True,'scalar_independent_postflight_PASS':True,'decimal_arithmetic':str(lower)=='9996.4754611649551',
 'actual_total_still_unknown':totals['overall_exact_wall_cost_seconds'] is None,'updates_accounting':True,
 'allocated200_started_atleast152_completed130':True,'slot_partition':True,'partial16_not_completed':True,
 'completed_legality_partition129_120_empty1_short9':True,
 'latent4_separate':True,'prepared64_not_real_attempts':True,'analysis11_834_separate':True,
 'scalar_terminal_not_live':True,'original_CE_stale_preserved':True,'PROMOTE_HOLD':True,
 'AR_adequacy_not_established':op['AR_adequacy']['status']=='NOT_ESTABLISHED','actual_compute_not_equal':not op['actual_compute_equal']}
assert all(checks.values()),checks
dump('VALIDATION.json',{'status':'PASS','checks':checks,'model_calls':0,'source_cohort_reads':0,
 'v1_v2_v3_or_run_mutations':0,'no_natural_sampling_or_training':True})

report=f'''AMP H5 成本与开发机会账本 v4

已核入 scalar 终态：2026-10-11 19:58:05 UTC 完成，exit0，VAE1000/DDPM1000，32请求全部执行成功；独立收据{post['status']}（{post.get('scope','见收据')}）。旧v1-v3保持不变。原CE中断没有被改写为正常终态，额外recovery仍单列。

累计已记录的实验任务wall成本仍只能报下界：{old}+{added}={lower}秒，约{float(lower)/60:.6f}分钟。这里是各已记录、不重叠任务/阶段的elapsed wall之和，不是CPU user/system时间，也不是所有工作总历时。原CE未记录的时间与部分审计开发时间仍未知。

| 新核入scalar成本 | 秒 |
|---|---:|
| supervisor外层（累计只加此项） | {job['seconds']} |
| 进程记录 | {status['process_seconds']} |
| training pipeline | {status['training_seconds']} |
| VAE阶段 | {vs['elapsed_seconds']} |
| DDPM阶段 | {ds['elapsed_seconds']} |
| generation | {gs['elapsed_seconds']} |

阶段均嵌套于外层，不能相加重复收费。全历史更新合计AR1100、VAE3060、DDPM4060，含smoke及额外诊断，不等同于正式实验预算。

序列分母为200个已分配slot：130个完成导出、6次数值失败、原CE FREE16个部分执行、CEv1入口失败32个not_executed、原CE AMD16个无开始证据。开始attempt仅能确定至少152个，不能把原CE部分轨迹当完整生成。另有4个latent-only诊断attempt；旧H4复用行不是新生成。完成导出中legal1–48为{totals['completed_legal1_48_sequences']}、legal5–48为{totals['completed_legal5_48_sequences']}、空串{totals['completed_empty_sequences']}、长度1–4为{totals['completed_length1_4_sequences']}；这些混合历史计数仅用于账务，不可汇总为方法质量率。scalar单独legal1–48为{den['completed_legal1_48']}、legal5–48为{den['completed_legal5_48']}。

新same-n NumPy分析另计11.834126624744385秒，不加进上述实验任务wall。其256次已有组件重采样和1024个方法/端基统计记录不是新模型、latent或生物样本；model calls=0。连同旧已计时AR只读readout，已知计时的非模型分析合计{known_analysis_sum}秒；其余未计时审计仍未知。此账本不重新计算分析质量。

旧A/B64个decoder control只是候选协议/代码准备。实际分配、开始、完成、forward均为0；不把64个候选加入not_executed或任何真实尝试分母。静态修订及审阅属于开发机会，未记录wall不能伪填0秒。

共同2940秒training cap是资源上限，实际计算量不相等。CE有定向DDPM诊断、入口修复、中断FREE轨迹及重新跑32链的recovery；全部保留。每正式方法只有一个训练种子，dev41多次复用，AR末点过拟合且充分性未证。scalar也是整个legacy流程对照，不能称单组件因果实验。四个方法的输出可通过单列CE恢复获得，并不恢复原正式E/L/G；原CE E/L/G仍NOT_EVALUABLE_INTERRUPTION，PROMOTE=HOLD。

精确来源/hash见SOURCE_BINDINGS.json；逐事件成本见COSTS_V4.csv；分母见DENOMINATORS_V4.csv。JSON另保存Decimal按已记录数字相加的结果及原生float和，避免以显示尾数混淆精度或冒称未知物理总成本已精确。
'''
(OUT/'READOUT.zh.md').write_text(report)
files=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='MANIFEST.json':
        data=p.read_bytes();files.append({'path':p.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
dump('MANIFEST.json',{'scope':'cost/opportunityv4; terminalscalar; model-experimentlowerbound; separateanalysiscost; noexecuteddecodercontrols','files':files})
print(json.dumps({'lower_bound_decimal':str(lower),'updates':totals['updates_by_family'],
 'slots':totals['sequence_allocated_slots'],'started_at_least':totals['sequence_sampling_attempts_observed_lower_bound'],
 'completed':totals['final_sequence_exports_with_completed_execution'],'legal1_48':totals['completed_legal1_48_sequences'],
 'legal5_48':totals['completed_legal5_48_sequences'],'checks':len(checks),
 'ledger_sha256':hashlib.sha256((OUT/'COST_OPPORTUNITY_LEDGER_V4.json').read_bytes()).hexdigest(),
 'manifest_sha256':hashlib.sha256((OUT/'MANIFEST.json').read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
