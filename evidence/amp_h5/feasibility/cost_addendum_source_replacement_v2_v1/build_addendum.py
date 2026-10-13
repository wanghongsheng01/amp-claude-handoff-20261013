"""Additive cost rollup after one completed source-v2 diagnostic.

No model/weight/source-cohort reads; v4 and all earlier files stay immutable.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCES = {}
def read(rel, kind='json'):
    assert not rel.startswith(('runs/formal_v2/categorical_ce/','runs/single_ce_execution_v2/'))
    p=ROOT/rel
    b=p.read_bytes()
    SOURCES[rel]={'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    if kind=='json': return json.loads(b)
    if kind=='csv': return list(csv.DictReader(b.decode().splitlines()))
    return b
def dump(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def dec(x): return Decimal(str(x))

ap=argparse.ArgumentParser()
ap.add_argument('--postflight',required=True,help='Root-relative immutable independent postflight JSON')
ap.add_argument('--evaluation-result',required=True,help='Root-relative completed read-only source diagnostic evaluator RESULT JSON')
ap.add_argument('--author-readout-result',required=True,help='Root-relative completed source diagnostic author readout RESULT JSON')
args=ap.parse_args()
assert not (OUT/'MANIFEST.json').exists(), 'Do not overwrite a sealed addendum'

oldbase='feasibility/cost_opportunity_ledger_v4/'
om=read(oldbase+'MANIFEST.json')
assert SOURCES[oldbase+'MANIFEST.json']['sha256']=='91ae0596763a3d09f5eacd02ecc0948d578bc7c79f576d0ba041f5f7b8ee1ddc'
for f in om['files']:
    assert hashlib.sha256(read(oldbase+f['path'],'bytes')).hexdigest()==f['sha256']
old=read(oldbase+'COST_OPPORTUNITY_LEDGER_V4.json')

run='runs/ce_source_replacement_v2/'
supervisor=read('runs/ce_source_replacement_execution_v2/SUPERVISOR_STATUS.json')
job=supervisor['jobs'][0]
status=read(run+'status.json')
generation=read(run+'generation_summary.json')
closing=read(run+'CLOSING_LEDGER.json')
slots=read(run+'REQUEST_SLOTS.json')
reused=read(run+'REUSED_DDPM_SLOTS.json')
rows=read(run+'generated.csv','csv')
contract=read(run+'contract.json')
post=read(args.postflight)
assert SOURCES[args.postflight]['sha256']=='01fe09cbce4c4d4d8cf6c04d617408de40162b02f4583cb2a0c095418535cac3'
assert str(post['status']).startswith('PASS'),post['status']
assert post['scope_id']=='AMP_H5_CE_SOURCE_REPLACEMENT_V2' and post['independent_check_count']==63
assert post['audit_model_forwards']==post['audit_test37_reads']==post['audit_dev41_reads']==0
assert supervisor['state']==job['state']==status['state']==generation['state']==closing['state']=='completed'
assert job['exit_code']==0 and job['seconds']==2.679257582873106
assert supervisor['plan_sha256']==status['plan_sha256']=='91e64d4ca473c1a56b2c66b285f0f5b8fd0ccba3884fe15ba465e574108bb46f'
assert status['scope_id']=='AMP_H5_CE_SOURCE_REPLACEMENT_V2'
assert generation['requested']==generation['completed']==slots['count']==status['completed_inputs']==96
assert generation['failed']==generation['not_executed']==status['failed_inputs']==status['not_executed_inputs']==0
assert status['new_training_updates']==status['DDPM_forward_attempts']==status['encoder_forward_attempts']==0
assert status['decoder_forward_attempts']==6 and status['prior_inverse_attempts']==status['prior_roundtrip_forward_attempts']==2
assert len(reused['rows'])==reused['old_rows_referenced']==status['reused_DDPM_rows']==generation['reused_prior_rows']==32
assert reused['new_generation_calls']==0
assert len(rows)==96 and len({r['sample_id'] for r in rows})==96
assert all(r['ranking_eligible']=='false' for r in rows)
groups=[]
for method,terminal in sorted({(r['method'],r['terminal']) for r in rows}):
    group=[r for r in rows if (r['method'],r['terminal'])==(method,terminal)]
    counts=Counter(r['status'] for r in group)
    assert len(group)==16 and set(counts)<=set(('ok','empty_sequence','no_eos_at_limit'))
    groups.append({'method':method,'terminal':terminal,'allocated_new_requests':16,
                   'started_requests':16,'completed_requests':16,'failed_requests':0,
                   'not_executed_requests':0,'status_counts':dict(counts),
                   'sample_id_set_sha256':hashlib.sha256('\n'.join(r['sample_id'] for r in group).encode()).hexdigest()})
assert len(groups)==6

analysis_sources=[
 ('same_n_independent_postflight','review/independent/LATENT_SAME_N_GROUP_REFERENCE_POSTFLIGHT_v1.json','audit_seconds'),
 ('four_arm_evaluation','feasibility/ce_recovery_four_arm_execution_v1/EVALUATION_RESULT.json','wall_seconds'),
 ('four_arm_readout_v4','feasibility/ce_recovery_four_arm_execution_v1/V4_READOUT_RESULT.json','wall_seconds'),
 ('four_arm_full_metrics_addendum','feasibility/ce_recovery_four_arm_execution_v1/ADDENDUM_RESULT.json','wall_seconds')]
new_analyses=[]
for name,path,field in analysis_sources:
    d=read(path)
    if 'returncode' in d: assert d['returncode']==0 and d['model_calls']==0
    else: assert d['status']=='PASS_SAVED_ARRAY_SAME_N_DESCRIPTIVE_ANALYSIS'
    new_analyses.append({'analysis_id':name,'source':path,'wall_seconds':d[field],
                         'cost_class':'non_model_analysis_elapsed_wall',
                         'new_model_requests':0,'included_in_model_job_cost':False})

source_eval=read(args.evaluation_result)
assert SOURCES[args.evaluation_result]['sha256']=='5df0d72cbf5f728f3e8555cb05df6487dcac8ab91a17335acc03de6bfd26dd28'
assert source_eval['returncode']==0 and source_eval['model_calls']==0
assert source_eval['test_access'] is False
assert source_eval['wall_seconds'] is not None
new_analyses.append({'analysis_id':'source_v2_readonly_evaluation','source':args.evaluation_result,
                     'wall_seconds':source_eval['wall_seconds'],'cost_class':'non_model_analysis_elapsed_wall',
                     'new_model_requests':0,'included_in_model_job_cost':False})
source_author=read(args.author_readout_result)
assert SOURCES[args.author_readout_result]['sha256']=='f28df5eb904c0f03542fd92f5bf9db527cba4b1188a1f1b92f93df8b1844b689'
assert source_author['returncode']==0 and source_author['model_calls']==0
assert source_author['test_access'] is False and source_author['metric_recomputations']==0
assert source_author['wall_seconds'] is not None
new_analyses.append({'analysis_id':'source_v2_author_readout','source':args.author_readout_result,
                     'wall_seconds':source_author['wall_seconds'],'cost_class':'non_model_analysis_elapsed_wall',
                     'new_model_requests':0,'included_in_model_job_cost':False})

# Two disjoint, actually timed audit sections; the summary sum is not a third charge.
audit_costs=post['independent_analysis_costs_seconds']
audits=[('source_v2_initial_audit_reviewer_label_error',
         'review/independent/CE_SOURCE_REPLACEMENT_TERMINAL_AUDIT_v2.json',
         'initial_audit_with_reviewer_label_error','FAIL',
         '7b2cab373086c7531885d60a9ef713565415e6016c4be73c0260e13477b1c8c1'),
        ('source_v2_corrected_complete_audit',
         'review/independent/CE_SOURCE_REPLACEMENT_TERMINAL_AUDIT_v2_r1.json',
         'corrected_complete_audit','PASS_CHECKS_PENDING_INDEPENDENT_INTERPRETATION',
         '19a6b240e095f8f4b333643f16c738008fb4e622733cd15bc3d0662a4da99df4')]
for name,path,field,expected_status,expected_sha in audits:
    audit=read(path)
    assert SOURCES[path]['sha256']==expected_sha and audit['status']==expected_status
    assert audit['audit_seconds']==audit_costs[field]
    assert audit['audit_model_forwards']==audit['audit_model_constructions']==audit['audit_new_sampling_requests']==0
    assert audit['test37_reads']==audit['dev41_reads']==0
    new_analyses.append({'analysis_id':name,'source':path,'postflight_source':args.postflight,
                         'wall_seconds':audit['audit_seconds'],'cost_class':'independent_audit_timed_section_wall',
                         'new_model_requests':0,'included_in_model_job_cost':False,
                         'timing_note':audit_costs['note'],
                         'audit_status':audit['status'],
                         'initial_failure_is_reviewer_only_not_model_failure':True})
assert abs(sum(dec(audit_costs[f]) for f in ['initial_audit_with_reviewer_label_error','corrected_complete_audit'])
           -dec(audit_costs['sum_measured_audit_process_sections'])) < Decimal('0.000000000000001')
assert audit_costs['other_interactive_review_and_report_wall_time'] is None
new_analyses.append({'analysis_id':'source_v2_other_interactive_review_and_report',
                     'source':args.postflight,'wall_seconds':None,
                     'cost_class':'independent_review_unmeasured_wall','new_model_requests':0,
                     'included_in_model_job_cost':False,
                     'timing_note':'Unmeasured interactive review and writing remains null; no overlap charge for the two timed sections.'})

prior_model=Decimal(old['totals']['all_accounted_disjoint_wall_cost_lower_bound_seconds_decimal'])
new_model=dec(job['seconds'])
combined_model=prior_model+new_model
prior_analysis=Decimal(old['non_model_analysis_costs']['known_timed_non_model_analysis_wall_seconds_decimal'])
analysis_increment=sum((dec(a['wall_seconds']) for a in new_analyses if a['wall_seconds'] is not None),Decimal('0'))
combined_analysis=prior_analysis+analysis_increment
assert analysis_increment==Decimal('39.97295353971807278')
assert combined_analysis==Decimal('53.70622158146245758')
counts={'allocated_slots':old['totals']['sequence_allocated_slots']+96,
 'started_attempts_observed_lower_bound':old['totals']['sequence_sampling_attempts_observed_lower_bound']+96,
 'completed_sequence_executions':old['totals']['final_sequence_exports_with_completed_execution']+96,
 'recorded_failed_sequence_executions':old['totals']['recorded_failed_sequence_executions'],
 'original_CE_partial_FREE_slots':16,'CEv1_entry_failure_not_executed_slots':32,
 'original_CE_AMD_no_start_evidence_slots':16,'latent_only_historical_attempts_separate':4,
 'new_DDPM_reused_rows_not_attempts':32,'old_H4_reused_rows_not_attempts':32,
 'scope':'Mixed historical/formal/posthoc scopes combined ONLY for execution bookkeeping; not a quality denominator.'}
assert counts['allocated_slots']==296 and counts['started_attempts_observed_lower_bound']==248
assert counts['completed_sequence_executions']==226 and 296==226+6+16+32+16
assert combined_model==Decimal('9999.154718747828206')
for entry in new_analyses[:4]: assert entry['wall_seconds'] is not None

timing={'supervisor_outer_wall_seconds':job['seconds'],
 'process_seconds_before_final_io':status['process_seconds_before_final_io'],
 'process_seconds_after_manifest_before_closing_ledger':closing['process_seconds_after_manifest_before_closing_ledger'],
 'phase_seconds_as_recorded':status['phase_seconds'],
 'scope_note':'All process/phase measures are nested in outer. Add outer exactly once. CPU elapsed wall, not CPU utilization or user/system time.'}
control={'scope_id':status['scope_id'],'attempt_id':status['attempt_id'],'state':'completed',
 'started_utc':job['started_utc'],'completed_utc':supervisor['completed_utc'],'exit_code':0,
 'independent_postflight_status':post['status'],'independent_postflight_source':args.postflight,
 'timing':timing,'new_requests':96,'completed':96,'failed':0,'not_executed':0,
 'groups':groups,'reused_DDPM_rows':32,'reused_DDPM_new_requests':0,
 'legal_sequence_count':None,'legality_note':'Execution completion/status is recorded here; literal legality is a separate evaluator metric and is not recomputed or inferred by this ledger.',
 'batch_forward_calls':{'decoder':6,'prior_inverse':2,'prior_roundtrip_forward':2,'DDPM':0,'encoder':0},
 'training_updates':0,'test_read':False,'dev_read':False,
 'train_source_read_started':status['train_source_read_started'],
 'train_source_read_validated':status['train_source_read_validated'],
 'fresh_conditional_or_independent_biological_samples_inferred':False,
 'method_boundary':'B empirical train posterior; C coordinate-permuted existing DDPM z; D reference-g conditional flow prior. Reused DDPM32 supplies context, not32 additional executions. All diagnostic groups excluded from formal quality ranking.',
 'original_CE_E_L_G':status['original_CE_E_L_G'],'exploratory_gates':status['exploratory_gates'],'PROMOTE':'HOLD'}

uncounted=[{'item':'old A/B64 candidate v1/v2 preparation','model_attempts':0,'wall_seconds':None},
 {'item':'source replacement v1 blocked by R-S1 static review','model_attempts':0,'wall_seconds':None},
 {'item':'v2 R-S1 repair, synthetic checks and planning/preflight','model_attempts':0,'wall_seconds':None},
 {'item':'transport outage, metadata writes/verification before actual launch','model_attempts':0,'wall_seconds':None},
 {'item':'other historical untimed code review/writing/analysis','model_attempts':None,'wall_seconds':None,
  'note':'No numeric cost inferred; model attempt totals come only from explicit execution evidence.'},
 {'item':'root/literature final reporting without a recorded timer','model_attempts':0,'wall_seconds':None}]
addendum={'schema':'AMP_H5_COST_ADDENDUM_SOURCE_REPLACEMENT_V2_V1',
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'status':'SEALED_TERMINAL_DIAGNOSTIC_COST_ADDITION','v4_unchanged':True,
 'v4_manifest_sha256':SOURCES[oldbase+'MANIFEST.json']['sha256'],
 'new_model_diagnostic':control,
 'model_job_wall_cost':{'previous_v4_lower_bound_decimal':str(prior_model),'new_outer_wall_decimal':str(new_model),
  'combined_lower_bound_decimal':str(combined_model),'combined_lower_bound_seconds':float(combined_model),
  'overall_exact_total_seconds':None,'original_CE_exact_total_still_unknown':True,
  'complete_total_cost_including_all_analysis_known':False},
 'non_model_analysis_cost':{'v4_known_timed_cost_decimal':str(prior_analysis),
  'v4_cost_not_charged_twice':'AR readout1.8991414169999998 + same-n author11.834126624744385 already in v4.',
  'newly_accounted_entries':new_analyses,'newly_accounted_timed_cost_decimal':str(analysis_increment),
  'combined_known_timed_cost_decimal':str(combined_analysis),
  'all_analysis_cost_seconds':None,'not_added_to_model_job_wall':True},
 'aggregate_execution_bookkeeping':counts,
 'unchanged_historical_training_updates':old['totals']['updates_by_family'],
 'unexecuted_preparations_and_unknown_costs':uncounted,
 'development_opportunity_note':'This is one additional posthoc model diagnostic and multiple extra saved-data analyses/reviews, not a fifth formal arm or replication. Shared2940cap does not imply equal actual compute; ARadequacy unproven, one trainingseed/method and reuseddev41 remain.',
 'no_quality_metrics_recomputed':True,'no_source_cohort_reads_by_ledger_author':True,
 'model_calls_by_ledger_author':0,'original_protocol_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD'}
dump('COST_ADDENDUM.json',addendum)
with (OUT/'NEW_CONTROL_DENOMINATORS.csv').open('w',newline='') as f:
    fields=['method','terminal','allocated_new_requests','started_requests','completed_requests','failed_requests','not_executed_requests']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(groups)
with (OUT/'NEW_ANALYSIS_COSTS.csv').open('w',newline='') as f:
    fields=['analysis_id','source','wall_seconds','cost_class','included_in_model_job_cost']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(new_analyses)
dump('SOURCE_BINDINGS.json',{'files':list(SOURCES.values()),'source_cohort_reads':0,
 'model_or_weight_loads':0,'test_reads':0,'old_file_edits':0})
stable=all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in SOURCES.values())
checks={'all_direct_sources_stable':stable,'v4_manifest_and_all_members_unchanged':True,
 'single_terminal_exit0_source_v2_job':True,'final_independent_postflight_bound':True,
 'source_readonly_evaluation_completed_and_timed':True,
 'source_author_readout_completed_and_timed':True,
 'both_source_audit_sections_counted_once':True,
 'reviewer_label_failure_not_natural_model_failure':True,
 '96_allocated_started_completed_no_failure':True,'6_groups16_each':True,
 '6_decoder_2_inverse_2_roundtrip_batches':True,'zero_training_encoder_DDPM':True,
 'reused32_not_new_attempts':True,'old64_and_source_v1_not_model_attempts':True,
 'model_outer_only_once':True,'decimal_combined_model_lower_bound':True,
 'mixed_scope296_248_226_bookkeeping_only':True,'known_analysis_not_model_wall':True,
 'v4_analysis_not_duplicated':True,'missing_costs_null_not_zero':True,
 'original_CE_unknown_and_partial_preserved':True,'original_ELG_NE_PROMOTE_HOLD':True}
assert all(checks.values()),checks
dump('VALIDATION.json',{'status':'PASS','checks':checks,'new_model_calls':0,'new_quality_statistics':0})
text=f'''AMP H5 source_v2 独立成本追加

仅追加这次真实完成的 source_v2 诊断，旧v4及此前报告均不改写。任务于{job['started_utc']}启动、{supervisor['completed_utc']}结束，exit0；独立后审{post['status']}。外层wall为{job['seconds']}秒。

新增96个请求全部完成，0失败、0未执行。B/C/D各32，按FREE/AMD各16；实际批调用为6次decoder、2次inverse flow、2次prior roundtrip forward，训练/encoder/DDPM调用均0。复用已有DDPM32行没有新增请求或模型调用。旧A/B64候选、source_v1/R-S1阻断、断连期间的元数据写入及v2静态修订都不是实际模型尝试。

累计实验任务wall下界：{prior_model}+{new_model}={combined_model}秒，约{float(combined_model)/60:.6f}分钟。进程记录{status['process_seconds_before_final_io']}秒、manifest之后closing之前{closing['process_seconds_after_manifest_before_closing_ledger']}秒及各phase均嵌套于外层，不能再次相加。原CE完整耗时仍未知，累计不能称精确总成本。

跨scope仅为执行记账的合计：296个分配slot，启动至少248，完成226，失败6，原CE部分执行16、CEv1未执行32、原AMD无开始证据16。原latent-only4另列。原四臂评价仍为128个请求；这96个诊断请求不并入正式质量分母，复用32行也不重复收费。训练更新仍AR1100/VAE3060/DDPM4060。

非模型分析与审阅独立计时，不加进上述模型任务wall：

| 新补列分析/审阅 | 秒 |
|---|---:|
'''
for a in new_analyses:
    text+=f"| {a['analysis_id']} | {a['wall_seconds'] if a['wall_seconds'] is not None else '未记录'} |\n"
text+=f'''
本次已知分析追加{analysis_increment}秒；加v4已计{prior_analysis}秒后，已知计时分析合计{combined_analysis}秒。v4的AR readout与same-n作者分析没有重复收费；其余未记录的审计/写作/基础设施时间保持null，不能伪填0。

source_v2首次审计因审计脚本自身误设Free/Amide标签而失败，修正为冻结合同的FREE/AMD后完成63项检查。两段计时分别计入分析，汇总字段不再重复相加；这次审计修正没有新增自然模型请求，也不计入模型失败数。完成请求与合法序列分母有别，本账本只记执行状态，合法性由已绑定评价另报。

该追加是一次posthoc来源替换诊断，不是新的正式方法臂或独立训练种子。参考训练后验、打乱已有DDPM潜变量坐标、借参考g的flow prior各有条件边界，不能作为新独立生物样本或升级公平优势证据。AR充分性未证、实际计算不等、dev41复用、原CE中断与额外recovery机会全部保留。原E/L/G仍NOT_EVALUABLE_INTERRUPTION，PROMOTE=HOLD。

来源及逐文件哈希见SOURCE_BINDINGS.json；批次分母见NEW_CONTROL_DENOMINATORS.csv；分析成本见NEW_ANALYSIS_COSTS.csv；本账本未加载权重、运行模型、读取test/cohort或重算质量。
'''
(OUT/'READOUT.zh.md').write_text(text)
files=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='MANIFEST.json':
        b=p.read_bytes();files.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
dump('MANIFEST.json',{'scope':'v4-preserving terminal source-v2 cost addendum; analysis separate','files':files})
print(json.dumps({'model_wall_lower_bound_decimal':str(combined_model),
 'new_analysis_timed_decimal':str(analysis_increment),'all_known_analysis_timed_decimal':str(combined_analysis),
 'new_requests':96,'cumulative_bookkeeping':counts,'checks':len(checks),
 'addendum_sha256':hashlib.sha256((OUT/'COST_ADDENDUM.json').read_bytes()).hexdigest(),
 'manifest_sha256':hashlib.sha256((OUT/'MANIFEST.json').read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
