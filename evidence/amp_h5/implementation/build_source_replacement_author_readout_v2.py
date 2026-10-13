#!/usr/bin/env python3
"""Read-only saved source-diagnostic report; no models, latent arrays, split reads or metric recomputation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/ce_source_replacement_v2'
EVAL=ROOT/'feasibility/ce_source_replacement_evaluation_v2'
OUT=ROOT/'feasibility/ce_source_replacement_author_readout_v2'
METHODS=['CE_DDPM_REUSED','EMPIRICAL_POSTERIOR_B','COORDINATE_PERMUTATION_C','REFERENCE_G_PRIOR_D']
LABELS={'CE_DDPM_REUSED':'既有CE DDPM32复用','EMPIRICAL_POSTERIOR_B':'B训练经验后验回放','COORDINATE_PERMUTATION_C':'C逐坐标置换B的采样z','REFERENCE_G_PRIOR_D':'D reference-g条件native prior'}

def record(p):
 p=Path(p).resolve();assert ROOT in p.parents and p.suffix not in ('.pt','.npz') and 'test37' not in str(p)
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def dump(p,d):
 with p.open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def fmt(value,digits=6):
 return "NA" if value is None else format(value,f".{digits}f")

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--postflight',required=True);ap.add_argument('--postflight-sha256',required=True);a=ap.parse_args();assert not OUT.exists()
 sources={}
 def read(p,binding=None,kind='json'):
  p=Path(p);r=record(p)
  if binding:
   assert r['sha256']==binding['sha256'],str(p)
   if 'bytes' in binding:assert r['bytes']==binding['bytes'],str(p)
  sources[str(p)]=r
  if kind=='binding':return None
  text=p.read_text()
  return list(csv.DictReader(io.StringIO(text))) if kind=='csv' else json.loads(text)
 post=read(Path(a.postflight),{'sha256':a.postflight_sha256});assert post['status']=='PASS' and post['scope_id']=='AMP_H5_CE_SOURCE_REPLACEMENT_V2' and post['generation_attempt_id']=='ce_source_replacement_v2' and post['read_only_frozen_evaluation_eligible'] is True
 prep=read(ROOT/'protocol/ce_source_replacement_evaluation_v2/EVALUATION_PREPARATION.json',{'sha256':'ced34988fc92145f1c6dd71dd2999b617527108864302644ed9ee0f47e1560c4'})
 for r in prep['frozen_sources']:read(Path(r['path']),r,'binding')
 read(Path(prep['sampling_seed_metadata']['binding']['path']),prep['sampling_seed_metadata']['binding'])
 read(Path(prep['prelocked_new_request_slots']['binding']['path']),prep['prelocked_new_request_slots']['binding'])
 runmanifest=read(RUN/'manifest.json');closing=read(RUN/'CLOSING_LEDGER.json');assert closing['state']=='completed' and closing['manifest_sha256']==sources[str(RUN/'manifest.json')]['sha256']
 def member(root,items,name,kind='json'):
  p=root/name;entries=[r for r in items if r['path']==str(p)];assert len(entries)==1;return read(p,entries[0],kind)
 status=member(RUN,runmanifest['files'],'status.json');summary=member(RUN,runmanifest['files'],'generation_summary.json')
 assert status['state']==summary['state']=='completed' and status['completed_inputs']==96 and status['failed_inputs']==status['not_executed_inputs']==0
 assert status['new_training_updates']==status['DDPM_forward_attempts']==status['encoder_forward_attempts']==0
 assert (status['decoder_forward_attempts'],status['prior_inverse_attempts'],status['prior_roundtrip_forward_attempts'])==(6,2,2)
 assert status['loaded_model_unchanged'] is True and status['inputs_unchanged'] is True and not status['finalization_errors'] and status['test_read'] is False
 supervisor=read(ROOT/'runs/ce_source_replacement_execution_v2/SUPERVISOR_STATUS.json');assert supervisor['state']=='completed' and len(supervisor['jobs'])==1 and supervisor['jobs'][0]['exit_code']==0
 em=read(EVAL/'MANIFEST.json');files=['generation_metrics.json','generation_sample_audit.json','reconstruction_metrics.json','reconstruction_row_audit.json','ar_teacher_forced_nll.json','exploratory_gates.json','run.json'];assert len(em['files'])==len(files)
 ev={n:member(EVAL,em['files'],n) for n in files};meta=ev['run.json'];gen=ev['generation_metrics.json'];audit=ev['generation_sample_audit.json']
 assert meta['rows_retained']==128 and meta['methods']==METHODS and len(gen['groups'])==8 and len(audit)==128 and meta['model_calls']==0 and meta['test_accessed'] is False
 assert ev['reconstruction_metrics.json']['groups']==[] and ev['reconstruction_row_audit.json']==[] and ev['ar_teacher_forced_nll.json']==[]
 allrows=[]
 for name,n in [('generated.csv',96),('reused_ddpm.csv',32)]:
  rows=member(RUN,runmanifest['files'],name,'csv');assert len(rows)==n
  binding=next(r for r in meta['sample_inputs'] if r['path']==str(RUN/name));assert binding==sources[str(RUN/name)]
  allrows+=rows
 # Do not follow any checkpoint/array path in engineering evidence.
 post_bindings=post.get('source_bindings',[])
 post_binding_checks=[]
 for n in ['generated.csv','reused_ddpm.csv','status.json','manifest.json','CLOSING_LEDGER.json']:
  matches=[r for r in post_bindings if r.get('path')==str(RUN/n)]
  if matches:
   assert len(matches)==1 and matches[0]['sha256']==sources[str(RUN/n)]['sha256'];post_binding_checks.append({'path':str(RUN/n),'sha256_match':True})
 oldindex=read(ROOT/'feasibility/CE_RECOVERY_FOUR_ARM_DELIVERY_v1.json',{'sha256':'72dc82556c3b0bcfe8f959724002bba614ff53871dd48c1f731b00009b84ad38'})
 b=next(r for r in oldindex['packages']['evaluation'] if Path(r['path']).name=='generation_metrics.json');oldgen=read(Path(b['path']),b)
 oldce={r['terminal']:r for r in oldgen['groups'] if r['method']=='CE21_DDPM'}
 equality={t:next({**g,'method':'CE21_DDPM'} for g in gen['groups'] if g['method']=='CE_DDPM_REUSED' and g['terminal']==t)==oldce[t] for t in ['FREE','AMD']};assert all(equality.values())
 rawgate=ev['exploratory_gates.json'];assert all('missing' in c for c in rawgate['generation_ce_vs_terminal_ar']['checks'].values())
 interpretation={'legacy_gate_artifact_interpretation':'NOT_APPLICABLE_SOURCE_DIAGNOSTIC','reason':'Frozen CLI default gate operands are original CE/MSE/AR method IDs, absent here. Raw false values are not a diagnostic failure or win/loss gate.','raw_legacy_gate_retained':True,'raw_legacy_gate_source':sources[str(EVAL/'exploratory_gates.json')],'original_CE_E_L_G':'NOT_EVALUABLE_PERMANENTLY','original_four_arm_recovery_screens':'UNCHANGED; not rerun or reinterpreted by this source diagnostic','new_winner_gate':None,'ranking':'B/C/D diagnostic references; D uses additional reference-g information and is excluded from c-only ranking','reconstruction_AR_NLL':'NOT_APPLICABLE_NO_NEW_ENCODER_OR_NLL','PROMOTE':'HOLD'}
 source_details=[]
 for method in METHODS:
  for terminal in ['FREE','AMD']:
   rs=[r for r in allrows if r['method']==method and r['terminal']==terminal];assert len(rs)==16
   refs=[str(r.get('reference_train_row_id','')) for r in rs if str(r.get('reference_train_row_id',''))!='']
   source_details.append({'method':method,'label':LABELS[method],'terminal':terminal,'requests':16,'new_decode_observations':0 if method=='CE_DDPM_REUSED' else 16,'reused_observations':16 if method=='CE_DDPM_REUSED' else 0,'condition_scopes':sorted({r['condition_scope'] for r in rs}),'ranking_eligible_values':sorted({r['ranking_eligible'] for r in rs}),'reference_train_row_ids':refs,'reference_row_unique_count':len(set(refs)),'reference_row_repeated_occurrences':len(refs)-len(set(refs)),'status_counts':dict(Counter(r['status'] for r in rs)),'empty_count':sum(not r['sequence'] for r in rs),'length_cap_hit_count':sum(len(r['sequence'])==48 for r in rs),'cap_hit_sample_ids':[r['sample_id'] for r in rs if len(r['sequence'])==48]})
 assert all(r['ranking_eligible_values']==['false'] for r in source_details)
 unchanged=[{**r,'unchanged':record(r['path'])==r} for r in sources.values()];assert all(r['unchanged'] for r in unchanged)
 costs={'new_diagnostic_status_phase_seconds':status['phase_seconds'],'new_diagnostic_process_before_final_io':status['process_seconds_before_final_io'],'new_diagnostic_closing_after_manifest_before_ledger':closing['process_seconds_after_manifest_before_closing_ledger'],'new_diagnostic_supervisor_wall_seconds':supervisor['jobs'][0]['seconds'],'new_diagnostic_supervisor':supervisor,'historical_CE_training_and_DDPM':'Already incurred; reused32 are not free independent new samples. Original interrupted CE generation lower bound >=451.935743s and recovery1115.496170s remain in closed four-arm cost record.','do_not_sum_nested_durations':True}
 limits=['Posthoc diagnostic designed after known generation/scale results; fixed one checkpoint, one paired epsilon bank and one selected training-row bank.','B is empirical training posterior replay containing seen sequence/conditioning information; copying or distribution proximity does not establish innovative or independent novel generation.','C permutes actual sampled B decoder-z coordinates within terminal, preserving finite coordinate marginals; it has no single donor sequence/g target and is not a new candidate model.','D uses the same B-selected training reference-g including residue/length-template information; it is not terminal-only, not a fair c-only competitor, and not independent new-peptide evidence.','B versus D also differs in posterior x/h information; differences cannot identify flow alone or a unique DDPM causal bottleneck.','Shared epsilon/row indices are paired random inputs, not independent training or biological replicates; no significance tests or new winner thresholds.','Full dev41 sequence metrics do not create missing dev posterior latent evidence.','Original four-arm results and CE interruption stay unchanged; no activity, safety, wetlab-validation, novelty or submission claim.']
 facts={'created_utc':datetime.now(timezone.utc).isoformat(),'scope_id':'AMP_H5_CE_SOURCE_REPLACEMENT_V2_READONLY_DESCRIPTION','posthoc_exploratory':True,'engineering_postflight':post,'source_details':source_details,'method_interpretations':prep['method_interpretations'],'counts':{'all_observations':128,'reused_DDPM':32,'new_B':32,'new_C':32,'new_D':32,'new_total':96,'new_encoder_calls':0,'new_DDPM_calls':0,'new_training_updates':0,'new_decoder_batches':6,'new_prior_inverse_batches':2,'new_prior_forward_roundtrip_batches':2},'full_generation_metrics':gen,'raw_sample_provenance':allrows,'all_frozen_sample_audit':audit,'gate_interpretation':interpretation,'raw_legacy_gate_record':rawgate,'reused_CE_full_groups_identical_to_closed_four_arm':equality,'source_completion':{'status':status,'closing_ledger':closing,'supervisor':supervisor},'costs':costs,'postflight_binding_checks':post_binding_checks,'source_bindings_and_preservation':unchanged,'source':record(__file__),'operations':{'model_calls':0,'metric_recomputations':0,'split_reads':0,'latent_array_reads':0,'checkpoint_reads':0,'test_reads':0},'limits':limits}
 OUT.mkdir();dump(OUT/'AUTHOR_FACTS.json',facts);dump(OUT/'GATE_AND_SCOPE_INTERPRETATION.json',interpretation)
 lines=['# AMP H5：B/C/D 来源替换诊断作者读出','','**128条来源观察=32条既有CE复用+96条新请求（B/C/D各32），不是128条新生成或独立重复。B/C/D不作模型排名；旧门不适用，原CE E/L/G永久NE，PROMOTE=HOLD。**','','本次只读评价使用固定train279/dev41和冻结421维MMD及全部原指标。新scope工程通过仅证明保存数据、算术路径、调用和记录完整，不是独立模型重放；roundtrip通过不能解释为prior质量优势或创新。生成与评价均无新训练、encoder或DDPM调用；新生成实际6次decoder batch、2次prior inverse及2次roundtrip forward。','','| 来源 | 信息与解释边界 |','| --- | --- |']
 for method in METHODS:lines.append(f'| {method} | {prep["method_interpretations"][method]} |')
 lines += ['', '| 来源 | 端基 | 请求 | 完成 | 空串 | 合法1–48 | 合法5–48 | 合法unique | copy同端/任意train | train MMD² | dev MMD² |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
 for r in gen['groups']:lines.append(f"| {r['method']} | {r['terminal']} | {r['requests']} | {r['execution_completed_count']} | {r['empty_count']} | {r['valid_1_48_count']} | {r['valid_5_48_count']} | {r['valid_1_48']['unique_count']} | {r['exact_copy_train_count_all_requests']}/{r['exact_sequence_copy_any_train_terminal_count']} | {fmt(r['valid_1_48_vs_train']['mmd2_linear_biased_fixed_features'],6)} | {fmt(r['valid_1_48_vs_development']['mmd2_linear_biased_fixed_features'],6)} |")
 lines += ['', '全部固定槽位保留；完成数与合法数不同，空串也是完成但非法1–48。分布分母只取冻结合法1–48集合，unique绝对数量不替代16请求分母。固定421特征MMD²为biased linear估计，JS为bit，KS无p值；未按结果择指标。','','| 来源 | 端基 | 参照 | AA JS | AA L1 | 二肽JS | 二肽L1 | 长度KS |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
 for r in gen['groups']:
  for label,key in [('train','valid_1_48_vs_train'),('dev','valid_1_48_vs_development')]:lines.append('| '+' | '.join([r['method'],r['terminal'],label]+[f"{fmt(r[key][k],6)}" for k in ['aa_jensen_shannon_bits','aa_l1_distance','dipeptide_jensen_shannon_bits','dipeptide_l1_distance','length_ks_statistic_no_pvalue']])+' |')
 lines += ['', '| 来源 | 端基 | AA熵 | 二肽熵 | AA熵距train | 二肽熵距train | 长度均值[min,max] |','| --- | --- | ---: | ---: | ---: | ---: | --- |']
 for r in gen['groups']:
  s=r['valid_1_48'];l=s['length'];lines.append('| '+' | '.join([r['method'],r['terminal']]+[f'{fmt(v,6)}' for v in [s['pooled_aa_entropy_bits'],s['pooled_dipeptide_entropy_bits'],r['pooled_aa_entropy_bits_absolute_gap_to_train'],r['pooled_dipeptide_entropy_bits_absolute_gap_to_train']]]+[f"{fmt(l['mean'],4)} [{l['min']},{l['max']}]"])+' |')
 lines += ['', '| 来源 | 端基 | 最近邻范围 | identity mean[min,max] | median | ≥70%数量/合法n |','| --- | --- | --- | --- | ---: | --- |']
 for r in gen['groups']:
  for label,key,count in [('同端train','nearest_train_identity_valid_1_48','nearest_train_at_least_70_percent_count_valid_1_48'),('任意train','nearest_any_train_terminal_identity_valid_1_48','nearest_any_train_terminal_at_least_70_percent_count_valid_1_48')]:
   n=r[key];lines.append(f"| {r['method']} | {r['terminal']} | {label} | {fmt(n['mean'],6)} [{fmt(n['min'],6)},{fmt(n['max'],6)}] | {fmt(n['median'],6)} | {r[count]}/{n['n']} |")
 lines += ['', '| 来源 | 端基 | 新/复用 | donor行数/unique | 48触顶 | status计数 |','| --- | --- | --- | --- | ---: | --- |']
 for r in source_details:lines.append(f"| {r['method']} | {r['terminal']} | {r['new_decode_observations']}/{r['reused_observations']} | {len(r['reference_train_row_ids'])}/{r['reference_row_unique_count']} | {r['length_cap_hit_count']} | {json.dumps(r['status_counts'],ensure_ascii=False)} |")
 lines += ['', 'B/D donor以reference_train_row_id记录并保留重复；C跨坐标来自多个B槽位，无单一对应训练序列或g。donor数量不是独立信息量。所有来源均用同CE J/PAD读出；48触顶不是AR式自然EOS事件。既有CE两组完整指标与原四臂包逐值相同。','','旧CLI附带的exploratory_gates.json缺原CE/MSE/AR操作数，其false仅为缺操作数产物；[门与scope语义](GATE_AND_SCOPE_INTERPRETATION.json)明确NOT_APPLICABLE_SOURCE_DIAGNOSTIC，不能写为本诊断失败或新胜负门。无新重建/AR NLL任务，空数组表示不适用。','',f"新诊断进程final I/O前 {fmt(status['process_seconds_before_final_io'],9)}秒，manifest后closing ledger前 {fmt(closing['process_seconds_after_manifest_before_closing_ledger'],9)}秒；外层supervisor {fmt(supervisor['jobs'][0]['seconds'],9)}秒。范围嵌套，不相加；已有DDPM32历史成本及原CE中断下界不会被重置为0。",'', '[完整指标、逐条来源、工程回执和成本](AUTHOR_FACTS.json) · [冻结评价包](../ce_source_replacement_evaluation_v2/MANIFEST.json)','','- '+'\n- '.join(limits)]
 with (OUT/'AUTHOR_READOUT.zh.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 dump(OUT/'MANIFEST.json',{'source':record(__file__),'files':[record(p) for p in sorted(OUT.iterdir()) if p.is_file()],'inputs':list(sources.values()),'original_CE_E_L_G':'NOT_EVALUABLE_PERMANENTLY','legacy_gates':'NOT_APPLICABLE_SOURCE_DIAGNOSTIC','PROMOTE':'HOLD'})
 print(json.dumps({'output':str(OUT),'observations':128,'reused':32,'new':96,'gate_interpretation':'NOT_APPLICABLE_SOURCE_DIAGNOSTIC','model_calls':0}))

if __name__=='__main__':main()
