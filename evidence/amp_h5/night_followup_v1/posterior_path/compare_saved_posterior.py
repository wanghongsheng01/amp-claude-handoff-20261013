#!/usr/bin/env python3
"""Saved CSVs only, frozen reconstruction scorer, no tensor/model imports."""
import sys
sys.dont_write_bytecode=True
import json,csv,hashlib,importlib.util,time,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
P=HERE/'INPUTS_AND_FORMULAS_PRELOCK.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def compact(stats,mode):
 keys=('rows','target_residues','rendered_aligned_correct','rendered_nonpad_micro','rendered_nonpad_macro','per_sequence_mean_nonpad_accuracy','length_absolute_error_sum','length_mae','length_exact_count','whole_rendered_exact_count','whole_rendered_exact_fraction','empty_count','status_counts','supported_aa_classes')
 out={k:stats[k] for k in keys};out['prediction_mode']=mode;out['frozen_scorer_hardcoded_mode']=stats['mode']
 out['pooled_nonpad_error_rate']=1-stats['rendered_nonpad_micro'];out['per_sequence_mean_nonpad_error_rate']=1-stats['per_sequence_mean_nonpad_accuracy']
 return out
start=time.perf_counter();protocol=read(P)
assert not (HERE/'RESULTS.json').exists()
for b in protocol['bindings']:assert sha(b['path'])==b['sha256'],b['path']
sp=ROOT/'implementation/evaluate_h5.py';spec=importlib.util.spec_from_file_location('frozen_posterior_scorer',sp);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
traintruth=e.dataset(protocol['train_cohort']['path'],'train',protocol['train_cohort']['sha256'],279)
devtruth=e.dataset(protocol['dev_cohort']['path'],'development',protocol['dev_cohort']['sha256'],41)
trainmeans=rows(protocol['train_mean_csv']);devmeans=rows(protocol['dev_mean_csv']);trainbyid={int(x['row_id']):x for x in trainmeans}
assert len(trainmeans)==len(trainbyid)==279 and set(trainbyid)==set(range(279))
fulltrain,_=e.reconstruction_metrics(trainmeans,traintruth);fulldev,_=e.reconstruction_metrics(devmeans,devtruth)
oldrec=read(protocol['existing_reconstruction_metrics'])['groups'];oldce=next(x for x in oldrec if x['method']=='CE21_DDPM' and x['split']=='development')
assert fulldev['rendered_aligned_correct']==oldce['rendered_aligned_correct']==497 and fulldev['target_residues']==oldce['target_residues']==701
assert fulldev['rendered_nonpad_micro']==oldce['rendered_nonpad_micro']
base=read(protocol['B_input_seal']);brows=[r for r in rows(protocol['B_generated_csv']) if r['method']=='EMPIRICAL_POSTERIOR_B']
assert len(brows)==len({r['sample_id'] for r in brows})==32
bbykey={(r['terminal'],int(r['terminal_sample_id'])):r for r in brows}
comparison=[];audit=[]
for terminal in ('FREE','AMD'):
 truth=[];ms=[];ss=[];donors=base['terminals'][terminal]['selected_row_ids']
 assert len(donors)==16
 for i,donor in enumerate(donors):
  b=bbykey[(terminal,i)];mean=trainbyid[donor];tr=traintruth[donor]
  assert b['sample_id']==f'EMPIRICAL_POSTERIOR_B_{terminal}_{i:03d}' and int(b['reference_train_row_id'])==donor
  assert mean['truth_sequence']==tr['sequence'] and mean['terminal']==tr['terminal_state']==terminal
  assert int(b['seed'])==422+('FREE','AMD').index(terminal) and int(b['length'])==len(b['sequence'])
  truth.append(tr)
  ms.append({'row_id':i,'terminal':terminal,'truth_sequence':tr['sequence'],'prediction_sequence':mean['prediction_sequence'],'status':mean['status']})
  ss.append({'row_id':i,'terminal':terminal,'truth_sequence':tr['sequence'],'prediction_sequence':b['sequence'],'status':b['status']})
 m,md=e.reconstruction_metrics(ms,truth);s,sd=e.reconstruction_metrics(ss,truth)
 summary={'terminal':terminal,'slot_count':16,'unique_donors':len(set(donors)),'repeated_draws_retained':16-len(set(donors)),'donor_rows_in_order':donors,'mean':compact(m,'saved_posterior_mu_decode'),'stochastic_B':compact(s,'saved_single_posterior_draw_B_decode')}
 summary['mean_minus_stochastic']={k:m[k]-s[k] for k in ('rendered_nonpad_micro','rendered_nonpad_macro','per_sequence_mean_nonpad_accuracy','length_mae','whole_rendered_exact_fraction')}
 comparison.append(summary)
 for i,(a,b) in enumerate(zip(md,sd)):
  donor=donors[i]
  audit.append({'sample_id':f'EMPIRICAL_POSTERIOR_B_{terminal}_{i:03d}','terminal':terminal,'slot_index':i,'donor_train_row_id':donor,
   'truth_sequence':truth[i]['sequence'],'mean_prediction_sequence':a['prediction_sequence'],'B_prediction_sequence':b['prediction_sequence'],
   'target_residues':a['target_residues'],'mean_correct_residues':a['correct_target_positions'],'B_correct_residues':b['correct_target_positions'],
   'mean_accuracy':a['rendered_nonpad_accuracy'],'B_accuracy':b['rendered_nonpad_accuracy'],'mean_minus_B_accuracy':a['rendered_nonpad_accuracy']-b['rendered_nonpad_accuracy'],
   'mean_length':len(a['prediction_sequence']),'B_length':len(b['prediction_sequence']),'mean_length_abs_error':a['length_absolute_error'],'B_length_abs_error':b['length_absolute_error'],
   'mean_exact':a['whole_rendered_exact'],'B_exact':b['whole_rendered_exact'],'mean_status':a['status'],'B_status':b['status']})
# The pooled32 comparator keeps the duplicate donor as another fixed draw slot.
pooledtruth=[{'sequence':a['truth_sequence'],'terminal_state':a['terminal']} for a in audit]
pooled={}
for mode,pkey,skey in [('mean','mean_prediction_sequence','mean_status'),('stochastic_B','B_prediction_sequence','B_status')]:
 rr=[{'row_id':i,'terminal':a['terminal'],'truth_sequence':a['truth_sequence'],'prediction_sequence':a[pkey],'status':a[skey]} for i,a in enumerate(audit)]
 value,_=e.reconstruction_metrics(rr,pooledtruth);pooled[mode]=compact(value,'saved_mu_decode' if mode=='mean' else 'saved_single_posterior_draw_B_decode')
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'SAVED_CACHE_COMPARISON_COMPLETE','protocol_sha256':sha(P),'analysis_script_sha256':sha(__file__),
 'prediction_route':'q(z|R,g,h)=Normal(mu,exp(logsigma)); no posterior flow. M=decoder(mu), S=decoder(mu+sigma*epsilon), same frozen J. g/h enter encoder; only g also enters prior_flow for training prior; decoder receives z and no teacher-forced R.',
 'dev70_90_identity':compact(fulldev,'dev_posterior_mean_mu_decode'),
 'fulltrain_mean_context':compact(fulltrain,'train_posterior_mean_mu_decode'),
 'cache_availability':protocol['cache_availability'],'paired_training_draw_comparison':comparison,'pooled32_context_only':pooled,
 'paired_raw_rows':audit,'slot_count':32,'unique_donor_count':len({a['donor_train_row_id'] for a in audit}),
 'checks':{'same_checkpoint_mean_and_B':protocol['checkpoint_tensor_digest']=='f57f91b37cac31d7640e089c2e941743f22ec2a9474d2beed7e8e6b1a444c789','B_all32_paired_to_true_donor':True,'AMD_duplicate117_retained':base['terminals']['AMD']['selected_row_ids'].count(117)==2,'source_inputs_unchanged':all(sha(b['path'])==b['sha256'] for b in protocol['bindings']),'old_dev_metric_exactly_reproduced':True},
 'inference_limits':['All32 are training reference slots, not dev, not independent training/biological replicates. AMD repeated donor117 retained.','This compares one existing stochastic posterior draw per slot with its donor mean. It does not estimate variance across posterior seeds.','No dev stochastic posterior reconstruction is cached; dev mean70.90% is not a stochastic result.','Supported-AA macro recall differs from per-sequence mean accuracy; both are separately labeled.','Frozen scorer hardcodes mode=posterior_mean; wrapper explicitly records that string separately and labels actual stochastic provenance without changing formulas.','No causal, generation-quality, native-prior-superiority or new-peptide claim follows from this cache comparison.','No flow/encoder/decoder/DDPM call, test access or historical checkpoint substitution occurred.','No5-point dev investment gate is computed: dev stochastic cache is MISSING. Original CE E/L/G remain NE; PROMOTE HOLD.'],
 'analysis_cost_seconds_before_export':time.perf_counter()-start,'model_calls':0,'new_random_draws':0,'test_read':False}
dump(HERE/'RESULTS.json',report)
lines=['# 已存后验随机重建与均值重建：训练 donor 配对','',f"70.90% 是当前冻结CE checkpoint 的 **dev posterior mean（μ直接解码）**：{fulldev['rendered_aligned_correct']}/{fulldev['target_residues']}={100*fulldev['rendered_nonpad_micro']:.4f}%，41条。它不是随机后验重建率。",'',
'随机缓存只有source-v2 B的32个训练抽样槽（FREE16、AMD16），覆盖31个唯一donor。AMD row117抽中两次，两个epsilon槽和同一donor的mean预测均保留；没有去重改分母。全train279随机重建和dev41随机重建均未缓存。','',
'| 端基 | 模式 | 正确/真实残基 | micro | supported-AA macro | 逐序列平均准确率 | 长度MAE | 全序列exact |','|---|---|---:|---:|---:|---:|---:|---:|']
for d in comparison:
 for key,label in [('mean','M：已有μ解码'),('stochastic_B','S：已有单次后验draw')]:
  x=d[key];lines.append(f"| {d['terminal']} | {label} | {x['rendered_aligned_correct']}/{x['target_residues']} | {x['rendered_nonpad_micro']:.4%} | {x['rendered_nonpad_macro']:.4%} | {x['per_sequence_mean_nonpad_accuracy']:.4%} | {x['length_mae']:.4f} | {x['whole_rendered_exact_count']}/16 |")
lines+=['',
'这里q是encoder输出的对角Gaussian，没有posterior flow；M=μ，S=μ+σepsilon，都直接经同一decoder/J。g经g_pre_net进入encoder，h也进入encoder；训练中prior_flow(z,g)只定义条件prior的坐标变换。decoder内部g=None，无teacher forcing。此比较是训练重建对后验抽样的敏感性，不能当dev泛化、独立重复、DDPM唯一瓶颈或新肽质量证据。','',
'计分沿冻结reconstruction scorer：micro以真实非PAD残基为分母，缺失预测位置算错；macro按真实支持的AA类别等权recall；额外预测长度由长度误差与wholeexact反映。结果JSON保留全部32个配对原始差异。旧scorer硬编码mode标签仅另列为函数元数据，随机分支实际模式已明确标注。','',
'本分析没有模型forward、随机采样或test读取，也不引入H0/H4旧checkpoint。dev随机缓存缺失，所以不计算dev的5百分点投入门。HOLD保持；后续补全R/M/S须另行锁定和独审。','']
(HERE/'READOUT.zh.md').write_text('\n'.join(lines))
manifest={'created_utc':report['created_utc'],'files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in [P,Path(__file__),HERE/'RESULTS.json',HERE/'READOUT.zh.md']], 'analysis_seconds_after_result_exports_before_manifest':time.perf_counter()-start,'actual_scoring_only_python_run':True,'agent_design_time_not_included':True,'model_calls':0,'new_random_draws':0,'source_inputs_unchanged':all(sha(b['path'])==b['sha256'] for b in protocol['bindings'])}
dump(HERE/'MANIFEST.json',manifest)
print(json.dumps({'status':report['status'],'protocol_sha256':sha(P),'results_sha256':sha(HERE/'RESULTS.json'),'manifest_sha256':sha(HERE/'MANIFEST.json'),'cost_seconds':manifest['analysis_seconds_after_result_exports_before_manifest'],'paired_training_summary':comparison},ensure_ascii=False))
