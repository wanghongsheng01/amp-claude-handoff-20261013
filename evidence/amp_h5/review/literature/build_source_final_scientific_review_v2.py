"""Read-only scientific audit of saved AMP H5 source-v2 text records.

Does not import torch/numpy, read cohorts/checkpoints/latent arrays, or run models.
Writes only this review's new JSON/Markdown files.
"""
import collections
import datetime
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).parent
BASE = 'SOURCE_REPLACEMENT_FINAL_SCIENTIFIC_REVIEW_v2'
checks = []
bindings = []

def read(rel):
    return json.loads((ROOT / rel).read_text())

def bind(rel, expected=None):
    p = pathlib.Path(rel)
    if not p.is_absolute():
        p = ROOT / p
    b = p.read_bytes()
    d = {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
    if expected is not None:
        assert d['sha256'] == expected, p
    bindings.append(d)
    return d

def check(name, value, detail=None):
    checks.append({'check': name, 'pass': bool(value), 'detail': detail})
    assert value, (name, detail)

index_rel = 'feasibility/CE_SOURCE_REPLACEMENT_EVALUATION_DELIVERY_v2.json'
index_binding = bind(index_rel, '4370fdb76d765cd190cac3d7b33013b87345a45dfd0c0ed51ffe2a63b86d7dab')
idx = read(index_rel)
refs = [d for pack in idx['packages'].values() for d in pack]
for key in ['preparation', 'contract', 'plan', 'engineering_postflight', 'source_run_manifest',
            'source_closing_ledger', 'source_supervisor_terminal', 'historical_four_arm_delivery']:
    refs.append(idx[key])
for d in refs:
    actual = bind(d['path'], d['sha256'])
    if 'bytes' in d:
        assert actual['bytes'] == d['bytes']
check('33 direct delivery-index text member hashes and available byte counts', len(refs) == 33)

ev = 'feasibility/ce_source_replacement_evaluation_v2/'
author = 'feasibility/ce_source_replacement_author_readout_v2/'
g = read(ev + 'generation_metrics.json')
samples = read(ev + 'generation_sample_audit.json')
f = read(author + 'AUTHOR_FACTS.json')
side = read(author + 'GATE_AND_SCOPE_INTERPRETATION.json')
post = read('review/independent/CE_SOURCE_REPLACEMENT_POSTFLIGHT_v2.json')
bind('review/independent/CE_SOURCE_REPLACEMENT_POSTFLIGHT_v2.json',
     '01fe09cbce4c4d4d8cf6c04d617408de40162b02f4583cb2a0c095418535cac3')
prior_review = read('review/literature/FOUR_ARM_FINAL_SCIENTIFIC_REVIEW_v1.json')
bind('review/literature/FOUR_ARM_FINAL_SCIENTIFIC_REVIEW_v1.json',
     '4fa10c677f8165152698243a99371e7ca22e4173529f9de64915bcb61b84e42a')
bind('review/literature/SOURCE_BCD_EXACT_CONTRACT_SCIENTIFIC_REVIEW_v2.json',
     '36489ec862ca52048a796525a0d3a7228f26b2db49f8938fe3dee8da62693bfb')
old_path = 'feasibility/ce_recovery_four_arm_evaluation_v1/generation_metrics.json'
bind(old_path, '7cd655ed623cd2d4d0751e1f06ab2e534361704e4c3fc755c7586ce9ee9eeac1')
old = read(old_path)
cost_path = 'feasibility/cost_addendum_source_replacement_v2_v1/COST_ADDENDUM.json'
cost = read(cost_path)
bind(cost_path, '8132da75896f07623f5389a56274bee82db4611d664288a2df1d53f3986a4a52')
err_path = 'feasibility/cost_addendum_source_c_erratum_v1/ERRATUM_SOURCE_C_v1.json'
err = read(err_path)
bind(err_path, 'a4c2c7f33db7a0f1d87c71c3990bae83e97070f22a4f39afd71947cbbd8fdda4')

check('author full metrics exactly equal frozen evaluator JSON', f['full_generation_metrics'] == g)
check('author all 128 sample audits exactly equal frozen evaluator JSON', f['all_frozen_sample_audit'] == samples)
check('completed engineering postflight with 63 checks', post['status'] == 'PASS' and post['independent_check_count'] == 63)
check('128 observations and eight groups, each with 16 requests',
      len(samples) == 128 and len(g['groups']) == 8 and all(x['requests'] == 16 for x in g['groups']))
check('unique observation IDs', len({s['sample_id'] for s in samples}) == 128)
check('32 reused plus 96 new with fixed model-call accounting',
      f['counts'] == {'all_observations':128,'reused_DDPM':32,'new_B':32,'new_C':32,'new_D':32,
      'new_total':96,'new_encoder_calls':0,'new_DDPM_calls':0,'new_training_updates':0,
      'new_decoder_batches':6,'new_prior_inverse_batches':2,'new_prior_forward_roundtrip_batches':2})
groups = {(x['method'], x['terminal']): x for x in g['groups']}
names = {'A':'CE_DDPM_REUSED','B':'EMPIRICAL_POSTERIOR_B',
         'C':'COORDINATE_PERMUTATION_C','D':'REFERENCE_G_PRIOR_D'}
details = {(x['method'],x['terminal']):x for x in f['source_details']}
raw = {x['sample_id']:x for x in f['raw_sample_provenance']}
check('all raw provenance rows agree with audit IDs/sequence/status/terminal', len(raw) == 128 and all(
    all(raw[s['sample_id']][k] == s[k] for k in ['method','sequence','status','terminal']) for s in samples))

group_checks = []
for key, x in groups.items():
    ss = [s for s in samples if (s['method'],s['terminal']) == key]
    d = details[key]
    ok = (len(ss) == x['requests'] == 16
          and dict(collections.Counter(s['status'] for s in ss)) == x['status_counts']
          and sum(s['empty'] for s in ss) == x['empty_count'] == 0
          and sum(s['valid_1_48'] for s in ss) == x['valid_1_48_count'] == 16
          and sum(s['valid_5_48'] for s in ss) == x['valid_5_48_count']
          and len({s['sequence'] for s in ss}) == x['valid_1_48']['unique_count'] == 16
          and sum(s['exact_sequence_copy_any_train_terminal'] for s in ss) == x['exact_sequence_copy_any_train_terminal_count']
          and sum(s['exact_sequence_plus_requested_terminal_copy'] for s in ss) == x['exact_copy_train_count_all_requests']
          and sum(s['length'] == 48 for s in ss) == d['length_cap_hit_count'])
    for nk, mk, ck in [('nearest_train','nearest_train_identity_valid_1_48','nearest_train_at_least_70_percent_count_valid_1_48'),
                       ('nearest_any_train_terminal','nearest_any_train_terminal_identity_valid_1_48','nearest_any_train_terminal_at_least_70_percent_count_valid_1_48')]:
        values = [s[nk]['identity'] for s in ss]
        ok &= sum(s[nk]['edge_at_70_percent'] for s in ss) == x[ck]
        ok &= abs(statistics.mean(values) - x[mk]['mean']) < 1e-12
        ok &= all(abs(s[nk]['matches']/s[nk]['aligned_columns'] - s[nk]['identity']) < 1e-12 for s in ss)
    ok &= x['train_same_terminal_n'] == (126 if key[1] == 'FREE' else 153)
    ok &= x['development_same_terminal_n'] == (18 if key[1] == 'FREE' else 23)
    group_checks.append({'method':key[0],'terminal':key[1],'pass':bool(ok)})
check('all eight group denominators/copy/nearest/native cap/reference n agree with saved rows', all(x['pass'] for x in group_checks), group_checks)
check('all observations explicitly excluded from diagnostic ranking', all(x['ranking_eligible'] == 'false' for x in raw.values()))
check('all new 96 are complete, legal 5..48 and no cap', all(s['status']=='ok' and s['valid_5_48'] and s['length']<48 for s in samples if s['method'] != names['A']))
check('A full groups are historical exact reuse except method label', all(
    {k:v for k,v in groups[names['A'],t].items() if k!='method'} ==
    {k:v for k,v in next(o for o in old['groups'] if o['method']=='CE21_DDPM' and o['terminal']==t).items() if k!='method'}
    for t in ['FREE','AMD']))
check('B and D reference indices agree including AMD repeated donor', all(
    details[names['B'],t]['reference_train_row_ids'] == details[names['D'],t]['reference_train_row_ids']
    and details[names['B'],t]['reference_row_unique_count'] == (16 if t=='FREE' else 15) for t in ['FREE','AMD']))
check('C has no single donor sequence or g target', all(not details[names['C'],t]['reference_train_row_ids'] for t in ['FREE','AMD']))
copies = [s for s in samples if s['exact_sequence_copy_any_train_terminal']]
check('one exact training copy retained at B AMD slot 002', len(copies)==1 and copies[0]['sample_id']=='EMPIRICAL_POSTERIOR_B_AMD_002')
check('source legacy gates marked not applicable, never new winner', side['legacy_gate_artifact_interpretation']=='NOT_APPLICABLE_SOURCE_DIAGNOSTIC' and side['new_winner_gate'] is None)
check('original E/L/G permanently NE and PROMOTE HOLD', side['original_CE_E_L_G']=='NOT_EVALUABLE_PERMANENTLY' and side['PROMOTE']=='HOLD')
check('no source reconstruction/AR NLL masquerading as observed results', read(ev+'reconstruction_metrics.json')['groups']==[] and read(ev+'reconstruction_row_audit.json')==[] and read(ev+'ar_teacher_forced_nll.json')==[])
check('source wall accounting agrees with completed author status and postflight', f['costs']['new_diagnostic_supervisor_wall_seconds']==2.679257582873106 and f['source_completion']['status']['state']=='completed')
check('cost C-origin error is explicitly corrected by immutable erratum', err['correct_source_C']['existing_DDPM_z_is_permuted'] is False and '/new_model_diagnostic/method_boundary' in err['structured_fields_with_error'] and err['numeric_or_execution_fields_changed'] is False)

# Exact saved scalar comparisons; these do not recompute features, distances or alignments.
comparisons = []
for left,right in [('B','A'),('C','B'),('D','A'),('D','B')]:
    for t in ['FREE','AMD']:
        for ref in ['train','development']:
            a=groups[names[left],t]['valid_1_48_vs_'+ref]
            b=groups[names[right],t]['valid_1_48_vs_'+ref]
            for metric,v in a.items():
                comparisons.append({'left':left,'right':right,'terminal':t,'reference':ref,
                    'metric':metric,'left_value':v,'right_value':b[metric],
                    'left_direction':'lower' if v<b[metric] else 'higher' if v>b[metric] else 'equal'})
check('B dev MMD exceeds A in both terminals', all(groups[names['B'],t]['valid_1_48_vs_development']['mmd2_linear_biased_fixed_features'] > groups[names['A'],t]['valid_1_48_vs_development']['mmd2_linear_biased_fixed_features'] for t in ['FREE','AMD']))
check('C versus B keeps mixed AMD dev length KS exception', sum(x['left']=='C' and x['left_direction']=='higher' for x in comparisons)==23 and sum(x['left']=='C' and x['left_direction']=='lower' for x in comparisons)==1)
check('all B/C/D AA and dipeptide entropy gaps to train exceed A', all(groups[names[a],t][k] > groups[names['A'],t][k] for a in ['B','C','D'] for t in ['FREE','AMD'] for k in ['pooled_aa_entropy_bits_absolute_gap_to_train','pooled_dipeptide_entropy_bits_absolute_gap_to_train']))

summaries=[]
for x in g['groups']:
    z={k:v for k,v in x.items() if k not in ['valid_1_48','valid_5_48']}
    for cohort in ['valid_1_48','valid_5_48']:
        z[cohort]={k:v for k,v in x[cohort].items() if k not in ['aa_counts','aa_frequency','dipeptide_counts','dipeptide_frequency','fixed_feature_mean']}
    summaries.append(z)

md = '''# AMP H5 来源替换 v2 独立科学终审

结论：冻结作者包的描述性报告可以封口；没有证据把经验后验或 reference-g 来源替换写成双端生成修复、独立新方法优势，或唯一 DDPM 因果瓶颈。无未解决的科学报告阻断项。成本包的 C 来源措辞曾错写为 DDPM-z，已另附不可变勘误；原包必须与勘误一起引用。

本审只读已保存 JSON/Markdown/日志，不读取 train/dev/test 原始队列、模型权重或 latent 数组，不调用模型，不重跑特征、距离或序列比对。独立核实联合索引的 33 个直接文本绑定、作者全指标与 128 条 audit 精确一致、A 两端除名称外与原四臂完整组逐值一致。工程后审的 63 项 PASS 是保存记录和算术核验，不能当作独立模型重放或质量证明。

128 条来源观察为 A 既有 CE DDPM 32 条复用，加 B/C/D 各 32 条新诊断；每来源每端 16 条，合计 96 次新 decoder 输入。新执行为 6 次 decoder batch、2 次 native prior inverse、2 次 roundtrip；没有新训练、encoder 或 DDPM forward。新 96 条全部完成、合法 5–48、无空串、无 48 触顶。A FREE 保留原 1 条 48 触顶、合法 5–48 为 14/16，AMD 为 15/16；全部 128 条均合法 1–48，每组 unique 为 16/16。触顶是 J/PAD 读出事件，不能混称自回归自然 EOS。

| 来源 | FREE dev MMD² | AMD dev MMD² | FREE/AMD 任意 train 最近邻 ≥70% | FREE/AMD 精确 train 复制 |
| --- | ---: | ---: | --- | --- |
'''
for a in ['A','B','C','D']:
    ff=groups[names[a],'FREE'];aa=groups[names[a],'AMD']
    md += f"| {a} | {ff['valid_1_48_vs_development']['mmd2_linear_biased_fixed_features']:.12f} | {aa['valid_1_48_vs_development']['mmd2_linear_biased_fixed_features']:.12f} | {ff['nearest_any_train_terminal_at_least_70_percent_count_valid_1_48']}/16；{aa['nearest_any_train_terminal_at_least_70_percent_count_valid_1_48']}/16 | {ff['exact_sequence_copy_any_train_terminal_count']}/16；{aa['exact_sequence_copy_any_train_terminal_count']}/16 |\n"
md += '''
A 为原 DDPM 来源复用。B 为带训练序列及其条件信息的训练经验后验回放；其双端 dev MMD² 都比 A 大，但部分 AA/二肽/长度距离比 A 小，不能概括为所有指标变差。B 的 AMD 精确复制发生在已保留槽位 002；高最近邻相似性和距离改善不能等同于新颖生成。所有 exact-copy 与最近邻分母均保留完整每组 16 条，同端与任意 train 两种范围独立报告。

C 在每个末端内，对 B **实际采样的 decoder-z** 的每个坐标跨 16 槽位置换，保持有限样本逐坐标边际，扰动联合组合；它不置换 DDPM-z，也不具有单一 donor/g/真值。C 比 B 双端 train/dev MMD² 都大；六类距离×两端×两参照中仍有一个反例：AMD dev 长度 KS 为 0.3695652173913043，低于 B 的 0.4320652173913043。C 的 AMD dev 二肽 JS/L1 也略低于 A。因此可说固定样本对坐标组合敏感，不能说所有指标全面崩溃，更不能证明某一特定相关结构或 DDPM 错配就是唯一原因。

D 是与 B 相同训练行的 reference-g 条件 native prior：标准正态先在 flow 的 zp 坐标，再 inverse(g) 到 decoder-z。g 含逐残基与长度模板信息，D 不是只知末端的公平候选，也不等同于直接给 decoder 标准正态。D 相对 A 的 dev MMD 为 FREE 略低、AMD 较高；相对 B 的 AA/二肽/长度指标仍混合。B 与 D 同 indices/epsilon 仅控制配对随机输入，后验还利用 x/h 信息，因此两者差异不能单独归于 flow。roundtrip 通过只支持数值往返一致，不能证明 prior 与目标分布匹配。

负面和混合结果必须一同保留：B/C/D 双端 AA、二肽熵距 train 均大于 A；C 的输出长度变窄为 FREE [17,23]、AMD [10,18]；B/D 的高 train 最近邻比例并未被独特序列数掩盖。B/D 的 donor 为 FREE 16 个不同训练行、AMD 15 个不同训练行（同一行重复一次），有放回重复保留。逐样本共同随机数、donor、两种末端都不是独立训练或生物实验重复。所有精确标量、两参照六类距离、合法 1–48/5–48 多样性与最近邻统计见本审 JSON；全频率向量保留在冻结原评价。

本诊断是看到既有生成/尺度结果后的固定 checkpoint、固定 epsilon 和固定训练 donor bank 的探索。MMD² 是冻结 421 维特征上的 biased linear 估计，不是生物活性；最近邻低或零精确复制也不能证明生物新颖性。dev41 的序列评价不能补造 dev posterior latent 对照，单组 n=16 不支持显著性或总体结论。既有文献中的 aggregate-posterior/prior mismatch 路线不能替本次实验确定原因。

原 CE 正式 E/L/G 因中断永久 NE；原四臂恢复描述性 L=PASS 4/4、G=FAIL 8/18 保持原义。本来源评价中旧 CLI 因缺原 CE/MSE/AR 操作数输出的 false 明确为 NOT_APPLICABLE_SOURCE_DIAGNOSTIC；不是新失败/胜负门。B/C/D 不参与 c-only 排名，也不回填原门或重建/AR NLL。原 CE/MSE 固定 β 的 loss-form 对照、scalar whole-pipeline 边界、AR1000 dev 过拟合、AR 充分性未立及不等计算成本均不因本诊断改变。PROMOTE=HOLD；无新颖性、活性、安全、湿实验前瞻或投稿就绪结论，wetlab N 未知不猜。

新诊断 outer 为 2.679257582873106 秒；2.4913654173724353 秒 process-before-final-I/O 与 2.4944359590299428 秒 closing-before-ledger 为嵌套口径，不相加。历史训练与 DDPM 的已耗成本保留；累计模型 wall 仍只知下界 9999.154718747828206 秒，原 CE 完整中断时间未知。评价、读出及审计耗时另列，当前科学复核没有完整计时，不补 0。

本轮足以否定“替换成训练经验后验即可同时消除双端差距”的简单解读；尚不能排除 latent 来源是多个因素之一，也不能锁定新的修复方案。可据此停止本轮追加探索并保留失败/混合证据；本审不授权训练、模型调用、新 seed 或新 scope。任何后续因果假设需另行明确，当前报告不借未来可能结果改写结论。

来源：[冻结交付索引](../../feasibility/CE_SOURCE_REPLACEMENT_EVALUATION_DELIVERY_v2.json)、[作者全指标](../../feasibility/ce_source_replacement_author_readout_v2/AUTHOR_FACTS.json)、[工程后审](../independent/CE_SOURCE_REPLACEMENT_POSTFLIGHT_v2.json)、[成本 C 来源勘误](../../feasibility/cost_addendum_source_c_erratum_v1/ERRATUM_SOURCE_C_v1.zh.md)。
'''
md_path=OUT/(BASE+'.zh.md')
json_path=OUT/(BASE+'.json')
assert not md_path.exists() and not json_path.exists(), 'Do not overwrite sealed review'
md_path.write_text(md)
result={
 'identity':'AMP H5 source replacement v2 independent scientific final review',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'review_status':'COMPLETED_INDEPENDENT_SCIENTIFIC_CLOSEOUT',
 'status':'PASS_REPORTING_WITH_EXPLICIT_INFERENCE_BOUNDARIES',
 'source_diagnostic_closeout_complete':True,
 'reporting_integrity':'SUPPORTED_WITH_COST_SOURCE_C_ERRATUM_BOUND',
 'blocking_scientific_mismatches':[],
 'new_scope_execution_authorized_by_this_review':False,
 'delivery_index':index_binding,
 'checks':checks,
 'source_bindings':bindings,
 'counts':f['counts'],
 'reference_denominators':{'train_FREE':126,'train_AMD':153,'dev_FREE':18,'dev_AMD':23,'per_source_per_terminal':16},
 'scalar_generation_summaries':summaries,
 'saved_distance_comparisons_no_new_metrics_or_gate':comparisons,
 'source_details':f['source_details'],
 'method_interpretations':f['method_interpretations'],
 'exact_copy_observations':[{'sample_id':x['sample_id'],'terminal':x['terminal'],'method':x['method'],'reference_train_row_id':x['raw_sample_metadata']['reference_train_row_id'],'same_terminal_copy':x['exact_sequence_plus_requested_terminal_copy'],'any_train_copy':x['exact_sequence_copy_any_train_terminal']} for x in copies],
 'resolved_reporting_issue':{'description':'Cost addendum C provenance wording incorrectly named existing DDPM-z; immutable erratum corrects actual sampled B empirical posterior decoder-z. Original sealed files unchanged; always cite erratum together.','erratum':err_path,'numeric_effect':False,'blocking_after_bound_erratum':False},
 'gate_interpretation':side,
 'original_four_arm_L':prior_review['L'],
 'original_four_arm_G':prior_review['G'],
 'AR_ADEQUACY':'NOT_ESTABLISHED_UNCHANGED',
 'PROMOTE':'HOLD',
 'verdict':{'simple_two_terminal_source_rescue':'NOT_SUPPORTED_IN_THIS_FIXED_BANK','coordinate_combination_sensitivity':'DESCRIPTIVELY_SUPPORTED_WITH_MIXED_METRICS','unique_DDPM_causal_bottleneck':'NOT_IDENTIFIED','c_only_source_ranking':'NOT_APPLICABLE','method_novelty_or_general_superiority':'NOT_ESTABLISHED'},
 'costs':{'new_source_outer_seconds':2.679257582873106,'nested_durations_do_not_sum':f['costs'],'model_job_wall_lower_bound':cost['model_job_wall_cost'],'scientific_review_elapsed_wall_seconds':None,'known_analysis_costs_separate':cost['non_model_analysis_cost']['combined_known_timed_cost_decimal']},
 'scientific_limits':f['limits']+['Every source group has only 16 observations per terminal; no p-values or significance claim. C disruption does not establish exact statistical independence or which dependencies caused changes.','No wetlab N inferred, no new prospective wetlab evidence.','Existing four-arm CE/MSE effective-KL-scale, scalar capacity/objective, AR adequacy and cost limitations remain unchanged.'],
 'operations':{'model_calls':0,'training_updates':0,'natural_random_draws':0,'checkpoint_reads':0,'latent_array_reads':0,'raw_cohort_reads':0,'test37_reads':0,'new_quality_metric_recomputations':0,'saved_row_count_and_transcription_arithmetic':True},
 'markdown':{'path':str(md_path),'bytes':md_path.stat().st_size,'sha256':hashlib.sha256(md_path.read_bytes()).hexdigest()},
}
json_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'all_pass':all(x['pass'] for x in checks),'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [json_path,md_path]]},ensure_ascii=False,indent=2))
