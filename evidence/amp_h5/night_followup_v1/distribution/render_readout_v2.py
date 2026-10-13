#!/usr/bin/env python3
"""Transcribe sealed descriptive statistics; no natural CSV/model access."""
import hashlib,json,time
from pathlib import Path
START=time.perf_counter();HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def dump(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
src=HERE/'results_v2';manifest=json.loads((src/'MANIFEST.json').read_text());s=src/'SUMMARY.json'
bindings={x['path']:x for x in manifest['files']};assert len(bindings)==len(manifest['files'])
for q in (s,src/'INDEX_SEAL.json'):
 b=bindings[str(q)];assert q.stat().st_size==b['bytes'] and sha(q)==b['sha256']
closing=json.loads((src/'CLOSING_LEDGER.json').read_text());execution=json.loads((HERE/'execution_v2/RESULT.json').read_text())
assert closing['state']=='completed' and closing['manifest_sha256']==sha(src/'MANIFEST.json') and execution['exit_code']==0 and not execution['outer_timeout'] and execution['analysis_closing_state']=='completed'
data=json.loads(s.read_text());assert data['operations']['model_calls']==0 and data['all_input_hashes_unchanged']
out=HERE/'author_v2';out.mkdir(exist_ok=False)
def num(x):return 'null' if x is None else f'{x:.9f}'
def qstr(x):
 if not x['defined']:return 'null'
 q=x['quantiles'];return f"{num(q['q025'])} / {num(q['q50'])} / {num(q['q975'])}"
lines=['# AMP H5 固定特征有限样本分布诊断 v2','',
'只读分析已完成。以下结果使用冻421维特征与原 biased linear MMD²，仅描述固定训练模型、现有train/dev划分和16请求批次。训练种子不确定性、总体置信区间及正式显著性均未估计；不新增胜负门。原CE正式E/L/G永久NOT_EVALUABLE、恢复描述门不改、PROMOTE=HOLD。','',
f"保留四臂128请求；2048次预锁重复，根种子20571。实际分析外层wall={execution['wall_seconds']:.9f}秒；模型、encoder、新生成、训练、latent数组和test读取均0。原四臂及B/C/D包均未改。",'',
'## 固定train/dev差异','',
'| 端基 | train/dev条数 | train/dev长度中位数 | train/dev MMD² | 长度块 | AA块 | 二肽块 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for t,c in data['cohort_descriptions'].items():
 m=c['train_development_MMD'];lines.append(f"| {t} | {c['train']['rows']}/{c['development']['rows']} | {num(c['train']['length_quantiles_linear']['q50'])}/{num(c['development']['length_quantiles_linear']['q50'])} | {num(m['total'])} | {num(m['length'])} | {num(m['AA'])} | {num(m['dipeptide'])} |")
lines+=['','完整长度计数/分位数、421维均值/标准差/极值和有符号train−dev均值差见SUMMARY.json。完整train有224组件，dev有33组件；其中跨端组件分别7/2个。','',
'## 真实train16到完整固定dev的经验参考','',
'每端固定n16，比较对象一直为完整dev FREE18/AMD23。组件方案对所有完整组件子集中恰16+16的可行子集均匀抽样，保留跨端组；它不等价于按序列均匀抽样。训练序列曾被网络看过、dev也被重复使用，这些参考不是模型未见验证、null或达标线。','',
'| 方案 | 端基 | MMD²均值 | 经验q2.5% / q50% / q97.5% |','| --- | --- | ---: | --- |']
for scheme,ts in data['train16_fixed_dev_reference'].items():
 for t,v in ts.items():lines.append(f"| {scheme} | {t} | {num(v['MMD_summary']['total']['mean'])} | {qstr(v['MMD_summary']['total'])} |")
lines+=['','| 方法 | 端基 | 固定生成16→dev MMD² | 序列参考中小于/并列/大于点值 | 组件参考中小于/并列/大于点值 |','| --- | --- | ---: | --- | --- |']
for method,ts in data['observed_four_arm'].items():
 for t,v in ts.items():
  a=data['train16_fixed_dev_reference']['sequence_without_replacement'][t]['generated_empirical_positions'][method];b=data['train16_fixed_dev_reference']['whole_component_exact_16_16'][t]['generated_empirical_positions'][method]
  lines.append(f"| {method} | {t} | {num(v['vs_development']['total'])} | {a['below']}/{a['exact_ties']}/{a['above']} | {b['below']}/{b['exact_ties']}/{b['above']} |")
lines+=['','这些计数仅为2048次经验参考中的位置，不能转换为显著性或生成优势。各样本点的有效n、状态、合法5–48、unique、对train的原距离及全部三块贡献见SUMMARY.json。','',
'| 端基 | 序列无放回解析期望MMD² | 2048次均值减解析期望 |','| --- | ---: | ---: |']
for t,v in data['sequence_SRSWOR_analytic_check'].items():lines.append(f"| {t} | {num(v['exact_expected_sample_mean_mmd']['total'])} | {num(v['Monte_Carlo_mean_minus_exact_expectation']['total'])} |")
lines+=['','解析式使用population mean分母N；有限n项=(N−16)/(16(N−1))×mean_i||φ_i−μ_train||²。只解释同一MMD估计器的有限样本正偏移，不用于组件方案，不新增比较阈值。','',
'## 固定拟合下CE/AR敏感性','',
'生成面板以完整请求槽位为单位；dev面板对33组件整体有放回重采样，各方法共享相同dev multiplicity，跨端组共享重复次数；联合面板复用这两组索引。下面是CE−AR的经验分位数，正值表示此次距离数值CE更高。它们不是总体CI或训练种子范围，不按是否跨零作检验。','',
'| 面板 | FREE ΔMMD² q2.5/q50/q97.5 | AMD ΔMMD² q2.5/q50/q97.5 | 两端等权均值分位数 |','| --- | --- | --- | --- |']
for panel,v in data['fixed_fit_bootstrap_sensitivity'].items():lines.append(f"| {panel} | {qstr(v['CE_minus_AR']['FREE']['total'])} | {qstr(v['CE_minus_AR']['AMD']['total'])} | {qstr(v['CE_minus_AR_equal_terminal_mean']['total'])} |")
for panel,v in data['fixed_fit_bootstrap_sensitivity'].items():
 n=v['dev_n_histogram'];ranges={t:[min(map(int,n[t])),max(map(int,n[t]))] for t in n}
 lines+=['',f"{panel}：dev有效n范围FREE={ranges['FREE']}、AMD={ranges['AMD']}；缺端次数={v['zero_terminal_repetitions']}。完整n频数和undefined计数留在SUMMARY，所有replicate值和索引完整封存。"]
lines+=['','dev估计对象是组件经验重采样后按序列加权的端内均值，分母随组件抽样变化；缺端保留null，不拒采补齐。样本单位不是残基或421坐标，n16中心协方差秩至多15。','',
'以上诊断不能把训练重放解释成创新，也不能证明完整序列分布相同、抗菌活性或安全性。AR_ADEQUACY仍NOT_ESTABLISHED；单训练种子、dev复用、不同实际成本和CE额外恢复仍限制方法结论。B/C/D未纳入此bootstrap，C不代表μ/logσ配对验证。','']
with (out/'READOUT.zh.md').open('x') as f:f.write('\n'.join(lines))
dump(out/'SOURCE_BINDINGS.json',{'inputs':[rec(s),rec(src/'INDEX_SEAL.json'),rec(src/'MANIFEST.json'),rec(src/'CLOSING_LEDGER.json'),rec(HERE/'execution_v2/RESULT.json')],'source':rec(Path(__file__)),'operations':'transcription of sealed results only; no natural CSV/model reads or new resampling'})
dump(out/'RENDER_RESULT.json',{'status':'completed','seconds_before_final_manifest_io':time.perf_counter()-START,'model_calls':0,'natural_statistics_recomputed':False})
dump(out/'MANIFEST.json',{'files':[rec(f) for f in sorted(out.iterdir()) if f.is_file()],'source':rec(Path(__file__))})
print(json.dumps({'output':str(out),'manifest_sha256':sha(out/'MANIFEST.json'),'elapsed_seconds':time.perf_counter()-START}))
