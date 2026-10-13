#!/usr/bin/env python3
"""Read saved four-arm results only; no model imports, split reads, or metric recomputation."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEAS = ROOT / 'feasibility'
EVAL = FEAS / 'ce_recovery_four_arm_evaluation_v1'
V4 = FEAS / 'ce_recovery_four_arm_readout_v4_v1'
OUT = FEAS / 'ce_recovery_four_arm_full_metrics_addendum_v1'
EXPECTED = {
    'implementation/evaluate_h5.py':'b894a7c56cd5c59c5fb77c046baa3c9873ae390d244082dd1f09998459e85417',
    'implementation/frozen_metrics_v1.py':'f65eec16391d98a382d415e7b879f99555a33b33398f8be704661d1c7c027480',
    'implementation/build_readout_v4.py':'8a762d82e9527c4cc4a8367a031f7c9a62a6ac8b150e579c286512cedfa94dc2',
    'implementation/vae_pipeline_formal_v2.py':'e97d2a766774461793cba1860455e4cb4c8d689656b60becc326a9f11297ca9b',
    'review/independent/CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json':'2450c62a2553d854685fee7dde300a2ee794746ad8af36bc4db814b9a472c484',
    'review/independent/MSE_V2_POSTFLIGHT_v1.json':'a0a410526b6a252ba1f73fd3957dea5ee20413bb6e5c264fca5d34a4eec97b71',
    'review/independent/SCALAR_V2_POSTFLIGHT_v1.json':'911a453786a351ac4365e30c818af37aaeac446c5c93159fedf17aab37396d5c',
    'feasibility/CE_RECOVERY_MSE_AR_THREE_ARM_DELIVERY_v1.json':'d0bb39e3f90c93b284acca9308a6f97737a3afc598a49908695c660d85e7f891',
    'feasibility/CE_RECOVERY_AR_PARTIAL_WITH_ADDENDUM_INDEX_v1.json':'bb0a2ec8a18742702d57f20fe559ece92fe9a0a81ec0734c7af0d084e58f0714',
    'runs/formal_v1/terminal_lstm/generation/samples.jsonl':'5127d3b4c342a0eb81c99c773a4f2f066d8fd167157dce8675dbff7e80e3b702',
}
METHOD_ROOTS = {
    'CE21_DDPM':('runs/formal_v2/categorical_ce','runs/ce_checkpoint_recovery_v1'),
    'MSE21_DDPM':('runs/formal_v2/raw_onehot_mse','runs/formal_v2/raw_onehot_mse'),
    'scalar_DDPM':('runs/formal_v2/scalar_legacy','runs/formal_v2/scalar_legacy'),
    'terminal_LSTM':('runs/formal_v1/terminal_lstm','runs/formal_v1/terminal_lstm'),
}

def record(p):
    p=Path(p).resolve()
    assert ROOT in p.parents and p.suffix not in ('.pt','.npz') and 'test37' not in str(p)
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def dump(p,obj):
    with p.open('x') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    assert not OUT.exists()
    sources={}
    def read(p,binding=None,kind='json'):
        p=Path(p);r=record(p)
        if binding:
            assert r['sha256']==binding['sha256'],str(p)
            if 'bytes' in binding:assert r['bytes']==binding['bytes'],str(p)
        sources[str(p)]=r
        if kind=='binding':return None
        t=p.read_text()
        if kind=='csv':return list(csv.DictReader(io.StringIO(t)))
        if kind=='jsonl':return [json.loads(x) for x in t.splitlines() if x.strip()]
        return json.loads(t)
    for rel,sha in EXPECTED.items():read(ROOT/rel,{'sha256':sha},'binding')
    em=read(EVAL/'MANIFEST.json')
    names=['generation_metrics.json','generation_sample_audit.json','reconstruction_metrics.json','reconstruction_row_audit.json','ar_teacher_forced_nll.json','exploratory_gates.json','run.json']
    assert len(em['files'])==len(names)
    def member(root,manifest,name):
        p=root/name;rows=[r for r in manifest if r['path']==str(p)];assert len(rows)==1;return read(p,rows[0])
    ev={n:member(EVAL,em['files'],n) for n in names}
    assert ev['run.json']['rows_retained']==128 and len(ev['generation_sample_audit.json'])==128 and len(ev['reconstruction_row_audit.json'])==960
    gen,rec,gates=ev['generation_metrics.json'],ev['reconstruction_metrics.json'],ev['exploratory_gates.json']
    assert len(gen['groups'])==8 and len(rec['groups'])==6
    vm=read(V4/'READOUT_MANIFEST.json');vf=member(V4,vm['outputs'],'EXECUTION_FACTS.json')
    read(V4/'AUTHOR_READOUT.zh.md',next(x for x in vm['outputs'] if x['path']==str(V4/'AUTHOR_READOUT.zh.md')),'binding')
    assert vf['input_snapshot_stable'] and vf['evaluation_manifest_integrity']['state']=='VERIFIED_CONSUMED_JSON'
    assert vf['original_protocol']['E']==vf['original_protocol']['L']==vf['original_protocol']['G']=='NOT_EVALUABLE'
    L,G=(vf['recovery_descriptive_screens'][x]['state'] for x in ('L','G'))
    receipts={}
    for key,filename,scope in [('CE21_DDPM','CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json','RECOVERY_ENGINEERING_ONLY'),('MSE21_DDPM','MSE_V2_POSTFLIGHT_v1.json','SINGLE_MSE_ENGINEERING_ONLY'),('scalar_DDPM','SCALAR_V2_POSTFLIGHT_v1.json','SINGLE_SCALAR_WHOLE_PIPELINE_ENGINEERING_ONLY')]:
        d=read(ROOT/'review/independent'/filename);assert d['status']=='PASS' and d['scope']==scope;receipts[key]=d
        for name in ['generated.csv','train_reconstruction.csv','dev_reconstruction.csv']:
            p=str(ROOT/METHOD_ROOTS[key][1]/name)
            a=[r for r in d['source_bindings'] if r['path']==p];b=[r for r in ev['run.json']['sample_inputs']+ev['run.json']['reconstruction_inputs'] if r['path']==p]
            assert len(a)==len(b)==1 and a[0]['sha256']==b[0]['sha256']
    oldidx=read(FEAS/'CE_RECOVERY_MSE_AR_THREE_ARM_DELIVERY_v1.json')
    oldbinding=next(r for r in oldidx['packages']['evaluation'] if Path(r['path']).name=='generation_metrics.json')
    old=read(Path(oldbinding['path']),oldbinding);prior={(g['method'],g['terminal']):g for g in old['groups']}
    carried={'/'.join((g['method'],g['terminal'])):g==prior[(g['method'],g['terminal'])] for g in gen['groups'] if g['method']!='scalar_DDPM'}
    assert len(carried)==6 and all(carried.values())
    oldcapidx=read(FEAS/'CE_RECOVERY_AR_PARTIAL_WITH_ADDENDUM_INDEX_v1.json')
    capbinding=next(r for r in oldcapidx['addendum'] if Path(r['path']).name=='CAP_HIT_AND_FULL_METRICS_ADDENDUM_v1.json')
    oldcap=read(Path(capbinding['path']),capbinding)
    native=read(ROOT/'runs/formal_v1/terminal_lstm/generation/samples.jsonl',kind='jsonl')
    native_by={r['sample_id']:r for r in native};assert len(native_by)==32
    term=[];configs={};initial={};statuses={}
    for method,(tr,gr) in METHOD_ROOTS.items():
        configs[method]=read(ROOT/tr/'config.json');statuses[method]=read(ROOT/gr/'status.json')
        if method!='terminal_LSTM':initial[method]=read(ROOT/tr/'vae_initialization.json')
        p=ROOT/gr/'generated.csv';binding=next(r for r in ev['run.json']['sample_inputs'] if r['path']==str(p));rows=read(p,binding,'csv');assert len(rows)==32
        for t in ('FREE','AMD'):
            a=[r for r in rows if r['terminal']==t];assert len(a)==16
            if method=='terminal_LSTM':
                for r in a:
                    n=native_by[r['sample_id']];assert all(str(n[k])==r[k] for k in ('sample_id','method','terminal','sequence','status'))
                    assert n['complete_attempt'] is True and n['length']==len(r['sequence'])
                ended=sum(native_by[r['sample_id']]['terminated'] is True for r in a)
                hits=[r['sample_id'] for r in a if native_by[r['sample_id']]['max_length_hit'] is True]
                semantics='native sampled EOS (terminated) versus max_length_hit; status=ok includes capped output'
            else:
                ended=sum(len(r['sequence'])<48 for r in a);hits=[r['sample_id'] for r in a if len(r['sequence'])==48]
                semantics=('clamp/round then first PAD; no sampled EOS' if method=='scalar_DDPM' else 'J-selected length/PAD suffix; no sampled EOS')
            term.append({'method':method,'terminal':t,'requests':16,'ended_count':ended,'length_cap_hit_count':len(hits),'cap_hit_sample_ids':hits,'status_counts':dict(Counter(r['status'] for r in a)),'semantics':semantics})
    ar= statuses['terminal_LSTM']['training_summary']
    methodinfo={
      'CE21_DDPM':{'parameters':initial['CE21_DDPM']['parameters'],'decoder_head_parameters':initial['CE21_DDPM']['decoder_head_parameters'],'target':'21 classes: AA0..19 and PAD20; all48 positions','reconstruction':'sum cross_entropy(raw,target)/(B*48)','density':'sum(logq-logp)/(B*48)','fixed_beta':1.0,'readout':'joint legal-prefix J on raw21 scores'},
      'MSE21_DDPM':{'parameters':initial['MSE21_DDPM']['parameters'],'decoder_head_parameters':initial['MSE21_DDPM']['decoder_head_parameters'],'target':'same 21 classes and all48 positions','reconstruction':'0.5*sum((raw-onehot(target))^2)/(B*48)','density':'same as CE','fixed_beta':1.0,'readout':'same joint legal-prefix J on raw21 scores'},
      'scalar_DDPM':{'parameters':initial['scalar_DDPM']['parameters'],'decoder_head_parameters':initial['scalar_DDPM']['decoder_head_parameters'],'target':'integer AA0..19 PAD20','reconstruction':'sum((raw-target)^2)/nonPAD_count (sum includes PAD positions)','density':'sum(logq-logp)/16','fixed_beta':1.0,'readout':'clamp0..20, round, firstPAD; fresh legacy arm, not historical H4 checkpoint'},
      'terminal_LSTM':{'parameters':ar['parameter_count'],'architecture':ar['architecture'],'optimizer':ar['optimizer'],'fixed_updates':ar['fixed_steps'],'batch_size':ar['batch_size'],'readout':'terminal-conditioned autoregressive sampled tokens/EOS, temperature1, max48','NLL_role':'teacher-forced diagnostic only; not compared to VAE objectives'}
    }
    costs={**oldcap['costs'],'MSE21_DDPM':receipts['MSE21_DDPM']['costs'],'scalar_DDPM':receipts['scalar_DDPM']['costs']}
    unchanged=[{**r,'unchanged':record(r['path'])==r} for r in sources.values()];assert all(r['unchanged'] for r in unchanged)
    facts={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Completed four-arm author supplement: saved frozen metrics, native termination semantics, exact method/cost descriptions; no new diagnostic scope','original_protocol_E_L_G':'NOT_EVALUABLE','recovery_descriptive_L':L,'recovery_descriptive_G':G,'PROMOTE':'HOLD','AR_ADEQUACY':'NOT_ESTABLISHED','method_info':methodinfo,'method_configs':configs,'phase_sources':METHOD_ROOTS,'requests':128,'reconstruction_rows':960,'full_generation_metrics':gen,'full_reconstruction_metrics':rec,'all_gate_subchecks':gates,'AR_teacher_forced_nll':ev['ar_teacher_forced_nll.json'],'termination':term,'AR_native_flags':native,'status_ok_comparable_for_natural_termination':False,'AR_monitor_history':oldcap['AR_fixed_checkpoint_monitor'],'AR_overfitting':oldcap['AR_overfitting_interpretation'],'costs':costs,'CE_MSE_coupling_review':receipts['MSE21_DDPM']['CE_MSE_coupling'],'prior_six_full_groups_unchanged':carried,'source_bindings_and_preservation':unchanged,'source':record(__file__),'operations':{'model_calls':0,'metric_recomputations':0,'split_reads':0,'test_reads':0,'checkpoint_reads':0},'limits':['Single fitted seed per arm; reused dev41; no p-values or independent generalization claim.','CE versus MSE is a same-capacity fixed-beta loss-form comparison, not reconstruction/KL scale-matched objective optimization; no beta sweep.','Scalar also differs in head capacity, reconstruction target/normalization, KL normalization and readout; it is not a one-factor CE control.','Same update count/budget caps are not equal realized compute or equal baseline tuning adequacy.','CE additional checkpoint recovery and original interrupted generation lower bound remain separate; unknown original total is never replaced by a sum.','AR overfitting remains evidence and AR_ADEQUACY stays NOT_ESTABLISHED; no stronger baseline superiority or new novelty claim.','No activity/safety/wetlab validation/submission claim; later diagnostics are not part of these four-arm observations.']}
    OUT.mkdir();dump(OUT/'FULL_METRICS_AND_NATIVE_CAP_v1.json',facts)
    lines=['# AMP H5：四臂完整指标与原生终止语义补充','',f'**恢复描述性 L={L}、G={G}；原正式 E/L/G 永久 NOT_EVALUABLE。AR_ADEQUACY=NOT_ESTABLISHED；PROMOTE=HOLD。**','','全部128请求、960条train/dev mean重建记录保留；无空串删除、重采或筛除。四臂均使用真实静止root，CE训练与恢复生成/重建分源；本表不含后续诊断。','',
      'CE/MSE均1681549参数、357参数输出头；共用21类、48位置、J读出、β=1及预锁随机批次。两者分别为CE/(B×48)与0.5×raw-onehot平方和/(B×48)，KL项均density/(B×48)。这识别固定β下损失形式的差异，不等于R/K有效尺度已匹配或已优化β。scalar是fresh旧式目标（1681209参数、17参数输出头），目标/归一化/KL尺度/读出同时不同，不能作单组件归因。AR为62501参数的2层hidden64 LSTM（token embedding32、terminal embedding8）。完整公式、配置与来源见JSON。','',
      '| 方法 | split | n | micro | AA macro | sequence mean | 长度MAE | exact |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rec['groups']:lines.append(f"| {r['method']} | {r['split']} | {r['rows']} | {r['rendered_nonpad_micro']:.6f} | {r['rendered_nonpad_macro']:.6f} | {r['per_sequence_mean_nonpad_accuracy']:.6f} | {r['length_mae']:.6f} | {r['whole_rendered_exact_count']} |")
    lines += ['', '| 方法 | 端基 | 请求 | 合法1–48 | 合法5–48 | unique | copy同端/任意train | train MMD² | dev MMD² |','| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    for r in gen['groups']:lines.append(f"| {r['method']} | {r['terminal']} | {r['requests']} | {r['valid_1_48_count']} | {r['valid_5_48_count']} | {r['valid_1_48']['unique_count']} | {r['exact_copy_train_count_all_requests']}/{r['exact_sequence_copy_any_train_terminal_count']} | {r['valid_1_48_vs_train']['mmd2_linear_biased_fixed_features']:.6f} | {r['valid_1_48_vs_development']['mmd2_linear_biased_fixed_features']:.6f} |")
    lines += ['', '分布指标按合法1–48样本、同端train/dev参照计算；固定421特征MMD²、JS单位bit，长度KS无p值。全部频率/长度分布及逐样本最近邻仍在冻结评价包。','','| 方法 | 端基 | 参照 | AA JS | AA L1 | 二肽JS | 二肽L1 | 长度KS |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in gen['groups']:
        for label,key in [('train','valid_1_48_vs_train'),('dev','valid_1_48_vs_development')]:lines.append('| '+' | '.join([r['method'],r['terminal'],label]+[f"{r[key][k]:.6f}" for k in ['aa_jensen_shannon_bits','aa_l1_distance','dipeptide_jensen_shannon_bits','dipeptide_l1_distance','length_ks_statistic_no_pvalue']])+' |')
    lines += ['', '| 方法 | 端基 | AA熵 | 二肽熵 | AA熵距train | 二肽熵距train | 长度均值[min,max] |','| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for r in gen['groups']:
        s=r['valid_1_48'];l=s['length'];lines.append('| '+' | '.join([r['method'],r['terminal']]+[f'{v:.6f}' for v in [s['pooled_aa_entropy_bits'],s['pooled_dipeptide_entropy_bits'],r['pooled_aa_entropy_bits_absolute_gap_to_train'],r['pooled_dipeptide_entropy_bits_absolute_gap_to_train']]]+[f"{l['mean']:.4f} [{l['min']},{l['max']}]"])+' |')
    lines += ['', '| 方法 | 端基 | 最近邻范围 | identity mean[min,max] | identity median | ≥70%数量/合法n |','| --- | --- | --- | --- | ---: | --- |']
    for r in gen['groups']:
        for label,key,count in [('同端train','nearest_train_identity_valid_1_48','nearest_train_at_least_70_percent_count_valid_1_48'),('任意train','nearest_any_train_terminal_identity_valid_1_48','nearest_any_train_terminal_at_least_70_percent_count_valid_1_48')]:
            n=r[key];lines.append(f"| {r['method']} | {r['terminal']} | {label} | {n['mean']:.6f} [{n['min']:.6f},{n['max']:.6f}] | {n['median']:.6f} | {r[count]}/{n['n']} |")
    lines += ['', '| 方法 | 端基 | 请求 | 终止/低于48 | 48触顶 | 语义 |','| --- | --- | ---: | ---: | ---: | --- |']
    for r in term:lines.append(f"| {r['method']} | {r['terminal']} | {r['requests']} | {r['ended_count']} | {r['length_cap_hit_count']} | {r['semantics']} |")
    lines += ['', 'AR FREE_001仍为status=ok但terminated=false/max_length_hit=true；CE FREE_009为no_eos_at_limit。scalar原生3条no_eos_at_limit全部保留。status_ok不能横比为自然终止率；CE/MSE J-PAD及scalar firstPAD也不是AR sampled EOS。','','| 门 | 端基/子项 | CE | 对照 | 方向 | 结果 |','| --- | --- | ---: | ---: | --- | --- |']
    for k,c in gates['reconstruction_ce_vs_matched_mse']['checks'].items():lines.append(f"| L | {k} | {c['left']:.6f} | {c['right']:.6f} | {c['operator']} | {'PASS' if c['pass'] else 'FAIL'} |")
    for t,checks in gates['generation_ce_vs_terminal_ar']['checks'].items():
        for k,c in checks.items():lines.append(f"| G | {t}/{k} | {c['left']:.6f} | {c['right']:.6f} | {c['operator']} | {'PASS' if c['pass'] else 'FAIL'} |")
    lines += ['', 'L通过4/4；G失败8/18：CE两个端基dev MMD均较AR高、合法5–48较少、AA/二肽熵距train较大。CE对MSE的重建改善不构成整体生成优势；FREE合法5–48 CE14/16，MSE和scalar均16/16。scalar双端AA/二肽熵距train均低于CE，其中FREE AA熵差0.030724也低于AR的0.157536，AMD二肽熵差0.264261低于AR的0.997935。CE双端MMD/AA与二肽JS-L1/长度KS距离低于scalar，但合法5–48与熵保护项并非同向；不能称CE全面优于scalar或AR所有指标均最好。原6组CE/MSE/AR全指标与三臂包逐值相同。', '', 'AR dev NLL在200步2.615253、预锁1000步3.169793，train最终1.643124，保留过拟合证据；不回选200步。teacher-forced NLL单列JSON，不与VAE目标数值横比。', '', '| 方法/阶段 | 秒数 | 范围 |', '| --- | ---: | --- |']
    for k,v in costs['AR'].items():
        if isinstance(v,(int,float)):lines.append(f'| AR/{k} | {v:.9f} | 纯更新、监测、pipeline、process分别记录；嵌套范围不相加 |')
    for m in ['MSE21_DDPM','scalar_DDPM']:
        for k,v in costs[m].items():
            if type(v) in (int,float):lines.append(f'| {m}/{k} | {v:.9f} | 原工程后审记录；嵌套范围不相加 |')
    for k,v in costs['CE'].items():
        if type(v) in (int,float):lines.append(f"| CE/{k} | {'≥' if k=='original_generation_lower_bound' else ''}{v:.9f} | 原训练/中断与新恢复分开 |")
        elif v is None:lines.append(f'| CE/{k} | UNKNOWN | 不用阶段相加替代未知总wall |')
    lines += ['', '同更新数/预算上限不等于实际等计算，也不证明AR调参充分。smoke、成熟度诊断、0更新入口失败、复核和分阶段执行的额外成本仍单列；后续诊断不回写这里。','','[完整JSON与来源哈希](FULL_METRICS_AND_NATIVE_CAP_v1.json) · [四臂v4读出](../ce_recovery_four_arm_readout_v4_v1/AUTHOR_READOUT.zh.md) · [原生cap历史链](../CE_RECOVERY_AR_PARTIAL_WITH_ADDENDUM_INDEX_v1.json)','','- '+'\n- '.join(facts['limits'])]
    with (OUT/'FULL_METRICS_AND_NATIVE_CAP_v1.zh.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    dump(OUT/'MANIFEST.json',{'source':record(__file__),'files':[record(p) for p in sorted(OUT.iterdir())],'inputs':list(sources.values()),'original_protocol_E_L_G':'NOT_EVALUABLE','recovery_descriptive_L':L,'recovery_descriptive_G':G,'PROMOTE':'HOLD'})
    print(json.dumps({'output':str(OUT),'requests':128,'reconstruction_rows':960,'L':L,'G':G,'prior_six_groups_unchanged':True}))

if __name__=='__main__':main()
