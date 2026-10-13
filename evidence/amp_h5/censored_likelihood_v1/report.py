from pathlib import Path
import json,csv,collections
R=Path(__file__).resolve().parent
result=json.loads((R/'run/RESULT.json').read_text());inv=json.loads((R/'INVENTORY.json').read_text());data=[json.loads(l)for l in (R/'DATA.jsonl').read_text().splitlines()]
def dump(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
contrasts={};foldrows=[];dist=[]
metrics=['group_peptide_MAE','group_peptide_RMSE','peptide_MAE','peptide_RMSE','row_MAE','row_RMSE']
for u in result['units']:
 m=result['units'][u];contrasts[u]={}
 for a,b in [('tail_ridge','endpoint_ridge'),('endpoint_ridge','point_ridge'),('tail_ridge','point_ridge'),('tail_ridge','point_mean')]:
  deltas=[v[a]['group_peptide_MAE']-v[b]['group_peptide_MAE']for v in result['fold_metrics'][u].values()]
  contrasts[u][a+'_minus_'+b]={'metrics':{k:m[a][k]-m[b][k]for k in metrics},'group_MAE_win_tie_loss':{'win':sum(d< -1e-12 for d in deltas),'tie':sum(abs(d)<=1e-12 for d in deltas),'loss':sum(d>1e-12 for d in deltas)}}
  for g,v in result['fold_metrics'][u].items():foldrows.append({'unit':u,'group':g,'contrast':a+'_minus_'+b,'MAE_delta':v[a]['group_peptide_MAE']-v[b]['group_peptide_MAE'],'RMSE_delta':v[a]['group_peptide_RMSE']-v[b]['group_peptide_RMSE'],'point_rows':v[a]['rows'],'point_peptides':v[a]['peptide_IDs']})
 for kind in ['point','right','left']:
  rs=[r for r in data if r['native_unit_group']==u and r['label_kind']==kind]
  for dim in ['peptide_id','sequence_family','source_family_group']:
   for value in sorted({r[dim]for r in rs},key=str):
    z=[r for r in rs if r[dim]==value];dist.append({'unit':u,'label_kind':kind,'dimension':dim,'value':value,'rows':len(z),'peptide_IDs':len({r['peptide_id']for r in z}),'exact_sequences':len({r['sequence']for r in z}),'families':len({r['sequence_family']for r in z}),'source_groups':len({r['source_family_group']for r in z}),'activity_ids':[r['activity_id']for r in z]})
dump('CONTRASTS.json',contrasts);dump('DISTRIBUTION.json',dist)
for name,rs in [('FOLD_CONTRASTS.csv',foldrows),('DISTRIBUTION.csv',dist)]:
 with (R/name).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
ex=json.loads((R/'EXCLUSIONS.json').read_text());dump('EXCLUSION_SUMMARY.json',{'total':len(ex),'formats':dict(collections.Counter(r['format']for r in ex)),'multiple_isolates_activity_ids':[12605],'no_imputation':True})
lines=['# 原生MIC删失观测似然：一次冻结开发性筛查','',
'本次结果未支持尾概率适配在既有点值验证子集上的收益。两个单位中，尾概率模型均差于界限端点对照、仅点值ridge与点值均值；与端点相比，三个权重口径的MAE/RMSE全部变差。全部负结果保留，没有追加模型、调参或seed。','',
'## 分布及可识别范围','',
'以下为可辨认ID/精确序列/操作性家族与来源组，并不声称真实生物学独立。原生单位始终分开，未换算浓度。','',
'|单位与标签|观测行|肽ID|精确序列|家族|来源组|','|---|---:|---:|---:|---:|---:|']
for u,d in inv.items():
 for k in ['point','right','left']:
  x=d[k];lines.append(f"|{u} / {k}|{x['rows']}|{x['peptide_IDs']}|{x['exact_sequences']}|{x['families']}|{x['source_groups']}|")
lines+=['','每个点值验证折的训练集至少保留4个删失来源组，满足预先固定的筛查覆盖底线。µM的G6701、G8221、G13803三个组仅有删失，保留为训练信息，不构成点值验证折或新验证样本。逐肽、家族和来源组分布见DISTRIBUTION.csv，逐折分母见FOLDS.json。','',
f"232条既有E.coli MIC中，本轮纳入173条，排除{len(ex)}条；排除格式计数见EXCLUSION_SUMMARY.json。所有不明区间和±均排除。12605同一界限明确关联两个临床分离株，无法确定单一观测事件，暂挂；161390仍保留原报点值，其泛称/拼写及20% CAMHB备注保留为语义警告，未假定与另一测量条件相同。",'',
'## 冻结比较','',
'采用此前已稳定执行的20AA组成+log长度线性ridge，共22系数；选择依据为简单、可运行、可审核，并非本轮成绩，且旧结果没有证明它稳健优于均值。所有臂无条件输入，不补造未知条件。主两臂的训练行、组→肽→记录权重、仅训练X标准化、容量、alpha=1惩罚与优化预算完全相同，只改删失损失。','',
'固定log10 MIC的Gaussian工作尺度σ=1；点值损失为0.5(μ−y)²，右删失为−logΦ(μ−b)。界限端点对照将b当作y，仅用于对照；连续Gaussian下>和≥事件概率相同，原字符仍保存。仅点值ridge/均值在点子集重新归一权重；本轮点值ridge共享全训练X的尺度，因此不是旧基线的精确复跑。','',
'各单位按完整来源家族组留出，只评估真实点值：21个µg/ml折、13个µM折。两主模型的对比隔离观测损失改变；点值→端点的对比同时改变附加观测和权重，不能解释为纯样本数因果效应。训练目标仅作各臂内部收敛诊断，不横比不相容likelihood分数。','',
'## 全部结果','',
'主指标为来源组→肽→记录等权的log10点值MAE，越低越好。','',
'|单位|点值均值|点值ridge|端点ridge|尾概率ridge|尾−端点|','|---|---:|---:|---:|---:|---:|']
for u,m in result['units'].items():
 vals=[m[a]['group_peptide_MAE']for a in ['point_mean','point_ridge','endpoint_ridge','tail_ridge']];delta=vals[3]-vals[2];lines.append('|'+u+'|'+ '|'.join(f'{v:.6f}'for v in vals)+f'|{delta:+.6f}|')
lines+=['','|单位|点值均值RMSE|点值ridge RMSE|端点RMSE|尾概率RMSE|尾优/平/劣来源组数（MAE）|','|---|---:|---:|---:|---:|---|']
for u,m in result['units'].items():
 vals=[m[a]['group_peptide_RMSE']for a in ['point_mean','point_ridge','endpoint_ridge','tail_ridge']];c=contrasts[u]['tail_ridge_minus_endpoint_ridge']['group_MAE_win_tie_loss'];lines.append('|'+u+'|'+'|'.join(f'{v:.6f}'for v in vals)+f"|{c['win']}/{c['tie']}/{c['loss']}|")
lines+=['','全部行等权、肽等权和来源组等权指标保存于run/RESULT.json；所有预定对比及逐组方向保存于CONTRASTS.json和FOLD_CONTRASTS.csv。未选择性引用较好口径、未进行显著性或独立泛化声明。','',
'## 执行、审核与界限','',
f"唯一自然运行完成102次拟合、34折、588条预测；所有优化均达到预定成功标志和最大绝对梯度≤1e-5。seed1729，CPU/BLAS单线程，无随机种子搜索。拟合与输出循环实测{result['elapsed_seconds']:.6f}秒（不含解释器导入），峰值RSS {result['peak_rss_bytes']/1e6:.2f} MB，低于600秒/1.5GB上限。",'',
'运行前独立科学预审683项、工程预审300项检查通过，均绑定FREEZE.json。运行后独立复算911/911项通过，详见independent/RESULT_AUDIT.json：588条预测、34折、102个拟合状态以及全部权重、组分母和指标均复核一致；没有再次自然优化。','',
'Gaussian尺度、未记录/未输入条件下的删失机制均未验证；点值标签出现本身受测定阈值和报告方式选择，故本结果只能检验已报告点值子集上的工作模型。无法评估潜在删失真值、仅删失来源组的点值预测或总体无偏风险；不能把这次失败说成尾概率似然理论错误。没有左删失自然样本，左尾仅有合成工程检查。','',
'全部材料仍为已暴露开发数据。未采集网络数据、未读取test37或原train/dev、未修改H0–H4或稿件；不声称创新、生物效力或生成优势。BACKUP_BLOCKED，仅本地封存。']
(R/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(contrasts,ensure_ascii=False))
