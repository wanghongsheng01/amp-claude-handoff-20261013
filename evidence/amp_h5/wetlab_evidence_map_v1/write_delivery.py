from pathlib import Path
import json,csv
R=Path(__file__).resolve().parent
rows=json.loads((R/'EVIDENCE_MAP_27.json').read_text());counts=json.loads((R/'COUNTS.json').read_text())
claims=[
{'id':'W01','status':'SUPPORTED_AS_SUPPLIER_REPORTED','claim':'27个编号各有COA、HPLC、MS、溶解度报告；序列、订单和lot可在供应商内部对应。','evidence':['EVIDENCE_MAP_27.json#synthesis.quality_evidence','QC_CHECKS.json'],'limits':'不是独立重测的化学身份或谱图解析，不证明这批lot进入特定MIC孔位。'},
{'id':'W02','status':'SUPPORTED_AS_REPORTED_SUMMARY','claim':'27×4 MIC汇总保留30个数值与78个N；16行全N。','evidence':['COUNTS.json','EVIDENCE_MAP_27.json#MIC'],'limits':'N含义不明，不能转阴性、>1000或未测；27是列出候选分母。'},
{'id':'W03','status':'SUPPORTED_DESCRIPTIVE_NUMERIC_VALUES','claim':'S.aureus列六候选低于32µg/mL：pp1-3=31.25、pp1-5=7.8125、pp2-8=15.625、pp3-11=15.625、pp3-13=7.8125、pp3-14=31.25。','evidence':['EVIDENCE_MAP_27.json#MIC.saureus_atcc_29213'],'limits':'数值记录算术，不是已复核的历史成功率或生成效率。'},
{'id':'W04','status':'SUPPORTED_REPORTED_FOUR_STRAIN_PROFILE','claim':'pp1-5在E.coli/P.aeruginosa/S.aureus/E.faecalis分别记录31.25/125/7.8125/125µg/mL。','evidence':['EVIDENCE_MAP_27.json:sample_id=pp1-5'],'limits':'可描述四株报告谱；不外推耐药临床株、机制、安全性或疗效。'},
{'id':'W05','status':'SUPPORTED_TWO_NUMERIC_VALUES_NOT_SPECIFICITY','claim':'pp2-8在S.aureus/E.faecalis分别记录15.625/62.5µg/mL；另两株为N。','evidence':['EVIDENCE_MAP_27.json:sample_id=pp2-8'],'limits':'N不能证明Gram-positive-specific；原稿/与16/63保留为版本表述。'},
{'id':'W06','status':'SUPPORTED_SUPPLIER_PURITY_WITH_EXCEPTION','claim':'26个COA报告95.0–99.1% HPLC纯度；pp1-7为80.3%，满足其≥80.0%订单规格。','evidence':['EVIDENCE_MAP_27.json#synthesis'],'limits':'不得声称全体≥95%，也不以低纯度自动排除或修改N；HPLC面积纯度非净肽质量分数证明。'},
{'id':'W07','status':'UNRESOLVED_HISTORICAL_RULE','claim':'历史6/30、后稿6/27及200/128/32阈值之间未统一。','evidence':['EVIDENCE_MAP_27.json#manuscript','COUNTS.json#thresholds'],'limits':'当前任一菌严格<128或<200为7而非6，不能事后选菌株/规则修成历史成功率。'},
{'id':'W08','status':'UNRESOLVED_VERSION_CONFLICT','claim':'pp2-3 E.coli旧截图N、两工作簿250；另五条截图空白后表有值。','evidence':['EVIDENCE_MAP_27.json#unresolved_conflicts'],'limits':'不判断修正/补测/重测，不把版本当重复。pp3-1截图序列F计数不确定性另保留。'},
{'id':'W09','status':'UNRESOLVED_ASSAY_AGGREGATION','claim':'pp1-1 Sa/Ef端点行均250/125/125，汇总分别250/125；E.coli M6 blank0.306199997663498。','evidence':['EVIDENCE_MAP_27.json:sample_id=pp1-1','MIC_RAW_ROWS.jsonl'],'limits':'不重新判MIC、不删除异常、不补重复类型/聚合规则。'},
{'id':'W10','status':'NOT_BOUND_TO_HISTORICAL_GENERATION','claim':'固定AMP GitHub代码存在，但27湿实验ID未恢复到历史run/condition/seed/checkpoint及1000→30筛选台账。','evidence':['GENERATION_CHAIN.json'],'limits':'代码默认值、近期H0–H5 checkpoint或相似序列不能替代历史来源。'},
{'id':'W11','status':'NOT_ESTABLISHED_MATCHED_GENERALIZATION_OR_NOVELTY','claim':'这些湿实验记录不能单独证明生成创新、可控谱、相对竞争模型优势或独立泛化。','evidence':['GENERATION_CHAIN.json','ORIGINAL_MANUSCRIPT_LOCATORS.json'],'limits':'原稿跨模型表来自不同实验/阈值，不构成同条件对照；本轮未重新做外部新颖性检索，不作不存在创新的定论。'},
{'id':'W12','status':'CANDIDATE_ONLY_FINAL_DISPOSITION_UNKNOWN','claim':'历史截图还列pp1-6、pp3-8、pp3-10，过程说明当时剩三条未完成。','evidence':['OUTSIDE_27_HISTORICAL_CANDIDATES.json'],'limits':'候选身份截图不证明已合成、最终失败、阴性或额外独立测试。'}]
(R/'CLAIM_REGISTER.json').write_text(json.dumps(claims,ensure_ascii=False,indent=2)+'\n')
report='''# 原 AMP 27 条湿实验肽：证据链映射

**现有材料支持一组真实可追溯到文件与样本编号的合成质控和MIC记录，可用于缩窄后的回顾性应用结果；尚不足以支撑“特定生成模型产生这些活性结果、可控谱或优于其他方法”的完整核心论证。** 主要缺口是历史生成/筛选来源绑定和测定交接语义；方法新颖性与匹配泛化对照也未因这些记录而得到证明。这不表示湿实验不存在。

本轮仅使用原授权AMP代码快照、下载材料及既有审计；不训练、不联网、不改原稿、不读取test37，也不把后来的H0–H5模型结果嫁接到旧湿实验。原文件均只读。

## 可复用交付

- **EVIDENCE_MAP_27.json / .jsonl**：27个逐样本对象，含序列、末端未知状态、订单/lot、COA/HPLC/MS/溶解度定位、四株原始MIC/单位/三行端点、原稿图表、版本冲突及各链条状态。
- **EVIDENCE_MAP_27.csv**：可筛选总表；复杂来源和读数以JSON为准。
- **MIC_RAW_ROWS.jsonl**：324行原工作表观测，包括梯度读数、NC/BLANK、端点与精确sheet/cell地址；不是324次独立实验。
- **CLAIM_REGISTER.json**：可写主张、证据与禁止外推边界，便于后续重写逐句引用。
- **SOURCE_INDEX.json**：原文件路径、SHA256及本轮读取角色；原工作簿逐格定位嵌入映射。

直接读取下载XLSX的ZIP/XML，核对4,608个已有审计单元格；108份供应商PDF哈希与旧清单一致，COA和溶解度文本中27条序列对应。HPLC/MS采用既有文字审计，未冒称重新独立化学解析全部谱图。两工作簿逻辑相同不算独立重复；原单位ug/ml与导出ug|ml分别保存，显示单位据原表写µg/mL。

## 链条实际闭合程度

|环节|现有证据|边界|
|---|---|---|
|生成→候选|有固定GitHub代码、稿件1000条及筛选条件叙述、历史30行候选截图|27/27尚未恢复具体run、checkpoint、seed、条件标签及逐阶段筛选分数；代码默认配置不是历史运行|
|候选→合成|27样本编号和序列可连供应商订单、lot及108份质控文件；历史截图26条序列明确相符，pp3-1含F计数读取不确定性|身份是文件/印刷编号关联；不把候选截图当完整原订单或生成池|
|合成→测定|27编号均连四菌MIC表|0/27恢复到lot→配液→plate/well交接；未知不等于没有发生|
|测定→汇总|108格=30数值+78个N，另324行梯度/端点记录|N、技术/生物重复、判读和聚合规则未闭合；不擅自重判|
|汇总→原稿|pp1-5、pp2-8由正文序列关联代表性图4/5、表2/3；27条均有后修订全表行定位|后修订表不是新增实验；预测结构不是实验结构或已验证逐样本结构来源|

27份COA修饰栏均为N/A，末端Free/Amide没有明确恢复，因此机器映射中N/C末端均为null。TFA salt、总重量和HPLC面积百分比保留，不自动变成纯肽净重或摩尔剂量依据。`ARCHIVE_SAMPLE_LINKAGE.csv`旧`mic_present=False`只指供应商归档内部不含MIC；本映射明确另有MIC工作簿，未误读为没有湿实验。

## 能如实进入重写稿的结果

1. 描述27个列出候选的供应商报告与四株原生µg/mL MIC汇总，保留全部N和负/未知结果，不能只展示亮点。
2. 六个S.aureus低数值记录：pp1-3 **31.25**、pp1-5 **7.8125**、pp2-8 **15.625**、pp3-11 **15.625**、pp3-13 **7.8125**、pp3-14 **31.25** µg/mL。它们是表中严格<32的记录，不是已核验的生成成功率。
3. pp1-5的四株记录为**31.25/125/7.8125/125**；pp2-8的Sa/Ef为**15.625/62.5**。可作报告谱实例，后者两个N不能证明革兰阳性特异性。原稿表3的31/125/8/125与/、/、16、63另存，数值近似不能代替原精度或建立舍入规则。
4. 26份COA纯度为95.0–99.1%；pp1-7为80.3%，符合其≥80%订单规格。不能写全体≥95%，也不能据此自动排除pp1-7或改变它的N。

## 保留而不裁决的矛盾

- **N未定义**；“上限1000”不足以把N统一解释为>1000。
- **pp2-3 E.coli：历史截图N／当前工作簿250**；五条历史空白转有值。两版本均保留，不推断重测或独立重复。
- **pp1-1：Sa与Ef端点行同为250/125/125，汇总却为250与125**；E.coli blank异常保留。没有重写成均值/中位数。
- **6/30、6/27、200/128/32不一致**：任一菌严格<32为6，<128或<200为7（新增pp1-1 Ef125）；Sa三个阈值均为6。只报透明算术，不事后挑规则让历史成功率成立。
- **pp3-1截图序列读取不确定性、三条额外候选最终状态、末端及测定批次**仍开放；不补造。

## 是否足以支撑生成方法应用稿

现有证据足以写出限定来源、保留冲突的“候选合成与报告MIC观察”部分，并证明有值得如实展示的数值结果。对原稿更强的生成方法主张，最先阻断的是**这些27样本与原模型/筛选批次的历史绑定**：未经绑定，无法把报告活性归因于某个方法版本或复核选择分母。其次，缺乏相同数据、条件、筛选预算和湿实验体系下的生成对照，原稿跨模型百分比表不能证明公平优势或泛化。新颖性是另一独立问题，本映射既不能补足它，也不据记录缺口断言没有创新。

因此可复用的是完整证据地图和缩窄的实验观察，不是已完成的生成方法优势证明。新H5输出、开发MIC回归成绩与旧27条湿实验互不替代；本轮没有要求用户补材料，没有正式投稿或外传。**BACKUP_BLOCKED**：交付仅本地保留。
'''
(R/'REPORT.zh.md').write_text(report)
lines=['# 27候选快速总表','','序列来自供应商COA；修饰均N/A、末端未明确。MIC次序E.coli / P.aeruginosa / S.aureus / E.faecalis，单位µg/mL；N原样保留。','','|编号|序列|COA纯度%|四株MIC原值|','|---|---|---:|---|']
for r in rows:
 vals=[r['MIC'][k]['summary_raw']for k in ['ecoli_atcc_25922','paeruginosa_atcc_27853','saureus_atcc_29213','efaecalis_atcc_29212']];lines.append(f"|{r['sample_id']}|{r['identity']['sequence_reported']}|{r['synthesis']['purity_percent_reported']}|{' / '.join(vals)}|")
(R/'MAP_INDEX.zh.md').write_text('\n'.join(lines)+'\n')
print('Report, claim register and quick index written')
