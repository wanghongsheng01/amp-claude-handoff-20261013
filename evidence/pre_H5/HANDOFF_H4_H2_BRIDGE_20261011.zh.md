# H4：H2 到真实生成的独立桥接

本轮已完成固定96个原始尝试的真实重放、两种读出的192个输出及完整评价。**工程重放通过，生成端观察混合，未证实一致改善；不能宣布端到端修复、完整方法优势或可投稿。** H3 两个扩容假说的负结果保持封存，本轮没有调用 H3 quadratic/RFF，也没有以其科学通过为先决条件。

## 实际完成与范围

- 新隔离包装器先按原 checkpoint、EMA、种子、条件、批量16和顺序重放全部96个原输出。六组历史 float32 raw48 按位一致（含符号位），原 tokens、序列、IDs、seeds/status 均一致；最大 raw 差0。只有全96通过后，才对本次保存的同一 hidden 应用冻结 H2 CE+J。
- 历史主运行没有保存 latent/hidden。本次保存了新重放的 latent、hidden、原raw、21类分数、49个候选长度得分及tokens；没有声称与不存在的旧latent文件作按位比较。96是共享上游请求数，192是两种读出数，不能当192次独立生成。
- 2次 inverse flow、1996次 DDPM逆过程批、6次WN、6次scalar projection、6次H2评分/J；零encoder forward、零训练更新、零test读取、零补采和重试。原统计控制没有新增生成或评价，不能将总量写成288个新latent。
- 固定CPU单线程，记录generation 1126.197799500秒、evaluation 12.288061875秒，合计1138.485861375秒（约19分钟），未到1800秒软上限。计时不包含随后final manifest哈希/写入及聊天等待，不能称独立外层全进程墙钟。原参数不变有运行时检查，独审另核checkpoint文件；没有独立重跑全部1996个神经前向。

## 所有六组结果

表内均为原scalar → H2。每组请求分母16，平均长度包含空串；MMD²使用各自合法1–48集合。FREE DDPM的MMD/熵/最近邻集合是16 → 15，其余五组16 → 16，因此该组不能被描述成相同成员集合上的质量差。

| 来源 | 全请求平均长度 | 合法1–48（/16） | 合法5–48（/16） | 开发集421特征MMD² |
| --- | --- | --- | --- | --- |
| FREE / 条件 flow | 25.1875 → 22.8125 | 16 → 16 | 16 → 16 | 0.023593 → 0.036678 |
| FREE / 训练后验池 | 21.9375 → 20.0625 | 16 → 16 | 16 → 16 | 0.021455 → 0.054878 |
| FREE / DDPM | 34.4375 → 27.1875 | 16 → 15 | 16 → 14 | 0.058085 → 0.094327 |
| AMD / 条件 flow | 17.1875 → 14.9375 | 16 → 16 | 16 → 15 | 0.022855 → 0.032073 |
| AMD / 训练后验池 | 15.3750 → 13.1250 | 16 → 16 | 16 → 15 | 0.026115 → 0.038629 |
| AMD / DDPM | 25.6875 → 18.6875 | 16 → 16 | 16 → 15 | 0.043836 → 0.035259 |

两种读出各96条全部保留。原读出合法1–48与5–48均为96/96；H2分别为95/96和91/96。H2的FREE DDPM保留1条空串，另外4条长度1–4；full48由10变7，full48仍可合法，不能等同执行失败。没有error或not_executed。每个head/source/terminal组内重复数均0，同端基及任意train的exact-copy数均0；小样本无exact-copy不证明不存在记忆或近邻复制。

所有六组的AA与二肽pooled entropy都下降，同端基及任意train的最近邻identity均上升。这只描述组成更集中和按冻结比对定义更接近训练序列，不自动等于更好、复制成立或生物效能。生成没有逐位真值，最近训练序列不能充当生成accuracy的金标。

**有利与不利方向同时保留：** train和开发集MMD²均为五升一降，唯一下降的是AMD DDPM；开发集二肽JS和L1六组都下降；train长度KS六组下降，开发集长度KS四降、AMD flow及后验池两组上升。开发集AA-JS五升一降，其他组成指标也有混合方向。不能概括为“所有分布变差”，也不能挑有利组或单指标宣布成功。G4事前就是全量描述覆盖，没有综合改善通过门，故不设事后MMD总体FAIL门。单训练seed、每组16请求、反复使用的val41不提供独立确认、显著性或活性证据。

完整12组表、6组差值表和全部分布字段见 [作者读出](diagnostics_h4_h2_generation_bridge_v1/AUTHOR_READOUT.zh.md)、[完整指标CSV](diagnostics_h4_h2_generation_bridge_v1/AUTHOR_METRICS_LONG.csv)、[全部分布CSV](diagnostics_h4_h2_generation_bridge_v1/AUTHOR_DISTRIBUTION_FULL.csv)。原始全量机器结果保留在两种head各自的evaluation/metrics.json；[根任务转写事实](H4_H2_BRIDGE_ROOT_READOUT_FACTS.json)明确不是独立数值复算。

## 独立核验与冻结

实现作者为 amp_mechanism_independent_readonly；架构审阅者为 amp_rebuild_runner；数学和指标审阅者为 amp_six_method_independent_review；根任务最终主张由 amp_final_science_review另审。架构事前641项、数学指标事前1744项、主张事前45项通过后，root才发精确绑定的START。

架构事后1037项主检查及64项辅助检查通过；独立数值核心1558项检查覆盖43,330字段（35,259数值）、全部192行、96,768个H2分数、12组指标及53,010个查询×train的完整NW比对。此处的PASS说明记录与计算可复核，不说明科学改善。作者表文与独审最终版本由各自最终清单/回执绑定，最终主张审查另存。

- [架构终审](diagnostics_h4_h2_generation_bridge_v1/architecture_review/FINAL_RECEIPT.json)
- [数学与指标终审](diagnostics_h4_h2_generation_bridge_v1/independent_review/FINAL_RECEIPT.json)
- [本轮主张状态](H4_H2_BRIDGE_CLAIM_STATUS_FINAL.json)
- [前代保全](H4_H2_BRIDGE_PREDECESSOR_PRESERVATION_AFTER.json)：507个文件哈希相同；完整H2 196文件、H3 186文件、旧被阻断H3桥接14文件，文件清单亦无新增/删除。

生成清单SHA256为 `305e1c7c47efadf85a92ce150c8d04b4d10a5387c685549a731b41db2cf711b9`；评价清单为 `46aa18fae1f16e6069e66d05cc1af8f756df4a1db99f669a6fe0e13f6c53f0cc`；预锁候选为 `f317c5f01ecfc399013097afc4184b0e19f8d0f82c8c98130dcf9110c8c635d7`。事前G4/软上限文字修订与旧候选均保留，模型运行后没有调容差、改变种子或重跑。

## 尚未成立的科学主张及最小依赖

本轮是捆绑的读出干预：原scalar末层17参数，H2类别末层357参数，并同时由first-PAD改为J边界。共享latent不等于相同头容量或耗时，不能把结果单独归因于CE、J或表示训练。H2/J没有显式真长度输入，但flow的逐残基g近乎标识序列并含长度；不代表去掉该条件信息。训练后验池抽样不是逐对象的held-out重建；DDPM生成仅给terminal。三个来源不能作同信息强基线排名。

尚缺的实质证据包括：完整方法与竞争生成器在同数据/条件/总资源下的比较；改变表示后的匹配端到端训练和生成证据；独立泛化确认；真实独立活性、毒性或安全性结果。标准CE/J/联合学习部件本身不是新算法。未知、区间及右删失MIC不变成阴性；本轮没有增加真人金标或生物实验。

当前桥接没有新增作者文件依赖。下一项可本地准备的是可审查的完整baseline适配及共同成本合同；[基线前置](h4_followup_baseline_design/FAIR_BASELINE_REQUIREMENTS.zh.md)仅给出LSTM_peptides与ProtWave-VAE两个候选，二者均NOT_YET_COMPARABLE，尚未运行、没有竞争性结果。冻结的[端到端可行性分析](h3_failure_analysis_readonly/END_TO_END_21HEAD_FEASIBILITY.zh.md)同样只是未来依据：若encoder变化，需重编码并重训匹配DDPM，不能直接称旧DDPM已修复。新训练的初始化、目标尺度、预算和判定必须另行冻结及独审；本轮没有启动。

本次新材料只在隔离目录；未修改冻结论文候选或原来源，未上传、push、正式提交、访问生产、读取凭据、付费调用或重试被拒鉴权；A10/GPU与API鉴权仍不作为已验证资源。**BACKUP_BLOCKED**：本地哈希清单不是云备份。没有投稿完成或录用承诺。
