AMP H5 source_v2 独立成本追加

仅追加这次真实完成的 source_v2 诊断，旧v4及此前报告均不改写。任务于2026-10-11T20:27:28.933294+00:00启动、2026-10-11T20:27:31.613386+00:00结束，exit0；独立后审PASS。外层wall为2.679257582873106秒。

新增96个请求全部完成，0失败、0未执行。B/C/D各32，按FREE/AMD各16；实际批调用为6次decoder、2次inverse flow、2次prior roundtrip forward，训练/encoder/DDPM调用均0。复用已有DDPM32行没有新增请求或模型调用。旧A/B64候选、source_v1/R-S1阻断、断连期间的元数据写入及v2静态修订都不是实际模型尝试。

累计实验任务wall下界：9996.4754611649551+2.679257582873106=9999.154718747828206秒，约166.652579分钟。进程记录2.4913654173724353秒、manifest之后closing之前2.4944359590299428秒及各phase均嵌套于外层，不能再次相加。原CE完整耗时仍未知，累计不能称精确总成本。

跨scope仅为执行记账的合计：296个分配slot，启动至少248，完成226，失败6，原CE部分执行16、CEv1未执行32、原AMD无开始证据16。原latent-only4另列。原四臂评价仍为128个请求；这96个诊断请求不并入正式质量分母，复用32行也不重复收费。训练更新仍AR1100/VAE3060/DDPM4060。

非模型分析与审阅独立计时，不加进上述模型任务wall：

| 新补列分析/审阅 | 秒 |
|---|---:|
| same_n_independent_postflight | 23.45660720905289 |
| four_arm_evaluation | 8.186495707999999 |
| four_arm_readout_v4 | 0.155796208 |
| four_arm_full_metrics_addendum | 0.08936612499999999 |
| source_v2_readonly_evaluation | 5.654053791 |
| source_v2_author_readout | 0.07601374999999999 |
| source_v2_initial_audit_reviewer_label_error | 0.9142335830256343 |
| source_v2_corrected_complete_audit | 1.4403871656395495 |
| source_v2_other_interactive_review_and_report | 未记录 |

本次已知分析追加39.97295353971807278秒；加v4已计13.7332680417443848秒后，已知计时分析合计53.70622158146245758秒。v4的AR readout与same-n作者分析没有重复收费；其余未记录的审计/写作/基础设施时间保持null，不能伪填0。

source_v2首次审计因审计脚本自身误设Free/Amide标签而失败，修正为冻结合同的FREE/AMD后完成63项检查。两段计时分别计入分析，汇总字段不再重复相加；这次审计修正没有新增自然模型请求，也不计入模型失败数。完成请求与合法序列分母有别，本账本只记执行状态，合法性由已绑定评价另报。

该追加是一次posthoc来源替换诊断，不是新的正式方法臂或独立训练种子。参考训练后验、打乱已有DDPM潜变量坐标、借参考g的flow prior各有条件边界，不能作为新独立生物样本或升级公平优势证据。AR充分性未证、实际计算不等、dev41复用、原CE中断与额外recovery机会全部保留。原E/L/G仍NOT_EVALUABLE_INTERRUPTION，PROMOTE=HOLD。

来源及逐文件哈希见SOURCE_BINDINGS.json；批次分母见NEW_CONTROL_DENOMINATORS.csv；分析成本见NEW_ANALYSIS_COSTS.csv；本账本未加载权重、运行模型、读取test/cohort或重算质量。
