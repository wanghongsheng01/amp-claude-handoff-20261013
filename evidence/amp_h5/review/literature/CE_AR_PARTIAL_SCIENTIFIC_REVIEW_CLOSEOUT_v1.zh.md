# AMP H5：CE＋AR partial 补充件回审关闭记录 v1

**结论：新补充件解决了此前 cap-hit、成本范围及完整负结果的呈现问题。关闭的是这些呈现问题；原科学限制与 HOLD 不变。** 原复核与报告未修改，使用时应引用原包与补充件的联合索引。

联合索引 SHA256 已核为 `bb0a2ec8a18742702d57f20fe559ece92fe9a0a81ec0734c7af0d084e58f0714`；其指向的补充 Markdown、JSON、manifest及原交付入口哈希均匹配。本回审核对文字及已保存数值，不替代工程审计。

| 原问题 | 回审结果 |
|---|---|
| cap命名及AR原生终止flag缺失 | **已解决（补充披露）**。两方法FREE均1/16长度48触顶，AMD均0/16；保留全部请求。明确AR `terminated=true` 是抽样EOS，CE长<48是J/PAD决策，机制不同；`status=ok`不可横比自然终止，时间预算`capped_by_terminal=false`也不代表无长度触顶。 |
| 成本范围容易混称 | **已解决**。AR纯更新12.268116秒、训练管线14.292998秒、生成0.020190秒、入口process15.109932秒分别标注，明确嵌套不可相加。CE中断生成下界、额外恢复成本、原总wall未知均保留，没有等实际计算声称。 |
| 负结果仅留JSON、正文不完整 | **已解决**。正文增加双端×train/dev的MMD²、AA/二肽JS与L1、长度KS，以及熵与熵差表、十个固定AR监测点。与此前复核保存数值逐字段一致；八项G失败未删除或改变。 |

补充成本表没有列AR外层supervisor的15.362345秒，但把15.109932秒正确标为入口process，没有将其冒称外层总耗时；外层值仍在原复核记录，故不构成此次成本命名问题的未解决项。

原复核W1/W2的呈现问题可关闭；W3“缺MSE/scalar、不能归因CE目标或形成四臂结论”仍是有效限制，不是补充件应补造的数据。原formal E/L/G永久NE，恢复L=NE、G=FAIL、AR充分性未建立、PROMOTE=HOLD全部保留。AR dev过拟合、固定末点、单拟合种子、每端16输出、dev复用及生物效能/新颖性未证等限制继续适用。

仅阅读指定补充件、联合索引及此前自己的科学复核记录；没有读取或使用当前MSE质量、没有模型/训练/新生成/指标重算/test读取，也没有修改旧文件、阈值或未来计划。此关闭记录不批准下一臂，不提升科学结论。

见 [机器可读关闭记录](CE_AR_PARTIAL_SCIENTIFIC_REVIEW_CLOSEOUT_v1.json)、[原复核](CE_AR_PARTIAL_SCIENTIFIC_REVIEW_v1.zh.md)与[联合索引](../../feasibility/CE_RECOVERY_AR_PARTIAL_WITH_ADDENDUM_INDEX_v1.json)。
