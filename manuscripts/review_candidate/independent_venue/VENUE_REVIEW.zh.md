# 独立期刊范围、指南及引用预审

核验日期：2026-10-12。审核者未撰写候选正文。本报告是内部独立代理审核，不是期刊同行评审，也不能代替人类作者的最终判断与签署。

建议采用 **Bioinformatics Original Paper 的初投审阅格式**，当前结论为 **NO_GO**。这是两项已授权目标中较匹配的候选格式，不是编辑接收判断。研究价值可以来自可复核的生成评测、失败边界或应用分析；不需要编造新模型全面胜出。但目前单次有限开发预算的“重建提升未转成生成改善”仅能支撑特定实现、分割和预算下的观察，尚不足以成为成熟benchmark或普遍规律。

Bioinformatics 官方首页及 Submission Online 明确允许初投不套期刊模板，PDF或LaTeX均可；审阅稿可用12pt、双倍行距、行号、嵌入图表的清晰通用格式。约5000词是Original Paper的组织尺度，不能把审阅稿页数当作7页期刊版式合规证据。Application Note服务于新软件/实现/数据库/服务的介绍，不是本研究补足科学贡献的捷径；本轮没有确认独立的Analysis类型。[官方初投要求](https://academic.oup.com/bioinformatics/pages/submission_online)、[官方指南](https://academic.oup.com/bioinformatics/pages/author-guidelines)、[期刊首页](https://academic.oup.com/bioinformatics)

Communications Biology Articles 接受长短原始研究，但要求研究带来新的生物学认识。约5000正文词（不含Methods）、最多10项图表及150–250词摘要只是格式建议。当前序列统计、开发性生成失败及不可与新模型绑定的历史27肽，不能靠格式调整弥补生物学主张不足。[范围](https://www.nature.com/commsbio/submit)、[内容类型](https://www.nature.com/commsbio/submit/content-types)、[格式](https://www.nature.com/commsbio/submit/formatting-guidelines)

## 四项直接比较先例

| 官方来源 | 已发表证据及与本稿的关系 | 不能借用的结论 |
|---|---|---|
| [Beierle等，2026，BioData Mining](https://link.springer.com/article/10.1186/s13040-026-00558-w) | 比较VAE/WAE/RNN/语言模型及采样，讨论指标偏好；明确重建集可能有同源序列，重建准确率不纳入生成质量。说明生成评测可以构成研究，也构成直接先例。 | 不能称本稿首次发现重建不等于生成，或把有限pilot包装为首次AMP模型评测。 |
| [Lavertu等，2026，QMAP，Scientific Reports](https://www.nature.com/articles/s41598-026-56004-8) | 五组预定同源隔离测试分割、共同任务比较及可用软件，报告进步有限等负结果；是负结论有研究价值的近邻。其单位转换、菌种合并与本稿原单位分析不同。 | 不能把本稿少量来源分组折叠评测称为QMAP复现，亦不能由其同行发表推断本稿足够。 |
| [Jin等，2025，AMPGen，Communications Biology](https://www.nature.com/articles/s42003-025-08282-7) | 文中给出生成—筛选—合成—抗菌测试；40候选、38合成、其中81.58%显示活性是该流程报告，非纯生成器命中率。 | 不可与历史27条记录直接比较成功率；也不能借其结果验证新H5模型。 |
| [Szymczak等，2023，HydrAMP，Nature Communications](https://www.nature.com/articles/s41467-023-36994-z) | 条件VAE、候选排序/动力学预筛与五菌株实验；摘要报告9条原型类似肽及6条非活性肽类似物的实验结果。 | 刊物不是Communications Biology；数值不作为本稿公平baseline。需使用已更正Fig.2。 |

这四例不是接收门槛或样本量规定。我的判断是：可将本稿写成可审阅的有限评测案例，但新意应限定在具体后验表示、解码目标及匹配预算下的失配证据；是否足以独立成文仍需科学审核，不能靠换“benchmark”标题解决。

## 必须修复或保留的提交阻止项

1. 完整披露单种子、开发集反复使用、预算和采样规模；不报告封存测试表现，不扩展为普遍生成规律。
2. 历史27肽只作独立历史材料；未定义N、阈值/分母冲突、重复类型不明不可填补；不得称其是新模型前瞻验证。
3. Witten & Witten是2019 bioRxiv预印本（10.1101/692681）。本地代码是缓存QMAP改编的PyTorch移植，不是原作者完整性能复现。
4. 全部实质AI参与需在Methods和投稿信中如实披露，包括代码、分析、写作及审核协助，不可声称仅润色；当前Nature框架要求人类保持判断与责任。代理审稿不能被写为人类独立同行审稿。人类最终确认尚未完成时，不写“all authors approved”。[Bioinformatics指南](https://academic.oup.com/bioinformatics/pages/author-guidelines)、[Nature政策](https://www.nature.com/nature-portfolio/editorial-policies/ai)、[当前出版商详细解释](https://group.springernature.com/gp/group/ai/ai-guidance-for-researchers-editors-reviewers)
5. 私有本地或Library交付不等于稳定公开研究归档。数据/代码可用性、授权与作者身份、机构、通讯作者、贡献、基金和利益冲突必须真实；待核条目是明确阻止提交项，不得用默认“无”抹平。

核心引用书目信息及逐条支持边界见 `REFERENCES_VERIFIED.json`。优先9篇，PRO-LDM只在有具体相关句时保留。Beierle与QMAP直接影响本稿新意判断，不能为压缩引用而删除。HydrAMP的2023更正（10.1038/s41467-023-40879-6）是图2重复问题，不外推为整篇结论失效。

访问限制：Bioinformatics主指南完整抓取因内容长度/网站challenge受限，使用官方索引摘录、首页和可读取的官方初投页交叉核验；Nature部分页跳转认证，未尝试登录或重试认证。官网当前AI说明中的copyediting及披露位置FAQ存在表述不一致，因此采取完整实际披露，不宣称豁免。未声称已核对所有投稿系统字段。未训练、未读取test37、未改原稿、未上传或投稿。

候选正文、补充和投稿信成稿后的引用—句子及格式复核待主代理交付文件。
