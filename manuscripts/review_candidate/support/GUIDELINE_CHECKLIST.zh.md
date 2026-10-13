# 候选格式与指南核查（2026-10-12）

候选采用 **Bioinformatics Original Paper，自由格式初投审阅布局**。Communications Biology要求的生物学推进与当前证据更不匹配；不预测录用概率，也不把改换文章类型当作科学缺口修复。

|项目|本包状态|官方依据/边界|
|---|---|---|
|初投模板|通用article.cls，明确非production模板|[Bioinformatics Submission Online](https://academic.oup.com/bioinformatics/pages/submission_online)允许初投不套模板|
|正文布局|12pt、双距、行号、图表内嵌|同上；16页审阅PDF不等于16页出版模板|
|篇幅|正文约5000词以内工作目标；精确提取统计在checks|[Author guidelines](https://academic.oup.com/bioinformatics/pages/author-guidelines)官方索引与可访问初投页；正式模板7页上限未宣称已通过|
|摘要结构|Motivation/Results/Availability and implementation/Supplementary information/Contact齐全|官方索引核验；精确摘要字数上限无法从受限全文完整确认，采用≤250词内部保守目标|
|补充材料|单一合并PDF，12页|官方指南补充合并要求|
|引用|9篇逐项真实性与支持范围核验|independent_venue/REFERENCES_VERIFIED.json；Witten明确bioRxiv预印本|
|AI披露|正文与投稿信均写实披露代码、分析、写作、代理检查；待真人核验|官方指南AI条款；未把AI代理当人类审稿|
|作者/经费/COI|保留历史信息并明确待确认|尚不满足最终作者声明，不制造贡献或无冲突承诺|
|数据与代码|局部审阅证据和源码已交，完整开放发布未成立|准确写明私有/未验证公开性；无上传|
|科学核心|有限实现个案可审阅，独特贡献及泛化/充分比较未确立|科学独审NO_GO，非排版问题|
|Communications Biology范围|当前不推荐为候选|[Submission](https://www.nature.com/commsbio/submit)、[文章类型](https://www.nature.com/commsbio/submit/content-types)、[格式](https://www.nature.com/commsbio/submit/formatting-guidelines)|
|Nature AI政策|核验现行框架，不沿用旧豁免套话|[Nature Portfolio AI](https://www.nature.com/nature-portfolio/editorial-policies/ai)、[Springer Nature guidance](https://group.springernature.com/gp/group/ai/ai-guidance-for-researchers-editors-reviewers)|

可访问范围限制：Bioinformatics主指南整页遇网页挑战；已用官方索引、官方初投页和主页交叉核验，不重试鉴权，不声称核完每个投稿系统字段。Nature部分页面跳转认证，未认证重试。详细逐条证据在independent_venue/GUIDELINES.json。

同类证据尺度：Beierle等2026的AMP生成多模型比较、QMAP 2026的定量回归基准已有更广覆盖；HydrAMP 2023与AMPGen 2025把选择与实验验证联系到其自身模型。本文不得借用这些论文的实验验证，也不能称首次提出重建与生成分开评价。具体匹配理由见independent_venue/VENUE_REVIEW.zh.md。
