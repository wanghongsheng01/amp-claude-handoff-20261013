# 问题 → 证据 → 修改 → 复核

本表追踪本轮独立审阅发现与修复，不把 AI 技术复核当作真人同行评审。

|问题|证据|修改|复核入口|
|---|---|---|---|
|把同容量重建比较与另一个生成基线混为排名反转|保存指标24/24 CE优于MSE，同时24/24 LSTM优于CE；LSTM无重建任务|改标题、摘要、引言、图注、讨论、结论、投稿信及决策表，明确“匹配改善成立，超越AR未成立”|independent_science/FINAL_REVIEW_v1.json；SAVED_ORDERING_CHECK.json|
|数据人数和哈希不足以说明来源/分区|原冻结队列协议与描述符协议|补DBAASP来源、active有限R/H规则、NW≥70%连通分组、历史暴露隔离、训练拟合尺度、端基非配对|科学终审SCI-02|
|补充MIC分组定义不完整|prepare_data.py的编辑距离与来源键|补Levenshtein≤20%较长序列长度，PMID或规范title+year合并|科学终审SCI-02|
|指标/解码边界未展开|原冻结指标和解码实现|补最短并列长度、单残基pair零块、pooled JS/L1/entropy与逐序列MMD区别|科学终审SCI-03|
|把DDPM会议条目与arXiv DOI混合|官方NeurIPS论文页|采用正式会议条目、页码和URL，去掉arXiv DOI|independent_venue/FINAL_REVIEW_v2.json|
|独立补充PDF缺基线引用|Witten为预印本，CNN来自QMAP改编|补Witten/QMAP/DBAASP参考文献；明确不是原作者性能复现|指南终审v2|
|图中第二/三面板纵轴标题拥挤|初编译第9页视觉检查|标题直接注明Free/Amide MMD²，移除重复纵轴标题；最终第10页灰度检查|checks/FINAL_VISUAL_AND_SURFACE.json|
|27表字号偏小且漂移到末尾|初补充第11页|放大至footnotesize，收窄列间距，放到独立第10页|最终逐页与表检查|
|决策表0.66219pt溢出|decision.log旧版本|略缩文本列宽，不改页面边距与字体|最终decision.log为0 overfull|
|原surface工具的Poppler bbox提取崩溃|保留各surface.stdout.txt与SIGABRT|明确用PyMuPDF词框代入原间距规则；原脚本与失败未改写|surface_fallback.py与fallback报告|
|AI披露机械检查要求“The authors verified”|原脚本固定字符串；真实人工核验未完成|保留完整事实披露和待人工核验，不添加虚假完成句；机械FAIL保留|指南终审v2；NO_GO决策|

不可通过文字修复的事项：一般性/独特科学贡献、独立确认及基线充分性、完整可复用发布、历史实验关联与测定语义、作者责任及授权。它们保持NO_GO，不列为已关闭。
