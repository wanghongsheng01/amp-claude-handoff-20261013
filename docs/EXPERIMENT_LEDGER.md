# 实验结论账本：先看这一页，再按需取证

所有分数属于各自冻结合同，不能跨行直接合并或当独立重复。工程/算术PASS与科学成功分开。原失败、部分完成、零更新启动错误及审计修正留在对应目录。

|阶段|真实结果与分母|结论边界 / 首选证据|
|---|---|---|
|原重建 H0–H2|原均值重建约15.12%；H2同容量357参数类别头+J约39.09%，长度MAE1.341→0.390|冻结表示上的读出诊断，不是端到端新生成方法；[H2交接](../evidence/pre_H5/HANDOFF_H2_20261011.zh.md)|
|H3非线性读出|quadratic/RFF扩容控制未成立|负结果不可被后续替换；[H3交接](../evidence/pre_H5/HANDOFF_H3_20261011.zh.md)|
|H4桥接|原96请求精确重放、双读出192结果；H2合法1–48为95/96、5–48为91/96；MMD五组变差一组改善，二肽JS六组改善|指标混合、非一致生成优势；[H4交接](../evidence/pre_H5/HANDOFF_H4_H2_BRIDGE_20261011.zh.md)|
|H5四臂|CE/MSE dev微准确率70.90%/12.55%，scalar15.83%；CE长度MAE0；4方法×2端型×16=128生成请求|CE改善匹配MSE，却24个距离比较均未胜AR。CE原生成中断永久NOT_EVALUABLE；另行恢复不算独立重复；[最终交接](../evidence/amp_h5/HANDOFF_H5_FOUR_ARM_AND_SOURCE_DIAGNOSTIC_FINAL.zh.md)|
|来源替换|A复用32，B/C/D各32新请求；经验后验B双端MMD未改善，C置换z结果混合、D含参考g额外内容|不能独因果归结DDPM/相关性；[作者读出](../evidence/amp_h5/feasibility/ce_source_replacement_author_readout_v2/AUTHOR_READOUT.zh.md)|
|R/M/S、分布诊断|保存后验路径与同n整组参考；经验敏感性范围不等CI/p值|未保存的dev latent不得假装已核；[posterior读出](../evidence/amp_h5/night_followup_v1/posterior_path/READOUT.zh.md)|
|β warmup|M等序列重建0.722796→0.657852，S8 0.661116→0.589099；四守卫全部失败|CLOSED_NEGATIVE，不追加种子/日程；[闭环索引](../evidence/amp_h5/night_followup_v1/WARMUP_CLOSURE_INDEX_v1.json)|
|AR+latent v3|A正确z的S8=.690802，B乱配=.091958，同A错配=.097533；ABC各1000更新，96新请求+32历史复用|信息门PASS、六组生成门FAIL；C更小/D历史，不说后验完全无信息；[最终索引](../evidence/amp_h5/ar_latent_v3/DELIVERY_INDEX_v1.json)|
|direct-z|R/Z各1000更新；96槽含历史C32。Z对R双端MMD/二肽JS均更差|FAIL_MECHANISM_SCREEN；8086保存序列指标复算PASS不消除checkpoint安全加载失败，完整独立管线INCOMPLETE；[v3终点](../evidence/amp_h5/direct_z_impl_v1/DELIVERY_DIRECT_Z_v3.json)|
|化学/CNN准备|75记录化学审核；测定材料质量基准未建立，质量→摩尔转换资格0；CNN实现与合成检查通过|结构可计算质量不等于测定浓度可换算；[交付](../evidence/amp_h5/chemistry_cnn_delivery_v1/DELIVERY.json)|
|原生单位MIC baseline|46/39点行、10/7来源组；17折、1700 CNN更新。质量组主MAE均值.811036/ridge.775373/CNN.819758；摩尔组.516993/.582247/.576262|CNN双组不胜均值；ridge微改善随权重反向，不能称稳健信号；[报告](../evidence/amp_h5/native_baseline_v1/RESULT_REPORT.zh.md)|
|censored-MIC|136详情阶段：173纳入观测=147点+26右界，34折、102拟合、588预测。尾-端点主MAE差+.023094/+.137381|两单位三个权重口径均恶化；工作模型失败不否定删失似然理论。不是旧46/39的复跑；[报告](../evidence/amp_h5/censored_likelihood_v1/REPORT.zh.md)|
|MIC根因只读诊断|组间均值方差占比90.8%/71.4%；线性跨组均值偏差超过组内收益。仅发现14条空白培养基计数问题|字段未进模型，旧预测不变/不重训；非条件因果效应或无特征信号证明；[报告](../evidence/amp_h5/root_cause_diagnostic_v1/REPORT.zh.md)|
|EB-context|train-only分组CV；dev等端型序列NLL partial49.16431、pooled48.58710、separate48.91772、fresh AR48.04302；384生成尝试全保留|NOT_SUPPORTED；historical AR57.96692仅历史参照，非正式PBCT复现；[最终结果](../evidence/amp_h5/eb_context_v1/DELIVERY.json)|
|约束生成候选|通用finite-state/HMM引导与twisted SMC等已有成熟先例|generic机制创新REJECT；未执行新自然实验；[决策](../evidence/amp_h5/constraint_analysis_v1/DECISION.json)|
|固定200数据收尾|详情200/200；1788 MIC/165ID，Ecoli318/158ID；相对146新增510MIC/78Ecoli，相对136新增546/86|仅数据交付闭合，未重跑上述模型；[一页附录](../data/fixed200/APPENDIX_1PAGE.pdf)、[数据说明](../data/fixed200/README.txt)|

来源、配置、种子、失败条件、预算与耗时均按各阶段原始协议/日志保存。累计早期166.65分钟是旧阶段下界，不能误当所有后续实验精确总耗时。某些大数组/逐复制矩阵仅提供未附清单，不能称“所有字节已打包”。
