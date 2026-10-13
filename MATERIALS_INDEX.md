# 材料目录（按需阅读）

|目录/文件|内容|阅读优先级|
|---|---|---|
|README / CLAUDE_HANDOFF_PROMPT / START_HERE|低token总览与边界|先读|
|docs/EXPERIMENT_LEDGER.md|全部阶段真实结果、失败终点、证据链接|先读|
|docs/NEXT_STEPS.md|未突破点、可证伪设计与停止标准|先读|
|manuscripts/review_candidate|原样保留的16页正文、12页补充、Overleaf源和既有审查|需要稿件时|
|manuscripts/latest_policy|替代旧决策页的Bioinformatics规则勘误、CommsBio缺口|涉及投稿时必读|
|manuscripts/historical_v6r3|H5前10页历史稿与源码|追溯历史主张时|
|manuscripts/EB_addendum|后续EB-context负结果补充|需要最新生成路线时|
|evidence/pre_H5|H0–H4交接/索引与原分割边界|追溯时|
|evidence/amp_h5/implementation + protocol + runs|作者训练实现、冻结合同、实际训练/生成日志|定位具体实现时|
|evidence/amp_h5/feasibility + review|四臂、恢复、来源诊断、文献对照与成本审查|核对因果/公平性时|
|evidence/amp_h5/night_followup_v1|后验R/M/S、分布诊断、warmup及失败记录|不要重复旧路线|
|evidence/amp_h5/ar_latent_v3 + direct_z_impl_v1|后续完整运行、指标、独立审计与安全加载失败|不要只看PASS字段|
|evidence/amp_h5/native_baseline_v1 + censored_likelihood_v1|原生单位预测与删失工作模型负结果、逐折日志|分母不等于最终200|
|evidence/amp_h5/root_cause_diagnostic_v1|只读误差分解、缺失计数勘误|没有新拟合|
|evidence/amp_h5/eb_context_v1|最新CV、AR公平机会、开发结果与审查|不是正式PBCT复现|
|evidence/amp_h5/wetlab_evidence_map_v1 + historical_binding_search_v1 + remote_history_binding_v1|27肽真实测量与未闭合来源链|作者事实待答|
|data/fixed200|最小公共分析表、原生单位、字典、clean_data.py、来源索引与一页增量附录|数据准入HOLD|
|external_dependencies|具体未附路径/大小/SHA/获取方式、原代码回收清单|执行前必查|
|environment|原始依赖锁/环境检查|不是当前机器新安装承诺|
|FILE_CATALOG.csv + MANIFEST_SHA256.txt|逐文件角色、来源与内容哈希|按文件查验|

实际日志保留目录层级。历史日志中的PENDING、BACKUP_BLOCKED、早期“下一步”不得覆盖随后已签署终点。源文件保留绝对路径是为了绑定证据；本包不是一键重训环境。
