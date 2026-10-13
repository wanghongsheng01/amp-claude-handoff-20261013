# 固定200 ID 单次 checkpoint 增量

本批请求 17/64，新增成功 10，累计 146/200；未尝试 47。状态 STOPPED，停止原因 THREE_CONSECUTIVE_TRANSFER_FAILURES。每ID本批一次，未新增或替换ID。

|项目|136版本|本批后|增量|
|---|---:|---:|---:|
|all_peptides.rows|136|146|10|
|all_peptides.families|93|102|9|
|all_peptides.source_family_components|46|52|6|
|all_MIC.rows|1242|1278|36|
|Ecoli_MIC.rows|232|240|8|
|ug_ml_native.point.rows|79|79|0|
|uM_native.point.rows|68|70|2|

完整肽/精确序列/家族/来源组增量见 DATA_INCREMENT.csv；200 ID逐项状态见 ALL_200_COVERAGE.csv，全部失败见 REQUESTS.json，测量单位与点值/删失/区间/±语义见 MEASUREMENT_SEMANTICS.json。旧136原文、原生准入和失败记录保留。分组为操作性关系，不等于独立生物样本。没有换算单位、标签筛选、训练或新假说。

剩余阻塞：Fixed200 coverage incomplete: 54 IDs lack successful details; batch stopped under THREE_CONSECUTIVE_TRANSFER_FAILURES。已交付两包快照未改。
