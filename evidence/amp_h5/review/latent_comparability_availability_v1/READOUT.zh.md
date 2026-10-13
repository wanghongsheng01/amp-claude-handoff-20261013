AMP H5 latent 可比性材料核查

CE/MSE 的 train posterior cache 可用；已检查的当前模型产物中，dev 的 μ、logσ、z 均未保存。CE恢复和MSE的 dev_reconstruction.csv 只保存 μ 解码结果，dev_monitor.jsonl 只保存聚合指标，不能替代 dev latent。此次没有为补缺执行任何模型前向。

- train cache：values float32[279,2,16,48]（μ/logσ）、labels int64[279]、row_ids int64[279]；无 sampled z、epsilon、component_id。
- 生成缓存：每臂每端16条，posterior_parameters 与 z 均保存；无原始 epsilon。
- 冻结 TRAIN_VAL_BINDING.json 可从 model_row_id 映射 component_id。train ID 为0..278；dev为279..319。重建CSV的dev row_id是局部0..40，不能直接当source ID或group ID。

冻结组件确有来源，但应称“协议定义的序列相似性连通组件”，不冒称已验证生物家族。建边条件是全局 Needleman–Wunsch（match+2、mismatch−1、线性gap−2、包含末端gap；按score、matches、负columns并列选择）所得 matches/all alignment columns≥0.70，再取传递连通分量。组件内不保证任意两条都≥70%；ID含原冻结marker成员，不能从发布子集重新推断。train279含224组，dev41含33组，组交集0，绑定字段逐行一致。

| train端基 | 序列数 | 含该端基的组数 | 该端基组大小分布 |
|---|---:|---:|---|
| FREE |126|109|1条×99组；2条×4；3条×5；4条×1|
| AMD |153|122|1条×97组；2条×19；3条×6|

7个组件跨端基，包含FREE12条、AMD11条；全组件最大5条。仅用整数DP证明：不拆组件的单个联合子集达到(FREE16,AMD16)可行，尚未选择任何子集。只有后续实际联合选取恰(16,16)，剩余reference才为110/137；两端各自抽选后取组件并集可能额外移走另一端样本，不能预设仍各减16。

比较单位应是一整条序列的latent张量。每端n16的中心化样本协方差秩上界为15；不能把16条生成的协方差与126/153条全train直接当成等量估计比较，亦不能把latent坐标/残基位置当独立样本。此次只核材料与整组件整数可达性，未抽样、未计算协方差或质量。

路径、字段、源码行和完整hash见 LATENT_AVAILABILITY.json 与 GROUP_BINDING_AVAILABILITY.json。未打开test数据、全pair/全graph文件或live scalar；未修改旧绑定产物。
