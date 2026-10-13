# AMP H5 开发集原始索引契约修复 v2

第一次单CE正式入口已执行且失败，不能抹去：`runs/single_ce_execution_v1`，退出2，用时1.074秒；`runs/formal_v1/categorical_ce`保存原状态、异常与32个not_executed请求。失败发生在读取开发集的身份检查，尚未构造/训练模型，更新0。原状态的development_read=false是成功校验后才置位的记录缺陷；实际上dev文件已打开，不能称未读dev。

根因已有作者与独立双重只读核查：绑定的train descriptor原始idx为0–278，dev原始idx为279–319。dev41逐位置序列、class和terminal全部对应，键唯一、多重集一致；没有数据或行序错配。旧guard错误要求每个split的idx从0开始。无需重排、重新选split、重算descriptor或修改源文件。

允许一次明确版本化的训练前契约修复，不在旧输出目录重试：新的formal_v2副本要求配置显式绑定train起始idx0、dev起始idx279，逐行同时核原始idx、序列、class、terminal、数量与文件hash。保留源记录idx，不重编号；评价导出仍使用split位置，与原cohort逐行对应。修正读操作记录，在打开前记录read_attempt，成功后另记validated，失败不能冒称未读。

修复统一适用于CE、MSE和scalar三个VAE臂。模型、head、目标、KL、scaler、数据、初始化、种子、批次、RNG调用、更新数、学习率、优化器、预算、解码、生成与监测规则均不变。独立审查必须确认相关计算函数与原版本AST相同，验证完整train/dev行映射，并确认错误offset仍会被拒绝；不运行新模型探针。

这不是一次根据质量结果调整模型的机会：第一入口没有任何模型更新或质量结果。失败、额外读取和入口成本仍记账；这是新增v2执行尝试，不能假称从未失败。原模型阶段不重试、不补采、不中途换版本的规则继续有效。

只有v2脚本、三个配置和单CE计划重新绑定哈希并取得新独立回执后才可执行一次fresh CE。沿用已完成的DDPM数值诊断之有限放行理由，但仍不是新VAE1000稳定保证。MSE/scalar和旧完整FORMAL_PLAN不获自动执行许可。原AR1000结果保持冻结，后续评价引用其原目录。H0–H4、稿件与test37不动；不创建夜间自动化，主线程跟进后续。
