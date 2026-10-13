# AMP H5 来源替换诊断：仅评价准备

状态：PREPARED_NOT_EXECUTED，不是模型或自然评价执行许可。等待新scope独审和根线程授权的执行完成，以及独立工程后审后，才可运行冻结评价器。

最终方法ID已与runner对齐：CE_DDPM_REUSED、EMPIRICAL_POSTERIOR_B、COORDINATE_PERMUTATION_C、REFERENCE_G_PRIOR_D。未来 `runs/ce_source_replacement_v1/reused_ddpm.csv` 是32条既有CE记录的可追溯引用；`generated.csv` 是B/C/D各32条、共96个新请求。每端固定16槽，空串、失败、重复、48触顶均保留。没有创建任何自然样本或结果文件。

B是训练经验后验回放，C是B采样decoder-z的端内逐坐标置换；二者不是新模型竞争优势。D使用同B训练索引的reference-g，包含额外训练残基和长度模板信息，不是terminal-only，不参与c-only方法排名。新成本与既有CE成本分开；128条观测不等于128条新生成或独立重复。

冻结train279/dev41精确绑定与未来命令在 `EVALUATION_PREPARATION.json`。seed列表示继承epsilon轨迹FREE422/AMD423；B索引选择523/524及C置换625/626另列。runner的最终seed文件尚待交付和hash核验；当前准备包不把此缺件当作已完成绑定。

继续报告冻结全部生成指标：train/dev421特征MMD²、合法1–48/5–48、unique、同端和任意train-copy/最近邻、AA与二肽JS/L1/entropy及其train差、长度KS与完整分布。请求分母始终是16；分布资格沿冻结实现，合法部分字符串即使来自失败请求也可能进入raw分布，因此必须同时显示完成数/失败数，失败源不能解释为完整质量。

CLI附带的旧 `exploratory_gates.json` 仅适用于原CE/MSE/AR名字。新方法缺操作数导致的false值标作NOT_APPLICABLE_SOURCE_DIAGNOSTIC，不是本诊断失败门；不设计新胜负门、不推断显著性。重建与AR NLL空数组也只表示本scope不适用，不能造观测。

纯合成接口8项检查已通过，无自然数据、模型、encoder、DDPM、训练、checkpoint或test37读取。原四臂交付、原CE永久NE及PROMOTE HOLD保持不变。
