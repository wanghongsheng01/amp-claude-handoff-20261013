# AMP H5 来源替换诊断 v2：仅评价准备

新 v2 运行根为 `runs/ce_source_replacement_v2`，评价输出预定 `feasibility/ce_source_replacement_evaluation_v2`。未执行自然模型或评价，没有创建未来run或结果目录。v1准备与原四臂交付保持封存。

v2 seed与96槽声明均已按实际文件SHA和内容核验；seed422/423、全部96行ID/种子、四method ID及所有指标均未改。32条复用CE引用与96条新B/C/D请求分账，未来合计128观察不能称128条新生成。

实际donor列为 `reference_train_row_id`，必须保留；v1的 `selected_train_row_id` 只是提案别名。B是训练经验后验回放，D含reference-g训练残基及长度模板额外条件，不是c-only公平候选，也不参与主排名。旧门文件缺操作数的false不作新诊断FAIL；不新增胜负门或显著性推断。

完成判定必须联合新scope终态status、独立closing ledger和supervisor最终退出记录。准入失败若没有CSV，依据预锁96请求和监督stderr记NE_ADMISSION_FAILURE，不造样本；已有forward后的finalization异常不得错报零模型准入失败，实际调用与失败证据保留，未完成源质量不可解释为完整结果。

等待root授权的新scope真实执行和独立工程后审后，才运行准备态命令。纯合成8检查沿用同接口v1结果；本次只核真实seed/slot绑定，未读自然序列、cohort、模型、checkpoint或test37。原CE E/L/G永久NE，PROMOTE HOLD。
