# AMP H5：CE恢复、MSE与AR三臂作者读出

**恢复描述性 L=PASS、G=FAIL；原正式 E/L/G 永久 NOT_EVALUABLE；AR_ADEQUACY=NOT_ESTABLISHED，PROMOTE=HOLD。**

本次仅执行冻结评价器的三臂评价，96个请求全部保留。三臂v4合同未创建、v4读出未执行；等待scalar终态后才使用全部真实root。scalar状态仅引用主线程“19:10:17已启动”的陈述，没有stat/读取其运行输出，也没有创建scalar观测。预备的withheld路径映射没有采用、没有目录或假run。

CE恢复与MSE各自独立工程回执均为PASS，MSE回执报告57检查及CE/MSE随机批次耦合核查通过；本件只转录这些独审结论，不把工程通过解释为质量成功。单seed、反复使用dev41及CE额外恢复attempt仍是限制；当前scalar许可在这些质量数值之前已确定。

| 方法 | split | 对象数 | 内容micro | AA支持类macro | sequence mean | 长度MAE | exact整串 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE21_DDPM | train | 279 | 0.735415 | 0.464623 | 0.755304 | 0.068100 | 21 |
| MSE21_DDPM | train | 279 | 0.120173 | 0.039339 | 0.116926 | 4.580645 | 0 |
| CE21_DDPM | development | 41 | 0.708987 | 0.427601 | 0.722796 | 0.000000 | 3 |
| MSE21_DDPM | development | 41 | 0.125535 | 0.035677 | 0.120047 | 5.560976 | 0 |

| L子项（dev mean，CE对MSE） | CE | MSE | 固定方向 | 结果 |
| --- | ---: | ---: | --- | --- |
| rendered_nonpad_micro | 0.708987 | 0.125535 | gt | PASS |
| rendered_nonpad_macro | 0.427601 | 0.035677 | gt | PASS |
| length_mae | 0.000000 | 5.560976 | le | PASS |
| whole_rendered_exact_count | 3.000000 | 0.000000 | ge | PASS |

L的4项均通过，但它只支持本固定预算匹配MSE比较；不是新颖性、成熟VAE最强基线优势或活性证明。生成方面，CE也未对MSE所有项占优：FREE合法5–48数量CE14/16低于MSE16/16。

| 方法 | 端基 | 请求/合法1–48 | 合法5–48 | unique | 同端/任意train-copy | train MMD² | dev MMD² |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| CE21_DDPM | AMD | 16/16 | 15 | 16 | 0/0 | 0.008950 | 0.007710 |
| CE21_DDPM | FREE | 16/16 | 14 | 16 | 0/0 | 0.017416 | 0.017328 |
| MSE21_DDPM | AMD | 16/16 | 10 | 16 | 0/0 | 0.092200 | 0.084847 |
| MSE21_DDPM | FREE | 16/16 | 16 | 16 | 0/0 | 0.079697 | 0.082335 |
| terminal_LSTM | AMD | 16/16 | 16 | 16 | 0/0 | 0.003158 | 0.007362 |
| terminal_LSTM | FREE | 16/16 | 16 | 16 | 0/0 | 0.004092 | 0.011567 |

所有分布指标以合法1–48集合为分母，对应端基的train/dev为参照。MMD²采用冻结421维固定特征；JS单位bit；KS无p值。

| 方法 | 端基 | 参照 | AA JS | AA L1 | 二肽JS | 二肽L1 | 长度KS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| CE21_DDPM | AMD | train | 0.094472 | 0.489628 | 0.293112 | 0.935708 | 0.255310 |
| CE21_DDPM | AMD | dev | 0.088451 | 0.425182 | 0.365594 | 1.069336 | 0.301630 |
| CE21_DDPM | FREE | train | 0.101694 | 0.517568 | 0.339680 | 1.019105 | 0.279762 |
| CE21_DDPM | FREE | dev | 0.099317 | 0.535324 | 0.417891 | 1.143709 | 0.263889 |
| MSE21_DDPM | AMD | train | 0.261880 | 0.922731 | 0.521821 | 1.395262 | 0.441176 |
| MSE21_DDPM | AMD | dev | 0.258718 | 0.888949 | 0.510380 | 1.366518 | 0.619565 |
| MSE21_DDPM | FREE | train | 0.282576 | 0.940774 | 0.543729 | 1.441071 | 0.473214 |
| MSE21_DDPM | FREE | dev | 0.295547 | 1.020415 | 0.574089 | 1.476348 | 0.458333 |
| terminal_LSTM | AMD | train | 0.021432 | 0.190054 | 0.244797 | 0.821077 | 0.182190 |
| terminal_LSTM | AMD | dev | 0.044849 | 0.328382 | 0.306125 | 0.913460 | 0.244565 |
| terminal_LSTM | FREE | train | 0.028243 | 0.259796 | 0.244611 | 0.810025 | 0.133929 |
| terminal_LSTM | FREE | dev | 0.059628 | 0.443252 | 0.371616 | 1.016355 | 0.201389 |

| 方法 | 端基 | pooled AA熵 | pooled二肽熵 | AA熵距train | 二肽熵距train | 长度均值[min,max] |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CE21_DDPM | AMD | 3.141149 | 5.940560 | 0.562840 | 1.194429 | 16.5000 [4,30] |
| CE21_DDPM | FREE | 3.267681 | 6.013477 | 0.666742 | 1.651367 | 17.1250 [2,48] |
| MSE21_DDPM | AMD | 1.918862 | 3.737209 | 1.785127 | 3.397780 | 8.7500 [1,17] |
| MSE21_DDPM | FREE | 2.014445 | 3.834027 | 1.919978 | 3.830817 | 11.8750 [5,22] |
| terminal_LSTM | AMD | 3.461642 | 6.137055 | 0.242347 | 0.997935 | 13.9375 [8,20] |
| terminal_LSTM | FREE | 3.776887 | 6.863672 | 0.157536 | 0.801172 | 21.0625 [10,48] |

| G子项（CE对AR） | 端基 | CE | AR | 固定方向 | 结果 |
| --- | --- | ---: | ---: | --- | --- |
| development_mmd2_strictly_lower | FREE | 0.017328 | 0.011567 | lt | FAIL |
| ce_all_requests_valid1 | FREE | 16.000000 | 16.000000 | eq | PASS |
| ar_all_requests_valid1 | FREE | 16.000000 | 16.000000 | eq | PASS |
| valid5_count_not_lower | FREE | 14.000000 | 16.000000 | ge | FAIL |
| valid_unique_count_not_lower | FREE | 16.000000 | 16.000000 | ge | PASS |
| same_terminal_train_copies_not_higher | FREE | 0.000000 | 0.000000 | le | PASS |
| any_terminal_train_copies_not_higher | FREE | 0.000000 | 0.000000 | le | PASS |
| aa_entropy_train_gap_not_larger | FREE | 0.666742 | 0.157536 | le | FAIL |
| dipeptide_entropy_train_gap_not_larger | FREE | 1.651367 | 0.801172 | le | FAIL |
| development_mmd2_strictly_lower | AMD | 0.007710 | 0.007362 | lt | FAIL |
| ce_all_requests_valid1 | AMD | 16.000000 | 16.000000 | eq | PASS |
| ar_all_requests_valid1 | AMD | 16.000000 | 16.000000 | eq | PASS |
| valid5_count_not_lower | AMD | 15.000000 | 16.000000 | ge | FAIL |
| valid_unique_count_not_lower | AMD | 16.000000 | 16.000000 | ge | PASS |
| same_terminal_train_copies_not_higher | AMD | 0.000000 | 0.000000 | le | PASS |
| any_terminal_train_copies_not_higher | AMD | 0.000000 | 0.000000 | le | PASS |
| aa_entropy_train_gap_not_larger | AMD | 0.562840 | 0.242347 | le | FAIL |
| dipeptide_entropy_train_gap_not_larger | AMD | 1.194429 | 0.997935 | le | FAIL |

G仍有8项失败：两个端基dev MMD均未低于AR、合法5–48数量较少、AA及二肽熵与train的距离更大。所有CE/AR完整group与此前二臂结果逐值一致；本次没有为改善结果修改指标或门槛。

AR原生cap语义继续有效：FREE_001为length48、terminated=false、max_length_hit=true但status=ok；CE FREE_009为length48/no_eos_at_limit。CE/AR均FREE触顶1/16、AMD0/16，不能横比status_ok为自然终止率；CE J/PAD与AR EOS机制不同。MSE本次两端最大长度22/17，没有48长度触顶。

AR纯训练更新12.268115932秒、监测0.569787959秒、训练pipeline wall14.292998417秒、生成0.020189667秒、入口process15.109931750秒，范围嵌套不相加。AR dev NLL从200步2.615253升至固定1000步3.169793、train降至1.643124，保留过拟合证据与AR_ADEQUACY=NOT_ESTABLISHED。

MSE记录训练1744.337824秒、生成1125.220597秒、process（final I/O前）2877.217766秒、supervisor outer2877.830812秒。CE原中断生成≥451.935743秒、原总wall未知、额外恢复生成1115.496170秒分别保留，不声称实际等计算。

完整原生flag、AR10点曲线及成本语义来自已封口的[补充链](../CE_RECOVERY_AR_PARTIAL_WITH_ADDENDUM_INDEX_v1.json)。旧补充件当时只有两臂，故其历史L不可判；本三臂新计算得到的L=PASS不会覆盖旧包。

完整所有指标、最近邻分布、全部门子项和来源hash见[AUTHOR_FACTS.json](AUTHOR_FACTS.json)；逐样本与逐重建行审计保留在[冻结评价包](../ce_recovery_mse_ar_evaluation_v1/MANIFEST.json)。

- One fitted seed per method and reused development41; no significance or independent generalization claim.
- CE uses a separately approved additional generation restart; original CE interruption remains and its total wall is unknown.
- L supports the fixed-budget matched CE-versus-onehot-MSE comparison only; it does not establish mature-baseline superiority or novelty.
- G remains a descriptive failure against the existing AR, whose adequacy remains unestablished and overfitting evidence remains disclosed.
- Scalar outputs have not been inspected; its running status comes only from root. No full four-arm result or v4 readout is claimed.
- No antimicrobial activity, safety or submission claim; old wetlab does not validate H5.
