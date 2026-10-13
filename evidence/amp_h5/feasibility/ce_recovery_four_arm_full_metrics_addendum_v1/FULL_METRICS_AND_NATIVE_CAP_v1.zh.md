# AMP H5：四臂完整指标与原生终止语义补充

**恢复描述性 L=PASS、G=FAIL；原正式 E/L/G 永久 NOT_EVALUABLE。AR_ADEQUACY=NOT_ESTABLISHED；PROMOTE=HOLD。**

全部128请求、960条train/dev mean重建记录保留；无空串删除、重采或筛除。四臂均使用真实静止root，CE训练与恢复生成/重建分源；本表不含后续诊断。

CE/MSE均1681549参数、357参数输出头；共用21类、48位置、J读出、β=1及预锁随机批次。两者分别为CE/(B×48)与0.5×raw-onehot平方和/(B×48)，KL项均density/(B×48)。这识别固定β下损失形式的差异，不等于R/K有效尺度已匹配或已优化β。scalar是fresh旧式目标（1681209参数、17参数输出头），目标/归一化/KL尺度/读出同时不同，不能作单组件归因。AR为62501参数的2层hidden64 LSTM（token embedding32、terminal embedding8）。完整公式、配置与来源见JSON。

| 方法 | split | n | micro | AA macro | sequence mean | 长度MAE | exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE21_DDPM | train | 279 | 0.735415 | 0.464623 | 0.755304 | 0.068100 | 21 |
| MSE21_DDPM | train | 279 | 0.120173 | 0.039339 | 0.116926 | 4.580645 | 0 |
| scalar_DDPM | train | 279 | 0.119145 | 0.075178 | 0.112077 | 1.326165 | 0 |
| CE21_DDPM | development | 41 | 0.708987 | 0.427601 | 0.722796 | 0.000000 | 3 |
| MSE21_DDPM | development | 41 | 0.125535 | 0.035677 | 0.120047 | 5.560976 | 0 |
| scalar_DDPM | development | 41 | 0.158345 | 0.080473 | 0.147073 | 1.341463 | 0 |

| 方法 | 端基 | 请求 | 合法1–48 | 合法5–48 | unique | copy同端/任意train | train MMD² | dev MMD² |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| CE21_DDPM | AMD | 16 | 16 | 15 | 16 | 0/0 | 0.008950 | 0.007710 |
| CE21_DDPM | FREE | 16 | 16 | 14 | 16 | 0/0 | 0.017416 | 0.017328 |
| MSE21_DDPM | AMD | 16 | 16 | 10 | 16 | 0/0 | 0.092200 | 0.084847 |
| MSE21_DDPM | FREE | 16 | 16 | 16 | 16 | 0/0 | 0.079697 | 0.082335 |
| scalar_DDPM | AMD | 16 | 16 | 16 | 16 | 0/0 | 0.032898 | 0.031325 |
| scalar_DDPM | FREE | 16 | 16 | 16 | 16 | 0/0 | 0.035209 | 0.039882 |
| terminal_LSTM | AMD | 16 | 16 | 16 | 16 | 0/0 | 0.003158 | 0.007362 |
| terminal_LSTM | FREE | 16 | 16 | 16 | 16 | 0/0 | 0.004092 | 0.011567 |

分布指标按合法1–48样本、同端train/dev参照计算；固定421特征MMD²、JS单位bit，长度KS无p值。全部频率/长度分布及逐样本最近邻仍在冻结评价包。

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
| scalar_DDPM | AMD | train | 0.174756 | 0.815530 | 0.559878 | 1.442512 | 0.594771 |
| scalar_DDPM | AMD | dev | 0.170141 | 0.765499 | 0.657743 | 1.513000 | 0.608696 |
| scalar_DDPM | FREE | train | 0.192892 | 0.826235 | 0.583719 | 1.445838 | 0.394841 |
| scalar_DDPM | FREE | dev | 0.206321 | 0.814374 | 0.683532 | 1.570983 | 0.534722 |
| terminal_LSTM | AMD | train | 0.021432 | 0.190054 | 0.244797 | 0.821077 | 0.182190 |
| terminal_LSTM | AMD | dev | 0.044849 | 0.328382 | 0.306125 | 0.913460 | 0.244565 |
| terminal_LSTM | FREE | train | 0.028243 | 0.259796 | 0.244611 | 0.810025 | 0.133929 |
| terminal_LSTM | FREE | dev | 0.059628 | 0.443252 | 0.371616 | 1.016355 | 0.201389 |

| 方法 | 端基 | AA熵 | 二肽熵 | AA熵距train | 二肽熵距train | 长度均值[min,max] |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CE21_DDPM | AMD | 3.141149 | 5.940560 | 0.562840 | 1.194429 | 16.5000 [4,30] |
| CE21_DDPM | FREE | 3.267681 | 6.013477 | 0.666742 | 1.651367 | 17.1250 [2,48] |
| MSE21_DDPM | AMD | 1.918862 | 3.737209 | 1.785127 | 3.397780 | 8.7500 [1,17] |
| MSE21_DDPM | FREE | 2.014445 | 3.834027 | 1.919978 | 3.830817 | 11.8750 [5,22] |
| scalar_DDPM | AMD | 4.067511 | 6.870728 | 0.363522 | 0.264261 | 23.1875 [16,48] |
| scalar_DDPM | FREE | 3.965147 | 6.811725 | 0.030724 | 0.853119 | 28.2500 [15,48] |
| terminal_LSTM | AMD | 3.461642 | 6.137055 | 0.242347 | 0.997935 | 13.9375 [8,20] |
| terminal_LSTM | FREE | 3.776887 | 6.863672 | 0.157536 | 0.801172 | 21.0625 [10,48] |

| 方法 | 端基 | 最近邻范围 | identity mean[min,max] | identity median | ≥70%数量/合法n |
| --- | --- | --- | --- | ---: | --- |
| CE21_DDPM | AMD | 同端train | 0.443089 [0.307692,0.571429] | 0.439286 | 0/16 |
| CE21_DDPM | AMD | 任意train | 0.454476 [0.333333,0.571429] | 0.450000 | 0/16 |
| CE21_DDPM | FREE | 同端train | 0.400203 [0.222222,0.523810] | 0.402834 | 0/16 |
| CE21_DDPM | FREE | 任意train | 0.423952 [0.222222,0.545455] | 0.444444 | 0/16 |
| MSE21_DDPM | AMD | 同端train | 0.464185 [0.142857,0.666667] | 0.466063 | 0/16 |
| MSE21_DDPM | AMD | 任意train | 0.485482 [0.142857,0.692308] | 0.483333 | 0/16 |
| MSE21_DDPM | FREE | 同端train | 0.459382 [0.363636,0.562500] | 0.461842 | 0/16 |
| MSE21_DDPM | FREE | 任意train | 0.500787 [0.400000,0.615385] | 0.500000 | 0/16 |
| scalar_DDPM | AMD | 同端train | 0.298147 [0.217391,0.380952] | 0.292892 | 0/16 |
| scalar_DDPM | AMD | 任意train | 0.311507 [0.224490,0.380952] | 0.304952 | 0/16 |
| scalar_DDPM | FREE | 同端train | 0.276786 [0.183673,0.363636] | 0.278889 | 0/16 |
| scalar_DDPM | FREE | 任意train | 0.288111 [0.183673,0.363636] | 0.287088 | 0/16 |
| terminal_LSTM | AMD | 同端train | 0.512850 [0.416667,0.666667] | 0.500000 | 0/16 |
| terminal_LSTM | AMD | 任意train | 0.524846 [0.428571,0.666667] | 0.514706 | 0/16 |
| terminal_LSTM | FREE | 同端train | 0.469031 [0.380000,0.588235] | 0.460606 | 0/16 |
| terminal_LSTM | FREE | 任意train | 0.490631 [0.380000,0.588235] | 0.490000 | 0/16 |

| 方法 | 端基 | 请求 | 终止/低于48 | 48触顶 | 语义 |
| --- | --- | ---: | ---: | ---: | --- |
| CE21_DDPM | FREE | 16 | 15 | 1 | J-selected length/PAD suffix; no sampled EOS |
| CE21_DDPM | AMD | 16 | 16 | 0 | J-selected length/PAD suffix; no sampled EOS |
| MSE21_DDPM | FREE | 16 | 16 | 0 | J-selected length/PAD suffix; no sampled EOS |
| MSE21_DDPM | AMD | 16 | 16 | 0 | J-selected length/PAD suffix; no sampled EOS |
| scalar_DDPM | FREE | 16 | 14 | 2 | clamp/round then first PAD; no sampled EOS |
| scalar_DDPM | AMD | 16 | 15 | 1 | clamp/round then first PAD; no sampled EOS |
| terminal_LSTM | FREE | 16 | 15 | 1 | native sampled EOS (terminated) versus max_length_hit; status=ok includes capped output |
| terminal_LSTM | AMD | 16 | 16 | 0 | native sampled EOS (terminated) versus max_length_hit; status=ok includes capped output |

AR FREE_001仍为status=ok但terminated=false/max_length_hit=true；CE FREE_009为no_eos_at_limit。scalar原生3条no_eos_at_limit全部保留。status_ok不能横比为自然终止率；CE/MSE J-PAD及scalar firstPAD也不是AR sampled EOS。

| 门 | 端基/子项 | CE | 对照 | 方向 | 结果 |
| --- | --- | ---: | ---: | --- | --- |
| L | rendered_nonpad_micro | 0.708987 | 0.125535 | gt | PASS |
| L | rendered_nonpad_macro | 0.427601 | 0.035677 | gt | PASS |
| L | length_mae | 0.000000 | 5.560976 | le | PASS |
| L | whole_rendered_exact_count | 3.000000 | 0.000000 | ge | PASS |
| G | FREE/development_mmd2_strictly_lower | 0.017328 | 0.011567 | lt | FAIL |
| G | FREE/ce_all_requests_valid1 | 16.000000 | 16.000000 | eq | PASS |
| G | FREE/ar_all_requests_valid1 | 16.000000 | 16.000000 | eq | PASS |
| G | FREE/valid5_count_not_lower | 14.000000 | 16.000000 | ge | FAIL |
| G | FREE/valid_unique_count_not_lower | 16.000000 | 16.000000 | ge | PASS |
| G | FREE/same_terminal_train_copies_not_higher | 0.000000 | 0.000000 | le | PASS |
| G | FREE/any_terminal_train_copies_not_higher | 0.000000 | 0.000000 | le | PASS |
| G | FREE/aa_entropy_train_gap_not_larger | 0.666742 | 0.157536 | le | FAIL |
| G | FREE/dipeptide_entropy_train_gap_not_larger | 1.651367 | 0.801172 | le | FAIL |
| G | AMD/development_mmd2_strictly_lower | 0.007710 | 0.007362 | lt | FAIL |
| G | AMD/ce_all_requests_valid1 | 16.000000 | 16.000000 | eq | PASS |
| G | AMD/ar_all_requests_valid1 | 16.000000 | 16.000000 | eq | PASS |
| G | AMD/valid5_count_not_lower | 15.000000 | 16.000000 | ge | FAIL |
| G | AMD/valid_unique_count_not_lower | 16.000000 | 16.000000 | ge | PASS |
| G | AMD/same_terminal_train_copies_not_higher | 0.000000 | 0.000000 | le | PASS |
| G | AMD/any_terminal_train_copies_not_higher | 0.000000 | 0.000000 | le | PASS |
| G | AMD/aa_entropy_train_gap_not_larger | 0.562840 | 0.242347 | le | FAIL |
| G | AMD/dipeptide_entropy_train_gap_not_larger | 1.194429 | 0.997935 | le | FAIL |

L通过4/4；G失败8/18：CE两个端基dev MMD均较AR高、合法5–48较少、AA/二肽熵距train较大。CE对MSE的重建改善不构成整体生成优势；FREE合法5–48 CE14/16，MSE和scalar均16/16。scalar双端AA/二肽熵距train均低于CE，其中FREE AA熵差0.030724也低于AR的0.157536，AMD二肽熵差0.264261低于AR的0.997935。CE双端MMD/AA与二肽JS-L1/长度KS距离低于scalar，但合法5–48与熵保护项并非同向；不能称CE全面优于scalar或AR所有指标均最好。原6组CE/MSE/AR全指标与三臂包逐值相同。

AR dev NLL在200步2.615253、预锁1000步3.169793，train最终1.643124，保留过拟合证据；不回选200步。teacher-forced NLL单列JSON，不与VAE目标数值横比。

| 方法/阶段 | 秒数 | 范围 |
| --- | ---: | --- |
| AR/train_updates_seconds | 12.268115932 | 纯更新、监测、pipeline、process分别记录；嵌套范围不相加 |
| AR/monitoring_seconds | 0.569787959 | 纯更新、监测、pipeline、process分别记录；嵌套范围不相加 |
| AR/training_pipeline_wall_before_manifest_seconds | 14.292998417 | 纯更新、监测、pipeline、process分别记录；嵌套范围不相加 |
| AR/generation_wall_before_manifest_seconds | 0.020189667 | 纯更新、监测、pipeline、process分别记录；嵌套范围不相加 |
| AR/entry_process_seconds | 15.109931750 | 纯更新、监测、pipeline、process分别记录；嵌套范围不相加 |
| MSE21_DDPM/training_seconds | 1744.337824208 | 原工程后审记录；嵌套范围不相加 |
| MSE21_DDPM/generation_seconds | 1125.220597417 | 原工程后审记录；嵌套范围不相加 |
| MSE21_DDPM/process_seconds_before_final_io | 2877.217765667 | 原工程后审记录；嵌套范围不相加 |
| MSE21_DDPM/supervisor_outer_seconds | 2877.830811875 | 原工程后审记录；嵌套范围不相加 |
| scalar_DDPM/training_seconds | 1742.769653959 | 原工程后审记录；嵌套范围不相加 |
| scalar_DDPM/generation_seconds | 1116.657168750 | 原工程后审记录；嵌套范围不相加 |
| scalar_DDPM/process_seconds_before_final_io | 2867.176046917 | 原工程后审记录；嵌套范围不相加 |
| scalar_DDPM/supervisor_outer_seconds | 2867.786129042 | 原工程后审记录；嵌套范围不相加 |
| CE/original_training_stage_subtotal | 1743.378091166 | 原训练/中断与新恢复分开 |
| CE/original_generation_lower_bound | ≥451.935742792 | 原训练/中断与新恢复分开 |
| CE/original_exact_total | UNKNOWN | 不用阶段相加替代未知总wall |
| CE/recovery_setup | 1.359153458 | 原训练/中断与新恢复分开 |
| CE/recovery_generation | 1115.496169917 | 原训练/中断与新恢复分开 |
| CE/recovery_export | 6.814929542 | 原训练/中断与新恢复分开 |
| CE/recovery_process_before_final_io | 1123.995185916 | 原训练/中断与新恢复分开 |
| CE/recovery_supervisor_outer | 1124.252043917 | 原训练/中断与新恢复分开 |

同更新数/预算上限不等于实际等计算，也不证明AR调参充分。smoke、成熟度诊断、0更新入口失败、复核和分阶段执行的额外成本仍单列；后续诊断不回写这里。

[完整JSON与来源哈希](FULL_METRICS_AND_NATIVE_CAP_v1.json) · [四臂v4读出](../ce_recovery_four_arm_readout_v4_v1/AUTHOR_READOUT.zh.md) · [原生cap历史链](../CE_RECOVERY_AR_PARTIAL_WITH_ADDENDUM_INDEX_v1.json)

- Single fitted seed per arm; reused dev41; no p-values or independent generalization claim.
- CE versus MSE is a same-capacity fixed-beta loss-form comparison, not reconstruction/KL scale-matched objective optimization; no beta sweep.
- Scalar also differs in head capacity, reconstruction target/normalization, KL normalization and readout; it is not a one-factor CE control.
- Same update count/budget caps are not equal realized compute or equal baseline tuning adequacy.
- CE additional checkpoint recovery and original interrupted generation lower bound remain separate; unknown original total is never replaced by a sum.
- AR overfitting remains evidence and AR_ADEQUACY stays NOT_ESTABLISHED; no stronger baseline superiority or new novelty claim.
- No activity/safety/wetlab validation/submission claim; later diagnostics are not part of these four-arm observations.
