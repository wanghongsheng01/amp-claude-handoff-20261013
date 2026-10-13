# AMP H5：B/C/D 来源替换诊断作者读出

**128条来源观察=32条既有CE复用+96条新请求（B/C/D各32），不是128条新生成或独立重复。B/C/D不作模型排名；旧门不适用，原CE E/L/G永久NE，PROMOTE=HOLD。**

本次只读评价使用固定train279/dev41和冻结421维MMD及全部原指标。新scope工程通过仅证明保存数据、算术路径、调用和记录完整，不是独立模型重放；roundtrip通过不能解释为prior质量优势或创新。生成与评价均无新训练、encoder或DDPM调用；新生成实际6次decoder batch、2次prior inverse及2次roundtrip forward。

| 来源 | 信息与解释边界 |
| --- | --- |
| CE_DDPM_REUSED | Byte-bound remapping of existing CE recovery32; preserve original source ID/method/path/SHA; no new DDPM or decoder calls. |
| EMPIRICAL_POSTERIOR_B | Training empirical posterior replay with replacement; posterior contains seen sequence and conditioning information. Not novelty or new model success. |
| COORDINATE_PERMUTATION_C | Within-terminal coordinate permutations of actual sampled B decoder-z; diagnostic of joint combination sensitivity, not a new generative method. |
| REFERENCE_G_PRIOR_D | Frozen CE reference-g conditional native prior with same B TRAIN row indices; extra residue and length-template conditioning. Not terminal-only; excluded from c-only ranking. |

| 来源 | 端基 | 请求 | 完成 | 空串 | 合法1–48 | 合法5–48 | 合法unique | copy同端/任意train | train MMD² | dev MMD² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| CE_DDPM_REUSED | AMD | 16 | 16 | 0 | 16 | 15 | 16 | 0/0 | 0.008950 | 0.007710 |
| CE_DDPM_REUSED | FREE | 16 | 16 | 0 | 16 | 14 | 16 | 0/0 | 0.017416 | 0.017328 |
| COORDINATE_PERMUTATION_C | AMD | 16 | 16 | 0 | 16 | 16 | 16 | 0/0 | 0.020093 | 0.016635 |
| COORDINATE_PERMUTATION_C | FREE | 16 | 16 | 0 | 16 | 16 | 16 | 0/0 | 0.021268 | 0.023325 |
| EMPIRICAL_POSTERIOR_B | AMD | 16 | 16 | 0 | 16 | 16 | 16 | 1/1 | 0.009699 | 0.013236 |
| EMPIRICAL_POSTERIOR_B | FREE | 16 | 16 | 0 | 16 | 16 | 16 | 0/0 | 0.010719 | 0.017889 |
| REFERENCE_G_PRIOR_D | AMD | 16 | 16 | 0 | 16 | 16 | 16 | 0/0 | 0.009689 | 0.012926 |
| REFERENCE_G_PRIOR_D | FREE | 16 | 16 | 0 | 16 | 16 | 16 | 0/0 | 0.010654 | 0.017108 |

全部固定槽位保留；完成数与合法数不同，空串也是完成但非法1–48。分布分母只取冻结合法1–48集合，unique绝对数量不替代16请求分母。固定421特征MMD²为biased linear估计，JS为bit，KS无p值；未按结果择指标。

| 来源 | 端基 | 参照 | AA JS | AA L1 | 二肽JS | 二肽L1 | 长度KS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| CE_DDPM_REUSED | AMD | train | 0.094472 | 0.489628 | 0.293112 | 0.935708 | 0.255310 |
| CE_DDPM_REUSED | AMD | dev | 0.088451 | 0.425182 | 0.365594 | 1.069336 | 0.301630 |
| CE_DDPM_REUSED | FREE | train | 0.101694 | 0.517568 | 0.339680 | 1.019105 | 0.279762 |
| CE_DDPM_REUSED | FREE | dev | 0.099317 | 0.535324 | 0.417891 | 1.143709 | 0.263889 |
| COORDINATE_PERMUTATION_C | AMD | train | 0.122045 | 0.676803 | 0.350194 | 1.055717 | 0.280229 |
| COORDINATE_PERMUTATION_C | AMD | dev | 0.098521 | 0.519462 | 0.361881 | 1.059682 | 0.369565 |
| COORDINATE_PERMUTATION_C | FREE | train | 0.137676 | 0.606502 | 0.354979 | 1.082787 | 0.398810 |
| COORDINATE_PERMUTATION_C | FREE | dev | 0.129432 | 0.648485 | 0.438122 | 1.213784 | 0.548611 |
| EMPIRICAL_POSTERIOR_B | AMD | train | 0.078213 | 0.420248 | 0.268884 | 0.881362 | 0.237745 |
| EMPIRICAL_POSTERIOR_B | AMD | dev | 0.094233 | 0.455526 | 0.361157 | 0.997068 | 0.432065 |
| EMPIRICAL_POSTERIOR_B | FREE | train | 0.116117 | 0.474348 | 0.307245 | 0.958281 | 0.199405 |
| EMPIRICAL_POSTERIOR_B | FREE | dev | 0.120101 | 0.532583 | 0.383615 | 1.077603 | 0.256944 |
| REFERENCE_G_PRIOR_D | AMD | train | 0.079074 | 0.424671 | 0.272814 | 0.895028 | 0.265523 |
| REFERENCE_G_PRIOR_D | AMD | dev | 0.094464 | 0.446330 | 0.369429 | 1.025758 | 0.494565 |
| REFERENCE_G_PRIOR_D | FREE | train | 0.131207 | 0.464999 | 0.327061 | 1.006531 | 0.199405 |
| REFERENCE_G_PRIOR_D | FREE | dev | 0.128821 | 0.502580 | 0.377115 | 1.056044 | 0.201389 |

| 来源 | 端基 | AA熵 | 二肽熵 | AA熵距train | 二肽熵距train | 长度均值[min,max] |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CE_DDPM_REUSED | AMD | 3.141149 | 5.940560 | 0.562840 | 1.194429 | 16.5000 [4,30] |
| CE_DDPM_REUSED | FREE | 3.267681 | 6.013477 | 0.666742 | 1.651367 | 17.1250 [2,48] |
| COORDINATE_PERMUTATION_C | AMD | 2.912647 | 5.418666 | 0.791343 | 1.716323 | 13.1875 [10,18] |
| COORDINATE_PERMUTATION_C | FREE | 3.015632 | 5.663339 | 0.918791 | 2.001505 | 20.6250 [17,23] |
| EMPIRICAL_POSTERIOR_B | AMD | 2.996869 | 5.550222 | 0.707121 | 1.584767 | 13.2500 [8,26] |
| EMPIRICAL_POSTERIOR_B | FREE | 3.121767 | 5.816873 | 0.812656 | 1.847971 | 18.6875 [9,35] |
| REFERENCE_G_PRIOR_D | AMD | 2.985869 | 5.491471 | 0.718121 | 1.643518 | 13.0625 [8,26] |
| REFERENCE_G_PRIOR_D | FREE | 3.095339 | 5.756568 | 0.839084 | 1.908276 | 18.5000 [9,35] |

| 来源 | 端基 | 最近邻范围 | identity mean[min,max] | median | ≥70%数量/合法n |
| --- | --- | --- | --- | ---: | --- |
| CE_DDPM_REUSED | AMD | 同端train | 0.443089 [0.307692,0.571429] | 0.439286 | 0/16 |
| CE_DDPM_REUSED | AMD | 任意train | 0.454476 [0.333333,0.571429] | 0.450000 | 0/16 |
| CE_DDPM_REUSED | FREE | 同端train | 0.400203 [0.222222,0.523810] | 0.402834 | 0/16 |
| CE_DDPM_REUSED | FREE | 任意train | 0.423952 [0.222222,0.545455] | 0.444444 | 0/16 |
| COORDINATE_PERMUTATION_C | AMD | 同端train | 0.497060 [0.357143,0.642857] | 0.500000 | 0/16 |
| COORDINATE_PERMUTATION_C | AMD | 任意train | 0.501524 [0.357143,0.642857] | 0.500000 | 0/16 |
| COORDINATE_PERMUTATION_C | FREE | 同端train | 0.445413 [0.360000,0.523810] | 0.452273 | 0/16 |
| COORDINATE_PERMUTATION_C | FREE | 任意train | 0.463941 [0.380952,0.550000] | 0.473389 | 0/16 |
| EMPIRICAL_POSTERIOR_B | AMD | 同端train | 0.761336 [0.526316,1.000000] | 0.759615 | 14/16 |
| EMPIRICAL_POSTERIOR_B | AMD | 任意train | 0.761336 [0.526316,1.000000] | 0.759615 | 14/16 |
| EMPIRICAL_POSTERIOR_B | FREE | 同端train | 0.661955 [0.461538,0.850000] | 0.689904 | 7/16 |
| EMPIRICAL_POSTERIOR_B | FREE | 任意train | 0.667393 [0.470588,0.850000] | 0.689904 | 7/16 |
| REFERENCE_G_PRIOR_D | AMD | 同端train | 0.740251 [0.473684,0.916667] | 0.750000 | 13/16 |
| REFERENCE_G_PRIOR_D | AMD | 任意train | 0.740251 [0.473684,0.916667] | 0.750000 | 13/16 |
| REFERENCE_G_PRIOR_D | FREE | 同端train | 0.639244 [0.444444,0.850000] | 0.630682 | 6/16 |
| REFERENCE_G_PRIOR_D | FREE | 任意train | 0.646316 [0.470588,0.850000] | 0.630682 | 6/16 |

| 来源 | 端基 | 新/复用 | donor行数/unique | 48触顶 | status计数 |
| --- | --- | --- | --- | ---: | --- |
| CE_DDPM_REUSED | FREE | 0/16 | 0/0 | 1 | {"ok": 15, "no_eos_at_limit": 1} |
| CE_DDPM_REUSED | AMD | 0/16 | 0/0 | 0 | {"ok": 16} |
| EMPIRICAL_POSTERIOR_B | FREE | 16/0 | 16/16 | 0 | {"ok": 16} |
| EMPIRICAL_POSTERIOR_B | AMD | 16/0 | 16/15 | 0 | {"ok": 16} |
| COORDINATE_PERMUTATION_C | FREE | 16/0 | 0/0 | 0 | {"ok": 16} |
| COORDINATE_PERMUTATION_C | AMD | 16/0 | 0/0 | 0 | {"ok": 16} |
| REFERENCE_G_PRIOR_D | FREE | 16/0 | 16/16 | 0 | {"ok": 16} |
| REFERENCE_G_PRIOR_D | AMD | 16/0 | 16/15 | 0 | {"ok": 16} |

B/D donor以reference_train_row_id记录并保留重复；C跨坐标来自多个B槽位，无单一对应训练序列或g。donor数量不是独立信息量。所有来源均用同CE J/PAD读出；48触顶不是AR式自然EOS事件。既有CE两组完整指标与原四臂包逐值相同。

旧CLI附带的exploratory_gates.json缺原CE/MSE/AR操作数，其false仅为缺操作数产物；[门与scope语义](GATE_AND_SCOPE_INTERPRETATION.json)明确NOT_APPLICABLE_SOURCE_DIAGNOSTIC，不能写为本诊断失败或新胜负门。无新重建/AR NLL任务，空数组表示不适用。

新诊断进程final I/O前 2.491365417秒，manifest后closing ledger前 2.494435959秒；外层supervisor 2.679257583秒。范围嵌套，不相加；已有DDPM32历史成本及原CE中断下界不会被重置为0。

[完整指标、逐条来源、工程回执和成本](AUTHOR_FACTS.json) · [冻结评价包](../ce_source_replacement_evaluation_v2/MANIFEST.json)

- Posthoc diagnostic designed after known generation/scale results; fixed one checkpoint, one paired epsilon bank and one selected training-row bank.
- B is empirical training posterior replay containing seen sequence/conditioning information; copying or distribution proximity does not establish innovative or independent novel generation.
- C permutes actual sampled B decoder-z coordinates within terminal, preserving finite coordinate marginals; it has no single donor sequence/g target and is not a new candidate model.
- D uses the same B-selected training reference-g including residue/length-template information; it is not terminal-only, not a fair c-only competitor, and not independent new-peptide evidence.
- B versus D also differs in posterior x/h information; differences cannot identify flow alone or a unique DDPM causal bottleneck.
- Shared epsilon/row indices are paired random inputs, not independent training or biological replicates; no significance tests or new winner thresholds.
- Full dev41 sequence metrics do not create missing dev posterior latent evidence.
- Original four-arm results and CE interruption stay unchanged; no activity, safety, wetlab-validation, novelty or submission claim.
