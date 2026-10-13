# CE 来源替换 v2：作者只读工程终态

本次新诊断已结束：监督器 exit0，96/96 新输入完成，失败/未执行均0；调用恰为6次decoder、2次inverse-flow和2次roundtrip forward，0训练、0encoder、0DDPM。另32条DDPM记录只引用既有CE recovery，不是新采样。模型与输入哈希保持。作者仅转录并核对JSON/CSV及字节哈希，不替代独立工程后审。

时间分别为：最终IO前2.491365秒、manifest后closing ledger前2.494436秒、监督器完整wall 2.679258秒。均在120/180秒预算内；这些是新诊断成本，旧DDPM生成成本保留在原尝试中。

| 来源 | 端基 | 请求/完成 | 状态 | 空串 | 长度48 | 长度范围 |
|---|---|---:|---|---:|---:|---|
| COORDINATE_PERMUTATION_C | FREE | 16/16 | ok:16 | 0 | 0 | 17–23 |
| COORDINATE_PERMUTATION_C | AMD | 16/16 | ok:16 | 0 | 0 | 10–18 |
| EMPIRICAL_POSTERIOR_B | FREE | 16/16 | ok:16 | 0 | 0 | 9–35 |
| EMPIRICAL_POSTERIOR_B | AMD | 16/16 | ok:16 | 0 | 0 | 8–26 |
| REFERENCE_G_PRIOR_D | FREE | 16/16 | ok:16 | 0 | 0 | 9–35 |
| REFERENCE_G_PRIOR_D | AMD | 16/16 | ok:16 | 0 | 0 | 8–26 |
| CE_DDPM_REUSED | FREE | 16/16 | no_eos_at_limit:1, ok:15 | 0 | 1 | 2–48 |
| CE_DDPM_REUSED | AMD | 16/16 | ok:16 | 0 | 0 | 4–30 |

两端原epsilon的saved-z逐字节复现记录均通过，B/C/g/permutation在任何forward前封存，D z在inverse后、decoder前封存。每端C有768个坐标边际检查记录。B/D共用训练索引；FREE16条无重复，AMD16条保留一次重复抽取（row117两次），未补采。D g使用原训练scaler的带符号残基zscore与零padding，保留原逐残基及长度模板信息。

D roundtrip最大绝对误差：FREE 7.153×10⁻⁷、AMD 6.557×10⁻⁷；相对预锁容差的最大归一误差为0.05926、0.04066，超限坐标0。它只说明此数值往返检查通过，不说明prior质量，不能单靠roundtrip验证g来源。

B是训练经验后验参照；C只考察该z组合对逐坐标置换的敏感性；D是reference-g conditional native-prior，额外使用训练残基和长度信息，不能进入terminal-only主排名或被称为独立新肽证据。本报告不计算MMD、train copy、近邻等质量统计，交由冻结evaluator。原CE E/L/G仍不可评估，诊断门NOT_APPLICABLE，PROMOTE HOLD。
