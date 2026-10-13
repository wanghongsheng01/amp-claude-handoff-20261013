# AMP H5 B/C/D v2：精确合同科学解释预审

**结论：PASS_STATIC_SCIENTIFIC_BOUNDARIES。** 已独立读取并核对 v2 源码、合同、候选清单和评价准备；在本次静态科学解释范围内，没有阻断性问题。这是精确绑定的科学边界预审，不是工程执行许可、自然结果后审或质量通过结论。正式运行仍需根线程的新计划及独立工程准入回执；本审查没有启动它们。

绑定文件为 `ce_source_replacement_v2.py`（SHA256 `e915f23cbf6157b20bd1d9a94fe831d326b0ab6a20b4013824224921a2bbf444`）、`CE_SOURCE_REPLACEMENT_CONTRACT_v2.json`（`7fcc1352f1902ea2ae2d72d8803f9a94defee97452b0a40a969e34c081e22ec8`）、候选 manifest（`6b11d9311314bbd70c1b4bca12c4964ecfbc220be4f5a03687dbef473e2ce37b`）及 v2 评价准备 manifest（`cae66f53b6caf54194f1269f28f6bced3bb30f2a5242d408d813b6490fa0c55f`）。完整成员核验与来源位置见同名 JSON。

| 静态核查 | 已冻结含义与源码位置 | 解释边界 |
| --- | --- | --- |
| 96新请求与32复用 | `allocated_rows`/`reused_rows`：B/C/D每端各16；32个既有DDPM记录仅改来源标识并保留原ID/hash，零新解码 | 128条将来观测不等于128次新生成；所有行均无排名资格 |
| B | `prepare_inputs:154–213`：同端train行池、带放回PCG64固定索引、整对μ/logσ、同一预锁epsilon，`z_B=μ+exp(logσ)ε` | 经验训练后验回放，含已见序列/条件信息；不是独立泛化或新模型胜利。重复donor保持，effective unique不能替换请求分母 |
| C | `permute_columns:134–143`、`verify_permutation:145–152`：对已采样B-z每个channel/position独立置换同端16行；检查双射、精确映射及每坐标float32位模式多重集 | 保留各坐标有限经验边缘并扰动联合组合；没有单一donor真值或单一g。C下降只支持该固定控制下的联合组合敏感性，不能指认A/DDPM的具体失配机制 |
| D | `execute:277–303`：同B索引的train reference-g，恢复原train scaler、带符号z-score、真实长度后精确零padding；`z_p=epsilon`经原flow逆变换到decoder-z，再正向roundtrip | 明示额外逐残基/长度条件，退出terminal-only排名。roundtrip只核数值可逆性；错误g也可能往返，因此不能证明条件正确或先验质量。直接decoder标准正态不等于D |
| 公共末端 | `execute:304–325`：两层输入seal后，六批次均同一冻结CE checkpoint、decoder与J；A/B/C不额外inverse flow | 输入来源干预可检验固定decoder下的来源差异；不能据此孤立归因某个尺度、标准化、DDPM目标或KL组件 |
| 失败与成本 | 固定120/180秒预算、D不可事后省略、全部96行预分配；无重试、重采、裁剪、白化或新训练 | 非有限值/empty/cap/失败/未执行必须保留，失败源不得当作完整质量比较；未知实际调用数不得归零 |

epsilon通过重建原CPU RNG轨迹并要求保存z按原始字节精确复现后才可进入模型阶段，不用 `(z−μ)/σ` 反推。A、B、D共用epsilon，C置换B已抽z；这是配对随机数，不能增加独立实验次数。B donor选择种子523/524、C置换种子625/626与epsilon种子422/423的职责分开。所有自然数值检查仍未执行；此处确认的是源码中存在且冻结的规则。

**v2保持科学操作，修复v1异常归因。** 我独立作AST文本比较：除`execute`和`main`外的16个函数与v1完全相同；`execute`越过准入包装后的整个body相同。差异仅版本路径/scope及异常处理。v2只把`validate_request`抛出的错误包装为`AdmissionFailure`，该分支才报告零模型调用；准入后的未捕获错误进入`execution_or_finalization_failure_phase_unknown`，保留96请求，将未知实际计数标null，尽力保留已持久化status及hash并明确可能陈旧、不得视为成功。v1被阻断、未执行，其候选件保持不变。这里没有替代独立工程审查的故障注入测试。

**未来评价的边界已对齐。** v2准备绑定真实seed/slots哈希，实际字段`reference_train_row_id`被保留；旧`selected_train_row_id`仅作为历史提议别名说明。指标维持全部冻结生成字段，包括双端train/dev的MMD、JS/L1、entropy差、长度KS、合法率、unique、train-copy和最近邻；不新增阈值或胜负门。请求分母每组16，分布资格仍按合法1–48并同时报告有效n和失败数。C不生成伪“重建真值”；无新reconstruction/encoder/NLL。旧CLI产生的缺操作数门值必须单列 `NOT_APPLICABLE_SOURCE_DIAGNOSTIC`，不能解释为本诊断FAIL或PASS。

只有实际终态、独立工程后审及status/closing ledger/supervisor exit的共同证据成立后，才可按冻结准备开展自然评价；中途写出的completed状态不够。随后任何A/B/C/D差异都仍是事后探索性、单模型/有限样本/训练参考条件下的诊断，不能改写已完成四臂的原E/L/G=NE、恢复L=PASS/G=FAIL、AR_ADEQUACY未建立及PROMOTE=HOLD。更不能宣称新方法、抗菌活性、安全性或投稿证据。

本次只读源码/JSON/清单并作哈希、槽位和AST核查，没有import候选模块，没有模型构造、forward、自然RNG、训练、生成、自然指标计算、cohort内容/权重/后验数组或test37读取。在静态核验时，v2 run目录、root plan与独立preflight路径均不存在；这一时点事实不是对之后执行状态的判断。

[概念解释边界预审](SOURCE_BCD_INTERPRETATION_BOUNDARIES_PREAUDIT_v1.zh.md)与[四臂科学终审](FOUR_ARM_FINAL_SCIENTIFIC_REVIEW_v1.zh.md)保持冻结。
