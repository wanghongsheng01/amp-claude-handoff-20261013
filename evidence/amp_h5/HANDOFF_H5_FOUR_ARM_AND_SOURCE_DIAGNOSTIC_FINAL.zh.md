# AMP H5 四臂实测与来源诊断最终交接

本轮预锁的有限实验已完成。**CE21+合法前缀 J 改善了观察到的重建；当前端到端生成整体优势未成立。** 真实可运行的 terminal LSTM 在本轮全部六类生成距离比较中优于 CE，充分优化的强基线仍未建立。后续来源替换没有把生成优势或单一因果瓶颈补出来。保留负结果，PROMOTE=HOLD；不升级为新颖性、抗菌活性、安全性或可投稿结论。

任务仅为抗菌肽 AMP。原仓库为 <https://github.com/3DHumanRehab/AMP>，只读材料根目录 `/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/`；身份、原 HANDOFF、H4/H2 交接及索引已核。H5 新文件全部位于本目录。训练279、开发41，test37未读取；H0–H4及稿件未编辑。7项原交接/索引哈希保全记录见 [保全核验](feasibility/H5_ROOT_PREDECESSOR_AND_FOUR_ARM_PRESERVATION_FINAL_v1.json)，其证明范围是列出的文件及既有保全链，不冒称重新扫描了全部历史文件。

执行使用本机 CPU、单线程、串行模型作业，预算由真实 train-only smoke 吞吐锁定。3个 VAE 各1000步、各自 DDPM1000步；AR1000步。每方法仅1配置、1训练种子、固定最终checkpoint、0次超参选择；VAE/AR固定10监控点，未按dev改终点。VAE/AR种子222、DDPM322、FREE/AMD生成422/423。共同训练上限2940秒、生成上限1800秒不等于相同实际算力或各路线充分调优。主实验生成仅给可获得的末端条件和自身随机数/AR前缀，不输入真实逐残基内容或真实长度。完整合同及实测命令由最终索引绑定。

四臂主读出有128个完成请求（4方法×2端基×16），以及3个VAE共960行 train/dev posterior-mean 重建。下表重建准确率的分母是dev的701个真实残基，长度MAE分母为41条。

| 方法 | dev重建micro | dev AA macro | 长度MAE | 整串exact/41 | FREE/AMD合法5–48 | FREE/AMD dev MMD² |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| CE21+J / DDPM | 497/701 = 70.90% | 42.76% | 0 | 3 | 14/16；15/16 | 0.017328；0.007710 |
| 同容量raw MSE21+J / DDPM | 88/701 = 12.55% | 3.57% | 5.5610 | 0 | 16/16；10/16 | 0.082335；0.084847 |
| scalar原式管线 / DDPM | 111/701 = 15.83% | 8.05% | 1.3415 | 0 | 16/16；16/16 | 0.039882；0.031325 |
| terminal LSTM | 不适用 | 不适用 | 不适用 | 不适用 | 16/16；16/16 | 0.011567；0.007362 |

全部8组合法1–48、unique均为16/16，任意train精确复制均0/16。MMD²使用冻结421维特征和biased linear估计；分布统计使用合法1–48集合，保留1–4短串和48触顶样本。六类距离为MMD²、AA/二肽JS及L1、长度KS；2端基×train/dev参考的24个对应比较中CE均低于MSE/scalar，却均高于AR。反向结果同样保留：scalar双端AA及二肽train熵差均小于CE，其中部分还小于AR；CE合法5–48也并非最好。CE dev还有C/D/E/H/M/N/Q/Y共8类AA零召回，不能将micro提升称为完整氨基酸覆盖。

CE与MSE使用相同1,681,549参数、357参数21类头、J、初始化、批次、RNG与训练日程，仅重建损失形式改变。CE与0.5×raw-onehot平方误差均除B×48，KL density也除B×48、β=1；固定β不意味着相同有效KL权重。scalar的17参数头、整数MSE、KL归一化和round/clamp/firstPAD均不同，是完整旧式管线对照，不能当同容量单组件消融。scalar MSE分子含全部48槽位，只有分母为非PAD数。AR为2层64维LSTM、62,501参数；dev NLL由step200的2.6153变为固定step1000的3.1698，存在过拟合，AR_ADEQUACY=NOT_ESTABLISHED。

原CE训练完成但第一次生成中断，FREE16只有部分链、AMD16无开始证据，原总耗时与终止原因未知。原E/L/G因此永久NOT_EVALUABLE，原目录、失败/partial及状态文件不回填。另行预锁的checkpoint recovery新增训练0、重新运行32条链，既非t200续跑，也非独立重复。恢复后的描述性L=PASS（4/4），G=FAIL（8/18子项失败，不是8条失败序列）。四臂科学终审已完成；见 [独立科学终审](review/literature/FOUR_ARM_FINAL_SCIENTIFIC_REVIEW_v1.zh.md)、[完整四臂交付](feasibility/CE_RECOVERY_FOUR_ARM_DELIVERY_v1.json)。该早期交付中的科学审核PENDING是历史快照，最终索引另绑定已完成审查，不改写它。

保存张量和源码显示DDPM生成原始μ/logσ通道，随后z=μ+εexp(logσ)直接送decoder；没有latent scaler，不应给已有decoder-z额外做flow inverse。原生prior的标准正态位于zp坐标，必须用冻结CE的prior_flow(zp,g,reverse=True)得到z，且g含参考训练残基及长度模板信息，不能由末端标签替代。静态路线未发现通道、logσ/方差、逆变换、EMA或标签错误；保存的6组重参数化z已按dtype/shape/原始字节精确重现。这些核查不能证明全部实现或统计建模无缺陷。

每端16个样本的中心协方差秩最多15，不做满秩W2、任意白化或把坐标当独立重复。按训练NW70%传递分量进行256次整组、同端且同n抽样；CE AMD的μ/logσ尺度偏离在此描述性对照下仍可见，但部分相关性差异的经验区间跨0，不能泛称全部结构异常。抽样参考都曾被网络训练看到，不是独立模型未见数据，也不是显著性检验。**没有保存的dev μ/logσ/z，缺失保持缺失；没有追加encoder补算。** 完整记录见 [同n整组参考读出](feasibility/latent_same_n_group_reference_v1/READOUT.zh.md)。

四臂闭环后，仅执行一次另行预锁的来源替换诊断：32条既有CE DDPM结果A复用，B/C/D各新增32条，共96条新请求、128条来源观察。B按同端训练后验有放回抽行并采样实际decoder-z；C在每个坐标跨16个B槽位独立置换，精确保留该坐标有限样本边缘值；D使用与B同样的训练行reference-g，标准正态经冻结CE原生prior inverse。各臂共用同CE decoder及J；B/C/A不额外inverse。B是训练经验重放，D有额外残基/长度模板条件，C无单一donor，不作为新模型排名或独立生物样本。

| 来源 | FREE dev MMD² | AMD dev MMD² | FREE/AMD合法5–48 | 任意train精确复制FREE/AMD | 任意train最近邻≥70%计数/16：FREE/AMD |
| --- | ---: | ---: | --- | --- | --- |
| A 既有CE DDPM复用 | 0.017328 | 0.007710 | 14/16；15/16 | 0；0 | 0/16；0/16 |
| B 经验后验 | 0.017889 | 0.013236 | 16/16；16/16 | 0；1 | 7/16；14/16 |
| C 置换B实际采样z | 0.023325 | 0.016635 | 16/16；16/16 | 0；0 | 0/16；0/16 |
| D reference-g prior | 0.017108 | 0.012926 | 16/16；16/16 | 0；0 | 6/16；13/16 |

新96条全部完成、各组unique16、无空串或48触顶；AMD有放回抽行的1次重复保留。B双端dev MMD均未改善，C双端MMD较B更高，D相对A结果混合且条件信息不公平。其他距离、熵、长度、复制和最近邻全部保留在 [来源诊断作者读出](feasibility/ce_source_replacement_author_readout_v2/AUTHOR_READOUT.zh.md)。B相对A的部分二肽距离等指标改善不抹去MMD、熵与复制结果；C相对B的AMD dev长度KS仍有改善（0.369565对0.432065），不能泛称全部指标变差。B/C/D双端AA及二肽train熵差均大于A。C的变化仅与联合坐标组合敏感性相容，不能唯一归因为相关结构或DDPM；B/D还存在posterior x/h信息差异。D双端roundtrip最大绝对误差约7.15e-7/6.56e-7，小于冻结容差，仅为数值路线核验，不是prior质量证据。

本诊断工程独立后审63项通过；实际调用6次decoder batch、2次prior inverse、2次roundtrip forward，0训练/encoder/DDPM。外层2.679257583秒、只读冻结评价5.654053791秒，仅执行一次。旧CLI在缺少主实验操作数时导出的门false有明确NOT_APPLICABLE_SOURCE_DIAGNOSTIC旁证，不能叫诊断新FAIL，也不能补主G。source_v1曾因异常分类问题在运行前被独立审查阻断，v2修订后才获准；v1没有自然模型尝试。首次后审脚本终端标签误写随后修正，两次审阅成本保留，不虚算为模型失败。完整包见 [来源诊断交付](feasibility/CE_SOURCE_REPLACEMENT_EVALUATION_DELIVERY_v2.json)；[独立科学终审](review/literature/SOURCE_REPLACEMENT_FINAL_SCIENTIFIC_REVIEW_v2.zh.md)23项报告核查通过，已完成并保留全部推断限制，无未解决的科学报告不一致。

累计模型作业wall**下界9999.154718747828206秒（约166.65分钟）**，不包括未记录的原CE完整耗时；不是精确总成本。已记录非模型分析53.70622158146245758秒独立列账，其他审阅/写作耗时未知不填0。跨scope执行账为296分配槽位、至少248启动、226完成、6数值失败、原CE16partial、32未执行、16无开始证据；原latent-only4另列。该跨scope账不能成为质量分母；四臂仍128，诊断新96，复用32不重复收费。训练更新累计AR1100/VAE3060/DDPM4060含smoke及成熟度诊断。见 [成本追加](feasibility/cost_addendum_source_replacement_v2_v1/READOUT.zh.md)。其中C来源在结构化字段、中文报告及两个脚本模板共4处误写为DDPM来源，由 [来源勘误附录](feasibility/cost_addendum_source_c_erratum_v1/ERRATUM_SOURCE_C_v1.zh.md)明确修正为“B实际采样的decoder-z”；数值及旧封存文件不改，旧22项成本核验未专项检查此来源语义。

VampPrior、Two-stage VAE和LSGM已为聚合后验/两阶段潜变量/潜空间score建模提供成熟先例，直接建模z本身不能作为新颖性。文献定位及边界见 [原始文献对照](review/literature/AGGREGATE_POSTERIOR_PRECEDENTS_ADDENDUM_v1.zh.md)。当前证据不足以授权把已知路线包装成新贡献；若之后研究direct-z，只能作为另行冻结的问题和实验，不自动继续开跑。本轮未多改KL/结构/数据来归因单一组件。

结束时已完成所有授权模型任务，已记录的source supervisor/model/caffeinate进程均退出，见 [最终进程快照](feasibility/H5_FINAL_PROCESS_SNAPSHOT_v1.json)。短暂工具连接故障已恢复，不是当前阻塞；未用A10未认证会话，未读/传聊天密码或新建凭据。无外部上传、push或正式投稿；本地备份边界仍为BACKUP_BLOCKED。既有自动化接口在委派环境不可用，未创建替代定时任务，不能声称会在本轮结束后继续后台运行。

交接后的证据缺口是单训练种子、每端16样本、反复使用dev41、AR充分性未证、实际算力不等、原CE中断及额外恢复机会、缺dev latent、湿实验N及材料冲突未解决。不能把旧湿实验冒用为新H5前瞻验证。当前结论是保留端到端重建目标修复的局部证据及生成否定结果；本轮已完成，不以继续probe代替方法证据。早晨与用户共同判断下一步之前不正式投稿。
