# AMP H5 来源替换 v2 独立科学终审

结论：冻结作者包的描述性报告可以封口；没有证据把经验后验或 reference-g 来源替换写成双端生成修复、独立新方法优势，或唯一 DDPM 因果瓶颈。无未解决的科学报告阻断项。成本包的 C 来源措辞曾错写为 DDPM-z，已另附不可变勘误；原包必须与勘误一起引用。

本审只读已保存 JSON/Markdown/日志，不读取 train/dev/test 原始队列、模型权重或 latent 数组，不调用模型，不重跑特征、距离或序列比对。独立核实联合索引的 33 个直接文本绑定、作者全指标与 128 条 audit 精确一致、A 两端除名称外与原四臂完整组逐值一致。工程后审的 63 项 PASS 是保存记录和算术核验，不能当作独立模型重放或质量证明。

128 条来源观察为 A 既有 CE DDPM 32 条复用，加 B/C/D 各 32 条新诊断；每来源每端 16 条，合计 96 次新 decoder 输入。新执行为 6 次 decoder batch、2 次 native prior inverse、2 次 roundtrip；没有新训练、encoder 或 DDPM forward。新 96 条全部完成、合法 5–48、无空串、无 48 触顶。A FREE 保留原 1 条 48 触顶、合法 5–48 为 14/16，AMD 为 15/16；全部 128 条均合法 1–48，每组 unique 为 16/16。触顶是 J/PAD 读出事件，不能混称自回归自然 EOS。

| 来源 | FREE dev MMD² | AMD dev MMD² | FREE/AMD 任意 train 最近邻 ≥70% | FREE/AMD 精确 train 复制 |
| --- | ---: | ---: | --- | --- |
| A | 0.017327803379 | 0.007709821688 | 0/16；0/16 | 0/16；0/16 |
| B | 0.017888733398 | 0.013236364505 | 7/16；14/16 | 0/16；1/16 |
| C | 0.023324997456 | 0.016634657842 | 0/16；0/16 | 0/16；0/16 |
| D | 0.017108436019 | 0.012926119282 | 6/16；13/16 | 0/16；0/16 |

A 为原 DDPM 来源复用。B 为带训练序列及其条件信息的训练经验后验回放；其双端 dev MMD² 都比 A 大，但部分 AA/二肽/长度距离比 A 小，不能概括为所有指标变差。B 的 AMD 精确复制发生在已保留槽位 002；高最近邻相似性和距离改善不能等同于新颖生成。所有 exact-copy 与最近邻分母均保留完整每组 16 条，同端与任意 train 两种范围独立报告。

C 在每个末端内，对 B **实际采样的 decoder-z** 的每个坐标跨 16 槽位置换，保持有限样本逐坐标边际，扰动联合组合；它不置换 DDPM-z，也不具有单一 donor/g/真值。C 比 B 双端 train/dev MMD² 都大；六类距离×两端×两参照中仍有一个反例：AMD dev 长度 KS 为 0.3695652173913043，低于 B 的 0.4320652173913043。C 的 AMD dev 二肽 JS/L1 也略低于 A。因此可说固定样本对坐标组合敏感，不能说所有指标全面崩溃，更不能证明某一特定相关结构或 DDPM 错配就是唯一原因。

D 是与 B 相同训练行的 reference-g 条件 native prior：标准正态先在 flow 的 zp 坐标，再 inverse(g) 到 decoder-z。g 含逐残基与长度模板信息，D 不是只知末端的公平候选，也不等同于直接给 decoder 标准正态。D 相对 A 的 dev MMD 为 FREE 略低、AMD 较高；相对 B 的 AA/二肽/长度指标仍混合。B 与 D 同 indices/epsilon 仅控制配对随机输入，后验还利用 x/h 信息，因此两者差异不能单独归于 flow。roundtrip 通过只支持数值往返一致，不能证明 prior 与目标分布匹配。

负面和混合结果必须一同保留：B/C/D 双端 AA、二肽熵距 train 均大于 A；C 的输出长度变窄为 FREE [17,23]、AMD [10,18]；B/D 的高 train 最近邻比例并未被独特序列数掩盖。B/D 的 donor 为 FREE 16 个不同训练行、AMD 15 个不同训练行（同一行重复一次），有放回重复保留。逐样本共同随机数、donor、两种末端都不是独立训练或生物实验重复。所有精确标量、两参照六类距离、合法 1–48/5–48 多样性与最近邻统计见本审 JSON；全频率向量保留在冻结原评价。

本诊断是看到既有生成/尺度结果后的固定 checkpoint、固定 epsilon 和固定训练 donor bank 的探索。MMD² 是冻结 421 维特征上的 biased linear 估计，不是生物活性；最近邻低或零精确复制也不能证明生物新颖性。dev41 的序列评价不能补造 dev posterior latent 对照，单组 n=16 不支持显著性或总体结论。既有文献中的 aggregate-posterior/prior mismatch 路线不能替本次实验确定原因。

原 CE 正式 E/L/G 因中断永久 NE；原四臂恢复描述性 L=PASS 4/4、G=FAIL 8/18 保持原义。本来源评价中旧 CLI 因缺原 CE/MSE/AR 操作数输出的 false 明确为 NOT_APPLICABLE_SOURCE_DIAGNOSTIC；不是新失败/胜负门。B/C/D 不参与 c-only 排名，也不回填原门或重建/AR NLL。原 CE/MSE 固定 β 的 loss-form 对照、scalar whole-pipeline 边界、AR1000 dev 过拟合、AR 充分性未立及不等计算成本均不因本诊断改变。PROMOTE=HOLD；无新颖性、活性、安全、湿实验前瞻或投稿就绪结论，wetlab N 未知不猜。

新诊断 outer 为 2.679257582873106 秒；2.4913654173724353 秒 process-before-final-I/O 与 2.4944359590299428 秒 closing-before-ledger 为嵌套口径，不相加。历史训练与 DDPM 的已耗成本保留；累计模型 wall 仍只知下界 9999.154718747828206 秒，原 CE 完整中断时间未知。评价、读出及审计耗时另列，当前科学复核没有完整计时，不补 0。

本轮足以否定“替换成训练经验后验即可同时消除双端差距”的简单解读；尚不能排除 latent 来源是多个因素之一，也不能锁定新的修复方案。可据此停止本轮追加探索并保留失败/混合证据；本审不授权训练、模型调用、新 seed 或新 scope。任何后续因果假设需另行明确，当前报告不借未来可能结果改写结论。

来源：[冻结交付索引](../../feasibility/CE_SOURCE_REPLACEMENT_EVALUATION_DELIVERY_v2.json)、[作者全指标](../../feasibility/ce_source_replacement_author_readout_v2/AUTHOR_FACTS.json)、[工程后审](../independent/CE_SOURCE_REPLACEMENT_POSTFLIGHT_v2.json)、[成本 C 来源勘误](../../feasibility/cost_addendum_source_c_erratum_v1/ERRATUM_SOURCE_C_v1.zh.md)。
