# H5：PepVAE、离散序列 VAE 与小型 peptide AR 的一手依据

状态：**静态文献核查完成；下述适配和对照均未执行，等待 root 冻结 H5 协议。** 仅本目录为本次写入范围；未训练、读 test、执行外部代码或联系作者。来源账本见 `SOURCE_LEDGER.json`。使用 claim-evidence 技能区分 VERIFIED / PARTIAL / PLANNED。本轮限三项先例，不改变 H0–H4 或既有 baseline 结论。BACKUP_BLOCKED：没有上传或云备份。

## 1. PepVAE 的确切身份与可借鉴边界

**原始来源 / 原问题（VERIFIED）。** Dean、Alvarez、Zabetakis、Walper、Malanoski，*PepVAE: Variational Autoencoder Framework for Antimicrobial Peptide Generation and Activity Prediction*，Frontiers in Microbiology **12:725727，2021-09-30**，[正式全文](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2021.725727/full)，DOI `10.3389/fmicb.2021.725727`。它将肽生成与 MIC 预测组合；Methods 的 Dataset / Variational Autoencoder / New Sequence Generation 及 Discussion 是本次定位。文中使用 vanilla LSTM VAE；排除 C 和 ≥40 AA，词表为 **19 AA + EOS + PAD**，不是当前 20 AA + PAD。其参考肽邻域采样也不是 terminal-only 条件生成。它区别于 Dean–Walper 2020 ACS Omega 前作（文末参考文献，DOI `10.1021/acsomega.0c00442`），不能混报版本。

**代码状态（PARTIAL）。** 已读正文、Data availability 和 Supplementary 入口，但尚未核得 PepVAE 生成器的公开官方代码 / commit；正文说明 binary models 可向作者索取。其 GitHub 链接是 GRAMPA 数据来源，不能据此宣称拿到了 PepVAE 实现。已阅材料未给出足以复现的 CE reduction、EOS/PAD mask 和 KL 归一化细节；本次不猜测，也不联系作者。

**AMP 迁移理由（本项目推论，PLANNED）。** 可借鉴“离散残基 + 显式终止”的对象表示及 VAE 生成框架；不能把排 C、短肽筛选、MIC 区域选择搬到当前队列。当前只有 FREE / AMD 作为允许的生成条件；没有可信的 MIC / 活性标签支持该论文的完整活性设计任务。H5 应称“受 PepVAE 等启发的 terminal-conditioned VAE”，不能称完整 PepVAE 重现，也不能把已有 categorical VAE 写成新颖性。

**可证伪控制（PLANNED）。** 使用相同冻结对象、条件、采样尝试数比较完整 H5 VAE 与第 3 项完整小 AR；分别报告 teacher-forced 拟合与真实先验生成。若前者重构改善但后者终止 / 长度 / 多样性没有改善，应否定“重构改善已迁移到生成”。没有本队列独立活性证据时，任何模型都不能获得“活性更好”的结论。精确 PepVAE reproduction 仍缺作者生成代码或完整实现细节；来源启发的独立实现不依赖将该缺口伪装为已补齐。

## 2. Bowman：离散序列 VAE、EOS 和潜变量失用

**原始来源 / 原问题（VERIFIED）。** Bowman 等，*Generating Sentences from a Continuous Space*，CoNLL **2016，pp.10–21**，[正式 PDF](https://aclanthology.org/K16-1002.pdf)，DOI `10.18653/v1/K16-1002`。PepVAE 引用的是其 2015 预印本；这里核查正式版本。印刷 p.11 Eq.(1) 给出序列 ELBO，p.12 Fig.1 标明 EOS，§3 为 LSTM 序列 VAE，§3.1 将似然项解释为交叉熵并讨论 decoder 忽略潜变量；p.13 说明训练时使用真实前一 token。该文解决自然语言全句连续表示与生成问题，未提供 AMP 生物学证据。本轮依据论文，不声称核过作者运行代码。

**AMP 迁移理由（本项目推论，PLANNED）。** 把词替换为 AA 后，next-token categorical likelihood 仍有明确概率意义；全局潜变量可尝试解释序列层面的变化。训练时 decoder 使用过去真值与生成时只用自己的历史是不同过程，必须分别检查。均值 posterior 与随机 posterior 的重构差异本身不足以诊断 collapse；需结合 KL、潜变量置换 / 屏蔽和先验生成，而且非零 KL 也不保证有用生物表示。

**可证伪控制（PLANNED）。** 与容量和预算已声明的小 AR 比较，检验 VAE 的潜变量是否贡献可重复的生成改善；可预先固定一次 z 正常 / 打乱或置零诊断，若输出和似然几乎不变，应收缩“使用潜变量”的主张。z 干预改变输入分布，不能单凭一次阴性判为数学意义上的 posterior collapse。论文的 annealing / word dropout 是已有训练手段，不自动成为 H5 设置；不据负结果临时扫 beta、加深或重开搜索。本项是先例和风险依据，非 H5 已有结果。

## 3. Müller：可小规模重实现的完整 AR peptide baseline

**原始来源 / 原问题（VERIFIED / PARTIAL）。** Müller、Hiss、Schneider，*Recurrent Neural Network Model for Constructive Peptide Design*，JCIM **2018，58(2):472–479**，DOI `10.1021/acs.jcim.7b00414`；[作者仓库](https://github.com/alexarnimueller/LSTM_peptides)、[README](https://raw.githubusercontent.com/alexarnimueller/LSTM_peptides/master/README.md)、[实现](https://raw.githubusercontent.com/alexarnimueller/LSTM_peptides/master/LSTM_peptides.py)。论文书目信息与摘要、官方代码已核；本轮未取得出版社全文，不声称全文复现。代码路径为 `main → SequenceHandler.one_hot_encode → Model.initialize_model / train / sample`：shifted next-token、LSTM/GRU、categorical CE、逐 token 采样；README 提供 2 层 × 64 units 示例。`sample` 以空格终止并丢弃过短输出，`train` 用内部 validation split；这两处不能直接沿用于本项目的公平评估。上述仅是所读 mutable master 的静态行为，不是运行失败结论。

**AMP 迁移理由（本项目推论，PLANNED）。** 小 AR 从 BOS 起生成整条序列，无需读出旧 FVAE hidden、目标 g/h 或参考肽，因此是完整生成比较对象。无需运行该历史依赖栈，可在当前已获授权环境中重实现其序列机制，并标注 `adapted terminal-conditioned AR`；此举不是“官方代码原样重现”。条件注入、显式 EOS / PAD 区分及训练划分适配是本项目修改，也不构成新架构贡献。

**最小规范与可证伪控制（PLANNED，具体超参由 root 冻结）。** 一套共享小 LSTM 或 GRU，输入为前一 token 和同一个 terminal 条件 c，输出为 20 AA + EOS，使用下节统一目标；从头训练，不加预训练库。64 units 有官方小模型示例依据，但 1 层 / 2 层、embedding、优化器、步数、checkpoint 规则及温度须先锁定；不能说任何选择已在此 CPU 验证可跑。与 VAE 共用 frozen train / val、条件分配、生成种子表、attempt 总数和报告口径；共享种子编号不能保证两个不同模型具有相同随机张量。若 AR 达到或优于 VAE，则目前没有证据支持 VAE 的额外复杂性。若只做过小、未收敛 AR，也不能把其失败包装为强 baseline 胜利。

## 4. H5 可共同采用的概率与公平性契约（本项目推导，不是三篇论文的完整原法）

设 `c ∈ {FREE, AMD}`，真序列 `a₁…aL`；输出词表固定为 20 AA + EOS。BOS 仅作输入；PAD 仅供 batch 填充，不与 EOS 共用 ID。训练输入为 `[BOS,a₁,…,aL]`，target 为 `[a₁,…,aL,EOS]`。EOS 参与损失，EOS 后 PAD 不参与；保留当前合法的 C 与完整长度范围。**若最大长度 48 AA，仍须为第 49 个 target 的 EOS 保留位置。** 词表数量相同不表示 token 语义相同，旧 21-head 权重不可按编号直接视为新 EOS head。

AR 定义 `p(a,EOS|c)=∏(t=1…L+1) p(x_t|x_<t,c)`。AR-VAE 可定义 `q(z|a,c)`、`p(z)` 与 `p(x_t|x_<t,z,c)`。每对象重构项为 `R_i=−Σ(t=1…L_i+1)log p(x_it|history,z,c_i)`，负 ELBO 为 batch 平均 `R_i + KL_i`（KL 对 latent 维求和）。每 token NLL 可另外报告；仅把 R 除以长度而保持 KL 不变会改变两项相对权重，不可再称相同目标。任何 beta / warmup 必须在新协议明写，不从来源补造默认值。

若 root 保留并行固定槽 VAE，上述 AR 分解不能冒充其原理依据。另一致性要求是显式定义 `p(L|z,c)` 与给定 L 的残基概率，并在训练和生成使用同一 joint 模型；标签缺失不能当 PAD / 阴性。此处只列必须澄清的数学契约，不另提出一轮结构搜索，也不修改 root 协议。

生成只能取得 c、随机数及已训练参数。原 FVAE 的逐残基 g、全序列派生 h、目标长度 / 序列、posterior 或参考肽不能作为 terminal-only 生成输入；encoder 为近似后验读入训练序列本身不等于泄漏，但不可把该路径称为先验生成。c 仅是终端化学标签，给出 AMD 不等于证明生成肽已合成或具备活性。

沿用上游冻结元数据的 **train 279 / val 41**（FREE 126/18、AMD 153/23）；本轮未重新打开序列或 test。val 已用于开发，不能改称未见外部检验。记录全部尝试、重复、过短 / 空串、EOS 缺失、达到长度上限等事件，不按成功数补采样。生成分布与 val 的比较没有逐对象配对目标，不能凭采样编号计算 target length MAE；重构任务才有对应真长度。

条件、数据和训练预算是不同公平维度：相同步数不保证相同 CPU 时间，参数数目也不是全部容量。root 应统一硬件 / 线程、数据曝光次数上限、调参次数、wall-time 上限和 checkpoint 规则，同时逐模型报告实际步数、时间、参数量、停止原因。当前没有小 AR 或新 VAE 的本机耗时 / 收敛证据，不能声称已实现同预算或已经具备竞争力。

**新旧生成模块边界。** 若 H5 更新 encoder / flow，旧 DDPM 拟合的 posterior-parameter 分布随之失配；不能直接接旧 DDPM 后宣称 H5 真生成。最小终端 cVAE 若使用标准先验，可不经旧 DDPM；若保留 DDPM 路线，需用新 encoder 对 frozen train 重编码，并重新拟合相应 posterior-parameter DDPM 及 EMA，再冻结新的采样链。此为分布对齐要求，非已完成工作。

**下一步最小前置。** root 锁定一种 VAE 输出 / 长度契约与一套小 AR、相同生成输入和资源上限；实现者检查 token round-trip、EOS mask、48 AA 边界、split 身份与条件可得性，独立审阅后再运行授权训练。这里的“可实现”是依赖和算法范围判断，运行可行性及科学增益仍为 PLANNED。未证实的 PepVAE 官方生成源码、准确 normalization 和原配置仍为 MISSING；不以猜测填补，也不把缺源码作为本地 AR 机制实现的虚构故障。
