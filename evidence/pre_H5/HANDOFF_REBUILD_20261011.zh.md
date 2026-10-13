# AMP 隔离重建：实际结果与科学边界

本交接只对应新的本地重建。来源纠正后的固定CPU 1000/1000训练与完整999日程采样已经完成，96条神经输出加96条统计控制全部保留。冻结论文 `candidate_v6r3`、原 `amp_commsbio_20261011/HANDOFF.zh.md`、历史数据和失败目录保持原样。当前不支持抗菌活性、方法优势、收敛或投稿准备完成的结论。

文中独审均指不同代理的代码、数据和证据复核，不是新增的真人语义金标或湿实验测定。

## 最先请求的三个回执

1. **最初 CPU 200/200 及完整999日程已实际完成。** VAE200更新224.7458秒；DDPM200更新约85.16秒。999噪声日程对应998次逆更新，2条DDPM采样48.4463秒。`runner/pilot_002/` 在训练后因JSON序列化失败，仍保留FAILED；`runner/pilot_002_posttraining_001/` 仅使用同批hash绑定新权重完成验证/采样，没有重训。首轮三次失败分别保留。六条神经输出全部长48、低复杂度，旧val非PAD准确率0/163。[首轮交接](SMOKE_HANDOFF.zh.md)；[六方法表](evaluation/pilot_002_combined_summary.csv)。
2. **DRAMP完整快照已下载和解析。** 抗菌表28,702记录、General表12,784记录；固定条件阶梯6157→5971→4626→4616→3540→3403→2160→592→560。其MIC-only注释层517还不是化学合格集；图和来源交叉核验后202记录、201序列、202化学对象。[逐步条件、排除及源hash](expanded_data_audit/HANDOFF.zh.md)。115条来自另一个DBAASP `qualification_v3`，不是DRAMP下载不足。
3. **A10做过既有通道检查，尚未验证硬件。** 平台资源GET返回200，但仅有192.168.2.30的DGX缓存；在有界普通源代码及资源检查中未找到A10已授权通道。14.103.24.42:22能连接并返回OpenSSH公开banner，向该端点发送的应用数据为0字节；没有SSH认证、密码读取或`nvidia-smi`。先前errno1为本地sandbox socket限制，不是服务器鉴权拒绝。[实际记录](A10_PUBLIC_SURFACE_002.json)。可选GPU迁移仍需用户自己完成首次安全登录或提供已授权代理；当前Mac工作不等待此项。

## 已完成的数据修复

DBAASP固定512卡片与24个搜索页共536个响应全部保全。原“所有provider必须精确图一致”规则仅留下115条且完全没有R/H，因此保留为HOLD，未执行长训练。全局互变异构枚举曾触及骨架及立体相关区域，v1保留BLOCKED。

新规则先预锁，再以完整原子双射验证限定的Arg胍基与His咪唑局部H/键表示关系。全体非局部键、边界、元素、电荷、同位素、自由基、立体中心均须保留；来源标签仅是观测声明。不同审阅者以NetworkX和邻居排列立体检查复核984个provider图、482个实际映射，9,512次立体中心检查通过。它认证受限的结构表示关系，不认证溶液中的优势互变/质子化态或实际测定样品。

最终357对象（FREE159、AMD198），含R218、H69；两类仅指来源对象的末端化学状态，不是菌种、活性强弱或真人标注。固定同源分量与已知暴露规则分为279train、41val、37sealed test。独立重算150,975对global NW比对，654边、388组件一致；跨分区同序列、≥70%同源边及已知暴露泄漏均为0。零泄漏结论只覆盖锁定的相似度定义和195个已知标记，不涵盖未知暴露或所有生物依赖。18个已暴露近邻对象被强制分配到train。公开源assay含后来划入test的记录已用于预锁资格审计；没有用模型test输出/评分调参。

**这些不是357个阳性金标。** 1,623条合格MIC-testing记录中1,185个数值、65个区间、373条右删失；48对象只有右删失。没有同一序列的FREE/AMD配对，不能据类别差异推断端基因果效应。描述符使用明确选定的参考结构；LabuteASA是新的明示约定，不冒充找回的历史ASA。

[数据交接](expanded_cohort/HANDOFF_EXPANDED_COHORT.zh.md) · [实际冻结入口](expanded_cohort/release_v2/runner_train_val/COHORT_FREEZE.json) · [独立结论](expanded_cohort/tautomer_equivalence_v2/independent_review/FULL_REVIEW.json)。

## 原模型与短诊断

原词表只有20AA与PAD20，没有独立EOS/BOS；decoder输出48个连续标量，经clamp/round后在首个20截断。因此旧`no_eos_at_limit`实为48位无PAD，可表示允许的48长度，不应单独判执行失败。较短输入的尾部PAD预测失败仍是真实长度重建问题。原目标是所有48位编号平方误差/非PAD数，加采样flow KL；不是分类交叉熵。KL只除latent16，batch大小改变相对权重。[PAD说明](diagnostics_generation_20261011/PAD_EOS_ADDENDUM.zh.md)。

相同初始化、相同步噪声、8fit+8已暴露开发holdout，四臂各160步，总42.53秒。下表为终点`decoder(mu)`，不是新队列的独立验证：

| 臂 | fit原MSE | holdout原MSE | fit非PAD准确率 | fit PAD召回 |
|---|---:|---:|---:|---:|
| A 原warmup、KL1、采样 | 481.235 | 486.323 | 2.94% | 0 |
| B 固定lr.0005、KL1、采样 | 81.868 | 91.885 | 4.71% | 0 |
| C 固定lr、KL0、采样 | 27.453 | 39.273 | 14.71% | 0 |
| D 固定lr、KL0、mu | 30.842 | 42.846 | 17.65% | 0 |

四臂全部未过预锁的95%非PAD准确率、95%PAD召回、长度MAE≤1联立门槛。A→B支持短预算预热不足影响学习。B趋近共同编号均值、跨对象信息使用弱，但零latent明显改变结果，不能断言标准正态后验完全坍塌。C利用更多对象信息却牺牲先验匹配；D的采样MSE534.814及方差均值62,526，说明较好的确定性重建不是VAE生成成功。全部640更新、32个原始评估、负结果及一次无训练的schema准备失败保留。

独审已重算raw重建/PAD/后验方差等；原sampled flow KL未保存中间密度，独审限日志有限性与loss恒等式，不声称完全独立复算。[实际结果](diagnostics_generation_20261011/RESULT.zh.md) · [独审](diagnostics_generation_20261011/independent_review/REVIEW.zh.md)。

[四臂学习和长度图](diagnostic_visuals/01_learning_and_decoding.png) · [均值与随机后验重建图](diagnostic_visuals/02_endpoint_mu_vs_sampled.png)。两图也提供同名PDF；没有平滑或虚构误差条。

## 固定实验身份与实际1000/1000结果

已发现并隔离修复runner学习率索引：旧r1使用f(u−1)，原trainer实际是f(max(0,u−2))，即前两次更新均f(0)。r2仅改这一行，经不同审阅者核对1000步日程。旧结果及原runner不改。

新公开队列试跑使用独立CPU身份，固定1000/1000更新、seed222、sampling master522、999噪声日程和每terminal每方法16请求。原MPS预算与原adapter协议保留；此运行是来源纠正后原目标的有界工程对照，不是收敛声明。CPU协议hash为`0f7a48bec35ab7acbea59544f47592da78193ce05b3380ab7ed57f09753a14d4`，runner r2 hash为`a6dd4e75d890aecb19efa6deeef7fd77228e303d932f8537605d018b3bb293b8`。

| 已完成阶段 | 实际用时 | 直接结果及边界 |
|---|---:|---|
| VAE1000更新，CPU batch32/尾23 | 1184.698131秒 | val41随机后验非PAD正确85/701=12.1255%；原MSE14.608984、非PAD MSE13.291452、采样flow KL/record0.316101 |
| DDPM1000更新，CPU batch8/尾7 | 545.832069秒 | val41、62,976元素、noise seed422，噪声MSE0.02116554；低值不证明可用生成分布 |
| 六个神经method×terminal组，各16请求 | 总1133.745414秒 | FREE DDPM565.609304秒、AMD DDPM567.723155秒；各完成998次逆更新，96请求全部执行、0失败/0未执行 |
| 冻结六方法联合指标 | 10.033150秒 | 96神经+96统计控制=192请求、12组，全数保留；未读取model test结果 |

运行状态为`PIPELINE_COMPLETE`，81文件运行清单及96条原始48位值已冻结。[运行目录](runner/dbaasp_v4_cpu_pilot_001/status.json) · [运行清单](runner/dbaasp_v4_cpu_pilot_001/files_manifest.json) · [联合结果表](evaluation/dbaasp_v4_cpu_001/RESULT.zh.md) · [完整指标CSV](evaluation/dbaasp_v4_cpu_001/summary.csv)。本次主run没有新增执行失败；前述旧失败与负结果不因此删除。源端化学/队列、预处理、预算均有变化，不能将本次结果与旧六条输出当成受控的单因子改善。

96条神经输出均为5–48位，各method×terminal的16条均不同，未精确复制train序列；字母合法性来自clamp/round构造，零精确复制也不能推出有用新颖性。4,608个原始连续输出中40个小于0、1,295个大于20，全部原值保留并按冻结的原解码处理。神经状态86条ok、10条`no_eos_at_limit`，后者全在DDPM（FREE7、AMD3），即达到允许的48位且没有PAD，并非执行失败。DDPM平均长度FREE34.4375、AMD25.6875；flow为25.1875/17.1875，aggregate为21.9375/15.375。全部原始输出保留。

不隐藏分布诊断的负结果：相对各自terminal验证集，DDPM的固定特征线性biased MMD²为FREE0.058085、AMD0.043836；Markov为0.014206/0.018602，shuffle为0.005569/0.009852。它们是单次小样本的描述性距离，不是带不确定性估计的排名、活性或等条件方法优势。统计控制中8条Markov输出短于5位，FREE另3条无结束符；memory32/32精确复制train，全部如实计入。

**阶段2容量诊断已完成并独审。** 仍用旧8fit/8holdout、两个fresh同初始化预锁1600步控制；3200更新耗时203.900195秒，全部finite，没有触发360秒软cap或失败。它是已看过开发结果后的新阶段，不冒充盲检，且不替换原四臂失败。两个臂的前160步日志及4个step160原始数组与原短诊断完全相同。

| 固定1600终点 | fit非PAD准确率 | fit PAD召回 | 长度MAE | 联立门 |
|---|---:|---:|---:|---|
| B 原KL1、采样训练、固定lr | 128/170（75.29%） | 213/214（99.53%） | 0.25 | 未过 |
| D KL0、mu确定性训练、固定lr | 170/170（100%） | 214/214（100%） | 0 | 通过，8/8完整重建 |

D证明在这个确定性控制下原结构可以记住8条及其长度，反驳了“没有独立EOS所以一定不能学长度”的解释。其随机后验重建MSE仍434.465、后验方差均值20,600.73，绝不是VAE生成成功。B在1600步重新利用了对象信息，因此160步信息弱现象不能升级为持续或不可逆的后验坍塌诊断。[阶段2结果](diagnostics_capacity_v2/RESULT.zh.md)。

真实320对象的独立构图、全部描述符与六层身份绑定通过32项检查；运行中实际scaler/config又通过20项检查。279条train、4,868残基的均值和标准差与独立两路复算完全一致，41val未参与拟合。[真实输入审查](expanded_preprocessing/independent_release2_actual_review/REVIEW.zh.md) · [实际scaler审查](expanded_preprocessing/independent_release2_actual_scaler_review/REVIEW.zh.md)。

三种控制的96条结果、882个Markov计数及全部来源/RNG链已独立重放；输出逐字节一致。[真实基线审查](evaluation/terminal_v2/actual_baseline_review_dbaasp_v4_001/REVIEW.zh.md)。这不意味着不同方法的条件信息或计算预算相同。

具体而言，条件flow先验使用训练序列导出的逐残基条件，后验聚合使用训练后验，DDPM仅接收请求的末端类别并另需1000次更新；三种统计控制只从train拟合。相关指标可检查复制、分布和失败模式，不能解释为严格同条件、同算力的方法优越性。末端标签也是请求状态，不是对生成物端基的实测认证。

## 新审阅者的只读机制诊断

未参与实现的新代理独立复算52组保存NPZ及相关日志，3,870项算数检查通过，87个静态输入未变；没有运行模型或更新权重。[完整报告](mechanism_independent_readonly/REPORT.zh.md)及[残基条件对照表](mechanism_independent_readonly/RESIDUE_CONDITION_INFORMATION.csv)保留全部可知与未知。

主run的85/701是固定噪声种子的**随机后验重建**，不是decoder(mu)准确率。主run没有保存逐对象mu重建、置换、零latent和完整flow密度，不能据低正确率、小KL或极值范围认定后验坍塌。其PAD误差只占总验证平方误差9.02%，但缺逐对象输出仍不能恢复准确PAD召回或长度误差。

已确认三项机制事实：（1）任意AA编号的连续MSE并非类别目标；（2）原KL只除latent16，B32与尾批B23相对权重不同；（3）新增train z-score后继承的global abs把输入改成距训练均值的绝对偏差。当前全局9维向量在abs前后未出现新的整向量精确碰撞，不能把符号丢失直接认定为质量根因。输入与目标数值尺度不同也不是自动的单位错误。

flow条件的20AA三描述符形成19个不同triplet，唯一碰撞I/L，其他18AA可逐位识别；尾部零填充还提供长度。这里只抽train条件，不构成已发现的跨分区泄漏，但它与terminal-only DDPM的信息量差异已得到具体输入证据。aggregate抽取训练后验，生成decoder不再次直接读取g。原mean-only加性flow的logdet为0，不能误诊为漏掉一般affine flow密度项。

最小后续动作按依赖排序：先在独立预锁的固定checkpoint诊断中保存mu/随机、预指定多次置换、零latent、逐对象后验与flow密度；再分别考察batch不变的KL归一、21类输出头及CE/PAD目标、随机噪声路径和signed/abs单因子对照。改变KL归一时必须区分归一语义与实际正则强度。DDPM噪声MSE还需与同x_t/t/epsilon上的无网络比例、train均值及高斯线性控制比较；目前没有这些控制的实测分数。以上均为**尚未执行的可证伪假说**，不覆盖当前固定run，也不盲目增加训练步数。

[VAE实际训练轨迹](main_training_visuals/01_vae_training_trace.png) · [DDPM实际训练轨迹](main_training_visuals/02_ddpm_training_trace.png)。5,000个绘图点逐一匹配日志，8份输入哈希未变；只展示最终验证值，没有补造逐步验证曲线。DDPM日志lr为scheduler.step后的下一步值，图中已注明。

科学开放项仍包括：原模型的随机生成与长度质量、KL与重建目标的相对尺度、独立复现与多seed统计、条件信息/算力不相等的比较边界、公开来源选择偏差和新生成肽真实活性。三种简单统计控制不覆盖有竞争力的closest-work比较，不能据此建立方法优势。旧27条湿实验不是新模型前瞻验证。当前本地来源、代码和计算推进无需新的作者资料；若要建立新生成候选的活性/安全性主张，仍需与具体候选及新模型身份绑定的真实独立证据。

## 最终本地验收与保全

实际执行与导出已由不同审阅者通过：[执行独审](runner/independent_architecture_review/actual_dbaasp_v4_cpu_001/REVIEW.zh.md)。核齐1000+1000更新、81文件、96神经请求及4,608个raw值。每类保存的是900、800、…、100、1十个逆扩散记录点；完整998次执行的证据是冻结无提前退出循环、到达1和成功产物，不冒称存在998条逐步日志。

[六方法独立指标审查](evaluation/terminal_v2/actual_six_method_review_dbaasp_v4_001/REVIEW.zh.md)通过489项检查：192请求、12组，独立穷举52,731次NW对齐，38,771个指标叶节点一致，最大浮点差1.9984e−15；204个CSV单元格、12行报告和96条raw解码转写全部通过。审阅者没有导入作者指标实现或执行模型。

[受保护文件复核](PROTECTED_INPUTS_20261011T120121.json)显示184/184未变，包括冻结论文、原交接和既有受保护实现；原失败和旧负结果保持可见。最终交接的确切版本及这些完成证据由`DELIVERY_CANDIDATE_MANIFEST.json`绑定，最后逐主张审阅另见`final_science_review/REPORT_FINAL.zh.md`。本地执行/导出/指标审查通过不等于科学投稿门通过。

本轮授权的固定运行、来源与实现修复、只读机制诊断及独立核验已经完成，无待补作者文件阻塞这批本地结果。未执行的H0及其后续单因子实验是下一个明确的本地科学工作；竞争性closest-work、多seed和新候选独立活性/安全性证据仍然开放。A10硬件仍未验证，当前结果全部来自Mac CPU；没有等待或重试鉴权。

**BACKUP_BLOCKED**。禁止上传、push、正式提交、生产访问、凭据读取及付费调用；本地hash保全不是云备份成功。
