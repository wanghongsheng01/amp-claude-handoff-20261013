# H5 独立设计审查：离散目标、长度一致性与 terminal-only AR

这是执行前的设计意见，**不是 PRELOCK PASS 或训练许可**。本次只读了冻结旧源码和既有证据；没有读取真实特征数组、运行模型、拟合、读取 test 或指定结果后阈值。H5 的新代码、预算、seed、数值门及 smoke 仍应由 root 事前锁定，经独审后 START。

最小可解释比较取决于要回答的问题：

| 问题 | 最小同架构新对照 | 可以归因的变化 |
|---|---|---|
| 显式合法前缀目标是否优于独立位置分类 | 同21输出头的 position CE vs legal-prefix NLL | 只改变重建目标的合法支持集/归一化，保留同一 J 解码 |
| 同容量回归到分类，再到长度一致目标，各自改变什么 | raw one-hot square → position CE → legal-prefix NLL，三臂同21头 | 两个预设相邻对照；不能用中间结果挑主臂 |
| 预算只容纳 square vs legal-prefix 两臂 | 可以做，但结论限于联合目标变更 | 同时改变回归/分类和长度结构，不能单独归因 CE 或结构归一化 |

建议第一项作为最小主问题；若论文还要单独解释“标量/回归到类别”的机制，才加入第三个 matched raw-onehot-square 臂。旧 scalar17参数/H4结果可保留为历史诊断，不能替代同21头的新对照。不要为恢复单因素措辞而隐去必要的修复，也不要为了多臂说明自行扩大本轮预算。

## 1. 实际源码约束

`runner/run_rebuild_v3r2.py:157` 的旧重建项为所有48位置的 scalar squared error 之和，除以当前 batch 非 PAD 残基数；PAD错误计入分子。`modules/vae.py:297` 的 sampled flow KL 为所有 batch×16×48 的 logq−logp 之和，仅除16，没有除 batch 或48。尾 batch 与序列长度分布会改变两项的相对权重。

实际 `FVAEDecoder.forward` 强制 `g=None`，WN只读 latent。encoder/flow训练会读序列及训练对象派生 g/h；当前 DDPM 训练拟合 encoder 的 mu/logsigma，生成时只读 terminal 标签和噪声。保留这个主生成路径不必在生成时传真实 g、原对象ID、训练对象长度或旧重建长度。flow prior与posterior aggregate可留历史诊断，但不能并入 terminal-only 主比较。

flow coupling 实际设 mean_only=True，属于单位体积的加性 coupling；不应仅因旧 KL表达式未出现一般 logdet 就宣称其漏项。sampled logq−logp 的单次估计可能为负，不能事后截零来制造非负 KL。

H5 必须从重建目标把梯度传回 encoder/decoder等应训练参数；它不同于 H2 冻结hidden只拟合头。随后重新拟合 DDPM仍是**两阶段生成训练**，并不意味着 DDPM损失反向贯通VAE的联合端到端优化。命名和论文措辞应区分这两件事。

## 2. 三种目标的精确定义

设每个对象的 score 为 a[t,k]，t=0..47，k为20AA加PAD；真实序列扩展为48位合法AA前缀/PAD后缀。相同21头、相同全部48位置权重、相同数值 dtype。下列 R 均先定义为每对象的未归一化重建量，避免混淆 class/time/batch reduction。

- Raw one-hot square：R_sq = 1/2 × Σ[t,k](a[t,k]−onehot(y[t],k))²。先不 softmax；这不是 softmax Brier，也不是原来的标量索引回归。按类别求和，不能默默 mean over21 导致额外1/21缩放。
- Position CE：R_CE = Σ[t](logsumexp_k a[t,k]−a[t,y[t]])。包含全部PAD位置；它是对全部21^48串的独立位置分布进行正规化。
- Legal-prefix NLL：定义长度支持 S，A_t=logsumexp_{k<20} a[t,k]，P_t=a[t,PAD]。Q_L=Σ[t<L]A_t+Σ[t≥L]P_t；logZ=logsumexp_{L∈S}Q_L；真实串得分 E_y=Σ[t]a[t,y[t]]；R_struct=logZ−E_y。这里对所有20^L个AA前缀求和，不能把 A_t 换成 max_AA。

共同推断用 J_L=Σ[t<L]max_AA a[t,k]+Σ[t≥L]P_t，然后在同一 S 上取 J 最大值、长度精确同分取最短、AA同分取最低索引。这是结构化类别分布的 joint MAP，也是 raw one-hot square 对合法one-hot序列集合的欧氏投影；后者不使 raw square scores 变成概率。

`logsumexp_L J_L`只对每个长度的一条最佳序列求和，遗漏其余序列，**不是**上述合法串归一化常数。长度 marginal MAP使用Q_L，joint sequence MAP使用J_L；不能换用其中一个后仍称同一解码器。

S建议事前明确为0..48以保留与H2/H4相同的空输出诊断；如果产品/科学问题另要求1..48，也可全臂一致采用，但排除空串是设计约束，不能作为学习后非空率提升证据。不能读到新结果后再改 min-length=1或5。PAD在合法串中承担后缀而非单个额外EOS位置；48长串也是合法事件。

零logits的合成检验应验证：position CE每位置log21；结构化Z=Σ[L∈S]20^L；J对所有长度同分按预定最短。这个例子同时可检出错误max归一化、遗漏L=48和多计PAD。

## 3. KL、尺度与归因

三臂应共享唯一写出的 batch reduction、KL系数及任何warmup/free-bits政策。两条可行路线不能混写：

1. 若目的严格保留旧KL，则每个新臂都原样使用同一 sampled sum/16，并承认 batch/tail依赖；比较结论只限新臂之间的目标变化。
2. 若为修复训练定义而按对象归一，则共同目标可写为 mean_B[R]+β×mean_B[Σlatent(logq−logp)]；也可将两项一起再除固定48。所有新臂采用同一版本和预定β政策，旧scalar→新模型的比较则是“类别头/目标与归一化修复的联合变化”，不能只称CE消融。

建议优先采用清楚的共享归一化，让CE vs结构化NLL的主对照只差R；是否保留旧式KL及β数值由root预注册。不能根据dev结果再调β或给某个目标更多机会。共同β不意味着各损失下相同有效KL压力；raw-square与CE的数值尺度/曲率本来不同，应保存R、KL及预定检查点的分项梯度范数/夹断比例，解释而不事后调平。

原头变为21输出后还必须明确输出轴(B,21,48)/(B,48,21)；不能把class数21当序列长度或用squeeze(1)后沿错轴CE。不能将round/argmax/J放进训练主损失并阻断重建梯度。H2 ridge强凸证书不适用于非凸端到端VAE或AR，有限梯度/训练下降不等于全局最优。

## 4. 初始化、噪声和选模机会

新VAE臂应共享一份可核对的初始完整21头模型状态，各自独立optimizer状态；相同结构间逐参数hash/值匹配。仅“相同seed”不足以证明共享初始化。共享固定batch对象顺序、尾batch政策、训练更新上限、lr/scheduler/clip、验证时点及checkpoint政策；validate使用独立RNG流，不能消耗训练流。若warm-start，所有臂共享同一初始来源并称为fine-tuning，不能称fresh training。

训练期可预锁同更新/对象位置的重参数化epsilon。不同目标更新后mu/logsigma不同，所以即使epsilon相同，z也不同。每个新encoder必须重新提取train/val后验并训练其配套DDPM；不能把旧H4 DDPM与新decoder拼接而称同一生成实验。每臂DDPM可共享初始化、batch索引、t、noise和condition-drop随机计划，但拟合目标随encoder改变。

生成期同DDPM初始Gaussian和逐步创新噪声只能称共同随机数耦合/相同随机请求索引。不同拟合DDPM生成的mu/logsigma和最终z可能不同，**不能称same latent**。与AR更不存在same latent；AR使用自身离散抽样随机流并单独记账。

主终点宜固定最后计划checkpoint，所有中途曲线仅诊断；如果root确需验证选择，必须事前固定同数量候选、共同选择规则并披露自适应dev使用。不可对候选选最佳seed/温度/decoder而给AR只一个默认点。本阶段预算与hyperparameter机会都由root先锁，不依据smoke表现临时续训、加seed或挑checkpoint。

## 5. 单元与有界短过拟合验收

先以合成对象验证：布局、one-hot class-sum、PAD/长度0/1/47/48、稳定logsumexp、合法集穷举概率归一化、J与穷举MAP、有限差分梯度、batch复制后的归一化不变性（或旧式KL预期缩放）、训练/推断不能触达test。对不同臂clone状态及显式噪声做输入身份测试。AR另核causal mask/shift、BOS不作为目标、只计首EOS而非PAD尾、teacher forcing不泄漏未来token。

短过拟合对象只可由train按预锁ID/规则选择，不能用val。记录固定预算内的loss、nonPAD micro/macro、长度误差、整串完全一致及分项梯度，保留失败。纯重建β=0小测试可检查数据/梯度接口，但不能冒充含KL完整目标验收；若需要该测试，应在授权smoke中另明确。重建均值模式与固定后验噪声模式要标明，不可挑较好模式。

本设计审查不替root规定样本数、更新数、成功率或损失阈值。root需在真正smoke前冻结这些有限预算和停止分支；未达预锁可学习性/数值门时停止更昂贵DDPM阶段并保留结果，不把未完成模型当成强baseline失败。数值异常/cap只允许保存最后已完成状态，不补样、不续训、不换seed。

## 6. AR公平范围与长度模型

AR可接terminal+BOS+已生成prefix；teacher forcing只在train拟合，生成API不得接原对象完整sequence、真长度或g/h。AR作者已同步其设计：L<48监督AA及首PAD/EOS，L=48仅监督48AA、不加EOS49；生成最多48步，允许48无PAD。这是一个有截断上界的正规长度模型，不应再给48长事件乘EOS49概率。若使用别的EOS政策，必须重写其likelihood和截断解释。

VAE和AR需共享train279/val41对象/terminal定义、合法长度支持、请求数和评价器。AR不必套J，因为其head依生成prefix递归变化；强行要求“相同J”不是合理同因子控制。共同J只用于同族VAE目标臂。AR比较应说同生成信息/数据/请求/评价/选择机会；参数容量、两阶段训练成本、自回归与DDPM采样成本应分别报告。

相同更新数或同一个最高墙钟cap不等于实际等成本。候选总预算必须计入VAE训练、重新编码、DDPM训练及采样；AR应获得事前明确且可辩护的总训练/调参机会。固定更新预算比较与实际算力对比是两个维度，不可只挑某一维度称严格公平。DDPM原EMA helper在2000步以前复制模型而非平滑；短pilot如未跨越该点，应保留记录，不宣传EMA平滑增益。

双方生成一律只按预锁terminal和随机请求，全部失败、空串、full48及所有分母保留；唯一性/熵、NW/复制、421特征MMD、JS/L1/KS全方向报告，不用单项形成新综合门。AR token NLL与VAE posterior reconstruction/ELBO或两阶段DDPM损失不是同一个可直接排名的likelihood。没有生成逐位真值，也没有活性标签或新湿实验，不能从相似度或合法性宣称生成accuracy、抗菌优势或独立新颖性。

待root/作者锁定：最小两臂还是三臂及主对照；S长度集合；共同KL reduction/β政策；初始化来源/共享随机计划；单位测试和短过拟合预算、停止分支；每臂VAE与配套DDPM训练预算；AR总训练/selection机会；最后checkpoint/温度/采样计划；可独立复算的逐步日志、原始scores/targets/lengths/ID/RNG/权重与保全清单。以上明确后再做候选静态预飞。本报告不阻止已获授权的准备工作，也不批准尚未冻结的真实训练。

**BACKUP_BLOCKED**。无额外spawn、网络、凭据、上传或前代写入。
