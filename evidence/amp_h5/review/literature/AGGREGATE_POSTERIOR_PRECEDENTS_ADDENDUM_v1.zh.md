# AMP H5：聚合后验、来源对照与成熟先例补充 v1

**三篇原文支持把聚合后验与生成先验的匹配单独研究，不能据此认定它是本次 AMP 生成负结果的原因。直接学习 decoder 潜变量 z 的分布已有成熟先例，本身不构成新颖性。** 本件只核原文与诊断边界，不增加forward、训练、生成或test访问，不读取活跃scalar。

| 用户指定先例 | 原文核实的位置与机制 | 本任务适用边界 |
|---|---|---|
| VampPrior（Tomczak、Welling，2018） | §3 Eq.(6–9)：先验交叉熵项涉及聚合后验；固定该分布、允许自由选择先验时，最优先验为聚合后验。实际方法用**可学习伪输入**上的变分后验混合近似。[正式原文](https://proceedings.mlr.press/v84/tomczak18a/tomczak18a.pdf) | 不是简单复用train均值或随机train donor；它改变可学习先验，并与encoder耦合。原文也讨论全数据混合的开销/过拟合风险。不能把冻结AMP的来源替换称为VampPrior复现或由该最优性推出AMP因果。 |
| Two-stage VAE（Dai、Wipf，2019；题名 *Diagnosing and Enhancing VAE Models*） | §4 Eq.(10)及三步方法：第一阶段学习表示；从其后验抽z；第二个独立VAE学习这些z的分布；经u→z→x生成。作者在其设定中强调分开训练。[作者原文](https://arxiv.org/pdf/1903.05789) | 区分重建/流形拟合与分布采样，但其理论涉及连续低维流形、Gaussian模型等假设，不自动覆盖离散AMP、有限279对象或条件flow。它不是“标准正态直接喂冻结decoder必然正确”的依据。 |
| LSGM（Vahdat等，NeurIPS 2021） | Fig.1、§3 Eq.(5–6)：encoder的z样本是潜空间score模型的目标，score模型作为VAE prior，生成z后进decoder；提出可同时学习encoder/decoder/score prior的目标。[正式原文](https://papers.nips.cc/paper_files/paper/2021/file/5dca4c6b9e244d24a30b4c45601d9720-Paper.pdf) | 学习z分布是成熟路线；LSGM并非对原始(μ,logσ)做当前AMP的固定DDPM。原文专门处理可学习encoder分布的目标项，不能把普通冻结编码DDPM等同其端到端训练或照搬其性能。 |

以下是基于本项目坐标审计的对照解释，不是上述论文替AMP做出的实验结论。

**来源替换先核坐标和train索引。** AMP有三个不同对象：参数对`r=(μ,logσ)`、真实decoder输入`z=μ+exp(logσ)ε`、基础先验坐标`z_p=F_g(z)`。比较来源时应固定同一decoder输入位置、shape/轴顺序、dtype、缩放及相同decoder/J。来自train的每个后验、噪声实例、terminal与条件g均须保留对应记录ID/索引，不可把不同对象的μ、logσ或g误配。后验均值和后验抽样也应明确区分。DDPM独立生成没有天然train真值配对，不能事后用最近邻或任意train序列赋予“重建正确率”。

**逐坐标shuffle是联合结构控制。** 在已经固定的同端基、同n的decoder输入矩阵`Z∈R^(n×768)`上，对每一列独立置换样本行，可以精确保留该列的经验边缘分布，同时扰动跨坐标联合结构；需保留每列donor索引。若所有列用同一行置换，仅重排完整样本，联合分布没有改变。若混端基打乱，terminal条件边缘也被改变；若在r而非z上打乱，经非线性重参数化后不能声称保留z边缘。有限n下独立置换不保证剩余相关性恰为零；性能变化只能支持该固定控制下联合结构的作用，不能单独证明原DDPM失配的机制。

**真实native条件先验与标准正态decoder控制分开命名。** 当前基础标准正态定义在z_p。给定真实可用g时，`z_p~N(0,I)`再经`F_g⁻¹`才是对应的native conditional-prior样本。借用某个train对象的g必须标明train条件诊断与配对索引，不能冒充只有terminal c的独立生成；不得暗用目标序列g。直接令decoder的`z~N(0,I)`是独立标准正态控制，不是上述原生先验，也不应与它合并统计。

**先核same-n参照，不能从距离跳到创新。** 比较高维潜变量的经验尺度、协方差或分布距离，须先固定相同n、同端基、同坐标和相同索引/抽样合同，保留有限样本参照及额外随机性。尚无这样的参考核验时，不据W2或协方差外观宣称机制、新颖性或提出whitening修复。统计差异、来源替换响应和端到端生成收益是不同层级的证据。

任何后续直接z建模仍须与相同生成信息的完整AR公平比较，保留实际训练/生成成本、调参机会、固定端点与已复用dev的限制。三篇先例没有替代这项要求，也没有解除原E/L/G的NE或当前G失败。本件不提出新head、多组件联合修改或新运行；当前scalar许可不受影响。

本地坐标依据：[上一份尺度与先验简报](LATENT_SCALE_PRIOR_PRIMARY_BRIEF_v1.zh.md)、[原静态坐标审计](../../feasibility/coordinate_audit/COORDINATE_AUDIT.zh.md)。
