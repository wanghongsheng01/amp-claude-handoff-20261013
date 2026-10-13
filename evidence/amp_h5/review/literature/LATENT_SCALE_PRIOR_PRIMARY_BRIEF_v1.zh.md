# AMP H5：潜变量尺度与先验坐标的一手先例简报 v1

**结论：已有成熟先例支持把扩散输入尺度作为明确、可逆的训练合同；没有先例或本轮证据证明“未做latent标准化”就是AMP当前错配的原因。** 本简报只核三个先例与当前坐标边界，不启动实验、不改变scalar许可。

## 先有的 AMP 证据

已完成静态审计：DDPM的目标是原始 `r=(μ,ℓ)`、`ℓ=logσ`，shape为`B×2×16×48`；encoder描述符g/h的train scaler不作用于r。生成先得到r，再用`z=μ+exp(ℓ)·ε`输入decoder。训练KL则把z经过依赖逐残基g的flow变到`z_p=F_g(z)`，在z_p坐标评价基础标准正态。[原坐标审计](../../feasibility/coordinate_audit/COORDINATE_AUDIT.zh.md)

CE恢复32请求均数值有限，但AMD生成σ范围为 **[0.110826,5.437218]**，对应train为 **[0.784264,1.248017]**；生成logσ标准差是train的 **5.48536倍**。这是同一冻结后审记录中的尾部/尺度描述，不是显著性检验、因果隔离或独立样本数768倍扩张。[CE恢复后审](../independent/CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json)

| 先例 | 一手证据与机制 | 对 AMP 的适用边界 |
|---|---|---|
| Ho等，DDPM，2020 | 原文§3.3将图像从0–255线性缩到[−1,1]；§4说明1000步线性β=1e−4…0.02是相对于该数据尺度选择；Eq.(4)给出前向均值/噪声关系。[原论文](https://arxiv.org/pdf/2006.11239) | 缩放并非新增范式；但[−1,1]不是逐维z-score，原文不要求所有目标标准正态，也没有证明AMP必须采用某种标准化。AMP的999槽及索引合同也不等于原文1000步。 |
| Rombach等，LDM官方实现 | `LatentDiffusion`的可选`scale_by_std`在首次训练batch取编码样本的整体std倒数；`get_first_stage_encoding`乘scale，`decode_first_stage`先除同一scale再解码。默认`scale_by_std=False`。[作者源码](https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py)；[已读取raw源码](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/ldm/models/diffusion/ddpm.py) | 核实的是对encoder样本z的全局标量缩放，并非对(μ,logσ)逐维中心化/白化，也不等于全train总体方差。不能照搬固定缩放常数或宣称所有LDM都自动这样做。 |
| Tolstikhin等，WAE，2018（所核作者修订版v4） | 论文定义聚合后验为先从数据分布抽X、再从Q(Z\|X)抽Z的边缘分布Q_Z；明确区分Q_Z与P_Z，并给出VAE正则的聚合KL与互信息分解。[作者论文v4](https://arxiv.org/html/1711.01558v4)，§2及附录A | “每对象后验”“聚合后验”“指定先验”不能互换；该理论不是诊断当前AMP已collapse或提出换WAE的证据。有条件flow需在相应条件/坐标内解释，不能把无条件恒等式直接套给错误坐标。 |

LDM本次机制核查以官方代码为准，未成功获取其arXiv PDF，不声称核过论文中的对应段落。代码URL是可变main，本简报未核定immutable commit；若成为后续实现依据，须单独固定版本。

## 从上述机制可推出、但尚未验证的边界

以下是数学/本项目推论，不是三篇来源对AMP的结论。

1. 固定前向式`r_t=√ā_t·r_0+√(1−ā_t)·ε`时，中心化某坐标的信号/噪声方差比为`ā_t·Var(r_0)/(1−ā_t)`。改变数据尺度会改变固定日程的有效信噪比；因此值得有控制地检验，却不能由末端尺度不合直接倒推原因。均值/方差规范化也不保证消除相关性、偏态或长尾。
2. 有限train上的聚合decoder后验是高斯混合`q_train(z)=N⁻¹Σ_i N(μ_i,diag(exp(2ℓ_i)))`，不是参数对r的经验分布。标准化r不会使该混合自动等于`N(0,I)`，也不会使生成参数分布自动拟合train。
3. AMP的原生基础先验在z_p。原生prior采样若回到decoder坐标，需要`z=F_g⁻¹(z_p)`及其逐残基g；terminal-only生成没有目标序列的真实g。直接在decoder坐标令`z~N(0,I)`可以命名为独立控制，**不能命名为原生flow-prior采样**，也不能暗用真实g补全条件。现有r→z生成路径不应无根据插入inverse-flow。

## 若后续提出标准化候选，最低合同条件

这些条件只约束未来候选的可解释性，不批准执行，也不预设其会赢：

- 先明确变换对象仍为r、统计轴和零方差处理；只用冻结train后验拟合变换T，保存统计与哈希，dev/test不参与。全train统计属于AMP的明示适配，不能冒称LDM官方首batch规则的原样复现。
- 训练与采样都在T(r)坐标运行同一新DDPM；生成后先施行同一`T⁻¹`，还原μ/logσ，再exp、重参数化及原decoder/J。改变坐标后须重新训练DDPM，不能给原始坐标训练的旧权重只在采样端加逆变换。
- 冻结VAE、表示语义、数据、其余损失/β、架构、日程、种子及预算合同；记录额外训练/生成成本与全部失败。AR的信息、预算及选择机会维持公平；既有过拟合及dev复用不能抹去。单次标准化干预的结果才可用于检验尺度因素，不能同时改多组件后归因。

当前只建立了原坐标链、实际尾部错配和三项成熟先例。未证明尺度是根因、标准化足以修复、decoder标准正态等于native prior，或任何新方法优势。scalar活跃输出未读取；原门与既有科学限制不变。
