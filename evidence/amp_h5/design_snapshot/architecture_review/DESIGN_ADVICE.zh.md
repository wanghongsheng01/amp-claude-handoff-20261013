# H5 独立架构建议：保留原方法的 terminal-only 端到端对照

状态：**静态设计建议，NOT_PREFLIGHT_APPROVED / NOT_STARTED**。没有导入/加载/运行模型、训练、生成或读取 test；不修改任何作者实现或 H0–H4 冻结产物。本建议不能替代根协议、独立预飞和 START。所有数值预算仍待根任务锁定。

## 建议的最小路径

保留原 embedding100、latent16、seq48、encoder/WN/flow、g/h 描述符及 train-only scaler。两个新臂均使用 `out_dim=21`，最后头同为 `Conv1d(16,21,1)`，357参数；同时从相同新初值端到端训练所有原上游参数。对照为 one-hot MSE 与 CE，主生成分别使用各自重新训练的 posterior-parameter DDPM。第三方法是从同一冻结 train、仅末端条件训练的完整 AR。原17参数 scalar/H4只能作历史锚点，不能声称与新两臂等容量。

代码亲核表明，这条路径不要求把 flow 改成 terminal 广播，也不要求删除 g/h：

- 原 `FVAE.forward` 训练 encoder 接收 x、g、h；KL比较同一后验与 g 条件下 flow prior。训练侧这些特征由同一可用序列/化学对象导出，必须披露其计算与信息来源。
- 原 `train_diff` 使用 VAE 的 μ/logσ 重编码 train/val，张量为 `[B,2,16,48]`，DDPM只接末端类别，并保留原10% label-drop训练机制。
- 原 DDPM 生成是 Gaussian noise、t、terminal → posterior参数 → 再采样 z → decoder；decoder `wn(z,g=None)` 不需要真实对象 g/h、源序列或真长度。此处可以与 AR 的 terminal-only 生成任务对齐。
- flow-prior 生成额外抽训练对象 g，posterior-aggregate抽训练对象后验，二者不能混入公平主比较或当作 terminal-only 强基线。如仍另列，只能是明确的信息不同诊断，不是新增未批准生成预算。

这保证相同原始训练对象和生成条件，不意味着两种方法的内部特征、架构、参数量或训练目标相同。可比较完整方法；不能把该比较解释成一个模块的纯因果效应。

## 目标、长度与归因

建议先明示一次共同的规范化修正，再比较两个匹配容量的重建目标，不扫描 β。令 D 为原实现逐坐标 `(logq-logp).sum()`，B为本批对象数：

```
K = D / (B * 48) = legacy_KL * 16 / (B * 48)
R_CE = sum_over_objects_positions CE21(scores, target) / (B * 48)
R_MSE = 0.5 * sum_over_objects_positions_classes (scores-onehot(target))² / (B * 48)
L = R + 1 * K
```

两臂 PAD 权重同为1，分母均包含48位置，MSE明确是21类平方和而不是再除以21。保留原条件密度、重参数化、flow及零 logdet 的实际结构；原 KL、D、新K、B与重建分子/分母一起记日志。小负 Monte Carlo 密度差不得截零、取绝对值或隐藏。尾批23与整批32不再因 KL 分母改变正则强度。

**这不是对旧总目标的纯全局单位换算。** 旧 scalar 重建除以非PAD数，旧KL除以16且随批大小变化；到H5改变了目标编码、头容量与相对KL尺度。它是两新臂共同的规范化修正，故不能把与旧scalar的差异单独归因于CE。相同β也不表示CE和MSE有相同有效约束强度。若根任务选择保留 legacy KL，技术上可执行，但须明确回答的是保留旧批依赖目标的替换诊断，不能私下在实现中混用两种定义。

两个新VAE臂共用 J 主解码：只根据各自生成scores选择0…48的合法AA前缀/PAD后缀，最短浮点 exact tie、AA最低索引tie。对CE是该因子化类别模型下的合法序列联合评分；对raw one-hot MSE是最近合法one-hot的欧氏投影，不应把其softmax当校准概率。firstPAD可作为同一已保存scores的附加机制读出，若采用需在协议预先列明，不增加模型调用或结果后挑主读出。PAD20是训练尾填充；未另增EOS，长度48且无PAD合法。不得使用真长度、旧重建长度或验证分布去决定单条生成长度。

AR应保留真正的自回归采样机制，不强套并行J。若采用末端分层两个子模型，BOS仅输入、输出为20AA+PAD，teacher forcing必须移位，推理只用自身已生成前缀；48位无PAD按同上限保留。词表、温度、multinomial/greedy、停止和失败约定均先冻结，不重采到凑齐合法数。两个子模型的训练/参数/采样成本合计为一个AR方法。实现若是对公开LSTM的PyTorch适配，必须绑定实际源版本及逐项差异，不能冒称未经核实的官方逐位复现。

## 独立训练与预算

每个端到端臂都会改变encoder、后验分布和decoder，因此必须以该臂最终冻结VAE重新编码train/val，再从头训练自己的DDPM。不能复用 H0/H4 posterior cache、旧DDPM或另一损失臂的DDPM。仅最后读出变而上游全冻结时才可讨论共用旧DDPM；那已不是本轮端到端实验。

两臂共享实际初始参数/缓冲区哈希、批次顺序、训练 seed、后验 epsilon 生成规则、学习率/clip/optimizer设置及固定终点。各自独立optimizer与结果目录；验证RNG隔离，val不能改变后续训练抽样、选终点或增预算。DDPM也使用各自fresh初值并固定共享的噪声/标签drop随机规则。相同随机流不保证不同已训练模型得到同一latent；不得沿用H4“两个头同一latent”的因果描述。

建议最小生成预算为一个预定sampling seed、每完整方法 FREE16+AMD16＝32 attempts。若方法为MSE→DDPM、CE→DDPM、AR，则共96 attempts。若改成3个sampling seeds，则每法96、共288，不是原预算；这仍只重复采样同一训练权重，不能称3个独立训练seed。三次完整训练又是另一个约3倍训练预算。根任务需明确选择，不由实现者自行扩展。

旧实测给出的粗成本依据：VAE1000更新1184.698秒/臂、DDPM1000更新545.832秒/臂，后验编码另约10.417秒；H4两terminal完整DDPM重放连同廉价flow/aggregate和readout记录共1126.198秒。用这些旧数作算术外推，两新臂各1000 VAE、1000 DDPM及一组两terminal完整采样约95–96分钟，再加AR和新评价；不是新21头实测或时间承诺。不得把H2缓存凸头的秒级拟合时间当端到端训练成本。

公平资源账户须将VAE、后验编码、DDPM和必要读出费用与AR全训练相比较；每方法相同1000步不等于同算力，也不能只计候选末头而漏掉上游。至少预锁每方法合计墙钟上限/固定终点、记录更新数、见过对象/有效token、参数量及采样成本。若优化未充分或某方法受预算限制，只能报告有界pilot，不能据其较差结果宣布方法被击败。当前279/41很小且val已反复开发，均不构成新的独立确认。

## 预飞必须检查的具体项

1. 锁实际协议、源快照、train/val对象和scaler；禁止旧通用目录遍历读取历史test。保留化学对象 `(sequence,class)` 身份，不跨terminal误去重或重新分区。
2. 精确证明两新VAE的357参数头及所有上游初值相同，只有重建目标分支不同；明确shared KL修正与旧目标差异。
3. 断言scores `[B,21,48]`、targets `[B,48]`、one-hot `[B,21,48]`，杜绝标量损失广播；检查PAD权重、分子分母、完整48位及长度48边界。
4. 亲核KL的原密度及16→B48换算，仅一次缩放且保留梯度；不误将logsigma当logvariance，不更改conditional flow或隐藏负估计。
5. 训练、验证和生成RNG隔离；相同源批次/epsilon规则、预定终点，无val择优、无失败后续跑/重抽。允许的smoke及其参数更新必须另入协议身份，不能暗中使用正式初值训练后再称fresh。
6. 各臂posterior cache必须绑定其最终VAE哈希，DDPM/checkpoint/EMA绑定该缓存、类标签和日程；原999项表对应998逆步/terminal，不能静默改步数。
7. 主生成入口只见terminal与自身噪声，不借train对象g/h、原型、真实前缀或长度；AR与DDPM全attempt台账和公共指标分母一致。
8. 全参数/梯度/optimizer有限性、原始scores/logits、失败slot、配置/seed/耗时/权重哈希可审；记录重建、KL及train/val内容/PAD/长度/wholeexact，不能仅看训练loss下降，也不预设CE胜出。
9. 区分训练重复与采样重复、训练成本与采样成本、工程完成与科学结论；任何终点比较保留每terminal与每seed结果，不挑有利集合。

**BACKUP_BLOCKED**。本文件仅给建议及可审验收需求，没有批准或执行H5的smoke、训练或生成。
