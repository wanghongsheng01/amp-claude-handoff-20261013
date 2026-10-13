# AMP H5 latent 参照设计预审：仅设计，不授权执行

当前 scalar 单次冻结运行继续；本文件不授权任何模型前向、训练、额外 DDPM 或重试。正式参照须等待实际证据、scalar 终态、独立的新 scope/计划及回执。原 CE E/L/G 不回填，PROMOTE 保持 HOLD。

## 随机量与 prior 的边界

CE/MSE 后审已从 CPU torch 的 422/423 种子重放随机调用：初始 `(16,2,16,48)` 一次，反向链 t>1 的同形噪声 997 次，随后 `(16,16,48)` 的 epsilon 一次。以最后这个 epsilon 计算 `mu+epsilon*exp(logsigma)` 与已保存 z 逐位相同。该证据支持固定调用图下最后随机 draw 与构成 DDPM posterior 的前面随机 draw 分离；不能用 `(z-mu)/sigma` 逆算替代原随机样本，因为浮点运算会丢失信息。

若复用最后 epsilon 作直接 `z~N(0,I)` 参照，后续应封存其文件、dtype/shape、SHA、种子、torch/CPU版本、调用次数及原 z 重构验证。CE/MSE/参照共享它属于配对的 common random numbers，不是增加独立种子或独立重复；固定每端基16、总32 slots，不挑样、不替换失败。

必须先明确模型坐标。原模型有 prior_flow；若标准正态先验定义在 zp，完整模型 prior 需 `z=inverse_prior_flow(zp,g)`，那么直接把 N(0,I) 送给 decoder 只能称“独立标准正态 latent 参照”，不能称已实现的完整 learned prior。不得猜测生成时不可获得的 g、偷偷用逐残基内容或真实序列派生属性补齐条件。等待静态 route 审计把坐标、flow 方向及 g 的可达性逐项锁定。

## 训练后验参照

从现有已封存 train posterior NPZ 按 terminal 划分 row_ids，然后有放回各抽16条。抽样算法、软件版本、seed和全部32索引必须在任何参照 decoder 输出之前封存；CE/MSE共享源记录索引与对应epsilon，保存重复、来源记录及每个slot的状态。不得按长度、重建或生成质量挑选记录。

使用 `z=mu_i+epsilon*exp(logsigma_i)`、同一已冻结最终 decoder、同一原生读出。它是“训练经验后验重采样/记忆参照”，不是独立生成的前瞻证据，也不能将对训练样本条件化的成功率当成新颖肽生成成功率。现有训练均值重建可另列为确定性参照，但不能替代带噪声后验重采样，或选择较好的一种呈现。

## 数值描述与可能解释

先锁定按 terminal 的 mu/logsigma/sigma/z 维度、汇总方法、均值/方差/尾部分位及相关性定义。区分“跨潜在坐标相关”“跨记录相关”与“训练/生成协方差”，不能将12288个潜在坐标当作12288个独立样本。每端基只有16次生成请求；高维完整协方差秩受限，不应根据结果挑选降维、核宽或显著性检验。训练均值和尺度只用于描述性标准化，不能反馈修改实际采样坐标。

若训练后验重采样可被 decoder 读出而 DDPM 生成后验较差，结果与 latent 匹配瓶颈相容，但仍受训练经验分布、posterior 不确定性、有限样本及读出偏置影响，不能单凭该对照宣称唯一因果或证明修改方案有效。直接标准正态参照则测量 decoder 对该坐标的反应，不自动检验完整 flow prior。

所有新增成本、失败和协议范围需另计。不得更改原 frozen 方法、KL、归一化/逆变换、噪声或条件；任何将来新参数化应另版本、各相关臂一致，并单独证明，不隐性 clip。test37 封存，不用 dev 选择索引、随机种子或控制版本。

## 后续回执必须绑定

- 真实最终 checkpoint、train posterior cache、模型/数据来源和完整 prior route 静态证据。
- 每个参照的准确名称、坐标、32固定slot、抽样索引、epsilon/RNG证据、固定decoder/readout、资源预算及失败保留规则。
- 无训练、无新DDPM、无新seed择优、无test、无自动重试；原E/L/G不可评估状态及已有额外成本保持。

独审依据为已封存 CE/MSE 工程证据与固定实现；本文件没有新增模型调用，也不构成可执行 preflight receipt。
