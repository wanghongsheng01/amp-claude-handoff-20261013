# Reference-g native-prior 诊断可行性判断

**判断：有条件可行，限额外诊断；当前未授权执行。** 公开使用预锁的训练 reference-g 后，可以诊断冻结 CE VAE 的原架构 learned conditional-prior 路径，不冒称 terminal-only 生成。应先完成四臂独立科学复核，再登记新 scope、精确输入/指标/失败条件及预算，独审最终代码与计划。

D 应严格为 `z_p=epsilon → CE.vae.prior_flow(z_p,g_j,reverse=True) → CE decoder(z) → 冻结 J`，g_j 使用与 B 相同的训练索引。名称为“冻结 CE VAE 的 reference-g conditional native-prior 诊断”，不得误称历史原 VAE checkpoint，也不能用 decoder(N(0,I)) 替代。原 flow 是 mean-only coupling；实际新代码仍须核验 reverse 参数、完整 block 顺序、张量形状和冻结权重。

g_j 必须来自已绑定训练逐残基描述数据，使用原训练 scaler、零填充和 tensorization，不重新 fit、不取目标真实序列条件、不造 c→g 映射。g 带有训练残基和长度模板信息，明显多于 terminal-only 条件；即使生成新序列，也不能称为独立新肽分布、无泄漏证明或主任务的公平候选。诊断应保留 reference row ID、来源与条件说明，不参与主排名或 PROMOTE。

B=`mu_j+exp(logsigma_j)*epsilon` 和 C 都已在 decoder-z 空间，直接同一 decoder/J，不加 inverse flow。C 对 B 已采样 z 在 terminal 内逐坐标跨16 slot 固定排列；这精确保留每坐标的16个经验值，同时改变跨坐标配对。必须先保存全部 permutation，不能依据结果挑排列。B→C 最多表明这个训练经验后验参照对联合组合的敏感性；不能排除其他分布差别，也不证明 DDPM 是唯一因果瓶颈。B↔D 同训练索引可对照后验与条件 prior 路径，但 B 还承载编码器吸收的 x/h 信息，不能把差异全归因于 flow 或某单一尺度。

计划边界可限定为：复用已有 DDPM32；B/C/D 各32，共96个新 decoder 输入，按每端16分批仅6次 decoder batch call；仅 D 的2次 inverse-flow batch call；零 encoder、零 DDPM、零训练，无追加种子或补采。所有索引、epsilon、g、B/C输入及排列应先封存并做字节/形状/来源核对，再进行任何模型 forward。D 输入 epsilon/g 必须先封存；D 输出 z 只能在其 inverse flow 后保存。共享 epsilon 是共同随机数配对，不是独立实验重复。

内层120秒、外层180秒可作为这项小规模诊断的有限候选预算，尚无实际吞吐保证；代码导入/数据核验/权重载入/输入封存/flow/decoder/导出都计入内层。超限或数值异常保留全部96槽、已完成输入及失败信息，不自动改参数、扩大预算或重新采样。任何输入不能合法复现或 g 路径不符，D 记不可评估；不能退回标准正态直解码并冒名 native-prior。

本判断没有模型调用或新随机样本。原 CE E/L/G 仍不可评估、PROMOTE HOLD；旧 A/B64 候选继续不执行。允许准备研究设计不等于批准新模型作业。
