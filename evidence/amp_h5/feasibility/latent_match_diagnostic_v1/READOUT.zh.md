# AMP H5：已保存 latent 匹配的作者只读诊断

本诊断是在已知生成负结果及已有尺度异常之后启动的事后探索。运行前冻结分析项仅用于约束改口和挑样，不是原实验的预注册确认性证据；原冻结协议保持原hash，明确补充见 protocol/LATENT_MATCH_DIAGNOSTIC_v1_POSTHOC_ADDENDUM.json。

本次证据支持一个具体定位：CE 已完成生成的后验参数在尾部及位置/通道结构上偏离训练编码后验，AMD 分组更明显；全局有限性或单一 RMS 不能概括这种偏移。它尚不能证明 DDPM 是生成未改善的唯一原因，因为现存重建导出只测了 z=mu，未测训练后验采样后的 decoder 行为。

先冻结 protocol/LATENT_MATCH_DIAGNOSTIC_v1.json（SHA256 c616a91f45f774fd1de47e7cd4675cea39f9bf3bee2abf39d6c597b24885b8f1），再用 NumPy/stdlib 全量计算。CE 使用原 formal_v2/categorical_ce 的训练 cache 和独立恢复 attempt 的生成；MSE 使用已完成的原单臂产物。24 项输入绑定在计算前后均未改变。零模型导入/权重反序列化/forward/训练/随机抽样，无 test/cohort 源读取，也未读取正在运行的 scalar。

每方法训练279条：FREE126、AMD153；每方法生成32条：每端16。每条潜变量有16通道×48位置。下表的比例是已保存坐标的描述性计数，坐标与位置存在依赖，不能把12,288个坐标当作12,288个独立样本。每端只有一个固定模型/种子批次，不做显著性或独立重复声明。

|方法/端基|logσ 标准差 生成/训练|μ 超同坐标训练 min–max|logσ 超同坐标训练 min–max|生成 z RMS / 训练 q(z) 解析 RMS|
|---|---:|---:|---:|---:|
|categorical_ce/FREE|1.606|15.67%|14.68%|0.939|
|categorical_ce/AMD|5.485|32.51%|29.18%|1.086|
|raw_onehot_mse/FREE|2.211|26.94%|22.74%|1.013|
|raw_onehot_mse/AMD|4.566|37.75%|30.75%|1.038|

训练 q(z) 仅使用 E[z]=mu、E[z²]=mu²+sigma² 的解析混合矩；没有新抽样，也没有伪造训练 z 的经验分位数。每条样本、完整分位数、范围、RMS、16通道/48位置 profile、16×48均值/标准差图及所有覆盖计数都保存在 LATENT_MATCH_READOUT.json。上述训练范围只是有限经验参考，不是合法潜空间边界；越界比例不能直接解释为失败率或统计拒绝。

CE AMD 的 mu 均值从训练0.02980到生成−0.19142；logσ标准差从0.03552到0.19483。sigma训练范围0.7843–1.2480，生成范围0.1108–5.4372。与此同时，sigma平均值仍接近1，z RMS由解析训练参考1.5701到生成1.7052。因此“都有限且总体RMS接近”不能支持后验匹配。

结构证据也并非只有幅度变化：CE AMD 训练/生成 mu 均值图相关0.9564，但 logσ均值图相关只有0.3157；按48位置聚合的 mu–logσ相关从0.8524变成0.0357。其全坐标相关仍接近0，去除每条均值后的相关也相近。这说明单个全局相关系数同样会遮蔽位置结构；这些都是描述性结果，未将相关改变作为因果证明。FREE 的偏移较小但并非零。MSE同样出现尾部偏移，因此当前证据不是 CE 独有 bug 的定位，也不构成任何方法生成质量的排名。

|方法|均值重建 train 微准确率|均值重建 dev 微准确率|train/dev 长度MAE|train/dev 完全重建条数|
|---|---:|---:|---:|---:|
|categorical_ce|73.54% (3580/4868)|70.90% (497/701)|0.0681 / 0.0000|21/279 / 3/41|
|raw_onehot_mse|12.02% (585/4868)|12.55% (88/701)|4.5806 / 5.5610|0/279 / 0/41|

这些是已有 CSV 的确定性 z=mu 重建参考，不是新肽生成成绩。源码 reconstruction_export 在592行说明 deterministic mean，并在598行调用 decode(values[...,0])。当前文件没有 sample reconstruction；raw/visible token两列只是同一次解码的原始/可见表示。CE训练 sigma 约1，均值点可读不保证整个 q(z|x)邻域可读，现有材料不足以测出这个差距。

随机来源边界：generated_*.npz保存 posterior_parameters、z、raw、tokens、lengths、length_scores，未单独保存epsilon。反算 (z−mu)/exp(logσ)只是带浮点误差的派生残差。本报告仅做其描述：FREE残差RMS约1.00525、AMD约0.99420；CE/MSE残差最大绝对差分别4.54e−7、8.51e−7，与共同随机流相容，但不能据此证明原epsilon逐位相等或统计独立。原独审已使用seed422/423的完整CPU随机消费序列核对过重参数化；本次没有复建任何RNG。

最小后续对照仅写入 CE_DECODER_INPUT_CONTROL_DRAFT.json，尚未授权执行。待scalar冻结队列终态并完成独审后，另定scope：CE固定32 paired slots（FREE16/AMD16）×2分支，共64输入、4次batch16 decoder forward，不跑DDPM或训练。未来先按原seed轨迹重建并保存真实epsilon及RNG/hash证据，先逐位复核保存的z；禁止用反算残差替代。另用固定PCG64 selector seeds523/524按端基有放回抽训练行，各16条，索引及状态必须在任何decoder输出前封存，重复保留。

分支A直接令z=epsilon，是decoder坐标的独立标准正态参照；端基仅是固定分组/种子标签，A本身不含端基条件。它不是原模型native flow prior：原先验密度作用在zp=prior_flow(z,g)，不能借真实逐残基g或临时零g来假装可生成的逆flow路线。分支B令z=mu_j+exp(logσ_j)epsilon，是训练经验后验/记忆参照，绝不记为新颖生成成功。两分支共享epsilon是配对共同随机数，不是额外独立重复。

现有文件能回答：保存的训练/生成参数及z是否有限、端基尺度与结构差异、确定性mean重建、原随机流来源。不能回答：独立标准正态输入的decoder行为、固定训练后验样本的decoder行为、完整native prior可生成路线、实际抗菌活性或未来泛化。前两项须队列结束后获得单独scope再执行；其余不由本诊断解决。

全样本保留，没有重新采样挑选、没有调KL/结构/数据/读出/超参、没有根据质量变更冻结scalar计划。原CE的E/L/G仍为NOT_EVALUABLE_INTERRUPTION；恢复生成及未来任何控制单独命名和计成本。PROMOTE保持HOLD。

复核入口：LATENT_MATCH_READOUT.json（全量）；COMPACT_READOUT.json（表格及分组摘要）；CE_DECODER_INPUT_CONTROL_DRAFT.json（非执行草案）；implementation/latent_match_saved_arrays_v1.py（无模型/无随机调用的计算脚本）。

静态路线证据另见 review/latent_route_static_v1/LATENT_ROUTE_STATIC_AUDIT.json（SHA256 cddbc4a3ee25926f6654afabfeef8448b77adc3d4620b0e9664cb60b6d2f1096）：已查当前路径的轴、logσ、尺度、EMA、CFG及flow先验坐标，未识别当前执行路径的坐标/逆变换实现错误；这不是无bug证明，也没有把观察到的不匹配归因于实现错误。未来RNG重建绑定CPU torch2.2.2；训练行selector绑定NumPy1.26.4/PCG64，版本不符须在任何抽样或forward前停止。
