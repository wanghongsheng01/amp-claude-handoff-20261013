# CE 来源替换诊断草案 v1（未授权、未执行）

本草案是在已知 CE 生成负结果与潜在尺度差异之后提出的事后探索。它保持四臂实验及所有旧候选不变；只设计一个新诊断 scope。原 CE E/L/G 永久保留为中断导致不可评估，PROMOTE HOLD。

| 来源 | 输入坐标与路线 | 新 decoder 输入 | 解释边界 |
|---|---|---:|---|
| 已有 DDPM | 复用 recovery 保存的 r、z、raw、tokens、序列；不重算 | 0（引用32） | 原有 terminal-only 路径 |
| B 训练经验后验 | 同端预锁 train row j，z=mu_j+exp(logσ_j)·epsilon；直接 decoder/J | 32 | 训练后验/记忆参照，不是新颖生成成绩 |
| C 联合结构置换 | B 的 decoder-z 在端内逐坐标对16 slots固定排列；直接 decoder/J | 32 | 精确保留各坐标16值的经验边际，改变跨坐标配对；不保证消除一切依赖 |
| D reference-g conditional native prior（待新 scope 允许） | z_p=epsilon，原 prior_flow.inverse(z_p,g_j)→z→同 decoder/J | 32 | 使用与B同索引的训练逐残基与长度模板条件；非 terminal-only，排除主排名 |

旧 A=decoder(N(0,I)) 不执行，也不冒名 native prior。B/C/DDPM 已处于 decoder-z，不能再加 inverse flow。D 是冻结 CE VAE 的条件 prior，不冒称历史 scalar checkpoint。

## 固定随机输入与来源

完整复建 FREE422/AMD423 的原 CPU torch2.2.2 RNG 轨迹：初噪一次、t998..2 共997次同形随机噪声，然后取最终 epsilon；t1不抽样。全程 fork_rng，保存状态与噪声摘要，必须先用 dtype+shape+raw bytes 验证 saved z=mu+epsilon·exp(logσ) 两端逐位一致，不能用残差逆算或只用 torch.equal。

B 使用 NumPy1.26.4 PCG64，FREE523/AMD524，从对应端按row_id排序的126/153行中有放回取16条，保留重复。C 使用独立 PCG64 FREE625/AMD626，固定 channel0..15、position0..47遍历，每坐标一次 permutation(16)，保存完整[16,48,16]索引和随机状态；不重抽固定点，不筛好排列。C 的每坐标原始float32 bit-pattern多重集合必须与B完全相同。共同随机数只提供配对，不增加独立生物重复。

D 从已绑定训练描述符、同B索引和原 train_feature_scaler.json 恢复 g：[16,3,48] float32，实际残基使用带符号 zscore，padding为0，无 abs、fit、encoder或目标/dev/test g。用冻结 tensorize 与独立逐数公式核对 g 的字节、padding及行来源。两端各一次 inverse-flow，再各一次 forward-flow roundtrip，逐元素误差必须 <=1e-5+1e-5·|epsilon|，保存全误差。容差是数值工程检查，不是 prior 质量门；错误g也能 roundtrip，所以 g 来源检查独立存在。

## 调用顺序、预算与失败

先封存 epsilon、索引、B/C z、完整排列和（若启用）g，再构建模型；只载入冻结CE权重。D 的 z 只能在inverse以后得到，因此另在所有 decoder 之前封存第二层输入。固定decoder顺序 B_FREE/B_AMD/C_FREE/C_AMD/D_FREE/D_AMD。上限为96新输入、6次decoder batch、2次inverse-flow batch、2次roundtrip forward-flow batch；0训练、0DDPM、0encoder。D 若在执行前因合法g或范围不获许可，明确NOT_IDENTIFIED；若已选D后运行失败，不能退回B/C-only成功。

候选预算内层120秒、外层180秒，包括导入、绑定校验、权重加载、随机轨迹复建、封存、flow、decoder与全部导出。目前无该新scope真实吞吐保证，不为此额外运行基准。所有异常、失败、中途not_executed与已完成输出完整保留，保存模型/输入前后身份与独立监督wall；不调参、不扩大预算、不补采、不自动重试。

未来评价另开scope，沿冻结evaluator全指标与train279/dev41引用，逐源16+16固定分母，报告失败、有效unique、train-copy、近邻及全MMD等。B/C/D均作为诊断，不能把复制改善当创新，不能据单次结果证明DDPM唯一瓶颈；没有dev latent仍明确缺失。此草案未创建可执行plan或receipt，未执行模型、随机抽样或质量评价。完整参数、来源真实路径和SHA见协议JSON。
