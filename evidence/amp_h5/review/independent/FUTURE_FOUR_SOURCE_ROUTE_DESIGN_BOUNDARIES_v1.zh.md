# AMP H5 后续四来源诊断的路径边界（设计意见，未授权执行）

本意见仅针对四臂完成并经独立审查之后，另行预登记的有限来源替换诊断。当前 A/B64 候选即使技术修复通过，也不获得执行许可。不得改写旧实验、原 CE 中断状态或 E/L/G；测试集继续封存。

## 正确坐标与条件

| 来源 | 实际坐标及合法路径 | 可支持的结论边界 |
|---|---|---|
| 已存 DDPM | terminal-only 条件下生成 (μ,logσ)，以固定 ε 形成 decoder-z，直接 decoder(z) | 保留真实已存32请求；是原候选路径 |
| 配对训练后验 | 固定 terminal 内训练行索引，以同 ε 形成 μᵢ+exp(logσᵢ)ε，直接 decoder(z) | 经验训练后验参照；含模型已见训练信息，不计新颖生成或候选优势 |
| 同索引库逐坐标打散 | 必须事前锁定打散层和排列单位；形成 decoder-z 后直接同 decoder | 联合结构敏感性的诊断，不能作为生成候选 |
| 真正原 VAE prior | zₚ~N(0,I)，随后 prior_flow(zₚ,g,reverse=True)，再 decoder(z) | g 是原逐残基描述条件，terminal-only 当前不可取得；未定义合法 g 机制前，该项在主任务下不可识别 |

“完整链”应指每个来源在原模型中所需的正确变换，不是机械地给所有输入增加 inverse flow。对已有 decoder-z 再逆 flow 会更改其坐标；先正 flow 再逆 flow 只是抵消变换，也不能据此声称同条件的科学对照。直接 decoder(N(0,I)) 只叫 decoder-space standard-normal 诊断，不能称原 VAE prior。

若将来明确使用预锁训练 reference-g，必须称为 empirical/reference-g conditional-prior 诊断：它利用逐残基训练条件，超出 c-only 信息集合。输出可以与训练序列不同，但没有因此建立独立新肽分布、无泄漏或主任务竞争优势；不得暗给目标真实逐残基内容。

## 打散控制必须预锁的单一含义

若要精确保留当前有限样本逐坐标边际，先对固定索引库和固定 ε 得到的 z 矩阵逐坐标跨 slot 排列，然后同 decoder；这保留每列的16个值，却改变跨坐标组合。排列应在 terminal 内固定，保存全部索引，避免排序或依据输出挑选。

若目标是后验参数的联合结构，应以每坐标的 (μ,logσ) 配对为整体跨训练行排列，再用已锁定 ε 采样。这样保留参数边际和同坐标 μ/σ 配对，但不保证采样 z 的有限样本边际完全相同。若分别打散 μ 与 logσ，还会破坏二者配对，形成另一种干预，不能混作唯一联合结构因素。最终协议须选定其中一种，不在看到输出后切换，也不无必要增加分支。

所有分支应固定同一个冻结 CE decoder/J、固定 slot 和噪声、全部输入先封存、权重前后相等；无重训、无追加 seed 择优。共同随机数有利于配对比较，但不是独立训练重复。两种后验输入之间的差异可以定位候选瓶颈，却不单独证明 DDPM 是唯一因果来源；标准正态、打散和 reference-g prior 都不能冒充已调好的强 baseline。

## 同尺寸经验参考

新参考可从冻结224个 sequence-similarity components 联合留出精确 FREE16/AMD16，不拆7个跨端 component。每端生成16和统计留出16都对同一剩余参考 FREE110/AMD137 计算已锁定统计量，禁止混用旧 full-train126/153 范围比例。

这里的留出是统计参考的留出，279行都参与过模型训练；不是模型未见验证集。uniform feasible component-subsets 不等于均匀序列或均匀单组 inclusion；256次重复是同一有限训练库的经验参考，不是256个独立生物重复、显著性检验或 p 值。component 是冻结 NW 相似性图传递闭包，不等于已验证生物家族。若没有保存 dev 后验，逐序列 dev latent 参照应记 MISSING，不能用均值解码字符串替代，也不能暗增 encoder forward。

依据：已封存 review/latent_route_static_v1/LATENT_ROUTE_STATIC_AUDIT.json、review/independent/LATENT_MATCH_DIAGNOSTIC_INDEPENDENT_REVIEW_v1.json，以及 review/latent_comparability_availability_v1/GROUP_BINDING_AVAILABILITY.json。本意见未读取 live scalar 或任何 test37 文件，未调用模型或生成新随机样本。
