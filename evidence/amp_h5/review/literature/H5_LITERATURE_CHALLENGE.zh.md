# AMP H5：有限文献核查与独立科学挑战

2026-10-11；审阅者 `/root/h5_literature`。只读 H0–H4、设计快照和公开一手来源；没有训练、采样、读取 test37、运行外部代码或修改旧产物。本意见不是运行回执，也不判断其他代理的进程状态。

**结论：继续 CE21 / 同容量 raw MSE21 的完整 VAE→自身重编码→自身 DDPM 对照，并加入 fresh scalar 与一个完整 terminal-only LSTM，方向合理；当前贡献只能定位为端到端目标修复实验，不能称已具有算法新颖性或强基线优势。** train279 / dev41 足以运行一次有限、可证伪的本地 pilot；不足以在单种子、反复使用的 dev 上确认泛化、生物效能或最先进性能。无需等作者新文件才能完成本轮。

## 核实的先例及适用边界

| 一手来源 | 本轮实际核实 | 对 H5 的直接含义 |
|---|---|---|
| [Bowman 2016](https://aclanthology.org/K16-1002.pdf)，正式论文 §2–3、Fig.1 | LSTM 序列 VAE、类别交叉熵、EOS，以及强 decoder 忽略 latent 的优化风险。 | CE/EOS/VAE 均是成熟机制。该语言任务的 collapse 经验是风险依据，不能诊断本 AMP 模型已经 collapse。 |
| [PepVAE 2021](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2021.725727/full)，Methods | 19 AA＋end＋padding；排除 C 和长度≥40；LSTM VAE；围绕参考肽相关 latent 邻域采样。作者称模型可索取。 | 它支持离散残基与终止建模，不支持照搬其筛选、MIC 条件或宣称准确复现。尚无本轮核实的完整官方生成器版本。 |
| [Müller 2018 作者代码](https://github.com/alexarnimueller/LSTM_peptides) 与 [实现](https://raw.githubusercontent.com/alexarnimueller/LSTM_peptides/master/LSTM_peptides.py) | shifted next-token、LSTM/GRU、类别 CE；[README](https://raw.githubusercontent.com/alexarnimueller/LSTM_peptides/master/README.md) 有 2×64 小模型示例。采样代码会丢弃过短结果，训练可内部划分验证集。 | 2×64 LSTM 是有来源、适合本轮的小型完整基线起点；terminal embedding、固定 split 与全 attempts 保留是本项目适配。不能照抄其短样过滤，不能称官方原样复现。 |
| [PRO-LDM 2025](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202502723)，§2.1 与训练方法；[作者仓库](https://github.com/AzusaXuan/PRO-LDM) | transformer/CNN 自编码器、平行 fitness regressor，以及对 latent z 的条件 DDPM；无标签时可无条件训练。 | “AE＋条件 latent diffusion”已经公开。改为 terminal 条件不是新范式。其论文大规模训练预算不能直接拿来代表 Mac/279 条件下重现可行。 |
| [AMP-Diffusion 作者仓库](https://github.com/programmablebio/amp-diffusion)，[公开 checkpoint 说明](https://raw.githubusercontent.com/programmablebio/amp-diffusion/main/note/README.md) | ESM-2 潜空间扩散。当前公开小版说明为 16.54M 参数、19,670 AMP 训练样本，并提供显式生成长度参数。 | 是 AMP 场景的直接近邻。其预训练/外部数据与显式长度输入不等于 H5 的从头 terminal-only 任务；不能原封不动加入同信息排名。其公开预测 MIC 表不是本队列实测。 |
| [AMPGen 2025](https://pubmed.ncbi.nlm.nih.gov/40447738/)，作者摘要与图注 | MSA 条件的顺序无关自回归 diffusion，继以理化筛选、XGBoost discriminator 和 LSTM scorer；生成器含外部预训练。 | 其额外条件和筛选链与本轮 c-only 无过滤任务不同。LSTM scorer 不是完整 AR 生成基线。本文未获取完整 Methods，不把图注当完整复现配置。 |

检索为围绕六个方法的有限核实，不是系统综述或穷尽新颖性检索。无关搜索命中未用作证据；Nature/Cell 全文部分请求失败，未绕过访问限制。源码网页是 mutable master/main，未验证 immutable commit 或下载字节哈希；H5 自有实现仍须独立冻结源码。

## 必须修正的一处文献解释：EOS49 不是所有模型的硬要求

旧 `design_snapshot/literature/PRECEDENTS.zh.md` 把“48 AA 必须再给第49个 EOS target”写成通用要求，过强。以下是本审阅者的数学推导，不是上述论文的原始结论。

1. **显式终止、可变长度模型**：每条序列概率包含最后 EOS；48 AA 的训练对象确实需要第49个 target。若采样最多48次，却把无 EOS 的48 AA 称“模型自然终止”，训练/报告不一致。
2. **明确有界的模型**：最长48 AA，第48个 AA 后按模型定义强制停止。L<48 时训练残基＋一次 EOS；L=48 时只训练48个残基。其概率为 `prod_t<=L p(a_t|prefix,c) * p(EOS|a_1:L,c)`（L<48），以及 `prod_t<=48 p(a_t|prefix,c)`（L=48）。这是有限生成树上的规范化分布，无需 EOS49。

两者都可以运行；选择后须固定训练/采样/统计合同。旧 AR 草稿采用第二种，不能单凭缺 EOS49 判它错误。PAD 与 EOS 可以在存储中共用编号，只要 first-stop 之后不计 loss、PAD 不进入活跃生成轨迹、BOS 不作输出、mask 和终止定义无歧义。AR 的48上限输出需单列 `cap_hit`，不能与自然 EOS 终止混称；其他臂的 full48 同样单列。短串、空串和 cap-hit 均保留，不补采。

固定48槽 VAE 的 CE＋J 是另一合同：CE 在20 AA＋PAD的固定格上训练，J 在合法 AA 前缀/PAD 后缀中作受限 MAP。它不是 AR 的 EOS 因子分解；在不改变目标的本轮不要另加长度头、长度先验或 EOS49 后再把收益单归因 CE。

## 四个完整臂应该回答三个不同问题

| 比较 | 可以回答 | 不能混入的主张 |
|---|---|---|
| CE21 vs raw one-hot MSE21，同上游/同21头/同 J，各自新 DDPM | 固定初始化、归一化、β、优化器、预算下，端到端目标形式的差异是否传到生成 | 同β不是等有效正则；不同损失裸值不可排名；不能宣称 CE 对所有尺度/β 普遍优越 |
| CE21 vs fresh scalar 完整管线 | 本轮声明预算和原方法复现/适配合同下的整体修复效果 | 包含头容量与读出改变，不能单归因 CE。fresh scalar 若改了原 loss 归一化/warmup，须称适配版并列差异 |
| CE21＋DDPM vs terminal-only LSTM | 当前任务下复杂 latent 管线相对直接生成器是否有可见收益 | 参数相等不是必要条件；不可用重建 accuracy 或 VAE/DDPM/AR 互异训练 loss 横比生成质量 |

旧快照 `baseline_ar/CONFIG_DRAFT.json` 实际为 GRU128（68,709参数）。父任务本轮要求 LSTM，应在新协议中选择且一致命名；不能拿 GRU 运行后称 LSTM。推荐只选一套有作者小模型示例依据的 LSTM（如2×64），以父任务实测吞吐和正式协议为最终准绳；不再并跑 GRU、分层双模型或多温度择优。

## 致命公平性问题与运行前检查

以下任一实际发生，会破坏相应比较；本意见列的是阻断条件，并不声称作者已经犯错。

- **生成信息越界**：任何臂输入真实逐残基 g、全序列 h、目标长度、dev/posterior 对象或参考肽。生成公开接口只应收 c、随机数、已训练参数；训练 encoder 读取自己的 train 序列是合法的后验学习。
- **新表示接旧 DDPM**：CE、MSE、fresh scalar 都须用各自 encoder 的 train 后验重新拟合自己的 DDPM；旧 checkpoint 只能历史锚点。相同数据 split 不等于相同 latent 分布。
- **预算给法制造弱 AR**：VAE＋train 重编码＋DDPM 的合计成本与完整 AR 比。仅每个组件都1000步，或让 AR 只用一小段时间后声称同资源，均不成立。固定步数用于 CE/MSE 机制控制；共同成本上限用于完整方法比较，这两个问题要分开报告。
- **未训练充分却声称强基线已败**：本轮至少记录 AR train/dev NLL 轨迹、达到的更新/数据曝光、实际学习率与停止原因。固定终点无改善或仍高速下降时，只能报告“该预算内的基线表现”，不能把不足收敛解释为模型本身弱。若允许 early stopping/checkpoint selection，所有臂的选择机会须事前有共同规则。
- **归一化与单组件归因不相容**：CE/MSE 的共同 K 求和域、位置分母、β、warmup、clip 要同；raw MSE 对21坐标求和后除 B×48，不能一臂多除21。CE相对fresh scalar涉及多项差异，应列改变清单。
- **吞掉失败**：合法集合大小变化、空串/过短、cap-hit、重复与 exact-train-copy 全报；MMD若只算合法集合，必须同时展示分母，不能把高 invalid 率清理后称好分布。
- **先看结果再放宽门**：主候选、终点、种子、attempts、指标方向和失败条件先冻结。不同模型同 seed 编号不构成共享 latent 配对实验。

推荐保留预锁 MMD/组成/长度/多样性和记忆保护项，但它们是 proxy。每个terminal仅16次请求、dev41已复用时，经验门PASS也不是统计优越性或真实活性证明。应保留全指标与所有负结果；不能只用MMD或二肽JS之一统称生成质量全面改善。

## 哪些可写，哪些仍是待证研究问题

**标准修复**：categorical CE、合法前缀投影、EOS/cap、terminal embedding、重新训练匹配DDPM、可审计全 attempts 和公平成本记录。它们可以形成可复核的修复证据与负结果，但不应包装成原创组件。

**值得检验但本轮尚未成立的贡献方向**：离散输出目标与合法支持集的失配，是否在端到端表示学习和二阶段生成中产生可重复、可定位的损害；以及后验参数扩散相对直接 latent 扩散究竟提供什么可测益处。后一问题本轮四臂没有隔离，不能因扩散对象不同于PRO-LDM就宣称创新有效。修复后若仍输给完整AR，应该直接承认当前复杂度尚无支持，并停止把局部probe收益升级成方法胜利。

本轮不增加新模型分支或大规模预训练baseline；先完成已授权四臂。没有新颖性、独立泛化与公平优势证据前，不称可投。未知 wetlab N 不推算，旧27肽实验不能冒用为新H5前瞻验证。
