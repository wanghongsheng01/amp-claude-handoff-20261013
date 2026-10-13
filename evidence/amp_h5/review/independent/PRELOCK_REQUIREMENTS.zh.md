# AMP H5 独立预审：必须锁定项

身份已核：这是抗菌肽 AMP 的 H5，旧 H0–H4、稿件与 test37 保持冻结。本报告仅静态读源码/协议；不是运行批准。源文件身份见 `INITIAL_SOURCE_BINDINGS.json`。

## 正式四臂合同

1. **任务**：train279/dev41、末端 FREE/AMD 元数据和最大 48 残基固定。生成 API 只接受 terminal 与独立随机流；不能取得源对象序列、真长度、逐残基 g、全局 h、近邻模板或 dev 对象。训练中原 VAE 对同一序列派生 g/h 可保留，须披露。
2. **臂**：CE21 为预选主候选；raw one-hot MSE21 为同容量控制；fresh scalar 为原方法新训参照；单一规范 terminal-only LSTM 为可执行竞争方法。旧 H4 不能充当本轮匹配 fresh scalar。若 scalar 沿用原目标和 first-PAD，CE 相对 scalar 的差异包括编码、容量、目标尺度与解码，不能单归因 CE。
3. **公式**：CE/MSE 使用相同完整 21 头/上游初值、相同 KL、相同全 48 位置分母。MSE 锁为 `0.5 * sum_classes(square(raw_score-onehot))`，不 softmax、不额外除 21。旧意见有缺 0.5 的文本冲突，应以新协议和实际代码为唯一运行合同。
4. **KL**：亲核原 sampled density sum 域与旧 `/16` 的含义；新两臂若使用 `D/(B*48)`，必须只转换一次，保留梯度及负 Monte Carlo 值。与旧 scalar KL/非PAD分母的差异明确记录。共同 beta 不等于相同有效正则强度。
5. **解码**：CE/MSE 共享真值无关 J；长度支持 0…48，精确同分取最短、AA 同分取最低索引。空串、短串、full48 全保留。若 scalar 保留原 round/clamp/first-PAD，写明完整方法差异。不得看结果后禁 PAD 或加最低长度。
6. **LSTM**：BOS 仅作输入、teacher forcing 严格右移；输出为 20AA+PAD/EOS。短序列只监督首 EOS、后续 PAD 遮蔽；48AA 事件没有第49位 EOS。生成使用自己的前缀、temperature=1、抽到 EOS 即停止；不拒采补样。新 2×64 LSTM 必须重新验证参数计数，旧快照是单层 GRU，不能把其已有单元测试当新结构已通过。
7. **阶段身份**：每个 fresh VAE 的最终参数生成其自己的 posterior cache；每臂 fresh DDPM 绑定该缓存。不得共享旧/另一臂 DDPM。VAE→DDPM 是两阶段训练，不能称联合 DDPM 梯度端到端优化。
8. **随机与初始化**：相同实际模型 state hash 而非仅相同 seed；optimizer 独立。batch 顺序、posterior epsilon、DDPM t/noise/drop、验证和生成随机流明确。验证不得改变训练 RNG。跨模型共同 seed 不称同 latent。
9. **预算与机会**：总训练成本含 VAE、重编码、DDPM、所有新拟合；AR 获得明确且可辩护的总机会。固定更新计划与实际 compute 分列；相同步数或相同最高 cap 不自动等于等成本。只一配置、固定末 checkpoint，不按 dev 选温度/seed/长度/读出或延训。
10. **终点与失败**：工程门、损失比较、完整生成比较分别锁。所有 attempt 保留，cap/数值故障保留失败状态及未执行 slots，不能删除或补采。单训练 seed/每 terminal 小样本/反复开发 dev 只支持 pilot 描述；不提供活性、毒性、新颖性或可投稿证据。

## Smoke 专项

根据 root 预告，smoke 仅验证 finite 与吞吐：三个 VAE 臂各 20 VAE+20 DDPM 更新，AR100；每法 FREE1+AMD1，权重全部弃用，不参加 formal，也不依结果质量筛选臂。需要协议将 train 180s、generation 450s 的归属写清（每阶段/每法），并定义在途调用完成与 cap 超限记录。smoke 不读取 dev 自然样本作模型输入或指标；如打开 dev 文件计算 hash，应诚实称身份校验读取，不称零文件读取。更简单是引用已有冻结 manifest 的 dev hash。

## 可复用评估器及限制

- `evaluation/frozen_tools/v1/amp_evaluate.py` 是 H4 真正 hash 绑定的指标底座，SHA-256 `f65eec16391d98a382d415e7b879f99555a33b33398f8be704661d1c7c027480`。`evaluate_group` 保留全请求分母，分布/近邻在 named legal cohort 上计算；421 特征、JS/L1/KS、唯一性、熵、exact copy、NW均可复用。
- `evaluation/terminal_v2/evaluate_terminal.py` 的 SHA-256 是 `f6d72248b8b28c652c175443d16e33a66c88bc200469253c9c71bfc88c563748`。其 `dataset` 校验 molecule/sequence hash、role、terminal/class、长度与去重，`by_terminal` 可复用。
- v2 CLI 的 `samples()` 用 `NEURAL_INDEX` 硬编码三个历史方法；不能直接传 CE/MSE/scalar/AR 名，也不能把 AR 假名为旧 DDPM。应在 H5 隔离 wrapper 中新增显式 H5 slot/seed 校验，仅复用已审公式。
- v2 输出里的 flow 信息不对等警告是历史专用文案，H5 wrapper 应写自己的信息与成本合同。不能把文案删除当公平已成立。
- same-terminal copy/NN 与 any-train-terminal copy/NN 均须保留；`valid_1_48` 分布集合在模型间不同时，应说明成员变化。MMD 只表示固定低阶特征均值距离，不能当活性或生成逐位 accuracy。
- 独立合成 NW/entropy oracle：`evaluation/independent_review/independent_oracles.py`。其 NW 默认分数与生产 evaluator 不同，调用时须明确传生产 match=2、mismatch=-1、gap=-2，并按生产 tie 规则检查，不可盲目默认值比较。
- 冻结 H4 输入绑定文件：`diagnostics_h4_h2_generation_bridge_v1/EVALUATION_INPUTS.json`。本次读取的是清单，尚未打开自然 train/val CSV 或 test。

独立审阅者不修改实现、不启动自然训练/生成、不读取 test。后续 receipt 只批准被源码/协议 hash 绑定的具体阶段，不把工程 PASS 提升成科学优势。
