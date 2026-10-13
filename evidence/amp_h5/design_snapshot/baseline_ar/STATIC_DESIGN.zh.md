# H5 terminal-only 小型 AR 基线（静态待锁定）

这里实现规范的小型自回归基线：输入仅为终端类别、BOS 与生成到当前的自身前缀。训练中的 teacher forcing 使用 train 序列；它不是生成时读取整条真实序列的许可。没有真实长度或逐残基性质接口。FREE=0、AMD=1 与冻结数据合同一致。

结构固定为 token embedding 22×32、terminal embedding 2×8、单层 GRU（输入40、hidden128）、线性21类输出，dropout=0。参数数为704+16+65280+2709=68709。GRU 隐状态从零开始；终端 embedding 在每个位置拼接。成熟 AR 原理的主来源由文献任务统一核验；这不是声称严格复现某一官方 LSTM_peptides 实现。

20AA 的顺序为 `ARNDCEQGHILKMFPSTWYV`，PAD/EOS=20，BOS=21且仅输入。L<48 的训练目标为 L 个残基与一个首次 PAD/EOS；之后的 PAD 不计损失。L=48 只有48个残基目标，不另监督 EOS49。因此48长度在 capped 分布中的似然不需要额外乘一个 EOS49。采样最多48步，允许48AA无PAD；空、短、重复及其他所有请求均保留。温度固定1，无样本拒绝或补采。

目前提供 `ARConfig`、`TerminalGRU`、`make_teacher_forcing_batch`、`sequence_nll`、`sample_batch`，以及 hash-bound `read_split`/`assert_split_isolation`。采样接口显式要求 labels、requests、seed，没有会扩容生成预算的默认值。每步为整个固定 batch 抽样，包括已终止行的吸收状态；保存 raw draws、实际 tokens、21维 logits、active-before mask、结束状态与 RNG state，使随机消耗规则可核。采样使用独立 CPU `torch.Generator`。

合成单元测试已通过38项：覆盖移位/掩码、PAD终止、48上限、空短样本保留、预测因果性、并行teacher-forcing与逐步路径一致、masked梯度为零、train/dev序列/对象/连通分量隔离，以及test角色/路径在读取前拒绝。测试仅使用合成数据，实际GRU只做合成前向，未做任何模型optimizer step，未读取真实数据。该结果支持实现合同，不支持生成性能。

建议优化器为 Adam lr=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，梯度全局L2 clip=1，batch32，masked token mean CE。以上优化预算仍待root统一锁定，不能先行真实训练；固定预算最终状态为唯一 checkpoint，无val温度、seed或checkpoint筛选。主协议应记录真实更新数、累计AA/EOS tokens、参数数、初始化/RNG、训练/生成时间及所有失败。不同模型只可称同terminal条件/同请求数；同更新数不等于同计算量。

最初96/方法及三个采样seed仅为讨论；root随后明确未锁定且禁止执行。当前较严格候选是三个方法各32请求（每终端16），总96；最终由root预注册，当前均未执行。

主张状态：可审计AR合同=已有合成验证但待独审；真实吞吐/过拟合=PLANNED；train拟合、开发质量、与候选对比=PLANNED；强方法优势/活性=未证。BACKUP_BLOCKED。
