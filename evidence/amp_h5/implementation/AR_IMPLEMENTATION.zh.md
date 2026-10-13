# AMP H5：terminal-conditioned LSTM baseline

唯一配置：token embedding 22×32，terminal embedding 2×8；每步拼接后进入 2 层、每层 64 单元 LSTM，dropout=0，线性输出 21 类。总参数 **62,501**。初始化由显式 seed 控制；隐藏状态为零。Adam(lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0)，全局梯度 L2 clip=1，batch32，每epoch独立生成器 shuffle，最后不足32的批保留。

作者仓库 [LSTM_peptides README](https://raw.githubusercontent.com/alexarnimueller/LSTM_peptides/master/README.md) 的示例提供两层×64单元的规模锚点；[作者实现](https://raw.githubusercontent.com/alexarnimueller/LSTM_peptides/master/LSTM_peptides.py) 使用 shifted next-token、LSTM/GRU、categorical CE 与逐步温度采样。本实现选择 LSTM，仅借鉴这些成熟机制；新增 terminal embedding、冻结分组划分、无输出拒采和本地 PyTorch 实现，不声称忠实复现 Müller et al., JCIM 2018, DOI 10.1021/acs.jcim.7b00414。源为2026-10-11阅读的 mutable master，未声明固定官方commit、出版社全文重现或官方代码运行。

与快照的一层128 GRU相比，这是事前单配置选择，不是观察训练结果后的替换。所有 snapshot 原文件保持不变。两层64并不使其成为经过充分调参的强baseline；正式1000更新约111个数据epoch是固定小模型参照，不能因wall cap更长无限追加训练。相同wall上限只定义资源包络，实际cost/quality仍需逐方法报告；不能称完全等计算。

## 概率与条件

AA字母表 `ARNDCEQGHILKMFPSTWYV`，PAD/首次EOS=20，BOS=21且仅输入。L<48 的目标为L个AA+首EOS；L=48只观察48AA，不强加EOS49。loss是所有观察token的平均CE。此为合法capped48分布：输出支持0..48，48AA输出是被上限截停、未观察EOS；empty与1..4长度保留。训练使用过去真值teacher forcing，生成仅使用terminal、BOS、自身前缀和显式随机数。没有真实长度、逐残基g、全序列h、参考序列或检索输入。

每terminal使用独立CPU生成器，温度1；每一步从当前整个terminal batch抽样，含已终止行，并用吸收EOS覆盖终止行的实际输出。保存raw_draws与active_before，便于区分RNG消耗和有效token。最后未执行位置的logits/probabilities均为0，不能当作已评价概率；`evaluated_steps`和mask是解释依据。实际概率全部保留，未筛选seed、温度或输出。

## API 与 root 调用

```python
from ar_pipeline import read_split, run_train, generate, teacher_forcing_evaluation

# p来自root已经冻结的protocol，不在本模块设置任何隐式数据路径。
train = read_split(p['train']['path'], role='train',
                   expected_sha256=p['train']['sha256'], expected_rows=279)
model, run = run_train(180.0, 100, 222, smoke_out / 'train',
                       train_bundle=train, shuffle_seed=222)
# 先检查run.status与finite；FAILED或未达到预定更新不能当作完整训练。
rows, generation = generate(model, {'FREE':422,'AMD':423}, smoke_out / 'generation',
                             smoke=True, cap_seconds=450.0)
```

正式调用 `run_train(cap_seconds, 1000, init_seed, out, train_bundle=train, shuffle_seed=...)`；传入 `dev_bundle=dev, monitor_every=100` 才开启最多10个train/dev NLL记录点，无best-dev保存、早停、学习率调度或参数搜索。监测消耗计入总训练cap；到cap不再启动新监测。默认smoke不传dev。独立终点评价调用 `teacher_forcing_evaluation(model, bundle)`，返回 `nll_sum`, `token_count`, `mean_nll`、AA/EOS分量与准确率；它是AR自身似然描述，不与VAE目标总数值比较。

正式 `generate(model, {'FREE':seed0,'AMD':seed1}, out)` 强制FREE16→AMD16；`smoke=True`唯一例外为FREE1→AMD1。返回 `(rows, summary)`，同时写 `samples.csv`、`samples.jsonl`、`raw_sampling.pt`、`GENERATION.json`。共同schema含 `sample_id,method='terminal_LSTM',terminal,sequence,status`，同时含同义`attempt_id,terminal_state`与长度、seed、EOS/cap标记。失败不替换请求：`error`, `partial_capped`, `not_executed`保留，`complete_attempt=False`；这些部分前缀不可冒充完整合法生成。

训练cap从setup前开始，在更新边界检查；生成cap在每terminal及每token前检查。仅在途操作和末点保存可越过软cap，记录actual/overrun。训练保存initial/final模型、优化器、shuffle RNG/order/cursor/epoch、每步行索引、梯度、loss、tokens与时间；没有恢复训练/重试接口。调用者还需检查同机是否已有活跃H5进程，并遵守串行root START。

## 验证和边界

`AR_SYNTHETIC_RESULT.json`是52项合成合同检查：prefix因果性、teacher-forcing/逐步一致性、mask梯度、EOS/cap、概率、RNG、失败/超时slot、hash与test路径拒绝、保存回读，以及可选monitor行为。仅合成102个optimizer更新；没有实际train/dev/test文件读取，未启动真实训练/生成或性能评价。它不是独立审阅，也不证明收敛或性能。

独审建议：核62,501参数；核训练token与EOS分母；从raw概率重放抽样；逐slot核样本数/seed/失败；核初末权重与freeze输入hash；核无dev选点/调参及test访问；单列wall cap与实际cost；将过拟合、未收敛或小样本证据限制留在最终结论。原H0–H4、湿实验与稿件不改，新计算不是旧湿实验的前瞻验证。无上传/提交，BACKUP_BLOCKED。
