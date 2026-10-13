# AMP H5：固定checkpoint恢复的分阶段读出 v4

**原 formal protocol E/L/G：NOT_EVALUABLE / NOT_EVALUABLE / NOT_EVALUABLE，永久保留，不由恢复结果回填。**
恢复工程：**PASS**（后审回执：RECOVERY_ENGINEERING_PASS）。恢复描述性 L/G：**PASS / FAIL**。
**AR_ADEQUACY = NOT_ESTABLISHED；PROMOTE = HOLD。**

评价包已消费JSON清单完整性：**VERIFIED_CONSUMED_JSON**。快照在读取期间未变不代表清单有效；失配包的原始值仅保留供核查，解释与描述性门均不可判。

原训练、恢复生成/重建按不同phase列源；复制训练日志或新增训练更新会使恢复记录检查失败。缺失方法不创建观测行。

| 方法 | training来源 | generation/reconstruction来源 | generation attempt |
| --- | --- | --- | --- |
| CE21_DDPM | /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/runs/formal_v2/categorical_ce | /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/runs/ce_checkpoint_recovery_v1 | ce_checkpoint_recovery_v1 |
| MSE21_DDPM | /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/runs/formal_v2/raw_onehot_mse | /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/runs/formal_v2/raw_onehot_mse | raw_onehot_mse_v2 |
| scalar_DDPM | /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/runs/formal_v2/scalar_legacy | /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/runs/formal_v2/scalar_legacy | scalar_legacy_v2 |
| terminal_LSTM | /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/runs/formal_v1/terminal_lstm | /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/runs/formal_v1/terminal_lstm | formal_v1/terminal_lstm |

| phase | 记录秒数 | 范围 |
| --- | ---: | --- |
| 原CE VAE训练 | 1179.031922 | 原阶段记录 |
| 原CE DDPM训练 | 564.346170 | 原阶段记录 |
| 原CE中断生成 | ≥451.935743（约≥451.936） | 下界；终止时刻、未记录工作量未知 |
| 原CE总wall | UNKNOWN | 不用阶段相加代替未知总wall |
| 新恢复加载/setup | 1.359153 | 新attempt记录，阶段可能含I/O差异 |
| 新恢复生成 | 1115.496170 | 新attempt记录，阶段可能含I/O差异 |
| 新恢复重建导出 | 6.814930 | 新attempt记录，阶段可能含I/O差异 |
| 新恢复进程至final status前 | 1123.995186 | 新attempt记录，阶段可能含I/O差异 |

原32槽位与新32槽位分别保留；不能写成64个已完成序列。恢复是额外attempt及no-retry条款的前瞻修订，不是t200无缝续跑。

| 已有方法 | terminal | 请求 | 合法1–48 | 合法5–48 | unique合法 | copy同端基/任意train | train MMD² | dev MMD² | 解释状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| CE21_DDPM | AMD | 16 | 16 | 15 | 16 | 0/0 | 0.008950 | 0.007710 | AVAILABLE_RECOVERY_DESCRIPTIVE |
| CE21_DDPM | FREE | 16 | 16 | 14 | 16 | 0/0 | 0.017416 | 0.017328 | AVAILABLE_RECOVERY_DESCRIPTIVE |
| MSE21_DDPM | AMD | 16 | 16 | 10 | 16 | 0/0 | 0.092200 | 0.084847 | AVAILABLE_RECOVERY_DESCRIPTIVE |
| MSE21_DDPM | FREE | 16 | 16 | 16 | 16 | 0/0 | 0.079697 | 0.082335 | AVAILABLE_RECOVERY_DESCRIPTIVE |
| scalar_DDPM | AMD | 16 | 16 | 16 | 16 | 0/0 | 0.032898 | 0.031325 | AVAILABLE_RECOVERY_DESCRIPTIVE |
| scalar_DDPM | FREE | 16 | 16 | 16 | 16 | 0/0 | 0.035209 | 0.039882 | AVAILABLE_RECOVERY_DESCRIPTIVE |
| terminal_LSTM | AMD | 16 | 16 | 16 | 16 | 0/0 | 0.003158 | 0.007362 | AVAILABLE_RECOVERY_DESCRIPTIVE |
| terminal_LSTM | FREE | 16 | 16 | 16 | 16 | 0/0 | 0.004092 | 0.011567 | AVAILABLE_RECOVERY_DESCRIPTIVE |

| 已有VAE dev mean | 对象数 | micro | AA支持类macro | sequence mean | 长度MAE | 整串exact | 解释状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CE21_DDPM | 41 | 0.708987 | 0.427601 | 0.722796 | 0.000000 | 3 | AVAILABLE_RECOVERY_DESCRIPTIVE |
| MSE21_DDPM | 41 | 0.125535 | 0.035677 | 0.120047 | 5.560976 | 0 | AVAILABLE_RECOVERY_DESCRIPTIVE |
| scalar_DDPM | 41 | 0.158345 | 0.080473 | 0.147073 | 1.341463 | 0 | AVAILABLE_RECOVERY_DESCRIPTIVE |

MMD²/JS/L1/长度KS/entropy/最近邻等完整指标及raw恢复门见 [EXECUTION_FACTS.json](EXECUTION_FACTS.json)；读出源与产物哈希见 [READOUT_MANIFEST.json](READOUT_MANIFEST.json)。

缺失状态：

- CE21_DDPM：生成组 AMD,FREE；重建split train,development；phase记录可用 True。缺失观测不当作零。
- MSE21_DDPM：生成组 AMD,FREE；重建split train,development；phase记录可用 True。缺失观测不当作零。
- scalar_DDPM：生成组 AMD,FREE；重建split train,development；phase记录可用 True。缺失观测不当作零。
- terminal_LSTM：生成组 AMD,FREE；重建split 无；phase记录可用 True。缺失观测不当作零。

独立后审通过仅指恢复工程范围；原协议不重命名为通过。训练/监测曲线源只在原training root，AR teacher-forced NLL与全部phase日志摘要保留JSON，不与VAE目标横比。

- original E/L/G are permanently NOT_EVALUABLE; amended attempt is separately identified and costed
- original FREE generation cost is at least 451.935743 seconds; termination time and total original wall time are unknown
- single fitted seed, reused development41, no significance or independent generalization evidence
- fixed original final checkpoint and seeds; curves do not select an endpoint or establish AR adequacy
- readout transcribes metrics and engineering records; only the bound postflight receipt supplies independent recovery engineering approval
- unverified evaluation MANIFEST disables interpretation and descriptive screens while retaining raw values; snapshot stability alone is insufficient
- no antimicrobial activity, safety, novelty, strong-baseline superiority or submission-readiness claim; old wetlab is not H5 validation
- BACKUP_BLOCKED; local source hashes do not constitute remote backup
