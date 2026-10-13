# CE v2 中断取证与零训练恢复草案

**原 CE v2 应记录为中断且未完成。** 原 runner 与 supervisor 的 `running` 状态均保持原样，不伪改为终态，不补写原 manifest，也不把可能的新恢复结果回填成原 attempt 成功。

根代理报告原 supervisor 60744/model 60747 已消失。作者的 `ps` 只读核验受本地权限限制，未能独立复核进程消失；终止原因、准确时间与最后未落盘工作量未知。以下证据来自已保存文件。

| 已保存证据 | 实际含义 |
| --- | --- |
| VAE 日志完整 1–1000、全部 loss 有限，阶段耗时 1179.032 秒 | 固定 VAE 训练终点已到达；不是生成成功证据 |
| DDPM 日志完整 1–1000、全部 MSE 有限，阶段耗时 564.346 秒 | 固定 DDPM 训练终点已到达 |
| VAE、DDPM、DDPM EMA 文件哈希均与保存时 JSON 一致 | checkpoint 文件完整性通过作者按字节核验；未反序列化模型 |
| FREE 最后进度 t=200、451.935743 秒 | FREE16 轨迹至少完成 998…200 共 799 个反向步；生成耗时至少此数 |
| 原 CSV 仍为 32 个 `not_executed` | 预分配导出状态尚未更新；不能说实际生成成本为零 |
| 无 AMD 开始日志，无任何最终序列/生成 posterior/recon/生成摘要/manifest | AMD 无开始证据；两个 terminal 均无可用最终序列 |

所有已记录文件的字节哈希清单与完整原状态、原 dev 曲线转录见 `EVIDENCE.json`。复核结束时，这份清单中的原文件哈希仍一致。没有读取 test、原始训练 CSV 或模型权重内容，没有模型调用。

## 恢复的最小必要边界

原流程没有在 t200 保存采样器的 x、随机状态或可继续的 sampler checkpoint，因此不能从 t200 接续。最小可行方案是使用已到达1000更新的原 VAE final state 和原 DDPM EMA state，**另建新 attempt，从原种子重新做 FREE16、AMD16 全部固定生成**，之后调用原 train/dev mean reconstruction 导出。

静态入口 `RECOVERY_ENTRY.py.draft` 提议：

1. 严格绑定原 checkpoint 文件及 tensor-state 哈希、原配置、冻结 formal_v2 runner 和保存的运行环境。
2. 只构造相同模型并加载最终权重；不读取 optimizer state，不执行训练或任何更新。
3. 调用原 `noise_schedule` 和原 `generate`，保持 batch16、seed422/423、terminal-only、999步 schedule 与998…1反向链、后验重参数化和 decoder 不变。无 clip、替换、补采、调参或挑结果。
4. 恢复原 `train_feature_scaler.json`，绕过 `FeatureScaler.__init__`，不重新 fit；按原编号合同读取 train279/dev41，调用原 `reconstruction_export`。
5. 所有状态、32个请求、失败张量、导出和 manifest 只写入新目录。原 attempt 保持中断记录。拟在加载后和导出后比对参数 tensor-state 哈希，确认权重未变。

重新构造模型时的 RNG 消耗不会决定生成噪声，因为冻结的 `generate` 在每个 terminal 内明确重置原采样种子。此判断来自静态代码；尚未进行加载或采样实测，不声称恢复结果已证明相同。

## 必须单独批准的协议偏离

该方案是**基础设施中断后的生成重做**，偏离 v2 原 no-retry 约定。即便使用相同 checkpoint、种子和生成函数，也不能将其包装成原 attempt 无中断完成、从 t200 继续或零额外成本。

`RECOVERY_CONTRACT_DRAFT.json` 暂拟新 attempt 生成 cap1800秒、outer cap1980秒、训练0更新。它们是待批准的新成本，不是原剩余预算：原 VAE1179.032 + DDPM564.346 + 部分生成至少451.936秒需继续单列；准确原总成本仍未知。原算法、种子、数据、目标或模型效果不构成调参理由。

独审须先判断此修复是否可接受；root 再明确授权一次新 attempt、增加的预算及具体执行。随后才可另建正式恢复实现与 supervisor plan，并获得新独立回执。当前 `.py.draft` 在入口无条件抛错且无恢复 CLI，合同 `authorized=false`，没有运行许可。

仍需独立确认进程已停止、无其他 AMP 模型运行并取得串行锁。新恢复结果须再次独审后才考虑 MSE/scalar，当前 `PROMOTE=HOLD`；不触碰 test、不宣称序列质量或可投稿。
