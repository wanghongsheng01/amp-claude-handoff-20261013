AMP H5 成本与开发机会账本 v4

已核入 scalar 终态：2026-10-11 19:58:05 UTC 完成，exit0，VAE1000/DDPM1000，32请求全部执行成功；独立收据PASS（SINGLE_SCALAR_WHOLE_PIPELINE_ENGINEERING_ONLY）。旧v1-v3保持不变。原CE中断没有被改写为正常终态，额外recovery仍单列。

累计已记录的实验任务wall成本仍只能报下界：7128.6893321229145+2867.7861290420406=9996.4754611649551秒，约166.607924分钟。这里是各已记录、不重叠任务/阶段的elapsed wall之和，不是CPU user/system时间，也不是所有工作总历时。原CE未记录的时间与部分审计开发时间仍未知。

| 新核入scalar成本 | 秒 |
|---|---:|
| supervisor外层（累计只加此项） | 2867.7861290420406 |
| 进程记录 | 2867.1760469167493 |
| training pipeline | 1742.7696539591998 |
| VAE阶段 | 1179.4492224999703 |
| DDPM阶段 | 563.3003288749605 |
| generation | 1116.6571687501855 |

阶段均嵌套于外层，不能相加重复收费。全历史更新合计AR1100、VAE3060、DDPM4060，含smoke及额外诊断，不等同于正式实验预算。

序列分母为200个已分配slot：130个完成导出、6次数值失败、原CE FREE16个部分执行、CEv1入口失败32个not_executed、原CE AMD16个无开始证据。开始attempt仅能确定至少152个，不能把原CE部分轨迹当完整生成。另有4个latent-only诊断attempt；旧H4复用行不是新生成。完成导出中legal1–48为129、legal5–48为120、空串1、长度1–4为9；这些混合历史计数仅用于账务，不可汇总为方法质量率。scalar单独legal1–48为32、legal5–48为32。

新same-n NumPy分析另计11.834126624744385秒，不加进上述实验任务wall。其256次已有组件重采样和1024个方法/端基统计记录不是新模型、latent或生物样本；model calls=0。连同旧已计时AR只读readout，已知计时的非模型分析合计13.7332680417443848秒；其余未计时审计仍未知。此账本不重新计算分析质量。

旧A/B64个decoder control只是候选协议/代码准备。实际分配、开始、完成、forward均为0；不把64个候选加入not_executed或任何真实尝试分母。静态修订及审阅属于开发机会，未记录wall不能伪填0秒。

共同2940秒training cap是资源上限，实际计算量不相等。CE有定向DDPM诊断、入口修复、中断FREE轨迹及重新跑32链的recovery；全部保留。每正式方法只有一个训练种子，dev41多次复用，AR末点过拟合且充分性未证。scalar也是整个legacy流程对照，不能称单组件因果实验。四个方法的输出可通过单列CE恢复获得，并不恢复原正式E/L/G；原CE E/L/G仍NOT_EVALUABLE_INTERRUPTION，PROMOTE=HOLD。

精确来源/hash见SOURCE_BINDINGS.json；逐事件成本见COSTS_V4.csv；分母见DENOMINATORS_V4.csv。JSON另保存Decimal按已记录数字相加的结果及原生float和，避免以显示尾数混淆精度或冒称未知物理总成本已精确。
