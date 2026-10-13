# AMP H5 CE v2 终态后审预锁清单

本文件在主线程尚未通知终态时准备。尚未读取 CE v2 的 live 输出，没有完成后审，也没有发放 MSE/scalar 许可。只读冻结源码、协议和旧已关闭证据；新后审脚本仅 AST 解析，未执行模型或自然结果检查。

执行入口为新文件 `audit_ce_v2_terminal.py --terminal-confirmed`。先要求模型 status 与 supervisor 均非 running，再进行下列已定检查。脚本拒绝覆盖其新输出 `CE_V2_TERMINAL_AUDIT.json`，不修改 DELIVERY_H5_DIAGNOSTIC_CE_RUNNING_INDEX.json 绑定的旧文件。

1. 校验当前唯一 single CE v2 计划、runner、config、所有计划绑定、实际命令、runtime 源码哈希及完整输出 manifest。无未声明重启、续训或权重替换。
2. 固定1000 VAE +1000 DDPM更新，逐步日志连续、数字有限。复核全部 VAE batch shuffle、原学习率和目标分母；首20步随机状态对照冻结 smoke。DDPM 全1000步 batch/noise/drop/LR 对照冻结1000 horizon诊断，检查 fresh init digest、保存 optimizer与 EMA 语义。该对照不把质量或 loss 值要求相等。
3. 固定10次 dev monitor，记录不选 checkpoint、RNG保持；最后监测 RNG 对照保存状态。数据只核定 exact train/dev 哈希、279/41输入身份、train-only scaler 和重建导出行身份。允许读取两份 exact hash-bound cohort，绝不读 test。无 test 判断来自源码、配置与产物路径审计，不冒称 OS syscall tracing。
4. 精确 FREE16/AMD16预定 ID/seed/status；空串、48上限及失败均保留，不删样、不补采。只 completed executions 纳入工程完整性；长度、unique、MMD和重建质量不作为继续臂的门槛。
5. 从已存 NPZ 复算 mu/logσ/sigma/z 范围、RMS、严格有限正sigma；以本次自身 train posterior 的 pooled 与 terminal-matched解析 q(z)尺度作描述。重放随机数 draw以核验末epsilon重参数化，复算 raw-logit J 解码；不实例化模型、不前向、不训练、不产生新生成请求。失败时额外读取已存失败阶段和可用张量摘要。
6. 核预算：VAE1920、DDPM1020、共同训练2940、生成1800、外层4920秒。逐更新起始检查遵守 soft cap；最后原子操作及checkpoint IO如有越界必须披露。任何不完整、未执行、数值异常或预算异常均保留，不靠额外预算或重试补齐。

自动结果最高为 PASS_ENGINEERING_CHECKS_PENDING_INDEPENDENT_SCALE_JUDGMENT；它不会授权后续臂。独立审查者再写新的尺度判断与终态后审说明。不得事后创造基于观察值的统计质量门槛。满足原先有限值、正scale、完整端点、预算和可读尺度规则后，最多建议对已冻结 MSE/scalar 配置分别新建串行计划及独立回执。全 FORMAL_PLAN 仍不授权；AR 原1000末点继续使用；AR_ADEQUACY=NOT_ESTABLISHED，PROMOTE=HOLD。

保留适应性成熟度诊断、原0更新契约失败及v2修复成本。单seed和16/terminal不足以证明新颖性、强基线优势、抗菌活性或投稿就绪。
