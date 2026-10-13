# Warmup 计划与根回执模板（未授权、未执行）

只引用封存候选 MANIFEST 888c0c… 与 CONTRACT 733ad7…。不改代码、数据、种子、损失或预算；不运行模型、RNG或新测试。固定VAE1000，β(step)=min(1,(step−1)/200)，同初始化/批次/CPU RNG/LR，保留10次原dev monitor；无DDPM、generation或额外RMS。

ROOT_LAUNCH_AUTHORIZATION_TEMPLATE.json 对齐封存wrapper的必需字段，并补齐score CLOSING_LEDGER、真实outer RESULT、INVESTMENT_SCREEN、manifest和后续独审绑定。当前完整dev门原始分数M=.7227961128580324，S8=.6611157984837004；精确差435643512761/7062926270400，比较必须以整数20n≥d为准。浮点wrapper admission只是相容性检查；只有独审据完整计数重算精确分数通过后才可签根回执。本模板记录现有数值，不冒称独审已完成。

根代理需等待集中 NIGHT_FOLLOWUP_JOINT_POSTFLIGHT_v1.json，填写评分后审/候选静态审同一文件真实hash；再做fresh ps/load检查。用真实信息写固定 ROOT_LAUNCH_AUTHORIZATION.json，填入plan bindings，最后由独审签exactplan的独立launcher回执。模板中的null/false/NOT_AUTHORIZED不得沿用为授权。计划与根回执不存在循环hash：静态集中审不引用未来plan，最后独审exactplan回执另存。

模型wrapper的outer_elapsed_seconds采集早于最终status/manifest写入，不能作为完整预算证据。实际supervisor job outer_timeout固定2100，VAE阶段1920；终态需联合wrapper状态+manifest+supervisor exit0/no-timeout/job.seconds≤2100。若终止宽限或cleanup使实际wall超限，完整报告成本并判NOT_EVALUABLE，不把旧wrapper completed称为预算PASS。完成训练亦不等于科学成功，四项root success guards和后续RMS仍需另scope。

所有候选冻结bindings、原scaler/trace/code-only82项均直接承接。未将source_manifest_excluded中9个非code旧文件加入计划hash读取；尤其不读legacy CSV或test37。单次串行，无retry；失联先查PID/stdout/状态，不能重发launch。

模板v2新增绑定根投入决定ROOT_WARMUP_INVESTMENT_DECISION_v1.json（bea1fdf…）；该决定不代替集中评分独审、fresh负载检查或独立exactplan回执。v1模板保留。
