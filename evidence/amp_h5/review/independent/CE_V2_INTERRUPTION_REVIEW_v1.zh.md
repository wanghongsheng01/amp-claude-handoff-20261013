# AMP H5 CE v2 中断取证与限定恢复判断

独立保存产物审计41项通过：VAE与DDPM均有连续1000更新、有限最终参数和optimizer、正确最终checkpoint哈希、预定seed/batch/noise/LR记录、10次保持RNG的dev监测。VAE阶段1179.032秒、DDPM阶段564.346秒，阶段小计1743.378秒；top-level总时间没有最终落盘，不能冒称恢复了其精确值。

独立只读ps查询没有列出60744/60747/60887。原supervisor/status仍是running且无退出码；所有原文件保留。FREE16请求已进入反向过程，记录确认至少完成t998到t200的799步，生成耗时至少451.936秒。AMD16没有启动证据。raw CSV仍32 not_executed，因为代码只在整个terminal批次完成或捕获异常后回写；这不等于32个都没有被尝试。没有最终生成NPZ、重建导出、manifest或失败张量。退出原因未知，不能把transport断连视为已证根因。原v2工程门为NOT_EVALUABLE_INTERRUPTION，不是PASS，也不能据此宣称模型数值失败。

首次取证脚本有一项过强审计假设：要求每个Adam state都到1000。它使初版报告记为HOLD，现保留初版并另写更正版本。实际DDPM label_emb.weight仅在条件未drop时参与forward，其Adam step894恰等于日志中894次条件更新，其余176个已有DDPM optimizer state均1000，VAE212个已有state均1000。全部有限；这不是模型修复或训练重试。条件分支已由冻结源码核实。

最终训练posterior的pooled sigma为[0.77155,1.25414]，解析q(z) RMS1.58201；FREE/AMD分别1.59632/1.57013。这里只能确认训练表示尺度，尚无生成尺度证据。

审查判断是**可以准备一项明确的新恢复attempt，但当前未授予执行许可**。已有最终VAE及DDPM EMA足以作为只读输入；不存在中间x/RNG，故必须从各端基原seed重新执行完整998步链，不能称接着t200续跑。允许范围限于：严格加载固定最终权重、恢复既有scaler.stats、原generation代码和FREE/AMD各16同seed请求、原train/dev固定导出；零训练、零参数或数据改变、不新选checkpoint、不clip、不补采。必须有新版本、新协议、新目录、新计划和独立回执。

这会明确偏离原no-retry规定，增加生成计算。原至少451.936秒、精确终止耗时未知，加新恢复全部耗时须分列；不能声称旧预算总量不变或实际等计算。新attempt仍以1800秒生成cap为上限，setup/export和outer预算须预先写定。可以用持有同一任务锁、单线程CPU、落盘日志且有界的独立session supervisor减小会话耦合，但这既不证明原死因，也不保证避免重现。

原E门持续NOT_EVALUABLE_INTERRUPTION。若恢复成功，另列RECOVERY_ENGINEERING，引用原训练与新生成各自的真实路径/hash，不能回填原v2或伪造其manifest。MSE/scalar此刻仍HOLD。只有新32请求完整执行、数值/正sigma/尺度/预算/状态与权重不变均独审完成后，才可考虑分别新建串行计划；不得按重建、长度、unique、MMD、JS好坏改许可或超参。PROMOTE=HOLD，AR充分性、新颖性、活性与投稿就绪均未建立。

本轮没有模型构造、前向、训练或生成；没有读取封存test37。只对已知train/dev文件做哈希，并对冻结源码/产物进行读取；这不是OS系统调用审计。
