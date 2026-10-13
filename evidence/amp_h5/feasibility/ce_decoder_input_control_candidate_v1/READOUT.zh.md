# CE-only decoder 输入诊断：候选实现，尚未执行

候选入口 implementation/ce_decoder_input_control_v1.py 与 protocol/CE_DECODER_INPUT_CONTROL_CONTRACT_v1.json 仅供预审。本次只导入 stdlib 候选模块、做静态/合成验证和文件哈希核验；无torch/numpy运行、无RNG复建、无模型构建/forward、无自然decoder输入、无live scalar读取。

32 paired slots（FREE16/AMD16）×2分支，共64固定输入，4个batch16调用，CPU1。内层总cap120秒，外层180秒。A直接z=epsilon，端基只是report group，非native learned flow prior。B使用固定PCG64 selector seeds523/524，端基内有放回训练行，z=mu_j+exp(logσ_j)epsilon，仅经验后验/记忆参照。原合法J原样复用，不进入flow inverse，不读取g/cohort/dev/test，不构建DDPM。

未来授权执行时，入口先核新的root明确授权plan、独立exact-scope receipt、scalar已封存终态与独审review的精确hash、任务锁及110项冻结输入。随后在global torch RNG的fork_rng隔离区复建seed422/423原完整消费顺序，保存真epsilon、全部原RNG状态和流digest；必须同时逐位重现两个端基保存的z。固定训练索引与全部64潜变量一并封存，完成以上检查后才构建/加载最终CE模型并执行decoder。

任何前置条件不满足，零decoder调用并FAIL；后续异常停止、保留已有张量、traceback及全部64成功/失败/not_executed槽，不补采、不换seed、不重试。保存模型前后digest、全部冻结输入后验hash与分阶段成本。源码没有自动评价或任何启动后续模型的入口。

原读出包保持原hash；新增 protocol/LATENT_MATCH_DIAGNOSTIC_v1_METRIC_CLARIFICATION.json 明确逐peptide宏平均与主评价器supported-AA macro不同，有限经验区间越界不代表support之外或失败。历史草案的local Generator说明保留，最终候选明确改为root/独审要求的global RNG+fork_rng。

未来plan必须是新的 protocol/CE_DECODER_INPUT_CONTROL_PLAN_v1.json，独立执行回执必须是 review/independent/CE_DECODER_INPUT_CONTROL_PREFLIGHT_RECEIPT_v1.json。现未创建二者。回执须绑定plan/contract/control_script/supervisor/launcher SHA并声明scalar_terminal_review_checked=true；final plan还须绑定已封存SCALAR_V2_POSTFLIGHT_v1.json及scalar SUPERVISOR_STATUS.json。具体argv和字段核验见validate_request。原CE E/L/G始终NE，控制是事后探索的新scope，PROMOTE HOLD。
