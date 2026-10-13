# AMP H5 Smoke 独立预审

**PASS，仅限绑定计划的真实 train-only 工程 smoke。** 不是正式实验批准、真实训练成功、收敛或科学优势结论。精确源码/配置/计划身份及可机器检查的批准字段见 `SMOKE_PREFLIGHT_RECEIPT.json`。

批准范围为三条 VAE 完整路径各 20 VAE + 20 DDPM 更新，以及单个末端条件 2×64 LSTM 的 100 更新；每方法 FREE1+AMD1，训练总软上限180s、生成总软上限450s、每子进程外层750s。旧源码、旧证据、稿件与test均不修改；smoke不打开dev，也不做分布评价或质量选择。所有smoke权重排除于formal初值。

亲自执行的独立合成核查：VAE30项通过，包括CE/MSE全初值一致、357参数头、全网1,681,549参数、两种重建公式、KL分母、encoder/decoder/flow梯度、scalar原生forward逐值等价及17参数头、J零/1/47/48长度与直接枚举/最近合法one-hot投影。LSTM24项通过，包括62,501参数、右移BOS、因果性、首EOS掩码/梯度、48AA无EOS49、独立随机流、强制空串/full48、零cap、途中cap和非有限失败slots。

此外由独审执行作者的合成benchmark：三臂各2次VAE参数更新、2次DDPM更新完成，有限性通过；它使用合成张量，没有自然记录、dev或test。该benchmark在工具返回运行session后，后续约0.68秒AR合成检查可能存在短暂CPU重叠，因此记录的step_seconds不用于公平吞吐或formal预算；后续预算以root串行真实smoke为准。审稿脚本第一次在作者将scalar更名scalar_legacy的窗口仍传旧名称，触发头计数断言；按冻结名称更新后30项全过，失败未涉及自然数据。

已复核关键修复：metadata train和descriptor train分别绑定，并逐行核对idx/序列/terminal；VAE两臂只更换重建目标且共享KL、学习率/批次/随机流。每臂重新编码自身后验并新训DDPM。DDPM阶段入口与编码batch检查总训练cap；生成只有terminal与自身噪声。AR wrapper提前写全请求槽位；上游失败不启动正常生成，剩余臂按固定次序继续，无补样或重试。所有计划8个源/配置绑定与当前文件匹配。

两条边界需保持：20更新的DDPM可能在完整998步反演后产生非有限logsigma/exp，此时是如实记录的smoke数值失败，不允许静默clip、更换sigma、缩短采样或换seed。正式运行必须另冻预算和门；小LSTM配置、成熟CE/J及有限pilot不能单独证明强竞争、新颖性或可投稿。`SMOKE_PROTOCOL`中的formal预算描述只作预告，root已讨论的AR固定1000更新和充分性诊断应写入独立formal协议后再审。

独审未启动真实训练/生成，未读取自然train/val/test记录。`BACKUP_BLOCKED`。
