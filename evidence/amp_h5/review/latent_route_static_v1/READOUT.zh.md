AMP H5 潜变量路径静态独立审计（v1）

本次未在已检查的冻结 H5 路径中发现 μ/logσ 通道交换、将 logσ 当作 log 方差、遗漏 latent 逆标准化、错载 EMA 或端基标签映射错误。这个结论只排除了已检查的具体实现不一致；不是“代码完全正确”的证明，更不能把已观察到的 latent 匹配差直接归因为某个算法组件。当前比较仍为 HOLD。

只读范围：冻结 runner、源配置、已完成 CE 训练/恢复元数据、CE/MSE 保存数组的 header。未读取 test、cohort 记录或 live scalar 内容；模型导入、权重加载、forward、训练、采样、RNG 重建和质量重算均为 0。现有文件未修改。共 22 项静态校验通过。

| 路径 | 核实事实 | 判断边界 |
|---|---|---|
| 输入标准化 | train279 全局向量、4868 实际残基拟合 population zscore；global 再 abs，residue 保留正负，padding 后置为0 | abs 为历史继承的非可逆信息折叠；不是 latent scaler，也不应在生成时逆变换 |
| DDPM 对象 | `[B,2,16,48]`，channel0=μ，channel1=logσ，末轴=位置；预测注入的 epsilon | 没有证据表明通道/位置错换；参数与位置（含 padding）全部建模 |
| latent 到 decoder | 生成 σ=exp(logσ)，z=μ+σε，直接 decoder(z) | 这条 posterior-parameter DDPM 路线没有漏 inverse flow；latent 也没有预先标准化 |
| noise schedule | 999 个系数，train t=1..998，reverse998..1，共998次；最后噪声置0 | 无999越界；abar0=.9999，未额外执行t0→完全clean，是继承端点约定，效果未知 |
| EMA | helper 在2000更新前复制当前权重；CE1000的 current/EMA tensor digest相同 | 不能称已发挥平滑平均效果；不是已发现的checkpoint混用 |
| CFG | H5仅调用有条件预测，无unconditional blend；源root diffusion.py同为cfg=-1 | 另一个 tasks/train_diffusion.py 是cfg=3推理、16通道、随机长度；不是相同入口合同 |
| 条件 | FREE0 / AMD1；DDPM 每批10%丢标签；生成只有端基与随机输入 | 软件条件可达不等于端基控制有效；序列字符串不能独自证明化学末端状态 |

精确实现入口：[F:154](/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/implementation/vae_pipeline_formal_v2.py:154)（scaler）、[F:347](/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/implementation/vae_pipeline_formal_v2.py:347)（cache）、[F:400](/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/implementation/vae_pipeline_formal_v2.py:400)（epsilon目标）、[F:480](/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/implementation/vae_pipeline_formal_v2.py:480)（生成）、[D:27](/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/corrected_code/AMP_VAEflow/modules/diff.py:27)（EMA）、[D:212](/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/corrected_code/AMP_VAEflow/modules/diff.py:212)（label embedding）。完整行范围、摘录和文件 SHA256 已随每条 finding 保存于 LATENT_ROUTE_STATIC_AUDIT.json。

标准正态所在的坐标必须区分。训练先把 decoder 坐标 z 送入 `prior_flow(z,g)` 得到 zp，再对 zp 求 N(0,I) 密度；decoder 接收 z。给定合法 g，所对应的 learned conditional prior 应是 zp~N 后逆 flow 回 z。活动 flow 使用 mean_only=True 的加性耦合，logdet为0，因此此处忽略返回的 logdet 不是漏 Jacobian 的证据。见 [F:198](/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/implementation/vae_pipeline_formal_v2.py:198)、[V:193](/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/corrected_code/AMP_VAEflow/modules/vae.py:193)、[FLOW:234](/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/corrected_code/AMP_VAEflow/modules/flow_base.py:234)、[FLOW:278](/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/corrected_code/AMP_VAEflow/modules/flow_base.py:278)。

g 是由训练序列逐残基内容生成的3维descriptor，不是端基 c 的函数；冻结H5没有定义 c→g 的生成机制。`g=None` 虽在底层可调用，但跳过了训练使用的条件，不可直接叫同一 learned prior。H4明确另设“抽取训练g后inverse flow”的经验条件路线。直接 `decoder(epsilon)` 只能叫 standard-normal latent control；decoder没有接terminal，不能继承原DDPM的端基条件性。见 [BUILD:113](/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/expanded_preprocessing/build_descriptors_v2.py:113)、[H4:125](/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/diagnostics_h4_h2_generation_bridge_v1/h4_run.py:125)、[V:142](/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/corrected_code/AMP_VAEflow/modules/vae.py:142)。

保存证据方面，CE恢复与MSE的generated NPZ均没有epsilon原始键；有posterior_parameters、z、raw、tokens、lengths、length_scores。posterior NPZ只含values。最终epsilon来自同一有状态RNG在逆扩散后的新randn_like调用，代码不据mu/logσ数值计算噪声；但本次没有做RNG重建，也不把相同种子当成跨臂逐位一致证明。`(z−μ)/exp(logσ)`仅是浮点派生残差，不保证还原原始epsilon。

现存train/dev重建都是μ的确定性解码。两个token字段是同一解码的表示，不能当作mean/sample两分支。训练后验重采样也保留了训练序列派生信息，应称 empirical training-posterior reference；它能作为诊断参考，却不是独立先验生成。标准正态control、训练后验reference、经验g的flow prior、端基DDPM生成应保留各自分母与来源，不能互换名称。

本审计未提出或应用新head、latent缩放、clipping、CFG、EMA或schedule调参。finite输出只表明数值/工程执行成功；究竟是DDPM拟合不足、后验分布难学、decoder敏感性或目标影响，仍需独立、预锁的小范围证据区分，不能靠静态检查断言原因。

冻结runner SHA256：`e97d2a766774461793cba1860455e4cb4c8d689656b60becc326a9f11297ca9b`。源文件与元数据完整绑定见 SOURCE_BINDINGS.json；保存数组结构见 SAVED_ARRAY_SCHEMAS.json；检查见 VALIDATION.json。
