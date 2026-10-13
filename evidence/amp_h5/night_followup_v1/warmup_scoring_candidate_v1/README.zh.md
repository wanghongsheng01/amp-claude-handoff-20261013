# Warmup RMS 评分与成对资源门候选（尚未执行）

原 score_rms_v2.py 只替换 SCOPE 常量为新 scope；所有评分数学、源函数和 CLI 逐字复用。新权重 RMS 完成且独立后审 PASS 后，才可由 root 授权单次 300s 内部/360s 外部评分。源模型应新编码320条，使用相同8个已保存 epsilon bank；不重抽。原 RMS、原评分、既有权重和稿件不改。

SCORING_CONTRACT.json 绑定模型候选合同和只变 scope 的 scorer。模型权重仍由 model 合同规定的后绑定文件+训练后审锁定；此准备包不授权训练、前向或自然评分。子模型合同如因独审变更，须保留此候选并另版本绑定，不静默替换。

PAIRED_GUARD_CONTRACT.json 绑定已通过联合独审的 baseline 与未来候选完整评分。paired_guard_v1.py 只读已保存的整数正确数、目标长度和 AA 支持数；不运行序列距离/描述符或模型。唯一资源成功门为：dev41 等权 S8 提升≥1/20、M准确率不降、M长度MAE不高于原0、S8 supported-AA macro不降。全用 Fraction，缺失/非有限/身份或receipt不一致一律NE。gap变化只描述，遵守根预锁四门优先于文献v2，不加第五门。资源门不能授权后续训练/生成或论证活性、新颖、可投稿。

评分完成需执行 RESULT state completed/exit0、CLOSING completed、300/360 caps和独立后审 PASS。未来候选评分后审须 scope_id AMP_H5_WARMUP_RMS_SCORING_V1，并绑定其 MANIFEST、CLOSING_LEDGER 和 execution RESULT 的绝对路径与哈希。成对读取器独立30s内部/60s外部预算，不重跑，不择优取draw；解释器完成后应单独留外层执行收据。

原基准 M 的72.27961128580324%是 mean_i(correct_i/length_i) 的逐序列等权准确率；70.89871611982882%是497/701残基 pooled micro。AA macro42.7601056550582%是第三种口径。历史M320逐条parity已通过，差异并非baseline变化。

所有新候选目前仅执行纯合成统计与静态代码身份检查。成本、最终作者addendum与一次本地归档等自然任务终态后集中封口；同盘归档不是独立灾备。
