# AMP H5 warmup 后 R/M/S：仅条件准备

本包未执行模型或自然数据评分，未读取自然 cohort、epsilon NPZ、test；没有生成随机数。合成 slot/AST/latebinding 检查 PASS49。真实 ROOT_PLAN.json、INDEPENDENT_PREFLIGHT.json、LATE_BOUND_TRAINING.json 尚不存在。

仅在唯一 warmup1000 成功、训练独审通过且真实 VAE时间≤1920秒 / supervisor wall≤2100秒后，root 方可绑定真实最终 checkpoint SHA/tensor digest；不能猜权重SHA，不能把训练完成当成此诊断已获授权。该晚绑定文件须由新 exact root plan 和独审 receipt 同时绑定。模板不可执行。晚绑定还要求训练独审覆盖checkpoint/metadata/summary/status/supervisor/执行合同的精确hash。

本导出代码由封存 ce_rms_reconstruction_v2.py 明确修订，保留原 code-only bootstrap、原 frozen posterior_parameters/decode/J 与原rawexports。新权重下全部320需重编码：train279分9批、dev41分2批，共11 encoder调用；内部抽到的320个z丢弃，并记录各批RNG及丢弃样本摘要，fork_rng恢复全局状态。这些内部工作样本不计作S结果。绝不复用旧train/dev q。

M320+S2560总2880请求；99 decoder调用。S读取原1422–1429八个保存epsilon银行，逐个核file/hash、float32[320,16,48]、epsilon tensor hash和原generator状态hash；不建立新Generator，不重新seed或randn。原train279→dev41行顺序及slot ID均保留。reference_rows.csv由同输入生成后须与旧完整baseline reference字节SHA完全相同；相异即在模型构建前停止。

新M不会与旧M预测强制相等：学习后的权重不同，评价是完整baseline/candidate配对分数。first32 M是99批内的首个正式批，保留一次，不额外pilot；只有有限/形状与时间估计 elapsed+8×首批秒数×98+10≤120 决定是否继续，无质量筛选。inner120/outer180冻结，所有请求含失败/未执行保持分母；失败不重试、不放宽预算。退出/manifest后账本及supervisor真实wall共同决定完成资格。

评分由另一代理单独准备，复用冻结 RMS metric core、score_rms_v2 的421序列特征与9项原始RDKit描述符，禁止模型评分循环。后续资源规则为同dev41/同8epsilon：S等序列准确率至少+1/20、M不降、M长度MAE≤0、S宏召回不降，全部满足才可考虑继续；gap缩小本身不构成成功。准确率用整数计数的精确Fraction比较。缺失描述符不能移除核心准确率分母。

历史CE原E/L/G未可评状态及其他负结果保持不变。该工作不是末端条件生成、独立训练复现、显著性、唯一因果、创新、抗菌功效或投稿证据。新模型q/RMS只在此单独限额内计算，不补写旧证据。
