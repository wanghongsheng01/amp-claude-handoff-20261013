# H5 端到端目标对照：静态设计，尚未真实执行

推荐主实验只设两个同容量端到端臂：`raw_onehot_mse` 与 `categorical_ce`，配独立作者的 terminal-only AR。保留现有 FVAE encoder、条件 flow、latent16×48、8层encoder/4层decoder；共同将最后17参数scalar投影换成357参数21类投影。两臂从同一个全新模型state克隆，使用相同训练记录顺序、批大小、更新机会和posterior随机流；共同J读出。唯一有意变化是重建损失。训练数据仍279、开发41，test37封存。

原代码已经有生成仅依赖terminal的 posterior-parameter DDPM。每个新端到端臂必须分别用自己的新encoder重提取train mu/logsigma，并训练自己的全新DDPM；不能复用H0/H4后验池、旧DDPM、另一臂DDPM或验证对象池。主生成条件只为0=FREE、1=AMD，外加内部噪声。原flow生成读逐残基g的路径和posterior aggregate不能加入公平主比较；保留旧结果作条件信息不等的历史参照。本稿最初讨论过把flow的g改为terminal广播，随后静态审查发现无需该额外干预，故不采用、不写入候选接口。

现有encoder的x、g(x)、abs(train-zscore h(x))全部保留为训练/重建输入。g/h来自该对象序列的确定性特征；它们不允许出现在DDPM或AR生成API中。保留不代表已证明这些特征/abs语义最优；本轮不同时解决该开放项。decoder本身仍不再次接收g。

共同损失单位建议（等待root冻结，不是执行授权）：令B为本批对象数、T=48、C=16、K=21，s为未softmax的21输出，y为含PAD20的onehot目标，D=Σ(log q(z|x,g,h)−log p(z|g))，求和覆盖B×C×T。

- MSE臂：R_MSE = 0.5 Σ_{b,t,k}(s−y)²/(B×48)。类别维求和，不额外除21；这是raw onehot平方损失，不是softmax Brier，也不是AA ordinal index误差。
- CE臂：R_CE = Σ_{b,t}[logsumexp(s)−s_y]/(B×48)。包含全部48格PAD；这是独立位置的categorical NLL，不是对合法前缀重新归一后的structured NLL。
- 两臂K_new=D/(B×48)，L=R+K_new，β固定1，不扫β、不anneal、不截断小负Monte Carlo KL。density路径、flow mean-only/logdet语义及logsigma解释不变。同时存D、B×48、legacy K_old=D/16、K_new及每对象KL。

该共同归一化修正让同内容批复制后目标不变；它不是把原损失整体乘一个常数。旧版原重建为全48格ordinal SSE/nonPAD、KL为D/16，batch32/末批23和长度会改变相对权重。从旧版到H5同时改变目标编码、头容量和相对KL尺度，不能声称“只改CE”。只在两新臂之间，其他共同协议相同，比较重建损失对完整学习系统的影响；仍不能保证固定优化预算下任一臂充分收敛。

两臂的J规则相同：每个候选长度L取前L格各自最高AA类分数、其后取PAD分数，选总分最大者；精确tie选最短L、AA tie选最小索引。对于raw onehot平方损失，这是到合法onehot序列集合的最近投影，因为每格平方损失等于与标签无关的Σs²+1−2s_y；对于CE，它是该factorized分布在合法前缀集合内的MAP。不得称MSE分数为校准类别概率。空串L=0、full48、全部失败均保留。firstPAD仅可在保存的同logits上作预锁附诊断，不增加模型调用、不代替J主读出。AR按其原生自回归抽样/EOS规则，不强套并行J。

亲查后的现有证据见EXISTING_EVIDENCE.json及14个源文件hash：

| 事实/问题 | 已验证证据 | 能推出/不能推出 |
| --- | --- | --- |
| ordinal语义 | 原scalar目标是0–20索引，padding也为20；MSE距离随AA任意排序改变 | 目标语义存在可行动问题；不是现有低质量的唯一因果证明 |
| posterior噪声 | H0均值106/701，三个随机样本85/74/80；均值本身仍弱 | 噪声确有损伤，不能说去噪声即可恢复内容，也不能低率⇒坍塌 |
| 对象信息仍被使用 | 同terminal×真长度19个移动对象、279残基，置换相对mu少17正确、MSE增加26.2113 | decoder使用对象相关latent；由于先验也依赖g，不排除条件后验坍塌 |
| 读出与终止 | H1类别头提高部分AA读出但premature PAD；H2 CE+J四开发mode正确274/208/185/200、长度MAE0.390/0.634/0.293/0.488 | J修复部分截断；不能把全部变化归因CE或已充分恢复表示 |
| 扩容量未稳定奏效 | H3 quadratic和RFF自身预锁门均失败 | 停止继续扩大固定hidden probe；不能数学证明hidden没有可恢复信息 |
| 重建到生成迁移 | H4两头共享96上游latent，H2合法1–48为95/96、5–48为91/96；MMD五升一降、val二肽JS六降、KS四降两升 | 观察混合，未证一致改善；尚未直接证明生成hidden分布失配是原因，更不能宣称活性 |
| 低DDPM噪声MSE | 旧val noiseMSE0.0211655 | 尚无最低信息解析噪声控制，不能据此证明学到有用生成分布 |

可证伪的主问题是：在相同21头、共同KL/训练噪声/预算且各自配套DDPM的系统中，CE相对raw-onehot MSE是否在预定全部重建与实际生成指标上呈现可描述改变；没有默认“CE更好”的结论。短过拟合若只降低PAD损失而非内容、出现nonfinite/梯度未到encoder或无法完成预算，应停止放大该候选，保留失败。训练/开发重建改善但生成分布仍混合时，不能把机制命名为表示已修复；DDPM拟合、posterior噪声、读出校准和优化充分性仍是开放项。任一实际结果不授权追加seed/挑checkpoint/换解码选赢家。

预算和选择机会建议待root锁：两个新VAE和AR各16 FREE+16 AMD，总96请求；一个训练seed、一个预定最终checkpoint/臂，不按开发指标挑选。两VAE共用训练初始化及posterior噪声；DDPM训练步序/noise机制相同但权重独立，生成可用共享噪声实现两系统的配对随机流。AR固定自己的随机流。相同terminal信息、请求数和模型选择机会可以匹配；AR与VAE/DDPM的参数量、训练更新含义和采样步数不同，应显式报参数/处理记录与token/实际秒数，不伪称同架构或同FLOPs。最终seed、更新数、wallcap与smoke阈值由新root协议事前冻结。

旧CPU1成本是测量而非新H5速度承诺：VAE1000步1184.698秒，DDPM1000步545.832秒，后验编码/设置10.417秒；H4的两terminal共1996 DDPM逆批及其他重放/读出总1126.198秒。若每个新臂仍1000+1000步并生成32条，两臂粗略外推约95.6分钟（不含AR、额外审计和环境变化）；这只是分配smoke/预算的依据，不能替代短吞吐测量。没有必要因这份估算自动增加步数。

现有源码可行性：FVAEDecoder已有out_channels参数，VAEModel(out_dim=21)即可构造21类输出，无需修改原类；decoder原squeeze(1)在21维时不删除类维，adapter明确接[B,21,48]并转[B,48,21]。候选显式继承encoder→samplez→decoder及flow密度计算，不调用原main/verify_code_copy/旧checkpoint读写；旧main固定ARCH为scalar且会训练/审计历史，因此不可直接复用入口。DDPM形状仍[2,16,48]，只在新后验数据上重建完整阶段。

当前candidate只有纯数学oracle、未执行的FVAE21适配器和29项stdlib/AST合成检查。没有真实模型构造、Torch导入、真实数据拟合、吞吐测量或生成。真实smoke需要独立审阅、root冻结协议与START。没有上传/push/提交/凭据/付费或生产访问；BACKUP_BLOCKED。
