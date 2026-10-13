# AMP H5：有限端到端 pilot（预算待真实 smoke 后冻结）

任务固定为抗菌肽序列生成的本地方法实验。主候选 CE21、同容量 half raw-onehot MSE21 控制、fresh scalar 原目标整管线，以及 terminal-conditioned 2×64 LSTM。H0–H4/稿件冻结，test37不访问。仅train279拟合，反复使用的dev41只提供开发证据。旧湿实验不验证新模型。

CE/MSE完整初值相同，末层各357参数，全网各1,681,549参数；共同合法前缀J，exact tie取最短长度与最低AA编号；只切换重建目标。R_CE=sum CE/(B*48)；R_MSE=0.5*sum(raw−onehot)^2/(B*48)；K=sum(logq−logp)/(B*48)，beta=1，负MC估计不截断。保留原训练g/h、flow、latent16、结构、zscore后abs、warmup/优化器。相同beta不等于相同有效正则，不能推出普遍目标优劣。

fresh scalar保留17参数末层、原整数MSE（全48格平方和/非PAD数）、legacy K/16、firstPAD读出；它相对CE涉及容量/尺度/读出多项差异，仅作完整方法对照。三个VAE均fresh，并各自重编码train、fresh训练自己的terminal条件DDPM，绝不接旧或其他臂DDPM。

LSTM固定2×64、token embedding32、terminal embedding8、Adam1e−3、clip1、dropout0。BOS右移，训练只计残基与首EOS；长度48只计48个AA，生成定义到48强制结束。温度1、无topk/拒采。属于有论文机制依据的本地适配，并非官方忠实复现或已充分优化的最强基线。

正式更新候选为三个VAE各1000+各自DDPM1000；AR1000（约111epochs，有作者100epoch示例锚点）。正式时间上限和phase allocation待四臂真实串行smoke吞吐按预声明公式冻结。相同总训练上限给予共同资源上界；实际计算量不相等，报告成本—质量点，不声称equal compute。训练每100更新作固定监测，各方法10点；不改变任何训练、checkpoint或超参，无best-dev选择。AR若仍明显下降或已过拟合，充分性标不确定，正结果不足以宣称强基线优势。

初值/训练批次/后验噪声seed222，DDPM322；各方法生成FREE422、AMD423，每terminal16，共32请求/方法，总128。不同模型同数字seed不构成共享随机变量或对象配对。模型训练串行单CPU线程；smoke成本独立计入总研究成本。固定final endpoint，不续跑，不补采，不筛除短串/空串/重复。失败与not_executed均占完整attempt分母。

生成可用信息仅terminal Free/Amide和模型自身随机数/AR自身前缀，不含真长度、源序列、g/h、近邻模板。生成序列没有逐对象真值。条件元数据不保证实物末端化学身份或抗菌活性。

工程E门：完整step端点、finite检查、数据/源码/初值/RNG/预算绑定、全attempt保留及无越界条件全部成立。失败保留，科学门不可判。L门（CE对MSE）：dev posterior mean的rendered非PAD micro、target支持AA类macro均严格提高，长度MAE不升，整串exact数不降。另报每序列均值防止macro歧义。只锁mean，不添加posterior采样模式。

G门（预选CE对AR）：分别FREE/AMD，双方全部16条合法1–48；CE相对AR的dev421-feature biased-linear MMD²均严格降低；合法5–48数、合法unique数不降；同terminal及任意train exact-copy数不升；AA与dipeptide pooled entropy到同terminal train参照的绝对差均不增。JS/L1/KS/entropy/最近邻等全部字段同时报告。工程/完整性不通过时NOT_EVALUABLE；否则任何条件失败均FAIL，不改门。不把proxy门等同统计检验、活性、新颖性或可投稿。

只有一套事前配置/训练seed/终点；此前H0–H4造成的累计选择机会无法抹去，单独披露。若CE/MSE重建改善却未过G，不能把原因直接称为latent OOD或collapse。若不胜完整AR，则当前复杂路线的收益未证；仍不能证明所有VAE/diffusion无效。

禁止正式投稿，早晨由用户共同确认后另议。禁止上传/push、凭据、安全设置变更或付费调用；BACKUP_BLOCKED。
