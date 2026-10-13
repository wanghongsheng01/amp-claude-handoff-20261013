# AMP H5 正式有限 pilot

本协议连同 FORMAL_PROTOCOL.json、formal_configs、SAMPLING_SEEDS.json、FORMAL_PLAN.json 共同冻结；旧 DRAFT 是设计历史。唯一任务为抗菌肽 AMP。train279 用于拟合，dev41 已反复用于开发，test37 禁止访问。H0–H4、稿件与原数据保持只读。

四臂为预选 CE21、同容量 half raw-onehot MSE21、fresh scalar 原目标整管线及条件两层 LSTM。CE/MSE 全网各 1,681,549 参数，末层各357参数，共同初值、批次、随机流、结构、KL和J解码，只切重建目标：CE求和除B×48，MSE为0.5×全21坐标平方和除B×48；K为sum(logq−logp)/(B×48)，beta1，不截断负MC估计。固定同beta不等于相同有效正则，结论仅限此规范化和预算。

scalar 保留17参数末层、整数MSE全48格平方和/非PAD数、原K/16及firstPAD解码；它相对CE包含容量、尺度与读出差异，只作完整方法比较。三VAE各fresh训练、各自重编码train并fresh训练自己的DDPM。保留原latent16、flow、g/h训练条件、train-only scaler、abs及1000步warmup。不得接旧DDPM或其他臂权重。fresh scalar不是旧整次运行的按位重放。

LSTM固定2×64、token embedding32、terminal embedding8、62,501参数、Adam1e-3、clip1、dropout0。BOS严格右移，残基和首EOS计loss；48AA不加EOS49，生成到48强制终止并单列触顶。完整温度1采样，无top-k、拒采或补采。它是有论文机制依据的本地适配，不称官方忠实复现。AR训练充分性预先固定NOT_ESTABLISHED；PROMOTE固定HOLD，曲线只描述，不按事后阈值升级。

真实train-only smoke中，三VAE各20+20更新均finite完成，但6个生成请求均在exp(logsigma)溢出；LSTM100更新及2请求完成。完整smoke不通过。失败的后验参数x已保留；正式副本只增加数值摘要和异常阶段张量，不改模型、损失、随机调用或采样。失败原因未证，不能断言更多训练必修复。数学/接口预审通过、已有1000步架构执行先例及事前1000/1000计划支持继续有限风险实验；不复用smoke权重。

按预先公式和三臂真实串行吞吐冻结：各VAE1000更新，随后各自DDPM1000更新；AR1000更新，约111轮，有作者100轮示例依据。VAE阶段1920秒，DDPM含重编码1020秒，共同训练上限2940秒/方法；生成上限1800秒/方法。在途原子操作可结束且记录超限，不启动下一个更新。任何未到固定终点、数值错误或超预算使对应工程门失败，不续跑或重试，其他预定臂继续。

所有模型单CPU线程、串行执行。每100次VAE或AR更新做一次固定开发监测，共10点，计入训练时间；DDPM不做新增dev监测。无best-dev选择，不按曲线改变终点、模型或超参。各方法一套配置、一训练seed、一固定末点、零调参；历史开发选择机会另披露。共同时间上限是资源上界，实际计算量不同，必须报告成本—质量点，不能宣称equal compute。

初值、VAE批次与后验噪声seed222，DDPM322；生成FREE422、AMD423，各方法各末端16请求，合计128。不同模型同数字seed不是共享潜变量或对象配对。生成信息只含terminal及自身随机数/AR前缀；禁止真长度、源序列、g/h、参考肽或dev对象。支持0–48，空串、短串、重复、触顶、失败均保留，不补采。terminal标签不证明实物化学身份或抗菌活性。

工程E门核查实际源码/数据/配置身份、固定更新端点、finite与预算、全部128请求、生成信息边界和CE/MSE同初值/批次/RNG。任何对应工程失败，科学比较为NOT_EVALUABLE。L门仅dev posterior mean：CE相对MSE的rendered非PAD micro及受支持AA类macro均严格提高，长度MAE不升，整串exact数不降；另报每序列均值避免macro歧义。

G门预选CE对AR，分别FREE/AMD：双方全部16条合法1–48；CE的dev421特征biased-linear MMD²均严格降低；合法5–48数和合法unique数不降；同末端及任意train的exact-copy数不升；AA与dipeptide pooled entropy到同末端train参照的绝对差不增。JS/L1/KS/熵/最近邻等全字段同时报告。任何一项不满足即FAIL，不改阈值或换主候选。门通过也只是此单seed、16请求/末端、复用dev的经验结果，无统计显著性、活性、新颖性或投稿就绪含义。

不把生成最近邻当真值；重建变好但生成失败不能直接诊断latent OOD或collapse。旧湿实验N不猜，旧27肽不能验证新H5。尚无最强现代基线复现或独立确认，算法新颖性未证。早晨与用户共同确认前不得正式投稿。禁止上传/push、凭据、安全设置修改、付费调用或鉴权重试；BACKUP_BLOCKED。
