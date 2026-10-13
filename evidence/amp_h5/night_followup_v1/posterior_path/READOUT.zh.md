# 已存后验随机重建与均值重建：训练 donor 配对

70.90% 是当前冻结CE checkpoint 的 **dev posterior mean（μ直接解码）**：497/701=70.8987%，41条。它不是随机后验重建率。

随机缓存只有source-v2 B的32个训练抽样槽（FREE16、AMD16），覆盖31个唯一donor。AMD row117抽中两次，两个epsilon槽和同一donor的mean预测均保留；没有去重改分母。全train279随机重建和dev41随机重建均未缓存。

| 端基 | 模式 | 正确/真实残基 | micro | supported-AA macro | 逐序列平均准确率 | 长度MAE | 全序列exact |
|---|---|---:|---:|---:|---:|---:|---:|
| FREE | M：已有μ解码 | 210/300 | 70.0000% | 44.6383% | 69.6170% | 0.0000 | 0/16 |
| FREE | S：已有单次后验draw | 195/300 | 65.0000% | 39.4864% | 64.4425% | 0.0625 | 0/16 |
| AMD | M：已有μ解码 | 177/213 | 83.0986% | 52.4506% | 84.7943% | 0.0625 | 4/16 |
| AMD | S：已有单次后验draw | 160/213 | 75.1174% | 43.1648% | 75.5089% | 0.1875 | 1/16 |

这里q是encoder输出的对角Gaussian，没有posterior flow；M=μ，S=μ+σepsilon，都直接经同一decoder/J。g经g_pre_net进入encoder，h也进入encoder；训练中prior_flow(z,g)只定义条件prior的坐标变换。decoder内部g=None，无teacher forcing。此比较是训练重建对后验抽样的敏感性，不能当dev泛化、独立重复、DDPM唯一瓶颈或新肽质量证据。

计分沿冻结reconstruction scorer：micro以真实非PAD残基为分母，缺失预测位置算错；macro按真实支持的AA类别等权recall；额外预测长度由长度误差与wholeexact反映。结果JSON保留全部32个配对原始差异。旧scorer硬编码mode标签仅另列为函数元数据，随机分支实际模式已明确标注。

本分析没有模型forward、随机采样或test读取，也不引入H0/H4旧checkpoint。dev随机缓存缺失，所以不计算dev的5百分点投入门。HOLD保持；后续补全R/M/S须另行锁定和独审。
