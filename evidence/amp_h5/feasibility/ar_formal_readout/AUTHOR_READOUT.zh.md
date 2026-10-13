# AMP H5：terminal LSTM 单臂描述性读出

已记录 fixed final 1000 更新、32 个生成请求（FREE/AMD 各16），全部原请求保留。该页只评价已完成的 AR；没有创建另外三臂观测，也没有方法比较。
**AR_ADEQUACY = NOT_ESTABLISHED；PROMOTE = HOLD；L/G = NOT_EVALUATED_NO_MATCHED_FORMAL_VAE_ARMS。**

| terminal | 请求 | 合法1–48 | 合法5–48 | unique合法1–48 | 同端基train-copy | 任意train-copy | 全请求平均长度 | train MMD² | dev MMD² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD | 16 | 16 | 16 | 16 | 0 | 0 | 13.937500 | 0.003158 | 0.007362 |
| FREE | 16 | 16 | 16 | 16 | 0 | 0 | 21.062500 | 0.004092 | 0.011567 |

MMD²使用冻结421特征，分布指标仅以合法1–48集合为分母；有效数与全请求分母一并报告。

| terminal | train AA-JS | dev AA-JS | train二肽JS | dev二肽JS | train长度KS | dev长度KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD | 0.021432 | 0.044849 | 0.244797 | 0.306125 | 0.182190 | 0.244565 |
| FREE | 0.028243 | 0.059628 | 0.244611 | 0.371616 | 0.133929 | 0.201389 |

全部AA/二肽L1、entropy、长度分布、同端基和任意train最近邻、逐样本合法性均保留于 GENERATION_METRICS.json / SAMPLE_AUDIT.json。无exact-copy只说明这32个输出中未观测到该事件，不证明不存在记忆。

| teacher-forced数据 | token分母 | NLL/token (nats) | teacher-forced token accuracy |
| --- | ---: | ---: | ---: |
| train | 5147 | 1.643124 | 0.491548 |
| development | 742 | 3.169793 | 0.203504 |

固定每100更新的全部十个监测点如下。仅描述已记录曲线，没有选择最优checkpoint或新增充分性阈值。

| step | train NLL/token | dev NLL/token | train teacher-forced accuracy | dev teacher-forced accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 2.583597 | 2.631441 | 0.195454 | 0.183288 |
| 200 | 2.458415 | 2.615253 | 0.229454 | 0.188679 |
| 300 | 2.343883 | 2.635472 | 0.271420 | 0.177898 |
| 400 | 2.218446 | 2.707511 | 0.308335 | 0.198113 |
| 500 | 2.105190 | 2.771109 | 0.346998 | 0.199461 |
| 600 | 2.000800 | 2.824545 | 0.377113 | 0.195418 |
| 700 | 1.905815 | 2.972322 | 0.391684 | 0.196765 |
| 800 | 1.805385 | 3.053019 | 0.432874 | 0.204852 |
| 900 | 1.712735 | 3.136477 | 0.472897 | 0.202156 |
| 1000 | 1.643124 | 3.169793 | 0.491548 | 0.203504 |

记录成本：训练pipeline 14.292998s；生成 0.020190s；方法进程 15.109932s。它们范围不同，不能相加声称独立外层墙钟或equal compute。

独立执行审计由另一代理进行，本页不代表其结论。评价复用冻结函数，仅打开已绑定train279/dev41；没有test37、模型调用、重采或原输出修改。

单seed、复用dev41和小样本不能证明AR已充分优化、强基线优势、显著性、抗菌活性、安全性或可投稿；旧湿实验不验证此新模型。BACKUP_BLOCKED：本地哈希不是远端备份。

完整指标：[GENERATION_METRICS.json](GENERATION_METRICS.json)；输入绑定：[SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)；输出清单：[MANIFEST.json](MANIFEST.json)。
