# 已保存指标的作者描述

128条来源观察由32条既有CE复用和96条新B/C/D请求组成。新96条全部完成、合法5–48、无空串、无48触顶；每个来源每端16条unique。独立工程后审通过仅指保存记录/路径/算术，不是独立模型重放或质量证明。

| 来源 | FREE dev MMD² | AMD dev MMD² | FREE/AMD train exact-copy | FREE/AMD 任意train最近邻≥70% |
| --- | ---: | ---: | --- | --- |
| CE复用 | 0.017327803379 | 0.007709821688 | 0/0 | 0/0 |
| B经验后验 | 0.017888733398 | 0.013236364505 | 0/1 | 7/14 |
| C逐坐标置换B-z | 0.023324997456 | 0.016634657842 | 0/0 | 0/0 |
| D reference-g native prior | 0.017108436019 | 0.012926119282 | 0/0 | 6/13 |

B的双端dev MMD均高于CE复用值；D的FREE略低而AMD较高。因此这组固定样本没有证明经验后验或reference-g替换能同时消除双端生成差距。B/D更接近训练序列的最近邻：AMD分别14/16和13/16达到70%identity，B还包含1条精确训练复制；这与它们使用训练后验或训练reference-g相符，不能作为新颖生成优势。

C逐坐标保留B实际采样z的经验边际，但双端dev MMD均高于B，长度分布也改变。这只能描述本固定训练经验样本对跨坐标组合的敏感性，不能证明DDPM是唯一因果瓶颈，不能推断显著性。C不存在单一训练donor或g目标。

B/D reference donor为FREE16个不同训练行、AMD15个不同训练行（重复1次保留）；共享epsilon、索引与重复样本不构成独立实验重复。D额外使用训练残基和长度模板reference-g，不是terminal-only，不参与c-only方法排名。所有B/C/D仅诊断来源，无新胜负门；旧CLI缺操作数的gate false明确为NOT_APPLICABLE_SOURCE_DIAGNOSTIC。

全部AA/二肽JS-L1-熵、长度KS、copy、最近邻、条件来源和逐样本信息见[完整作者读出](../ce_source_replacement_author_readout_v2/AUTHOR_READOUT.zh.md)。原四臂结果及原CE永久NE保持不变，PROMOTE HOLD；本件没有活性、安全性、湿实验前瞻、新颖性或投稿就绪结论。
