# AMP · Claude handoff

**状态：研究交接，科学 NO_GO；不是投稿包。** 原稿未改，未启动新实验。只涉及抗菌肽 AMP。

已完成有界训练、生成、MIC 预测与独立核验；尚未证明新方法优于强公平基线。27 肽湿实验存在，但历史生成—筛选—送样链未完全恢复，不能作为新 H5 方法的验证。

**最短阅读顺序（不要默认通读历史日志）：**

1. [给 Claude 的首轮提示（≤800字）](CLAUDE_HANDOFF_PROMPT.txt)
2. [START_HERE：问题、方法与证据边界](START_HERE.md)
3. [实验结论账本](docs/EXPERIMENT_LEDGER.md) → 只按需要进入所链接的日志
4. [下一步与停止标准](docs/NEXT_STEPS.md)

需要材料时查 [材料目录](MATERIALS_INDEX.md)、[未附清单](external_dependencies/NOT_INCLUDED.md)、[编译与复现入口](docs/REPRODUCIBILITY.md)。

- [当前审阅 PDF](manuscripts/review_candidate/pdfs/main.pdf) / [补充](manuscripts/review_candidate/pdfs/supplement.pdf) / [Overleaf 源码](manuscripts/review_candidate/source/)
- [最新期刊政策勘误](manuscripts/latest_policy/BIOINFORMATICS_POLICY_ERRATUM_v2.zh.md)：当前实质 AI 起草方式不符合 Bioinformatics；换刊不消除科学缺口。
- [最终200公共数据](data/fixed200/)：数据交付闭合；不是已通过污染隔离的新训练集。原生单位分开、不作摩尔换算。

仅用户控制的独立私有仓库交付，不直接向 Claude 发送。`MANIFEST_SHA256.txt` 核验本包内容；旧记录中的 BACKUP_BLOCKED 是当时权限状态，不撤销历史事实，也不覆盖此次明确的私有仓库交付授权。

[独立内容核查](verification/independent/PACKAGE_REVIEW.md)已完成；[5份稿件编译记录](verification/LATEX_COMPILATION.json)与[1699份来源复制核验](verification/SOURCE_COPY_VERIFICATION.json)可查。这些检查不改变科学 NO_GO。
