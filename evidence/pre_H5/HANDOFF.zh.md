# AMP 当前交接入口：v6-r3

**已完成本地可修项目与不同代理独审；科学、作者和目标刊门槛仍未通过，不能称可投稿。** 当前候选是v6-r3，旧v5/v6/v6-r2保持冻结，旧顶层入口另存于本检查点。

- [最终PDF，10页](editorial_corrections/candidate_v6r3/build/main.pdf)；[TeX](editorial_corrections/candidate_v6r3/source/main.tex)；[25文件冻结清单](editorial_corrections/frozen_candidate_manifest_v6r3.json)。PDF SHA-256：`83cc43957e078f4379e2a1029519b6df0a87ade1a61f5c3f37bf397096f904d8`。
- [完整交接、完成证据及六组最小依赖](review_checkpoint_v6r3_20261011/CHECKPOINT.zh.md)；[最终哈希核验](review_checkpoint_v6r3_20261011/FINAL_EVIDENCE.json)。
- [外部44+6项逐项回复](editorial_corrections/v6r3_checks/ISSUE_RESPONSE_FINAL.zh.md)，另列4个本轮内部发现与修复；不把条目数当独立缺陷数、实验数或计算关闭率。
- [不同代理最终页审](editorial_corrections/independent_review_v6r3/REVIEW.zh.md)：重新检查全部20页，限定页面/差异通过。
- [不同代理代码独审](editorial_corrections/independent_review_v6r2_code/REVIEW.zh.md)：29项提供测试、21项独立测试及新建toy微例通过，补丁4/4匹配、原代码90/90不变。只通过输入接口子项；没有Torch/模型运行。
- [实际隔离补丁与最小命令](editorial_corrections/v6r2_substantive/input_contract/README.zh.md)；[完整27×4 MIC表与逐格来源](editorial_corrections/v6r2_substantive/mic_full_table/HANDOFF.zh.md)。
- [指定独审任务接收材料](review_checkpoint_v6r3_20261011/REVIEW_HANDOFF.zh.md)：本地材料已准备；不声称已发送消息或获得该任务认可。

本轮实质完成：输入解析和dtype边界修复、新建明确toy接口例；全27个MIC候选ID/值/N及严格阈值分母；AMPGen/PRO-LDM机制定位；临床/性能越界表述撤回；无run/样本绑定的旧结构、重构和比较图表保存在历史存档并退出主文证据展示；AI与可用性如实披露。图、词距和PDF字符映射亦已修订并独审。

MIC中30个数字、78个N、16行全N；严格<32/<128/<200任一菌计数为6/27、7/27、7/27，S. aureus均6/27。两份相同工作簿不算独立重复，N不当阴性，27不是全生成池分母。pp1-7纯度80.3%满足原订单≥80%的规格，没有自动排除。

仍需真实run/config/log/权重绑定、完整数据/标签/独立split、1000→30→27筛选交接、MIC判定/聚合/原plate与版本变化、样品配液/批次映射，以及匹配baseline/控制/消融。作者贡献/经费/COI、完整AI史及真实人工核验、AMPGen等跨稿资源关系、目标刊/类型也未闭合。收窄主张和修复接口不能替代这些科学证据。

原surface脚本的词距、字体、glyph和日志阻塞项为零；**完整gate仍FAIL**，因为真实人工最终核验尚未完成。不同AI代理技术互查不是真人作者责任或期刊同行评审。

追索输入计数为14条检查记录、13个唯一路径：8个派生报告/索引/文本、5个DOCX/XLSX；不是14个独立实验。各子包测试/文件数另有自己的范围。

**BACKUP_BLOCKED**：用户禁止上传/push，未远端备份、正式提交、联系编辑、访问生产、读取凭据、付费调用或重试被拒认证。所有新稿、补丁与证据只有本地保存；哈希不等于独立备份。
