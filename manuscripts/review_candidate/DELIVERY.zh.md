# AMP 完整审阅候选交付

**本工作包已完成；正式投稿仍为 NO_GO。** 候选格式为 Bioinformatics Original Paper 的自由格式初投稿审阅布局，不是官方production模板，不声称录用概率。Communications Biology目前更缺乏对应的生物学推进证据。

最强且准确的结论：在固定开发实验中，CE21相对同容量MSE21改善了重建和六类低阶生成距离；但仍未超越已运行的terminal LSTM。这两层比较不能写成“重建与生成排名反转”。同类文献已有生成评测与重建/生成分离先例，本稿没有制造新模型创新。

交付：

- `build/main.pdf`：16页完整英文正文；Methods、Results、Discussion、limitations、AI与可用性声明完整。
- `build/supplement.pdf`：12页合并补充，完整距离表、执行成本、全部已完成负向诊断、独立MIC监督实验、原27肽完整表及冲突。
- `AMP_H5_Overleaf_review_candidate.zip`：13文件，自包含可编译源码；已在新目录解压并本地干净编译4份PDF。未上传Overleaf。
- `build/cover_letter.pdf`：1页投稿信草稿，明确未发送且未经作者确认；没有虚构独家投稿或全作者同意。
- `build/decision.pdf`：1页GO/NO_GO。
- `support/GUIDELINE_CHECKLIST.zh.md`：两期刊官方要求、可验证范围与访问限制。
- `support/REVIEW_FIXES.zh.md`：问题→证据→修改→复核。
- `independent_science/FINAL_REVIEW_v1.json` 与 `independent_venue/FINAL_REVIEW_v2.json`：未参与正文撰写的审核者终签与哈希。
- `evidence/`、`verify_packaged_numbers.py`：已保存结果的少量独立可复算材料；不是完整模型再训练发布包。

不存在以“未来补实验”代替的空白结果章节。作者贡献、经费关系、COI、资源重叠、完整AI史及最终真人审核仍明确未决，不能称最终投稿稿。

真正阻止提交的事项：局部个案相对已有评测的独特科学价值及适用范围、重复dev/单seed与基线充分性、完整数据/代码/权重的可复用发布安排；如恢复应用效力声称，还须历史模型—筛选—样品链和MIC语义。人工作者责任与声明不能由AI补造。原27肽没有绑定新H5模型。

检查边界：4份PDF终版编译无overfull、未定义引用或缺字，所有30页和主图灰度已检查。原surface脚本的Poppler bbox提取在3份PDF崩溃，失败保留，采用显式PyMuPDF词框替代后间距/字体/glyph检查通过。主稿AI披露机械检查仍为FAIL，因为脚本要求“The authors verified”而真人核验尚未完成；没有为通过检查伪造该句。完整投稿surface门不宣称PASS。

本轮没有新训练、自然采样、test37读取、原稿或P19改动、数据重新采集、上传、push或正式提交。原v6-r3及已交付研究快照不变。新成果仅本地保全，**BACKUP_BLOCKED**。无需提升到ultra解决当前问题；缺口来自证据与真人责任，不是增加推理额度即可填补。
